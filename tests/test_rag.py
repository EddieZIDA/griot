"""Tests de la récupération sur une vraie collection Chroma temporaire.

Les embeddings sont des vecteurs 2D fabriqués à la main : la requête
vaut toujours [1, 0], et chaque chunk est placé à une similarité cosinus
choisie. Aucun appel à l'API Gemini.
"""

import math

import chromadb
import pytest

import rag

DAY = 86400
T0 = 1785974400  # 2026-08-06 00:00:00 UTC


def _vector(similarity: float) -> list[float]:
    return [similarity, math.sqrt(1 - similarity**2)]


@pytest.fixture
def collection(tmp_path, monkeypatch):
    client = chromadb.PersistentClient(path=str(tmp_path))
    col = client.get_or_create_collection(
        "test", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_collection", lambda: col)
    monkeypatch.setattr(rag, "embed_query", lambda text: [1.0, 0.0])
    return col


def _add(col, url, pays, source, day, chunks):
    """chunks : liste de (texte, similarité avec la requête)."""
    col.add(
        ids=[f"{url}_{i}" for i in range(len(chunks))],
        embeddings=[_vector(sim) for _, sim in chunks],
        documents=[text for text, _ in chunks],
        metadatas=[
            {
                "url": url,
                "pays": pays,
                "source": source,
                "titre": f"Titre {url}",
                "date_publication": f"2026-08-{6 + day:02d}T10:00:00+00:00",
                "date_publication_ts": T0 + day * DAY + 36000,
                "chunk_index": i,
                "n_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ],
    )


def test_search_renvoie_le_texte_complet_d_un_article_court(collection):
    _add(collection, "a", "mali", "RFI", 0, [("début", 0.9), ("suite", 0.5)])
    (article,) = rag.search_articles("q", "Mali")
    assert article["contenu"] == "début\n\nsuite"
    assert article["score"] == pytest.approx(0.9, abs=1e-3)
    assert article["date"] == "2026-08-06"


def test_search_dedoublonne_par_article_et_trie_par_score(collection):
    _add(collection, "a", "mali", "RFI", 0, [("a0", 0.80), ("a1", 0.85)])
    _add(collection, "b", "mali", "RFI", 0, [("b0", 0.90)])
    urls = [x["url"] for x in rag.search_articles("q", "Mali")]
    assert urls == ["b", "a"]


def test_search_ecarte_les_resultats_trop_loin_du_meilleur(collection):
    _add(collection, "bon", "mali", "RFI", 0, [("x", 0.90)])
    _add(collection, "moyen", "mali", "RFI", 0, [("x", 0.80)])
    _add(collection, "loin", "mali", "RFI", 0, [("x", 0.70)])
    urls = [x["url"] for x in rag.search_articles("q", "Mali")]
    assert urls == ["bon", "moyen"]


def test_search_sous_le_seuil_absolu_renvoie_vide(collection):
    _add(collection, "a", "mali", "RFI", 0, [("x", 0.30)])
    assert rag.search_articles("q", "Mali") == []


def test_search_filtre_par_pays_et_sans_pays(collection):
    _add(collection, "m", "mali", "RFI", 0, [("x", 0.9)])
    _add(collection, "s", "senegal", "RFI", 0, [("x", 0.9)])
    assert [x["url"] for x in rag.search_articles("q", "Sénégal")] == ["s"]
    assert {x["url"] for x in rag.search_articles("q")} == {"m", "s"}


def test_search_filtre_par_date(collection):
    _add(collection, "j0", "mali", "RFI", 0, [("x", 0.9)])
    _add(collection, "j2", "mali", "RFI", 2, [("x", 0.9)])
    found = rag.search_articles("q", "Mali", "2026-08-08", "2026-08-08")
    assert [x["url"] for x in found] == ["j2"]


def test_search_article_long_garde_le_debut_et_le_passage_trouve(collection):
    big = "mot " * 600  # 2400 caractères par chunk
    _add(
        collection,
        "long",
        "mali",
        "RFI",
        0,
        [("DEBUT " + big, 0.6), ("MILIEU " + big, 0.2), ("FIN " + big, 0.9)],
    )
    (article,) = rag.search_articles("q", "Mali")
    contenu = article["contenu"]
    assert contenu.startswith("DEBUT")
    assert "FIN" in contenu and "MILIEU" not in contenu
    assert "[...]" in contenu
    assert len(contenu) <= rag.SEARCH_MAX_CHARS + 10


def test_search_date_invalide_leve_une_erreur(collection):
    with pytest.raises(ValueError, match="AAAA-MM-JJ"):
        rag.search_articles("q", "Mali", "hier")


def test_latest_trie_par_date_decroissante(collection):
    _add(collection, "vieux", "mali", "RFI", 0, [("x", 0.1)])
    _add(collection, "recent", "mali", "Maliweb", 3, [("x", 0.1), ("y", 0.1)])
    _add(collection, "autre", "senegal", "RFI", 5, [("x", 0.1)])
    articles = rag.latest_articles("Mali")
    assert [a["url"] for a in articles] == ["recent", "vieux"]
    assert "score" not in articles[0]
    assert articles[0]["contenu"] == "x\n\ny"
    assert [a["url"] for a in rag.latest_articles()][0] == "autre"


def test_latest_limite_le_nombre_par_source(collection):
    for i in range(6):
        _add(collection, f"rfi{i}", "mali", "RFI", i, [("x", 0.1)])
    _add(collection, "mw", "mali", "Maliweb", 0, [("x", 0.1)])
    sources = [a["source"] for a in rag.latest_articles("Mali")]
    assert sources.count("RFI") == rag.LATEST_MAX_PER_SOURCE
    assert "Maliweb" in sources


def test_latest_filtre_par_periode(collection):
    _add(collection, "j0", "mali", "RFI", 0, [("x", 0.1)])
    _add(collection, "j1", "mali", "RFI", 1, [("x", 0.1)])
    found = rag.latest_articles("Mali", "2026-08-07", "2026-08-07")
    assert [a["url"] for a in found] == ["j1"]


def test_base_vide(collection):
    assert rag.search_articles("q", "Mali") == []
    assert rag.latest_articles("Mali") == []
