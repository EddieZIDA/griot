import ingest


def _text(n_paragraphs: int, words: int) -> str:
    return "\n".join(
        f"P{i} " + " ".join(["mot"] * (words - 1)) for i in range(n_paragraphs)
    )


def test_chunk_text_respecte_la_taille_cible():
    chunks = ingest.chunk_text(_text(8, 120), target_words=350)
    assert len(chunks) == 4
    assert all(len(c.split()) <= 350 for c in chunks)
    # Rien n'est perdu ni dupliqué.
    assert sum(len(c.split()) for c in chunks) == 8 * 120


def test_chunk_text_article_court_un_seul_chunk():
    assert ingest.chunk_text("Un seul paragraphe.") == ["Un seul paragraphe."]


def test_chunk_text_paragraphe_geant_non_coupe():
    chunks = ingest.chunk_text(_text(1, 900), target_words=350)
    assert len(chunks) == 1


def test_embedding_inputs_ajoute_titre_et_chevauchement():
    chunks = ["premier " * 100, "second " * 100]
    inputs = ingest.embedding_inputs("Mon titre", chunks)
    assert inputs[0].startswith("Mon titre\n\n")
    assert "[...]" not in inputs[0]
    assert inputs[1].startswith("Mon titre\n\n[...] premier")
    overlap = inputs[1].split("\n\n")[1]
    assert overlap.count("premier") == ingest.CHUNK_OVERLAP_WORDS
    assert inputs[1].endswith(chunks[1])


def test_embedding_inputs_sans_titre():
    assert ingest.embedding_inputs("", ["texte"]) == ["texte"]


def test_stable_id_deterministe():
    assert ingest.stable_id("https://a.b/c") == ingest.stable_id("https://a.b/c")
    assert ingest.stable_id("https://a.b/c") != ingest.stable_id("https://a.b/d")
    assert len(ingest.stable_id("x")) == 16


def test_load_sources_ignore_les_todo(tmp_path):
    path = tmp_path / "sources.csv"
    path.write_text(
        "pays,url\nMali,https://a/rss\nMali,TODO\nMonde,\n", encoding="utf-8"
    )
    assert ingest.load_sources(str(path)) == [
        {"pays": "Mali", "url": "https://a/rss"}
    ]


def test_store_article_classe_par_sujet(monkeypatch):
    """Les métadonnées portent le pays du sujet et celui du flux."""
    stored = {}

    class FakeCollection:
        def upsert(self, **kwargs):
            stored.update(kwargs)

    monkeypatch.setattr(
        ingest, "embed_documents", lambda client, texts: [[0.0]] * len(texts)
    )
    n = ingest.store_article(
        None,
        FakeCollection(),
        url="https://lefaso.net/a",
        titre="FIFA : Infantino sous pression",
        source="leFaso.net",
        pays="Burkina Faso",
        date_iso="2026-08-06T10:00:00+00:00",
        date_epoch=1786010400,
        text=_text(6, 120),
    )
    assert n == len(stored["ids"]) == len(stored["documents"]) == 3
    meta = stored["metadatas"]
    assert [m["chunk_index"] for m in meta] == [0, 1, 2]
    assert all(m["n_chunks"] == 3 for m in meta)
    assert all(m["pays"] == "monde" for m in meta)
    assert all(m["pays_flux"] == "burkina faso" for m in meta)
