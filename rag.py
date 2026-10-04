"""Recherche dans la base Chroma alimentée par ingest.py.

Deux modes, exposés au LLM par chat.py :

- `search_articles` : recherche sémantique sur un sujet précis.
- `latest_articles` : les articles les plus récents, sans requête, pour
  les questions de type "journal" ("que se passe-t-il au Mali ?"). Une
  recherche vectorielle sur le mot "actualité" ne ressemble à aucun
  article en particulier et renvoyait des résultats arbitraires.

Dans les deux cas le LLM reçoit le texte de l'article (reconstitué à
partir de ses chunks), pas un extrait de 500 caractères.
"""

import os
from functools import lru_cache

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

from utils import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    date_to_epoch,
    normalize_pays,
)

load_dotenv()

# Nombre d'articles renvoyés au LLM et taille maximale du texte de chacun.
# 6 x 4000 caractères, soit environ 6 000 tokens par appel d'outil : large
# pour Gemini Flash, et assez pour restituer contexte, chiffres et
# citations.
SEARCH_TOP_K = 6
SEARCH_MAX_CHARS = 4000
LATEST_TOP_K = 10
LATEST_MAX_CHARS = 1800
# Dans un journal, évite qu'un seul média occupe toute la sélection.
LATEST_MAX_PER_SOURCE = 4

# Filtre de pertinence. Les similarités cosinus de gemini-embedding sont
# tassées (un article hors sujet obtient facilement 0.60), donc le seuil
# absolu ne fait qu'écarter le bruit évident ; c'est surtout l'écart au
# meilleur résultat qui trie. Valeurs à recalibrer si les réponses
# manquent d'articles (baisser) ou en citent de hors sujet (monter).
MIN_SCORE = 0.50
SCORE_MARGIN = 0.12


@lru_cache(maxsize=1)
def get_genai_client() -> genai.Client:
    return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


@lru_cache(maxsize=1)
def get_collection():
    """Collection Chroma, ouverte au premier appel (pas à l'import).

    Permet d'importer ce module sans clé API ni base, et aux tests de
    substituer une collection temporaire.
    """
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    return chroma.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_query(text: str) -> list[float]:
    """Embed une requête utilisateur (task_type RETRIEVAL_QUERY)."""
    response = get_genai_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    return response.embeddings[0].values


def _build_where(
    pays: str | None,
    date_debut: str | None,
    date_fin: str | None,
    first_chunk_only: bool = False,
) -> dict | None:
    conditions: list[dict] = []
    if pays and normalize_pays(pays) not in ("tous", "tout"):
        conditions.append({"pays": normalize_pays(pays)})
    if date_debut:
        conditions.append(
            {"date_publication_ts": {"$gte": date_to_epoch(date_debut)}}
        )
    if date_fin:
        conditions.append(
            {
                "date_publication_ts": {
                    "$lte": date_to_epoch(date_fin, end_of_day=True)
                }
            }
        )
    if first_chunk_only:
        conditions.append({"chunk_index": 0})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def _article_texts(
    urls: list[str],
    max_chars: int,
    matched: dict[str, set[int]] | None = None,
) -> dict[str, str]:
    """Reconstitue le texte de chaque article à partir de ses chunks.

    Article court : texte intégral. Article plus long que max_chars : le
    début (chapeau, qui pose le sujet) puis les passages qui ont matché
    la requête, séparés par "[...]", dans la limite de max_chars.
    """
    if not urls:
        return {}
    rows = get_collection().get(
        where={"url": {"$in": urls}}, include=["documents", "metadatas"]
    )
    by_url: dict[str, dict[int, str]] = {}
    for document, metadata in zip(rows["documents"], rows["metadatas"]):
        index = int(metadata.get("chunk_index", 0))
        by_url.setdefault(metadata["url"], {})[index] = document

    texts: dict[str, str] = {}
    for url, chunks in by_url.items():
        ordered = sorted(chunks)
        full = "\n\n".join(chunks[i] for i in ordered)
        if len(full) <= max_chars:
            texts[url] = full
            continue
        wanted = sorted({ordered[0]} | ((matched or {}).get(url) or set()))
        parts: list[str] = []
        previous = None
        for index in wanted:
            if index not in chunks:
                continue
            if previous is not None and index != previous + 1:
                parts.append("[...]")
            parts.append(chunks[index])
            previous = index
        text = "\n\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + " [...]"
        texts[url] = text
    return texts


def _to_article(metadata: dict, contenu: str, score: float | None) -> dict:
    article = {
        "titre": metadata.get("titre", ""),
        "source": metadata.get("source", ""),
        "pays": metadata.get("pays", ""),
        "date": (metadata.get("date_publication") or "")[:10],
        "url": metadata.get("url", ""),
        "contenu": contenu,
    }
    if score is not None:
        article["score"] = score
    return article


def search_articles(
    query: str,
    pays: str | None = None,
    date_debut: str | None = None,
    date_fin: str | None = None,
    top_k: int = SEARCH_TOP_K,
) -> list[dict]:
    """Articles les plus pertinents pour `query`, du plus au moins proche.

    Les erreurs (quota d'embedding, base inaccessible) ne sont pas
    avalées : une liste vide signifierait "aucun article", ce qui est
    faux et ferait répondre à Griot qu'il n'a rien sur le sujet.
    """
    where = _build_where(pays, date_debut, date_fin)
    results = get_collection().query(
        query_embeddings=[embed_query(query)],
        # Large : plusieurs chunks du même article remontent ensemble.
        n_results=min(top_k * 5, 50),
        where=where,
        include=["metadatas", "distances"],
    )
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    if not metadatas:
        return []

    best_score = 1 - distances[0]
    threshold = max(MIN_SCORE, best_score - SCORE_MARGIN)

    # Regroupe les chunks par article ; le score d'un article est celui
    # de son meilleur chunk (le premier rencontré, résultats déjà triés).
    selected: dict[str, tuple[dict, float]] = {}
    matched: dict[str, set[int]] = {}
    for metadata, distance in zip(metadatas, distances):
        score = 1 - distance
        url = metadata.get("url", "")
        if not url or score < threshold:
            continue
        if url not in selected:
            if len(selected) >= top_k:
                continue
            selected[url] = (metadata, round(score, 4))
        matched.setdefault(url, set()).add(int(metadata.get("chunk_index", 0)))

    texts = _article_texts(list(selected), SEARCH_MAX_CHARS, matched)
    return [
        _to_article(metadata, texts.get(url, ""), score)
        for url, (metadata, score) in selected.items()
    ]


def latest_articles(
    pays: str | None = None,
    date_debut: str | None = None,
    date_fin: str | None = None,
    top_k: int = LATEST_TOP_K,
) -> list[dict]:
    """Articles les plus récents (du plus récent au plus ancien).

    Sans période, renvoie les derniers articles disponibles même s'ils
    datent : le champ `date` permet au LLM de le signaler honnêtement
    plutôt que de présenter une actualité ancienne comme celle du jour.
    """
    where = _build_where(pays, date_debut, date_fin, first_chunk_only=True)
    rows = get_collection().get(where=where, include=["metadatas"])
    metadatas = sorted(
        rows["metadatas"],
        key=lambda m: m.get("date_publication_ts", 0),
        reverse=True,
    )

    selected: list[dict] = []
    per_source: dict[str, int] = {}
    for metadata in metadatas:
        source = metadata.get("source", "")
        if per_source.get(source, 0) >= LATEST_MAX_PER_SOURCE:
            continue
        per_source[source] = per_source.get(source, 0) + 1
        selected.append(metadata)
        if len(selected) >= top_k:
            break

    texts = _article_texts([m["url"] for m in selected], LATEST_MAX_CHARS)
    return [_to_article(m, texts.get(m["url"], ""), None) for m in selected]
