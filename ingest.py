"""Pipeline d'ingestion : flux RSS -> texte complet -> embeddings -> Chroma.

Usage :
    python ingest.py            # ingère les nouveaux articles des flux
    python ingest.py --reindex  # convertit l'ancienne collection (v1) vers
                                # la nouvelle (v2), sans rien retélécharger
    python ingest.py --reclassify  # recalcule le pays de chaque article
                                   # déjà ingéré d'après son sujet (aucun
                                   # appel à l'API, instantané)

Les deux commandes sont relançables sans doublon : un article déjà
présent dans la collection est ignoré. Si le quota Gemini est atteint en
cours de route, relancer plus tard reprend là où le traitement s'est
arrêté.
"""

import csv
import hashlib
import os
import re
import sys
import time
import urllib.request
from urllib.parse import quote

import chromadb
import feedparser
import trafilatura
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from utils import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    LEGACY_COLLECTION_NAME,
    classify_pays,
    clean_text,
    is_press_review,
    normalize_pays,
    parse_date,
    source_name,
)

load_dotenv()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SOURCES_PATH = "config/sources.csv"
CHUNK_TARGET_WORDS = 350
# Fin du chunk précédent ajoutée au texte vectorisé (pas au texte stocké)
# pour qu'une phrase à cheval sur deux chunks reste retrouvable.
CHUNK_OVERLAP_WORDS = 50
MIN_ARTICLE_WORDS = 50
MAX_EMBED_RETRIES = 5
EMBED_RETRY_BASE_DELAY = 15  # secondes, doublé à chaque tentative
HTTP_TIMEOUT = 20
# Plusieurs sites (RFI, France 24...) refusent l'agent par défaut de
# Python/feedparser et renvoient un flux vide ou une erreur 403.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def load_sources(path: str) -> list[dict]:
    """Charge les flux depuis le CSV (colonnes : pays, url)."""
    with open(path, encoding="utf-8") as f:
        return [
            row
            for row in csv.DictReader(f)
            if row.get("url") and row["url"].strip().upper() != "TODO"
        ]


def stable_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def http_get(url: str) -> bytes:
    """GET avec un User-Agent de navigateur.

    L'URL est encodée : "rfi.fr/fr/tag/sénégal/rss" contient des accents
    qu'urllib refuse tels quels.
    """
    safe_url = quote(url.strip(), safe=":/?&=%#+,;@~!$'()*")
    request = urllib.request.Request(
        safe_url, headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
        return response.read()


def fetch_feed(url: str):
    """Télécharge puis parse un flux. Lève une erreur s'il est inutilisable."""
    feed = feedparser.parse(http_get(url))
    if not feed.entries:
        reason = feed.bozo_exception if feed.bozo else "aucune entrée"
        raise RuntimeError(str(reason))
    return feed


def fetch_article_text(url: str) -> str | None:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        # Second essai avec un User-Agent de navigateur.
        try:
            downloaded = http_get(url).decode("utf-8", errors="replace")
        except Exception:
            return None
    return trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )


def feed_entry_text(entry) -> str | None:
    """Texte fourni par le flux lui-même (content:encoded ou description).

    Sert de repli quand la page de l'article est inaccessible : mieux
    vaut le résumé du flux que pas d'article du tout.
    """
    candidates = [c.get("value", "") for c in entry.get("content", [])]
    candidates.append(entry.get("summary", ""))
    raw = max(candidates, key=len, default="")
    if not raw:
        return None
    extracted = trafilatura.extract(f"<html><body>{raw}</body></html>")
    return extracted or clean_text(re.sub(r"<[^>]+>", " ", raw))


def chunk_text(
    text: str, target_words: int = CHUNK_TARGET_WORDS
) -> list[str]:
    """Accumule les paragraphes jusqu'à ~target_words par chunk."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for paragraph in paragraphs:
        words = len(paragraph.split())
        if current and current_words + words > target_words:
            chunks.append("\n\n".join(current))
            current, current_words = [], 0
        current.append(paragraph)
        current_words += words
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def embedding_inputs(titre: str, chunks: list[str]) -> list[str]:
    """Texte réellement vectorisé pour chaque chunk.

    Titre de l'article + fin du chunk précédent + chunk. Sans le titre,
    un chunk de milieu d'article ("Il a ajouté que...") n'a plus de
    sujet et ne remonte sur aucune requête. Le texte stocké dans Chroma
    reste le chunk seul, pour pouvoir reconstituer l'article sans
    répétitions.
    """
    inputs = []
    for i, chunk in enumerate(chunks):
        parts = [titre] if titre else []
        if i > 0:
            tail = chunks[i - 1].split()[-CHUNK_OVERLAP_WORDS:]
            parts.append("[...] " + " ".join(tail))
        parts.append(chunk)
        inputs.append("\n\n".join(parts))
    return inputs


def embed_documents(
    client: genai.Client, texts: list[str]
) -> list[list[float]]:
    """Embed des textes, avec retry/backoff sur quota Gemini dépassé (429)."""
    for attempt in range(MAX_EMBED_RETRIES):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=EMBEDDING_DIM,
                ),
            )
            return [e.values for e in response.embeddings]
        except errors.ClientError as e:
            is_last_attempt = attempt == MAX_EMBED_RETRIES - 1
            if e.code != 429 or is_last_attempt:
                raise
            delay = EMBED_RETRY_BASE_DELAY * (2**attempt)
            print(f"  [QUOTA] 429, nouvelle tentative dans {delay}s...")
            time.sleep(delay)
    raise RuntimeError("embed_documents : nombre de tentatives épuisé")


def is_quota_error(error: Exception) -> bool:
    return isinstance(error, errors.ClientError) and error.code == 429


def already_ingested(collection, url: str) -> bool:
    existing = collection.get(where={"url": url}, limit=1, include=[])
    return len(existing.get("ids", [])) > 0


def store_article(
    client: genai.Client,
    collection,
    *,
    url: str,
    titre: str,
    source: str,
    pays: str,
    date_iso: str,
    date_epoch: int,
    text: str,
) -> int:
    """Découpe, vectorise et enregistre un article. Renvoie le nb de chunks.

    `pays` est le pays du flux ; le pays enregistré est celui du sujet
    (voir utils.classify_pays). Le pays du flux est conservé dans
    `pays_flux` pour pouvoir reclasser plus tard si les règles changent.
    """
    chunks = chunk_text(text)
    pays_flux = normalize_pays(pays)
    pays_sujet = classify_pays(pays_flux, titre, text, url)
    embeddings = embed_documents(client, embedding_inputs(titre, chunks))
    base_id = stable_id(url)
    collection.upsert(
        ids=[f"{base_id}_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks,
        metadatas=[
            {
                "pays": pays_sujet,
                "pays_flux": pays_flux,
                "source": source,
                "titre": titre,
                "url": url,
                "date_publication": date_iso,
                "date_publication_ts": date_epoch,
                "chunk_index": i,
                "n_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ],
    )
    return len(chunks)


def open_collection(name: str = COLLECTION_NAME):
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    return chroma, chroma.get_or_create_collection(
        name, metadata={"hnsw:space": "cosine"}
    )


def run_ingestion() -> None:
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    _, collection = open_collection()
    sources = load_sources(SOURCES_PATH)
    print(f"{len(sources)} flux à traiter\n")

    report: list[tuple[str, str, str]] = []
    totals = {"nouveaux": 0, "deja": 0, "ignores": 0, "erreurs": 0}
    quota_exhausted = False

    for source in sources:
        if quota_exhausted:
            break
        feed_url = source["url"].strip()
        pays = source["pays"].strip()
        try:
            feed = fetch_feed(feed_url)
        except Exception as e:
            print(f"[ERREUR flux] {feed_url}: {e}")
            report.append((pays, feed_url, f"ERREUR : {e}"))
            continue

        feed_title = feed.feed.get("title", "")
        counts = {"nouveaux": 0, "deja": 0, "ignores": 0, "erreurs": 0}

        for entry in feed.entries:
            url = entry.get("link")
            if not url:
                continue
            titre = clean_text(entry.get("title", ""))
            if is_press_review(titre):
                counts["ignores"] += 1
                continue
            if already_ingested(collection, url):
                counts["deja"] += 1
                continue
            try:
                text = fetch_article_text(url)
                if not text or len(text.split()) < MIN_ARTICLE_WORDS:
                    fallback = feed_entry_text(entry)
                    if fallback and len(fallback.split()) > len(
                        (text or "").split()
                    ):
                        text = fallback
                if not text or len(text.split()) < MIN_ARTICLE_WORDS:
                    # Trop court pour être utile (ex. article réduit à
                    # une vidéo intégrée) : pas un échec technique.
                    counts["ignores"] += 1
                    continue
                raw_date = entry.get("published") or entry.get("updated", "")
                date_iso, date_epoch = parse_date(raw_date)
                store_article(
                    client,
                    collection,
                    url=url,
                    titre=titre,
                    source=source_name(url, feed_title),
                    pays=pays,
                    date_iso=date_iso,
                    date_epoch=date_epoch,
                    text=text,
                )
                counts["nouveaux"] += 1
            except Exception as e:
                print(f"  [ERREUR article] {url}: {e}")
                counts["erreurs"] += 1
                if is_quota_error(e):
                    # Inutile d'insister : chaque article suivant
                    # échouerait après plusieurs minutes d'attente.
                    print(
                        "Quota Gemini épuisé. Relancer `python ingest.py` "
                        "plus tard : les articles déjà ingérés seront "
                        "ignorés."
                    )
                    quota_exhausted = True
                    break

        for key in totals:
            totals[key] += counts[key]
        report.append(
            (
                pays,
                feed_url,
                f"{counts['nouveaux']} nouveaux, {counts['deja']} déjà "
                f"présents, {counts['ignores']} ignorés, "
                f"{counts['erreurs']} en erreur",
            )
        )

    # Rapport flux par flux : un flux qui ne rapporte jamais rien (erreur
    # ou 0 nouveau à chaque passage) doit être corrigé ou retiré de
    # config/sources.csv, sinon le pays concerné s'appauvrit en silence.
    print("\n--- Rapport par flux ---")
    for pays, feed_url, status in report:
        print(f"[{pays}] {feed_url}\n    {status}")
    print(
        "\n--- Total ---\n"
        f"Articles ingérés         : {totals['nouveaux']}\n"
        f"Articles déjà présents   : {totals['deja']}\n"
        f"Articles ignorés         : {totals['ignores']} "
        "(trop courts ou revues de presse)\n"
        f"Articles en erreur       : {totals['erreurs']}\n"
        f"Chunks dans la collection: {collection.count()}"
    )


def run_reindex() -> None:
    """Convertit l'ancienne collection (v1) vers la nouvelle (v2).

    Les flux RSS ne contiennent que les articles récents : repartir de
    zéro ferait perdre l'archive déjà constituée. Le texte des articles
    est donc relu depuis l'ancienne collection, puis redécoupé et
    revectorisé au nouveau format. L'ancienne collection n'est pas
    supprimée.
    """
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    chroma, collection = open_collection()
    try:
        legacy = chroma.get_collection(LEGACY_COLLECTION_NAME)
    except Exception:
        print(f"Aucune collection {LEGACY_COLLECTION_NAME!r} à convertir.")
        return

    rows = legacy.get(include=["documents", "metadatas"])
    articles: dict[str, dict] = {}
    for _id, document, metadata in zip(
        rows["ids"], rows["documents"], rows["metadatas"]
    ):
        url = metadata.get("url", "")
        if not url:
            continue
        index = int(_id.rsplit("_", 1)[-1])
        article = articles.setdefault(url, {"meta": metadata, "chunks": {}})
        article["chunks"][index] = document

    print(f"{len(articles)} articles dans l'ancienne collection\n")
    done = skipped = present = failed = 0
    for url, article in articles.items():
        meta = article["meta"]
        titre = clean_text(meta.get("titre", ""))
        if is_press_review(titre):
            skipped += 1
            continue
        if already_ingested(collection, url):
            present += 1
            continue
        chunks = article["chunks"]
        text = "\n".join(chunks[i] for i in sorted(chunks))
        try:
            store_article(
                client,
                collection,
                url=url,
                titre=titre,
                source=source_name(url, meta.get("source", "")),
                pays=meta.get("pays_flux") or meta.get("pays", ""),
                date_iso=meta.get("date_publication", ""),
                date_epoch=int(meta.get("date_publication_ts", 0)),
                text=text,
            )
            done += 1
            if done % 25 == 0:
                print(f"  {done} articles convertis...")
        except Exception as e:
            print(f"[ERREUR] {url}: {e}")
            failed += 1
            if is_quota_error(e):
                print(
                    "Quota Gemini épuisé. Relancer `python ingest.py "
                    "--reindex` plus tard : la conversion reprendra ici."
                )
                break

    print(
        "\n--- Résumé réindexation ---\n"
        f"Articles convertis          : {done}\n"
        f"Déjà convertis              : {present}\n"
        f"Revues de presse écartées   : {skipped}\n"
        f"En erreur                   : {failed}\n"
        f"Chunks dans la collection v2: {collection.count()}"
    )


def run_reclassify() -> None:
    """Recalcule le pays (par sujet) de tous les articles déjà ingérés.

    Ne touche qu'aux métadonnées : aucun embedding recalculé, aucun
    appel à l'API. À relancer après une modification de PAYS_MARKERS.
    """
    _, collection = open_collection()
    rows = collection.get(include=["documents", "metadatas"])
    articles: dict[str, list[tuple[str, str, dict]]] = {}
    for _id, document, metadata in zip(
        rows["ids"], rows["documents"], rows["metadatas"]
    ):
        articles.setdefault(metadata.get("url", ""), []).append(
            (_id, document, metadata)
        )

    moves: dict[tuple[str, str], int] = {}
    for chunks in articles.values():
        chunks.sort(key=lambda c: int(c[2].get("chunk_index", 0)))
        meta = chunks[0][2]
        pays_flux = meta.get("pays_flux") or meta.get("pays", "")
        text = "\n".join(document for _, document, _ in chunks)
        pays_sujet = classify_pays(
            pays_flux, meta.get("titre", ""), text, meta.get("url", "")
        )
        if meta.get("pays") == pays_sujet and meta.get("pays_flux"):
            continue
        collection.update(
            ids=[_id for _id, _, _ in chunks],
            metadatas=[
                {**m, "pays": pays_sujet, "pays_flux": pays_flux}
                for _, _, m in chunks
            ],
        )
        if meta.get("pays") != pays_sujet:
            key = (meta.get("pays", ""), pays_sujet)
            moves[key] = moves.get(key, 0) + 1

    print(f"{len(articles)} articles examinés")
    if not moves:
        print("Aucun article reclassé.")
    for (before, after), count in sorted(moves.items()):
        print(f"  {before} -> {after} : {count}")


if __name__ == "__main__":
    if "--reindex" in sys.argv[1:]:
        run_reindex()
    elif "--reclassify" in sys.argv[1:]:
        run_reclassify()
    else:
        run_ingestion()
