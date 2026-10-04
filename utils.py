"""Utilitaires et constantes partagés par ingest.py, rag.py et chat.py."""

import html
import re
import unicodedata
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

# --- Constantes communes à l'ingestion et à la recherche ---
#
# Une seule définition ici plutôt qu'une copie dans ingest.py et une autre
# dans rag.py : si le modèle ou la dimension divergent entre les deux, la
# recherche renvoie du bruit sans aucune erreur visible.
CHROMA_DIR = "chroma_db"
# v2 : les chunks sont vectorisés avec le titre de l'article et portent
# chunk_index / n_chunks. L'ancienne collection "griot_news" n'a pas ces
# champs ; `python ingest.py --reindex` la convertit sans rien retélécharger.
COLLECTION_NAME = "griot_news_v2"
LEGACY_COLLECTION_NAME = "griot_news"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768

PAYS_COUVERTS = ["Burkina Faso", "Senegal", "Mali", "Côte d'Ivoire", "Monde"]

# Nom lisible d'un média à partir du domaine de l'article. Les titres de
# flux RSS sont souvent inutilisables tels quels ("Politique Archives -
# Burkina24.com - Actualité du Burkina Faso 24h/24", "Actualités
# intelligence-artificielle"...).
SOURCE_NAMES = {
    "burkina24.com": "Burkina24",
    "lefaso.net": "leFaso.net",
    "sidwaya.info": "Sidwaya",
    "lepays.bf": "Le Pays",
    "rfi.fr": "RFI",
    "allafrica.com": "AllAfrica",
    "france24.com": "France 24",
    "africanews.com": "Africanews",
    "bbc.com": "BBC Afrique",
    "bbc.co.uk": "BBC Afrique",
    "maliweb.net": "Maliweb",
    "abidjan.net": "Abidjan.net",
    "actuia.com": "ActuIA",
    "lemondeinformatique.fr": "Le Monde Informatique",
    "01net.com": "01net",
    "clubic.com": "Clubic",
}

# Marqueurs (sans accents, en minuscules) signalant qu'un texte parle d'un
# pays couvert : nom, gentilé, grandes villes. Sert à classer un article
# selon son SUJET et non selon le flux d'où il vient. Liste volontairement
# courte et sans termes ambigus ; à compléter au besoin.
PAYS_MARKERS = {
    "burkina faso": [
        "burkina", "faso", r"burkinabe\w*", "ouagadougou", "ouaga",
        "bobo-dioulasso", "bobo dioulasso", "koudougou", "ouahigouya",
        "banfora", "kaya", "tenkodogo", "fada n'gourma",
    ],
    "senegal": [
        "senegal", r"senegalais\w*", "dakar", "thies", "ziguinchor",
        "casamance", "touba", "kaolack",
    ],
    "mali": [
        "mali", r"malien\w*", "bamako", "tombouctou", "gao", "kidal",
        "mopti", "segou", "sikasso", "kayes",
    ],
    "cote d'ivoire": [
        "cote d'ivoire", r"ivoirien\w*", "abidjan", "yamoussoukro",
        "bouake", "san-pedro", "korhogo",
    ],
}
# Sources dont les flux par pays sont déjà triés par sujet (rubrique
# "Sénégal" d'AllAfrica, tag "mali" de RFI...). Leurs articles locaux ne
# nomment pas toujours le pays ("Kolda : 44 villages réclament
# l'électricité") : on fait confiance au flux au lieu de les envoyer dans
# "monde". Les journaux nationaux généralistes (leFaso.net, Le Pays...)
# ne sont pas dans cette liste car ils couvrent aussi l'étranger.
FLUX_PAR_SUJET = ("allafrica.com", "rfi.fr")

_MARKER_RE = {
    pays: re.compile(r"(?<!\w)(?:" + "|".join(markers) + r")(?!\w)")
    for pays, markers in PAYS_MARKERS.items()
}

JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


def normalize_pays(pays: str) -> str:
    """Normalise une valeur de pays : minuscules, sans accents.

    Fait correspondre "Sénégal" et "Senegal" quelle que soit la graphie
    utilisée dans le CSV, le system prompt ou la question de l'utilisateur.
    """
    normalized = unicodedata.normalize("NFKD", pays.strip().lower())
    return "".join(c for c in normalized if not unicodedata.combining(c))


def mentioned_pays(text: str) -> set[str]:
    """Pays couverts mentionnés dans un texte (clés normalisées)."""
    normalized = normalize_pays(text or "").replace("\u2019", "'")
    return {p for p, regex in _MARKER_RE.items() if regex.search(normalized)}


def _domain_in(url: str, suffixes: tuple[str, ...]) -> bool:
    domain = urlparse(url).netloc.lower()
    return any(domain == s or domain.endswith("." + s) for s in suffixes)


def classify_pays(pays_flux: str, titre: str, text: str, url: str = "") -> str:
    """Pays d'un article d'après son sujet, et non d'après son flux.

    Deux règles simples, sans appel LLM :

    1. Le titre nomme exactement un pays couvert : c'est le sujet
       ("Mali : un journaliste condamné" dans un journal burkinabè va
       dans "mali").
    2. Sinon, un article d'un flux national qui ne mentionne nulle part
       ce pays (ni nom, ni gentilé, ni grande ville) traite d'un sujet
       étranger : il va dans "monde" (ex. un éditorial du Pays sur la
       FIFA). Règle non appliquée aux sources de FLUX_PAR_SUJET.

    Dans tous les autres cas, le pays du flux est conservé. Une simple
    mention en passant suffit à garder un article dans son pays : la
    règle préfère rater un reclassement que d'en faire un faux.
    """
    flux = normalize_pays(pays_flux)
    in_title = mentioned_pays(titre)
    if len(in_title) == 1:
        return next(iter(in_title))
    if (
        flux in PAYS_MARKERS
        and not _domain_in(url, FLUX_PAR_SUJET)
        and flux not in mentioned_pays(f"{titre}\n{text}")
    ):
        return "monde"
    return flux


def clean_text(text: str) -> str:
    """Décode les entités HTML ("ao&#xfb;t" -> "août") et resserre les espaces."""
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def source_name(article_url: str, fallback: str = "") -> str:
    """Nom lisible du média : table SOURCE_NAMES, sinon repli, sinon domaine."""
    domain = urlparse(article_url).netloc.lower().removeprefix("www.")
    for suffix, name in SOURCE_NAMES.items():
        if domain == suffix or domain.endswith("." + suffix):
            return name
    return clean_text(fallback) or domain


def is_press_review(titre: str) -> bool:
    """Vrai pour les "revues de presse" multi-pays (AllAfrica).

    Ces articles agrègent une dizaine de brèves sur toute l'Afrique : une
    fois découpés, un séisme en Égypte se retrouve étiqueté "Mali" parce
    qu'il vient du flux Mali. Ils sont écartés à l'ingestion.
    """
    return normalize_pays(titre).startswith("revue de presse")


def parse_date(date_str: str) -> tuple[str, int]:
    """Parse une date de flux RSS en (iso, epoch_utc).

    Deux formats rencontrés : RFC 822 (`<pubDate>`, la majorité des flux)
    et ISO 8601 (`<dc:date>`, cas de lefaso.net). epoch_utc sert au
    filtrage par plage dans Chroma ($gte/$lte n'acceptent que des
    nombres). Renvoie ("", 0) si la date est absente ou illisible.
    """
    if not date_str:
        return "", 0
    parsed = None
    try:
        parsed = parsedate_to_datetime(date_str)
    except (TypeError, ValueError):
        pass
    if parsed is None:
        try:
            parsed = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            return "", 0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed_utc = parsed.astimezone(timezone.utc)
    return parsed_utc.isoformat(), int(parsed_utc.timestamp())


def date_to_epoch(date_str: str, end_of_day: bool = False) -> int:
    """Convertit une borne "AAAA-MM-JJ" en timestamp Unix UTC.

    end_of_day=True positionne l'heure à 23:59:59 (borne de fin de
    période) plutôt qu'à 00:00:00 (borne de début). Lève ValueError avec
    un message explicite si le format est mauvais : l'erreur remonte au
    LLM, qui peut alors corriger son appel.
    """
    try:
        parsed = datetime.strptime(date_str.strip(), "%Y-%m-%d")
    except ValueError:
        raise ValueError(
            f"Date invalide : {date_str!r}. Format attendu : AAAA-MM-JJ."
        ) from None
    parsed = parsed.replace(tzinfo=timezone.utc)
    if end_of_day:
        parsed = parsed.replace(hour=23, minute=59, second=59)
    return int(parsed.timestamp())


def today_label() -> str:
    """Date du jour en UTC, lisible : "samedi 2026-10-03".

    UTC explicite : les 4 pays couverts sont en GMT, et les filtres de
    date de rag.py travaillent aussi en UTC.
    """
    now = datetime.now(timezone.utc)
    return f"{JOURS[now.weekday()]} {now.date().isoformat()}"
