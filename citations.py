"""Numérotation et mise en forme des citations, sans dépendance à l'UI.

Le LLM ne rédige pas la liste des sources. Chaque article renvoyé par un
outil reçoit un `numero` ; le LLM le cite tel quel, et c'est le code qui
construit le bloc de sources final.

Les numéros sont uniques sur toute une conversation (et non remis à 1 à
chaque appel d'outil) : le LLM peut appeler les outils plusieurs fois
dans un même tour, ou citer dans une relance un article obtenu au tour
précédent, sans que deux articles différents portent le même numéro. À
l'affichage, ils sont ramenés à 1, 2, 3... dans l'ordre d'apparition.

La référence est l'historique de la conversation lui-même (résultats
d'outils stockés dans les tours chatlas) : elle survit à un redémarrage
et suit le changement de conversation dans le panneau d'historique.
"""

import json
import re
from typing import Any, Callable, Iterable

TOOL_NAMES = {"search_news", "latest_news"}

# Accents graves optionnels (le LLM les oublie parfois) ; le (?!\() évite
# de prendre un lien Markdown "[1](...)" pour une citation.
CITATION_RE = re.compile(r"`?\[(\d+(?:\s*,\s*\d+)*)\]`?(?!\()")

Article = dict[str, Any]


def articles_from_turns(turns: Iterable[Any]) -> dict[int, Article]:
    """Articles déjà numérotés dans les résultats d'outils d'une conversation."""
    articles: dict[int, Article] = {}
    for turn in turns:
        for content in getattr(turn, "contents", []):
            request = getattr(content, "request", None)
            if getattr(request, "name", None) not in TOOL_NAMES:
                continue
            value = getattr(content, "value", None)
            if isinstance(value, str):
                # Valeur resérialisée en texte après rechargement d'une
                # conversation depuis le disque.
                try:
                    value = json.loads(value)
                except ValueError:
                    continue
            if not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, dict) and isinstance(
                    item.get("numero"), int
                ):
                    articles[item["numero"]] = item
    return articles


class CitationRegistry:
    """Attribue les numéros d'articles pour UNE session de chat.

    `get_turns` renvoie les tours de la conversation courante (en
    pratique `client.get_turns` de chatlas). `_pending` ne couvre que les
    résultats du tour en cours, pas encore inscrits dans l'historique.
    """

    def __init__(self, get_turns: Callable[[], Iterable[Any]]):
        self._get_turns = get_turns
        self._pending: dict[int, Article] = {}

    def known(self) -> dict[int, Article]:
        """Tous les articles citables : historique, puis tour en cours."""
        return {**self._pending, **articles_from_turns(self._get_turns())}

    def number(self, articles: list[Article]) -> list[Article]:
        """Ajoute un `numero` à chaque article (le même si déjà connu)."""
        known = articles_from_turns(self._get_turns())
        top = max(known, default=0)
        # Les entrées en attente désormais inscrites dans l'historique
        # (ou issues d'une conversation quittée) ne servent plus.
        for n in [n for n in self._pending if n <= top]:
            del self._pending[n]
        known.update(self._pending)
        by_url = {a["url"]: n for n, a in known.items()}
        next_number = max(known, default=0) + 1

        numbered = []
        for article in articles:
            numero = by_url.get(article["url"])
            if numero is None:
                numero = next_number
                next_number += 1
                by_url[article["url"]] = numero
            item = {"numero": numero, **article}
            self._pending[numero] = item
            numbered.append(item)
        return numbered


def strip_trailing_blockquote(text: str) -> str:
    """Retire une liste de sources que le LLM aurait écrite malgré la
    consigne : un bloc de lignes "> " en toute fin de texte."""
    lines = text.rstrip().split("\n")
    idx = len(lines) - 1
    found_quote = False
    while idx >= 0:
        stripped = lines[idx].strip()
        if stripped.startswith(">"):
            found_quote = True
            idx -= 1
        elif stripped == "" and found_quote:
            idx -= 1
        else:
            break
    return "\n".join(lines[: idx + 1]).rstrip() if found_quote else text


def render_citations(text: str, articles: dict[int, Article]) -> str:
    """Reconstruit la réponse finale à partir des citations du LLM.

    Chaque `[N]` valide devient un lien cliquable vers l'article,
    renuméroté 1, 2, 3... dans l'ordre d'apparition ; un numéro qui ne
    correspond à aucun article réel est supprimé. Le bloc "Sources" est
    généré à partir des seuls articles effectivement cités. Sans effet
    si `articles` est vide (aucun outil appelé dans la conversation).
    """
    if not articles:
        return text
    body = strip_trailing_blockquote(text)
    display: dict[int, int] = {}

    def repl(match: re.Match[str]) -> str:
        links = []
        for raw in match.group(1).split(","):
            n = int(raw.strip())
            if n not in articles:
                continue
            shown = display.setdefault(n, len(display) + 1)
            links.append(f"[`[{shown}]`]({articles[n]['url']})")
        if not links and not match.group(0).startswith("`"):
            # Nombre entre crochets sans accents graves et inconnu des
            # sources : ce n'est pas une citation, on n'y touche pas.
            return match.group(0)
        return " ".join(links)

    body = CITATION_RE.sub(repl, body)
    # Une citation supprimée peut laisser une espace avant la ponctuation.
    body = re.sub(r" +([.,;:!?])", r"\1", body)
    if not display:
        return body

    lines = []
    for n, shown in display.items():
        article = articles[n]
        source = (article.get("source") or "").strip()
        titre = (article.get("titre") or "").strip() or article["url"]
        date = (article.get("date") or "").strip()
        label = " · ".join(part for part in (source, date) if part)
        prefix = f"{label} : " if label else ""
        lines.append(f"> **[{shown}]** {prefix}[{titre}]({article['url']})")
    return body + "\n\n> **Sources**\n" + "\n>\n".join(lines)
