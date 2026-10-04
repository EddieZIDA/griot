import os
from typing import Any
from urllib.parse import quote

from dotenv import load_dotenv
import chatlas as ctl
from shiny import App, run_app, ui
from shinychat import Chat as ChatUI, chat_ui

# HistoryOptions n'est pas réexporté par shinychat.__init__ mais c'est le
# type documenté du paramètre `history=` de `shinychat.Chat` : import
# direct depuis son module. Version de shinychat épinglée dans
# requirements.txt pour cette raison.
from shinychat._history import HistoryOptions

import rag
from citations import CitationRegistry, render_citations
from limits import RateLimiter, guard_client
from utils import today_label

load_dotenv()

# Supprime le rendu des appels d'outils côté shinychat (l'élément n'est
# jamais créé, ce n'est pas un masquage CSS).
os.environ.setdefault("SHINYCHAT_TOOL_DISPLAY", "none")

MODEL = "gemini-2.5-flash"


def build_system_prompt() -> str:
    """System prompt avec la date du jour.

    Reconstruit à chaque ouverture de session (voir `server`) : calculé
    une seule fois à l'import, "aujourd'hui" restait figé à la date de
    lancement de l'application, et "hier" devenait faux dès le lendemain.
    """
    return f"""\
Tu es Griot, un journaliste-présentateur d'actualité en français. Tu couvres
le Burkina Faso, le Sénégal, le Mali, la Côte d'Ivoire, ainsi que
l'actualité internationale/panafricaine et l'actualité tech/IA francophone
(catégorie "Monde").

Nous sommes aujourd'hui le {today_label()} (UTC).

# Tes outils

Tu ne réponds JAMAIS à une question d'actualité de mémoire, et tu ne fais
jamais de recherche web. Tu passes TOUJOURS par l'un de ces deux outils :

- `latest_news` : les articles les plus récents. À utiliser pour les
  questions générales, sans sujet précis : "quoi de neuf au Mali ?", "le
  journal du Burkina", "que s'est-il passé hier ?", "les nouvelles du
  monde".
- `search_news` : recherche sur un sujet précis : une personne, un
  événement, un thème ("la libération de Moussa Mara", "le prix du coton",
  "les nouveaux modèles de DeepSeek"). Rédige `query` comme une phrase
  descriptive riche en mots-clés, pas comme un seul mot vague.

N'hésite pas à appeler les outils plusieurs fois avant de répondre : un
appel par pays si la question en mentionne plusieurs, plusieurs
formulations si le sujet a plusieurs facettes, ou `latest_news` puis
`search_news` pour approfondir un sujet repéré. Si l'utilisateur demande
plus de détails sur un point, relance `search_news` sur ce point précis
plutôt que de reformuler ce que tu as déjà dit.

Paramètre `pays` : utilise EXACTEMENT l'une de ces valeurs :
"Burkina Faso", "Senegal", "Mali", "Côte d'Ivoire", "Monde".
- "Monde" regroupe l'actualité internationale/panafricaine et la tech/IA.
- Omets `pays` pour chercher dans toute la base (utile quand le sujet
  n'est rattaché à aucun pays précis).
- Si la question porte sur un pays hors de cette liste, tente une
  recherche sans `pays` : la presse panafricaine en parle peut-être. Si
  rien de pertinent ne revient, dis clairement que tu ne couvres pas
  encore ce pays.

Paramètres `date_debut` / `date_fin` (format "AAAA-MM-JJ") : à renseigner
seulement si la question contient une référence temporelle ("hier", "cette
semaine", "le 5 août"). Calcule la période à partir de la date du jour
ci-dessus. Les deux peuvent être identiques pour cibler un seul jour.

# Comment rédiger

Tu racontes l'actualité comme un bon journal radio : précis, dense,
agréable à lire. Le champ `contenu` de chaque article contient son texte :
exploite-le vraiment.

- Commence par une phrase d'ouverture qui donne le fait principal ou la
  tendance du moment.
- Regroupe ensuite par sujet, avec un court intertitre en gras par sujet
  quand il y en a plusieurs. Classe du plus important au plus anecdotique.
- Pour chaque sujet, donne la matière : qui, quoi, quand, où, les chiffres,
  les noms et fonctions des personnes, les déclarations marquantes, et
  surtout le contexte (pourquoi c'est arrivé, ce que ça change, ce qui est
  attendu ensuite) quand l'article le fournit. Vise 3 à 6 phrases par
  sujet, pas une seule ligne.
- Quand plusieurs articles parlent du même sujet, croise-les : complète
  l'un par l'autre et signale les divergences.
- Indique la date des faits. Si les articles les plus récents datent de
  plusieurs jours ou semaines, dis-le explicitement ("les derniers
  articles dont je dispose remontent au...") : ne présente jamais une
  information ancienne comme si elle était du jour.
- Un journal burkinabè peut traiter un sujet étranger (football mondial,
  autre pays) : ne le présente pas comme une actualité du Burkina Faso.
  Écarte-le, ou place-le à la fin sous un intertitre "Ailleurs".
- Ignore les articles renvoyés qui sont hors sujet par rapport à la
  question, même s'ils figurent dans les résultats.
- Si l'utilisateur demande une réponse courte, fais court. Sinon, préfère
  une réponse développée à une réponse sèche. Ne remplis jamais avec des
  généralités : tout ce que tu écris doit venir des articles.

# Fiabilité

- Tu ne t'appuies que sur le `contenu` des articles renvoyés par tes
  outils. N'invente jamais un fait, un chiffre, une date ou une citation,
  et n'ajoute pas de connaissances extérieures.
- Chaque fait est suivi de sa source, sous la forme du champ `numero` de
  l'article entre crochets et entouré d'accents graves : `[3]`, ou
  `[3, 7]` pour deux articles. Réutilise le `numero` EXACTEMENT tel que
  l'outil l'a renvoyé, sans renuméroter (l'application s'en charge à
  l'affichage).
- N'écris jamais toi-même de liste de sources en fin de réponse :
  l'application la génère à partir des numéros que tu as cités.
- Si un outil renvoie une liste vide, dis-le honnêtement ("Je n'ai pas
  d'article sur ce sujet pour le moment", ou "Je n'ai pas d'article sur le
  Burkina Faso pour hier").
- Si un outil renvoie une erreur, ne dis pas qu'il n'y a pas d'article :
  explique qu'un problème technique empêche la recherche et invite à
  réessayer dans un instant.
"""


def create_client() -> tuple[ctl.Chat, CitationRegistry]:
    """Crée le client LLM et le registre de citations d'UNE session.

    Un client par session de navigateur : avec un client global, deux
    onglets (ou deux personnes sur le réseau local) partageaient le même
    fil de conversation et les mêmes numéros de sources.
    """
    client = ctl.ChatGoogle(model=MODEL, system_prompt=build_system_prompt())
    registry = CitationRegistry(client.get_turns)

    def search_news(
        query: str,
        pays: str | None = None,
        date_debut: str | None = None,
        date_fin: str | None = None,
    ) -> list[dict]:
        """
        Recherche les articles de presse sur un sujet précis, parmi les
        articles déjà ingérés localement (jamais de recherche web).

        Parameters
        ----------
        query : str
            Le sujet recherché, en français, sous forme de phrase
            descriptive riche en mots-clés (ex. "libération de l'ancien
            Premier ministre Moussa Mara").
        pays : str, optional
            "Burkina Faso", "Senegal", "Mali", "Côte d'Ivoire" ou
            "Monde". Omettre pour chercher dans toute la base.
        date_debut : str, optional
            Début de la période, format "AAAA-MM-JJ". Omettre si la
            question n'a pas de référence temporelle.
        date_fin : str, optional
            Fin de la période, même format.

        Returns
        -------
        list of dict
            Articles du plus pertinent au moins pertinent. Champs :
            numero (à citer tel quel), titre, source, pays, date, url,
            contenu (texte de l'article), score. Liste vide si aucun
            article pertinent.
        """
        return registry.number(
            rag.search_articles(query, pays, date_debut, date_fin)
        )

    def latest_news(
        pays: str | None = None,
        date_debut: str | None = None,
        date_fin: str | None = None,
    ) -> list[dict]:
        """
        Renvoie les articles les plus récents, sans sujet précis. Pour
        les questions générales : "quoi de neuf ?", "le journal", "que
        s'est-il passé hier ?".

        Parameters
        ----------
        pays : str, optional
            "Burkina Faso", "Senegal", "Mali", "Côte d'Ivoire" ou
            "Monde". Omettre pour tous les pays.
        date_debut : str, optional
            Début de la période, format "AAAA-MM-JJ". Omettre pour
            obtenir simplement les derniers articles disponibles.
        date_fin : str, optional
            Fin de la période, même format.

        Returns
        -------
        list of dict
            Articles du plus récent au plus ancien. Champs : numero (à
            citer tel quel), titre, source, pays, date, url, contenu
            (début de l'article). Vérifie le champ date : les derniers
            articles disponibles peuvent être anciens. Liste vide si
            aucun article sur la période.
        """
        return registry.number(rag.latest_articles(pays, date_debut, date_fin))

    client.register_tool(search_news)
    client.register_tool(latest_news)
    return client, registry


# Marque visuelle de Griot : deux arcs radiant d'un point central (voix qui
# porte, tradition orale), en terracotta et indigo, réutilisée pour le
# favicon et l'avatar de l'assistant dans le chat.
GRIOT_MARK_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <circle cx="16" cy="16" r="3.5" fill="#C1622B"/>
  <path d="M21 16a5 5 0 0 1-5 5" stroke="#C1622B" stroke-width="2"
        fill="none" stroke-linecap="round"/>
  <path d="M24.5 16a8.5 8.5 0 0 1-8.5 8.5" stroke="#C1622B"
        stroke-width="2" fill="none" stroke-linecap="round" opacity="0.55"/>
  <path d="M11 16a5 5 0 0 0 5 5" stroke="#2B3A67" stroke-width="2"
        fill="none" stroke-linecap="round"/>
  <path d="M7.5 16a8.5 8.5 0 0 0 8.5 8.5" stroke="#2B3A67"
        stroke-width="2" fill="none" stroke-linecap="round" opacity="0.55"/>
</svg>
"""

FAVICON_HREF = "data:image/svg+xml," + quote(GRIOT_MARK_SVG)

# shinychat n'expose aucun point d'extension Python pour traduire les
# libellés de son panneau d'historique (vérifié : aucun paramètre
# labels/i18n/locale dans Chat()/HistoryOptions()), les chaînes sont
# figées en anglais dans le bundle JS compilé. Un MutationObserver côté
# client est le seul levier disponible. Ouvre aussi le panneau par
# défaut (docked) puisque le composant natif ne propose pas d'état
# "ouvert au chargement".
HISTORY_UI_SCRIPT = """\
(function () {
  "use strict";

  var TEXT_MAP = {
    "History": "Historique",
    "New": "Nouvelle conversation",
    "Today": "Aujourd'hui",
    "Last 7 days": "7 derniers jours",
    "Previous": "Plus anciennes",
    "No conversations yet": "Aucune conversation pour le moment",
    "No conversations found": "Aucune conversation trouvée",
    "Rename": "Renommer",
    "Delete": "Supprimer",
    "Delete?": "Supprimer ?",
    "Cancel": "Annuler"
  };
  var ATTR_MAP = {
    "Close history": "Fermer l'historique",
    "New conversation": "Nouvelle conversation",
    "Search conversations": "Rechercher dans les conversations",
    "Conversation actions": "Actions sur la conversation",
    "Rename conversation": "Renommer la conversation",
    "Cancel delete": "Annuler la suppression",
    "Confirm delete": "Confirmer la suppression"
  };
  var PLACEHOLDER_MAP = {
    "Search\\u2026": "Rechercher\\u2026"
  };

  function translateEl(el) {
    if (!el || el.nodeType !== 1) return;
    var label = el.getAttribute && el.getAttribute("aria-label");
    if (label && ATTR_MAP[label]) {
      el.setAttribute("aria-label", ATTR_MAP[label]);
    }
    var ph = el.getAttribute && el.getAttribute("placeholder");
    if (ph && PLACEHOLDER_MAP[ph]) {
      el.setAttribute("placeholder", PLACEHOLDER_MAP[ph]);
    }
    el.childNodes.forEach(function (node) {
      if (node.nodeType === 3) {
        var trimmed = node.textContent.trim();
        if (TEXT_MAP[trimmed]) {
          node.textContent = node.textContent.replace(
            trimmed, TEXT_MAP[trimmed]
          );
        }
      } else {
        translateEl(node);
      }
    });
  }

  function openHistoryDefault() {
    var trigger = document.querySelector(".shiny-chat-history-trigger");
    var alreadyOpen = document.querySelector(".shiny-chat-history");
    if (alreadyOpen) return;
    if (trigger) {
      trigger.click();
    } else {
      setTimeout(openHistoryDefault, 150);
    }
  }

  function init() {
    var observer = new MutationObserver(function (mutations) {
      mutations.forEach(function (m) {
        m.addedNodes.forEach(translateEl);
      });
    });
    observer.observe(document.body, { childList: true, subtree: true });
    setTimeout(openHistoryDefault, 350);
  }

  // Le script est chargé dans <head> : document.body n'existe pas
  // encore à ce stade, donc on attend DOMContentLoaded plutôt que
  // d'exécuter immédiatement.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
"""

# Version plus détaillée de la même marque (points en bout d'arc), pour un
# usage logo à côté du titre "Griot" plutôt qu'en petite icône.
GRIOT_LOGO_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40">
  <circle cx="20" cy="20" r="4.2" fill="#C1622B"/>
  <path d="M26.5 20a6.5 6.5 0 0 1-6.5 6.5" stroke="#C1622B"
        stroke-width="2.4" fill="none" stroke-linecap="round"/>
  <circle cx="26.5" cy="20" r="1.3" fill="#C1622B"/>
  <path d="M31 20a11 11 0 0 1-11 11" stroke="#C1622B" stroke-width="2.4"
        fill="none" stroke-linecap="round" opacity="0.55"/>
  <circle cx="31" cy="20" r="1.3" fill="#C1622B" opacity="0.55"/>
  <path d="M13.5 20a6.5 6.5 0 0 0 6.5 6.5" stroke="#2B3A67"
        stroke-width="2.4" fill="none" stroke-linecap="round"/>
  <circle cx="13.5" cy="20" r="1.3" fill="#2B3A67"/>
  <path d="M9 20a11 11 0 0 0 11 11" stroke="#2B3A67" stroke-width="2.4"
        fill="none" stroke-linecap="round" opacity="0.55"/>
  <circle cx="9" cy="20" r="1.3" fill="#2B3A67" opacity="0.55"/>
</svg>
"""


def simple_title(turns: list[dict[str, Any]]) -> str | None:
    """Titre de conversation = premier message utilisateur, tronqué.

    Pas d'appel LLM (contrairement au title="auto" par défaut de
    chatlas/shinychat) : simple, gratuit, et évite de consommer le quota
    API déjà serré sur le tier gratuit à chaque nouvelle conversation.
    """
    for turn in turns:
        if turn.get("role") != "user":
            continue
        contents = turn.get("contents")
        if isinstance(contents, list):
            text = "".join(
                c.get("text", "")
                for c in contents
                if isinstance(c, dict) and c.get("content_type") == "text"
            )
        else:
            text = str(turn.get("content", ""))
        text = text.strip()
        if text:
            return text[:45] + ("…" if len(text) > 45 else "")
    return None


# Par défaut l'historique est commun à tous les navigateurs (scope fixe
# "local") : pratique pour un usage personnel sur une seule machine.
# Avec GRIOT_MULTI_USER=1 dans .env, chaque navigateur a son propre
# historique (scope=None : jeton stocké dans le navigateur). À activer si
# l'application est ouverte à plusieurs personnes. Attention : changer ce
# réglage masque les conversations enregistrées sous l'autre mode.
MULTI_USER = os.environ.get("GRIOT_MULTI_USER", "0") == "1"

# Limites d'usage (voir limits.py). Désactivées par défaut en local ;
# activées par le Dockerfile pour la démo publique.
LIMITER = RateLimiter.from_env()

HISTORY_OPTIONS = HistoryOptions(
    store="file",
    scope=None if MULTI_USER else "local",
    title=simple_title,
)

THEME = (
    ui.Theme(preset="shiny")
    .add_defaults(
        body_bg="#FAF6F0",
        primary="#C1622B",
        secondary="#2B3A67",
        link_color="#2B3A67",
        gray_600="#5f6368",
    )
)


def launch_app(
    *,
    launch_browser: bool | None = None,
    port: int | None = None,
    host: str | None = None,
) -> None:
    """Lance l'interface Shiny de Griot.

    Par défaut, l'adresse d'écoute vient de l'environnement :
    GRIOT_HOST (127.0.0.1 en local ; 0.0.0.0 pour être joignable depuis
    l'extérieur, cas d'un déploiement) et GRIOT_PORT (0 = port libre
    choisi automatiquement). Le navigateur ne s'ouvre qu'en local.
    """
    if host is None:
        host = os.environ.get("GRIOT_HOST", "127.0.0.1")
    if port is None:
        port = int(os.environ.get("GRIOT_PORT", "0"))
    if launch_browser is None:
        launch_browser = host in ("127.0.0.1", "localhost")

    def app_ui(request):
        return ui.page_fillable(
            ui.head_content(
                ui.tags.link(rel="icon", href=FAVICON_HREF),
                ui.include_css("www/styles.css", method="inline"),
                ui.tags.script(HISTORY_UI_SCRIPT),
            ),
            ui.div(
                ui.div(
                    ui.HTML(GRIOT_LOGO_SVG),
                    ui.h1("Griot", class_="griot-title"),
                    class_="griot-brand",
                ),
                ui.p(
                    "L'actualité du Sahel et d'ailleurs, racontée comme "
                    "un griot : vérifiée, jamais inventée.",
                    class_="griot-tagline",
                ),
                class_="griot-header",
            ),
            chat_ui(
                "chat",
                messages=[],
                icon_assistant=ui.HTML(GRIOT_MARK_SVG),
                placeholder="Écrivez votre message...",
            ),
            title="Griot",
            theme=THEME,
            fillable_mobile=True,
        )

    def server(input):
        # Client et registre propres à la session ; le system prompt est
        # donc construit ici, avec la date du jour.
        client, registry = create_client()
        guard_client(client, LIMITER.new_session())
        chat_ui = ChatUI("chat", client=client, history=HISTORY_OPTIONS)

        # transform_assistant_response est déprécié dans shinychat 0.6.0
        # mais reste le seul point d'accroche pour reconstruire le texte
        # final après le streaming. Version épinglée dans
        # requirements.txt ; à revoir lors d'une montée de version.
        @chat_ui.transform_assistant_response
        async def _finalize(content: str, chunk: str, done: bool) -> str:
            if not done:
                return content
            return render_citations(content, registry.known())

    app = App(app_ui, server)
    run_app(app, launch_browser=launch_browser, port=port, host=host)


if __name__ == "__main__":
    launch_app()
