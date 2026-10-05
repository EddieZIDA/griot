---
title: Griot
emoji: 📰
colorFrom: yellow
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Griot

**Agent conversationnel d'actualité ouest-africaine, par RAG strict et sources citées**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LLM](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-4285F4)
![Base vectorielle](https://img.shields.io/badge/Base%20vectorielle-Chroma-orange)
![Tests](https://img.shields.io/badge/tests-pytest-success)
![Docker](https://img.shields.io/badge/Docker-Hugging%20Face%20Spaces-2496ED?logo=docker&logoColor=white)

> Couverture : Burkina Faso, Sénégal, Mali, Côte d'Ivoire, et l'actualité
> internationale et tech/IA francophone (catégorie "Monde") · 15 médias
> reconnus · Ingestion automatique toutes les 6 heures

---

## En bref

Griot répond en français aux questions d'actualité, par exemple "Quoi de
neuf au Burkina Faso cette semaine ?" ou "Que dit la presse sur l'IA en ce
moment ?".

**Principe non négociable : RAG strict.** Griot ne répond jamais à partir de
sa mémoire ni via une recherche web. Il ne répond qu'à partir d'articles de
presse déjà ingérés et vectorisés (Chroma), et cite ses sources. S'il n'a
aucun article pertinent, il le dit.

| Choix de conception | Pourquoi |
|---|---|
| Réponses uniquement à partir d'articles ingérés | Pas d'hallucination sur l'actualité : chaque affirmation renvoie à un article |
| Citations numérotées par le code, pas par le LLM | Un numéro de source ne peut pas être inventé |
| Deux outils de récupération (`search_news`, `latest_news`) | Une question sur un sujet précis et une demande de "journal" ne se traitent pas de la même façon |
| Filtre de pertinence (score minimal et marge) | Mieux vaut dire "je n'ai rien" que citer un article hors sujet |
| Classement par pays selon le sujet de l'article, sans appel LLM | Un journal burkinabè qui parle de l'étranger ne pollue pas le fil "Burkina Faso" |
| Limites par visite et par jour sur la démo | Tous les visiteurs partagent le même quota Gemini |

## Stack technique

- **LLM** : Gemini 2.5 Flash, via `chatlas` (appel d'outils)
- **Embeddings** : `gemini-embedding-001`
- **Base vectorielle** : Chroma (locale)
- **Ingestion** : flux RSS (`feedparser`), extraction du texte complet (`trafilatura`)
- **Interface** : Shiny et shinychat
- **Déploiement** : Docker, Hugging Face Spaces, GitHub Actions (déploiement et ingestion planifiée)
- **Qualité** : pytest, tests sans appel à l'API (embeddings fabriqués)
- **Environnement** : Python 3.12

---

## Architecture

- `ingest.py` : lit `config/sources.csv`, récupère les articles des flux RSS,
  extrait leur texte complet (`trafilatura`, avec repli sur le contenu du
  flux), les découpe en chunks, les vectorise (`gemini-embedding-001`, titre
  de l'article inclus) et les stocke dans `chroma_db/`. Affiche un rapport
  flux par flux.
- `rag.py` : deux modes de récupération.
  - `search_articles` : recherche sémantique sur un sujet précis, avec
    filtre de pertinence.
  - `latest_articles` : articles les plus récents, pour les questions de
    type "journal".
  Dans les deux cas, le texte de l'article est reconstitué à partir de ses
  chunks avant d'être transmis au LLM.
- `chat.py` : lance Griot (`chatlas.ChatGoogle`, `gemini-2.5-flash`) avec
  les outils `search_news` et `latest_news`. Le LLM choisit l'outil, le
  pays et la période. Les citations sont numérotées et mises en forme par
  le code, pas par le LLM. Interface web via Shiny/shinychat.
- `citations.py` : numérotation des articles et construction du bloc
  "Sources", indépendantes de l'interface.
- `utils.py` : constantes communes (collection, modèle d'embedding) et
  fonctions partagées (dates, noms de sources, nettoyage, classement d'un
  article par pays).

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Renseigner `GOOGLE_API_KEY` dans `.env` (clé Gemini).

## Utilisation

1. Compléter `config/sources.csv` (colonnes : `pays`, `url`). Valeurs de
   `pays` : Burkina Faso, Senegal, Mali, Côte d'Ivoire, Monde.
2. Ingérer les articles :
   ```bash
   python ingest.py
   ```
   Relançable sans doublon. À relancer régulièrement (idéalement chaque
   jour) : les flux RSS ne contiennent que les articles récents, ce qui
   n'est pas ingéré à temps est perdu.
3. Lancer Griot :
   ```bash
   python chat.py
   ```

### Migration depuis l'ancienne base

Les articles ingérés avant la refonte sont dans la collection `griot_news`.
Pour les convertir au nouveau format (`griot_news_v2`) sans les
retélécharger :

```bash
python ingest.py --reindex
```

Relançable : si le quota Gemini est atteint, la conversion reprend où elle
s'est arrêtée.

## Classement des articles par pays

Le pays d'un article est celui de son **sujet**, pas celui du flux
(`utils.classify_pays`, sans appel LLM) :

1. si le titre nomme exactement un pays couvert, l'article y est classé ;
2. sinon, un article d'un journal national qui ne mentionne nulle part son
   pays (nom, gentilé, grandes villes) est classé dans "monde".

Les flux déjà triés par pays (AllAfrica, tags RFI : `FLUX_PAR_SUJET`) ne
sont pas soumis à la règle 2. Le pays du flux reste dans la métadonnée
`pays_flux`. Pour reclasser les articles déjà ingérés (après la mise à jour,
ou après une modification de `PAYS_MARKERS`) :

```bash
python ingest.py --reclassify
```

Aucun appel à l'API : seules les métadonnées changent.

## Ingestion automatique (Windows)

Double-cliquer une fois sur `scripts\install_task.bat` : crée la tâche
planifiée "Griot Ingestion", qui lance `scripts\run_ingest.bat` toutes les
6 heures. La sortie de chaque passage s'ajoute à `logs\ingest.log`.

- Vérifier : `schtasks /Query /TN "Griot Ingestion"`
- Supprimer : `schtasks /Delete /TN "Griot Ingestion" /F`

La tâche ne tourne que si le PC est allumé et la session ouverte.

## Déploiement de la démo (Hugging Face Spaces)

Trois éléments, tous gratuits :

- un **dépôt GitHub** avec le code ;
- un **Space** Hugging Face (SDK Docker) qui fait tourner l'application ;
- un **dépôt de données** Hugging Face privé qui contient `chroma_db/`, car
  le disque d'un Space gratuit est effacé à chaque redémarrage.

Deux workflows GitHub Actions font le lien : `deploy-space.yml` pousse le
code vers le Space à chaque modification de `main`, et `ingest.yml` lance
l'ingestion toutes les 6 heures, republie la base et redémarre le Space.

### Où vont les clés

La clé Gemini n'est jamais dans le dépôt ni dans l'image Docker : `.env` est
exclu par `.gitignore` et `.dockerignore`. Elle est fournie par la
plateforme, sous forme de variable d'environnement.

| Où | Secrets | Variables |
|---|---|---|
| Space Hugging Face (Settings) | `GOOGLE_API_KEY`, `HF_TOKEN` | `GRIOT_DB_REPO` |
| Dépôt GitHub (Settings, Secrets and variables, Actions) | `GOOGLE_API_KEY`, `HF_TOKEN` | `GRIOT_DB_REPO`, `GRIOT_SPACE_ID` |

### Mise en place

1. Créer un jeton Hugging Face avec droit d'écriture, puis un Space vide
   (SDK Docker).
2. Publier la base une première fois depuis le PC, avec `HF_TOKEN` et
   `GRIOT_DB_REPO` renseignés dans `.env` :
   ```bash
   python sync_db.py push
   ```
3. Renseigner les secrets et variables du tableau ci-dessus.
4. Pousser le code sur GitHub (branche `main`) : le déploiement part seul.

### Protection du quota Gemini

Tous les visiteurs utilisent la même clé. Le `Dockerfile` active donc des
limites (voir `limits.py`), modifiables dans les variables du Space :

- `GRIOT_MAX_MESSAGES_PER_SESSION` (10) : questions par visite ;
- `GRIOT_MAX_MESSAGES_PER_DAY` (150) : questions par jour, tous visiteurs
  confondus ;
- `GRIOT_MAX_INPUT_CHARS` (2000) : longueur d'un message.

Si Gemini renvoie malgré tout une erreur de quota, le visiteur voit un
message clair. Il est aussi conseillé de dédier une clé à la démo, distincte
de la clé de développement.

## Plusieurs utilisateurs

Chaque session de navigateur a son propre client LLM et sa propre
numérotation de sources. L'historique des conversations reste commun par
défaut (usage personnel). Pour un historique séparé par navigateur, ajouter
dans `.env` :

```
GRIOT_MULTI_USER=1
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Les tests n'appellent jamais l'API Gemini : la récupération est testée sur
une collection Chroma temporaire avec des embeddings fabriqués.

## Réglages utiles

- `rag.py` : `SEARCH_TOP_K`, `SEARCH_MAX_CHARS`, `LATEST_TOP_K`,
  `LATEST_MAX_CHARS` (quantité de texte transmise au LLM), `MIN_SCORE` et
  `SCORE_MARGIN` (filtre de pertinence).
- `ingest.py` : `CHUNK_TARGET_WORDS`, `MIN_ARTICLE_WORDS`.
- `utils.py` : `SOURCE_NAMES` (nom affiché de chaque média),
  `PAYS_MARKERS` et `FLUX_PAR_SUJET` (classement par pays).

## Limites connues

- Le classement par pays repose sur des mots-clés : un article local qui ne
  cite ni le pays ni une grande ville peut partir à tort dans "monde".
- Les tests ne couvrent pas l'interface ni les appels réels à Gemini.
- Ne pas lancer deux ingestions en même temps sur la même base.

---

## Auteur

**Wend Kouni Eddie Eliel ZIDA**, élève-ingénieur Big Data & IA à l'ISGA Rabat.
[Portfolio](https://eddiezida.github.io) · [LinkedIn](https://www.linkedin.com/in/wend-kouni-eddie-eliel-zida-501815260/) · [GitHub](https://github.com/EddieZIDA)
