"""Synchronise chroma_db/ avec un dépôt de données Hugging Face.

Sur un hébergement gratuit (Hugging Face Spaces), le disque est remis à
zéro à chaque redémarrage : la base ne peut pas vivre dans le Space. Elle
est donc stockée dans un dépôt de données privé, téléchargée au démarrage
de l'application et mise à jour par l'ingestion planifiée.

Usage :
    python sync_db.py pull     # télécharge la base (au démarrage du Space)
    python sync_db.py push     # publie la base locale (après une ingestion)
    python sync_db.py restart  # redémarre le Space pour qu'il recharge la base

Variables d'environnement :
    GRIOT_DB_REPO   dépôt de données, ex. "EddieZIDA/griot-db"
    HF_TOKEN        jeton Hugging Face (lecture pour pull, écriture sinon)
    GRIOT_SPACE_ID  Space à redémarrer, ex. "EddieZIDA/griot" (restart)

Sans GRIOT_DB_REPO, chaque commande ne fait rien : en local, la base reste
simplement dans chroma_db/.
"""

import os
import sys

from dotenv import load_dotenv

from utils import CHROMA_DIR

load_dotenv()


def _api():
    from huggingface_hub import HfApi

    return HfApi(token=os.environ.get("HF_TOKEN"))


def pull(repo_id: str) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        snapshot_download(
            repo_id,
            repo_type="dataset",
            local_dir=CHROMA_DIR,
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"Base téléchargée depuis {repo_id}")
    except RepositoryNotFoundError:
        # Dépôt absent ou jeton sans accès : l'application démarre quand
        # même, avec une base vide, plutôt que de planter.
        print(
            f"[ATTENTION] Dépôt {repo_id} introuvable ou inaccessible "
            "(vérifier GRIOT_DB_REPO et HF_TOKEN). Démarrage sans base."
        )


def push(repo_id: str) -> None:
    api = _api()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=CHROMA_DIR,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Mise à jour de la base Griot",
        ignore_patterns=[".cache/**"],
        # Supprime du dépôt les fichiers qui n'existent plus en local.
        delete_patterns="*",
    )
    try:
        # Une base de plusieurs dizaines de Mo republiée plusieurs fois par
        # jour ferait vite grossir l'historique du dépôt : on n'en garde
        # qu'une version.
        api.super_squash_history(repo_id, repo_type="dataset")
    except Exception as error:
        print(f"[INFO] Historique non compacté : {error}")
    print(f"Base publiée vers {repo_id}")


def restart() -> None:
    space_id = os.environ.get("GRIOT_SPACE_ID")
    if not space_id:
        print("GRIOT_SPACE_ID non défini : aucun Space à redémarrer.")
        return
    _api().restart_space(space_id)
    print(f"Space {space_id} redémarré")


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else ""
    if command not in ("pull", "push", "restart"):
        print(__doc__)
        return 2
    if command == "restart":
        restart()
        return 0
    repo_id = os.environ.get("GRIOT_DB_REPO")
    if not repo_id:
        print("GRIOT_DB_REPO non défini : synchronisation ignorée.")
        return 0
    (pull if command == "pull" else push)(repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
