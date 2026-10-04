# Image de la démo Griot pour Hugging Face Spaces (SDK Docker).
# Aucune clé dans l'image : GOOGLE_API_KEY et HF_TOKEN sont fournis par les
# "Secrets" du Space, sous forme de variables d'environnement.
FROM python:3.12-slim

# Hugging Face Spaces exécute le conteneur avec l'utilisateur d'UID 1000.
RUN useradd --create-home --uid 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .
# Dossiers écrits à l'exécution (base téléchargée, historique des chats).
RUN mkdir -p chroma_db .shinychat && chown -R user:user /app
USER user

ENV PYTHONUNBUFFERED=1 \
    GRIOT_HOST=0.0.0.0 \
    GRIOT_PORT=7860 \
    GRIOT_MULTI_USER=1 \
    GRIOT_MAX_MESSAGES_PER_SESSION=10 \
    GRIOT_MAX_MESSAGES_PER_DAY=150

EXPOSE 7860

# La base est téléchargée depuis le dépôt de données avant le lancement.
CMD ["sh", "-c", "python sync_db.py pull && python chat.py"]
