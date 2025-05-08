FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Installer dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Dossier de travail
WORKDIR /app

# Copier les fichiers
COPY requirements.txt .

# Installer dépendances Python
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Télécharger les ressources NLTK
RUN python -m nltk.downloader punkt averaged_perceptron_tagger wordnet

# Télécharger le modèle SpaCy français
RUN python -m spacy download fr_core_news_md

# Copier le reste du projet
COPY . .

# Exposer le port Flask
EXPOSE 5000

# Lancer l'application avec gunicorn (production)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
