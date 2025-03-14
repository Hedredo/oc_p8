# Utiliser une image Python 3.10 slim comme base
FROM python:3.10-slim

# Installer les packages système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier les dépendances
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application
COPY . .

# Définir la variable d'environnement PORT
ENV PORT=7860

# Exposer le port utilisé par l'application Gradio
EXPOSE 7860

# Lancer l’application Gradio
CMD ["python", "main.py"]