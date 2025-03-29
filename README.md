# Documentation du Projet

Ce projet est structuré autour de trois composants principaux : les **Notebooks**, l'**API** et le **Front**.

## Notebooks

Les notebooks situés dans le dossier [notebooks](notebooks/) contiennent des analyses de données, des explorations et des visualisations.  
- **Fonctionnalité:** Ils permettent de tester des algorithmes, d'explorer les données et de valider des hypothèses de travail.  
- **Utilisation:** Ouvrez les notebooks dans un environnement compatible (Jupyter Notebook ou JupyterLab) et exécutez les cellules dans l'ordre pour reproduire les analyses.  
- **Dépendances:** Le projet utilise UV (ASTRAL) pour la gestion des packages et de Python. Toutefois, à cause de la version downgradée de Tensorflow sur le notebook modelisation_p2, les groupes de dépendances d'UV ne permettent pas de gérer deux versions différentes de Tensorflow.<br>
UV doit être installé sur le système pour exécuter les notebooks : https://docs.astral.sh/uv/getting-started/installation/<br>
Il suffit d'éxecuter un uv sync pour synchroniser les dépendances du projet avec l'environnement virtuel.<br>
	- Pour les dépendances liées au notebook modelisation_p1, il est nécessaire d'utiliser `uv pip install -r requirements-dev.txt`
	- Pour les dépendances liées au notebook modelisation_p2, il est nécessaire d'utiliser `uv pip install -r requirements-sm.txt`<br>

## API

Le dossier [app](app/) contient l'implémentation de l'API du projet. L'API a été developpée en utilisant FastAPI.
- **Fonctionnalité:** L'API expose des endpoints pour interagir avec les données et les fonctionnalités du projet.  
- **Structure:**  
  - `main.py` : Point d'entrée de l'application.  
  - `classes.py` et `utils_and_constants.py` : Contiennent des classes utilitaires et des constantes utilisées dans toute l'API.  
  - Sous-dossier `model/` : Contient la sauvegarde du modèle entraîné sous format keras.
- **Exécution:** Pour démarrer l'API en local, lancez `fastapi run main.py` en vous assurant que toutes les dépendances dans [app/requirements.txt](app/requirements.txt) sont installées.<br>
Il est préférable de lancer le container Docker de l'API dont l'image sur DockerHub est `hedredo/segmentation-api:latest`.<br> 

## Front

Le dossier [front](front/) abrite la partie frontend du projet.  
- **Fonctionnalité:** Cette partie sert l’interface utilisateur et se connecte à l’API pour afficher et interagir avec les données du projet. L'interface utilise Gradio.
- **Structure:**  
  - Contient des fichiers de configuration comme `.env`, `constants.py` et des fichiers Docker (ex: `Dockerfile`, `.dockerignore`) pour définir l’environnement de déploiement.  
  - Le code frontend comprend la logique d'affichage et les services pour appeler l'API.  
- **Déploiement:** Vous pouvez déployer le frontend en utilisant Docker avec le container sur DockerHub est `hedredo/segmentation-front:latest`.<br>
Il est aussi possible d'éxécuter le notebook en local et de configurer le fichier `.env` pour pointer vers l'API en local ou directement vers le service Azure.<br>
- **Variables d'environnement:** Le fichier `.env` contient des variables d'environnement nécessaires pour le bon fonctionnement de l'application en local. Sur Azure, il est nécessaire de les définir dans le service de l'application.<br>

## Chaîne CI-CD

Le projet utilise une chaîne CI-CD pour automatiser le déploiement des containers de l'API et du frontend qui sont hébergés séparemment sur Azure.
- **GitHub Actions:** Les workflows sont définis dans le dossier `.github/workflows/`.
- **Docker:** Les fichiers Docker sont présents dans les dossiers `app/` et `front/` pour construire les images nécessaires au déploiement.
- **Azure:** Le projet est configuré pour déployer automatiquement les images Docker en les pushant sur DockerHub et sont mises à jour sur la version latest sur les services Azure.<br>
Le déploiement de l'API et du frontend est géré par des workflows GitHub Actions qui s'exécutent à chaque push sur un des deux dossiers.<br>


## Données d'entraînement

Le dossier [data](data/) contient les données utilisées dans le projet. Il n'est pas inclus dans le dépôt Git pour des raisons de taille.<br>
Les données sont téléchargeables aux liens suivants :
- https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+8+-+Participez+%C3%A0+la+conception+d'une+voiture+autonome/P8_Cityscapes_gtFine_trainvaltest.zip
- https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+8+-+Participez+%C3%A0+la+conception+d'une+voiture+autonome/P8_Cityscapes_leftImg8bit_trainvaltest.zip

Pour plus d'information sur le dataset Cityscapes, vous pouvez consulter la page officielle : https://www.cityscapes-dataset.com/.<br>
Pour reproduire la structure des fichiers utilisés lors de l'entraînement, il est nécessaire de dézipper les fichiers téléchargés dans le dossier `data/` et de suivre ensuite les étapes dans le notebook `notebooks/modelisation_p1.ipynb`.<br>


