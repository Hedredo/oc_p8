# oc_p8
**Parcours IA Engineer - Projet 8 OpenClassrooms :** Traitez les images pour le système embarqué d’une voiture autonome<br>
**Date de début du projet :** 17/01/2025<br>
**Date de fin estimée du projet :** 14/03/2025

## Description du projet
Je suis ingénieur IA pôle R&D de Future Vision Transport > conception de systèmes embarqués de computer vision

4 != spécialisations :
- S1: Acquisition des images en temps réel
- S2: Traitement des images
- S3: **Segmentation des images** > Ma Spécialité
- S4: Système de décision

Modélisation:
- Classification supervisée avec Ground Truth Fine > Pixel Level Segmentation en 8 catégories

Input :
- 1 dossier avec images brutes en couleur
- 1 dossier avec GroundTruth Fine (labels segmentation niveaux de gris/instances segmentation binaire/visualisation segmentation couleurs/polygons coordinates JSON)  
- Utiliser uniquement les 8 catégories principales et pas les 32 sous-classes + VOID

Output :
- Image en couleur avec la segmentation + pixel annotation par 8 classes de couleur + void ?


Etapes clés du projet:
- Tensorlow Keras framework
- Pipeline entrainé sur les 8 catégories principales
- Concevoir une API de prédiction (FastApi ou Flask)
- Déploiement de l'API sur le cloud
- Interface UI streamlit déployée sur le cloud (Azure)


## CONTENU DE LA PRESENTATION DU PROJET
Pendant la soutenance, l’évaluateur ne jouera aucun rôle en particulier. Vous lui présenterez l’ensemble de votre travail. 

- Présentation (20 minutes) 
	- Présentation du contexte, des objectifs, des principes de segmentation et des mesures de performance qui seront utilisées pour comparer les modèles (5 minutes)
	- Présentation des différents modèles, simulations et comparaisons des modèles (10 minutes).
	- Mise en production d’un modèle (5 minutes) :
		- Architecture API et application Web et démarche de mise en production sur le Cloud choisi par l’étudiant
		- démonstration de fonctionnement de l’application et de la prédiction de segmentation d’une image (mask).


## Livrables
1. Les scripts développés sur un notebook permettant l’exécution du pipeline complet :
Ce livrable vous servira à présenter le caractère “industrialisable” de votre travail en particulier le générateur de données.
2. Une API (Flask ou FastAPI) déployée sur le Cloud (Azure, Heroku, PythonAnywhere ou toute autre solution), pour exposer votre modèle entraîné et qui recevra en entrée une image et retournera le mask prédit (les segments identifiés par votre modèle) :
Ce livrable permettra à Laura d’utiliser facilement votre modèle.
3. Une application (Flask, Streamlit) de présentation des résultats qui consomme l’API de prédiction, déployée sur le Cloud (Azure, Heroku, PythonAnywhere ou toute autre solution). Cette application sera l’interface pour tester l’API et intégrera les fonctionnalités suivantes :  affichage de la liste des id des images disponibles, lancement de la prédiction du mask pour l’id sélectionné par appel à l’API, et affichage de l’image réelle, du mask réel et du mask prédit :
Ce livrable permettra d’illustrer votre travail auprès de vos collègues
4. Une note technique de 10 pages environ contenant une présentation des différentes approches et une synthèse de l’état de l’art, la présentation plus détaillée du modèle et de l’architecture retenue, une synthèse des résultats obtenus (incluant les gains obtenus avec les approches d’augmentation des données) et une conclusion avec des pistes d’amélioration envisageables  :
Ce livrable vous servira à présenter votre démarche technique à vos collègues.
5. Un support de présentation (type Power Point) de votre démarche méthodologique (30 slides maximum) :
Ce livrable vous permettra de présenter vos résultats à Laura.


## Learning
- Check Ressources OC à utiliser au début : https://openclassrooms.com/fr/paths/795/projects/1517/resources

- Helping ressources :
    - `Cours CS231n`
        - [X] CNN : https://cs231n.github.io/convolutional-networks/
        - [X] Understanding & visualizing CNN : https://cs231n.github.io/understanding-cnn/
        - [X] Transfer Learning : https://cs231n.github.io/transfer-learning/
    - `Data generator for images` :
        - [X] https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    - `Metrics for segmentation` :
        - [X] https://medium.com/@nghihuynh_37300/understanding-evaluation-metrics-in-medical-image-segmentation-d289a373a3f
        - [X] https://ilmonteux.github.io/2019/05/10/segmentation-metrics.html
        - [X] https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
        - [X] Loss & Metrics https://www.kaggle.com/code/yassinealouini/all-the-segmentation-metrics#Bonus-3:-metrics-libraries
        - [X] https://www.geeksforgeeks.org/what-are-different-evaluation-metrics-used-to-evaluate-image-segmentation-models/
        - [X] Keras segmentation metrics : https://keras.io/api/metrics/segmentation_metrics/
    - `Arithmetics` :
        - Guide to Convolution arithmetics -31 p- : https://arxiv.org/pdf/1603.07285.pdf
    - `Image Segmentation with pretrained model with HuggingFace & Tensorflow`
        - [X] https://www.geeksforgeeks.org/image-segmentation-using-tensorflow/
        - [X] tensorflow : https://www.tensorflow.org/tutorials/images/segmentation
        - Librairie segmentation models : https://github.com/qubvel/segmentation_models
        - Load a pretrained model from huggingface : https://huggingface.co/docs/transformers/tasks/semantic_segmentation
        - SOTA models huggingface : https://huggingface.co/blog/mask2former
        - Fine-tuning segformer : https://huggingface.co/blog/fine-tune-segformer
        - Deeplabv3 + MobileNetV2 : https://www.tensorflow.org/tfmodels/vision/semantic_segmentation

    - `UNET` : 
        - [X] https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
    - `Albumentations` : 
        - [X] https://albumentations.ai/docs/examples/example/


## Workflow
https://docs.google.com/document/d/1kIbxaqzdPZqHEJRxwuFCAar-G8W78VFRgDUEF5Mesjw/edit?invite=CI_q4swC&tab=t.0
