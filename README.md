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
    - DEEPSEEK : https://www.datacamp.com/blog/deepseek-r1?utm_source=marketo&utm_medium=email&utm_campaign=250128_1-012825newsletter_2-b2b_3-all_4-na_5-bl_6-deepseek_7-en_8-emal-mk_9-na_10-bau_11-email&utm_content=blast&utm_term=lead-news&mkt_tok=MzA3LU9BVC05NjgAAAGYTGlVnPRJCH2k2X9MdUIIyqH8eTVHXb-2m8bzXkiGTU4DAFuCDKVBjwXThnNgSL_P6U1Cqa985CCug5Cvx0jfShQ-MEvPWa-6X9XtTCrpavWoqg
    - Model + Metrics + Display avec changement d'encodeur en supplément : https://www.geeksforgeeks.org/image-segmentation-using-tensorflow/
    - Guide to Convolution arithmetics -31 p- : https://arxiv.org/pdf/1603.07285
    - Cours CS231n :
        - CNN : https://cs231n.github.io/convolutional-networks/
        - Understanding & visualizing CNN : https://cs231n.github.io/understanding-cnn/
        - Transfer Learning : https://cs231n.github.io/transfer-learning/
    - Image Segmentation with pretrained model with HuggingFace & Tensorflow
        - https://www.youtube.com/watch?v=oL-xmufhZM8
        - https://huggingface.co/docs/transformers/tasks/semantic_segmentation
        - SOTA models huggingface : https://huggingface.co/blog/mask2former
        - Fine-tuning segformer : https://huggingface.co/blog/fine-tune-segformer
    - UNET : https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
    - Albumentations : https://albumentations.ai/docs/examples/example/
- Analyse pré-exploratoire et préparation des données
    - Regarder l'équilibre des catégories avant le train-test-split ? Possible avec np.unique_counts(img_array) sur chaque image (se servir du multiprocessing)
    - Crée le train test split avec shuffle et 0.2

## Workflow
https://docs.google.com/document/d/1kIbxaqzdPZqHEJRxwuFCAar-G8W78VFRgDUEF5Mesjw/edit?invite=CI_q4swC&tab=t.0


## To-Do
- Prétraitement des données
    - Normalisation des images RGB (vérifier si c'est fait avec load_img)

- Datagenerator (tf.keras.utils.PyDataset) : https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    - Dans le init, argument pour dossier + str endswith pour les images et les masks (`le def doit être dans le init`)
    - A partir de çà générer avec une fonction interne les indexes des images pour le shuffle
    - Faire une classe DataGenerator unique
        - avec multiprocessing as a function to call internally et ajouter un arg external_multiprocessing. Si external_multiprocessing = True, set use_multiprocessing = False dans le DataGenerator
        - avec un argument pour le transform to categorical (si true, alors on fait le to_categorical sur les labels)
    - Ajouter la normalisation sur le RGB si nécessaire dans load_img
    - Ajouter la mapping (en valeur fixe dans init) sur le mask avec np.vectorize + le dictionnaire
    - Ajouter le on_epoch_end pour shuffle les données à chaque epoch sur la liste des indexs (créer une fonction interne pour l'extraction des indexes)
    - Mettre uniquement le path du dossier dans le DataGenerator et non les images directement (pour pouvoir changer de dataset plus facilement) - cela permets de récupérer les indexes des images pour le shuffle
    - Ajouter la partie data augmentation (rotation, flip, zoom, etc.)
    - Appliquer le data loader ensuite sur train - val - test

- Datagenerator (tf.data.Dataser)
    - Adapter le DataGenerator pour tf.data.Dataset
    - Internaliser les fonctions dans la classe
    - Ajouter le mapping (en valeur fixe dans init) sur le mask avec tf.lookup
    - Remplacer la partie mapping avec np vectorize par tf.lookup (voir email pour le code)

- Modele
    - Tester avec fit generator pour les perfs ? : https://www.geeksforgeeks.org/keras-fit-and-keras-fit_generator/  ou https://stackoverflow.com/questions/55531427/how-to-define-max-queue-size-workers-and-use-multiprocessing-in-keras-fit-gener

- Ajouter les métriques IoU et Dice
    - Ajouter les métriques IoU et Dice dans le modèle
    - Ajouter les métriques IoU et Dice dans le callback