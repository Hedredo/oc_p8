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


## Composition du pipeline d'entraînement et d'industrialisation
Dans un **pipeline de segmentation** pour la vision embarquée, la situation classique est la suivante :

1. **Entrée (Input) du modèle :**  
   - Généralement, on ne fournit **que l’image brute (en couleurs)** au réseau de neurones.  
   - Les différentes variantes d’annotations (binaire, niveaux de gris, couleurs) servent uniquement de **labels** (vérités terrain) pour l’entraînement ou l’évaluation.  

2. **Annotations (Labels) utilisées lors de l’entraînement :**  
   - Les annotations dites *FINE* (très précises) correspondent à des masques de segmentation, où chaque pixel de l’image est associé à sa classe (véhicule, piéton, route, trottoir, etc.).  
   - Les différentes représentations (binaire, niveaux de gris ou couleurs) sont juste trois manières de stocker la même information de classe par pixel. Au final, on convertit souvent l’une de ces représentations en un **index de classe** par pixel. Par exemple :  
     - Classe “route” = 0  
     - Classe “piéton” = 1  
     - Classe “véhicule” = 2  
     - etc.  
   - Ce sont ces masques qui vont servir de référence (label) pour calculer la perte (loss) et ajuster les poids du réseau lors du training.

3. **Sortie (Output) du modèle :**  
   - Après l’entraînement, pour une nouvelle image en entrée, le modèle va produire **une carte de segmentation prédite**.  
   - Cette carte est, en interne, souvent un tenseur de probabilités par pixel (ex. dimensions \[height, width, nombre_de_classes\]). Chaque pixel est assorti d’un score pour chaque classe.  
   - En post-traitement, on choisit la classe la plus probable pour chaque pixel, et on peut reconstruire une **image segmentée** (en couleurs ou en niveaux de gris, selon ce qu’on préfère pour la visualisation).  


## Architecture du modèle
Pour faire de la **segmentation sémantique** (ou instance/panoptic), on utilise généralement un **backbone** pré-entraîné sur une tâche de classification (par exemple ResNet, VGG, MobileNet, etc.) pour extraire des caractéristiques (features) de l’image. Ensuite, on ajoute un **“head” de segmentation”** (ou *decoder*) qui va convertir ces features en une carte de segmentation.

En d’autres termes :

1. **Backbone (Encoder)**  
   - Réseau initialement conçu pour de la classification (ResNet, MobileNet, etc.).  
   - Rôle : extraire une représentation riche et hiérarchique de l’image.  
   - Souvent **pré-entraîné** sur de larges bases de données (ImageNet), pour accélérer et améliorer la convergence.

2. **Head ou architecture de segmentation (Decoder)**  
   - Partie additionnelle (decoder, ASPP, pyramid pooling, etc.) qui opère sur les features du backbone.  
   - Rôle : produire un masque de segmentation (en général, de la même dimension spatiale que l’image d’origine, ou proche).  
   - Exemples de “heads” : U-Net (avec upsampling + skip connections), DeepLab (avec ASPP et convolutions dilatées), PSPNet (avec pyramid pooling), etc.

**Donc, oui**, dans la majorité des cas on combine :
- un **backbone** (issu d’un réseau de classification) pour l’extraction de caractéristiques,  
- et un **module de segmentation** (le “head”) pour redimensionner et classer chaque pixel.

C’est pourquoi on voit souvent des notations du type **“DeepLabv3+ avec backbone ResNet-50”** ou **“U-Net avec backbone VGG”**.


## Métriques du projet (Pixel Segmentation)
 
### 2.1 Intersection over Union (IoU)
C’est la métrique la plus standard pour évaluer la segmentation sémantique. Aussi appelée **Jaccard Index**, elle est définie pour chaque classe comme :

IoU = TP / (TP + FP + FN)

- **TP** (True Positive) : Pixels correctement prédits comme appartenant à la classe.  
- **FP** (False Positive) : Pixels prédits comme appartenant à la classe alors qu’ils sont dans une autre classe ou qu’ils sont “void”.  
- **FN** (False Negative) : Pixels appartenant effectivement à la classe mais prédits comme autre chose ou “void”.  

On calcule cette IoU pour chaque classe (ou catégorie), puis on moyenne les valeurs pour obtenir :
- **mIoUclass** : la moyenne de l’IoU sur toutes les *classes* fines.  
- **mIoUcategory** : la moyenne de l’IoU sur les *catégories* plus globales (par exemple, “vehicules”, “piétons”, “infrastructure”, etc.).

> **Remarque** : Les pixels annotés comme “void” (aucune classe assignée) ne sont pas pris en compte dans la métrique.

**Biais vers les grandes classes** : la IoU standard “pèse” naturellement les grosses instances de classe plus lourdement que les petites, car un grand nombre de pixels d’une classe peut faire monter ou descendre le score.

---

### 2.2 Instance-level Intersection over Union (iIoU)

Pour corriger en partie le biais précédemment mentionné (qui peut nuire aux petites classes), Cityscapes introduit une métrique pondérée : **iIoU** (pour *instance-level IoU*). Sa formule ressemble à la IoU standard, mais les notions de *vrais positifs* (TP) et de *faux négatifs* (FN) sont **pondérées** par la taille moyenne des instances de la classe.


iIoU = iTP / (iTP + FP iFN)

- **iTP** : comme TP, mais chaque pixel vrai positif est pondéré par un facteur qui dépend de la *taille moyenne* de la classe vs. la taille de l’instance annotée.  
- **iFN** : même idée (faux négatifs pondérés).  
- **FP** ne change pas, car les faux positifs n’ont pas d’ID d’instance associé dans ce type de segmentation purement sémantique.  

Ensuite, on procède de la même façon pour le *moyennage* :  
- **iIoUclass** : moyenne de l’iIoU sur toutes les classes fines.  
- **iIoUcategory** : moyenne de l’iIoU sur les catégories plus globales.

**Pourquoi pondérer ?**  
- Les petites instances (ex. piétons, cyclistes) peuvent avoir un grand impact sur la sécurité (typiquement pour la conduite autonome), mais ne comptent pas beaucoup de pixels.  
- Cette pondération évite que les classes représentant de très grandes surfaces (ex. route, bâtiment) dominent la métrique.

---
