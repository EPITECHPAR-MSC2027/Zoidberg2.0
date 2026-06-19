# Zoidberg 2.0 — Logistic Regression

## Présentation du projet

Ce projet s’inscrit dans le cadre du projet d’école **Zoidberg 2.0**. L’objectif est de classifier des radiographies thoraciques afin d’identifier différents cas liés à la pneumonie.

Cette partie du projet concerne l’entraînement et l’évaluation d’un modèle de **Logistic Regression** pour une classification en trois classes :

- **NORMAL** : radiographie sans pneumonie visible ;
- **BACTERIA** : radiographie associée à une pneumonie bactérienne ;
- **VIRUS** : radiographie associée à une pneumonie virale.

L’objectif de cette approche est d’évaluer les performances d’un modèle de machine learning classique, simple et interprétable, sur un problème de classification d’images médicales.

Le notebook principal de cette partie est :

```text
pneumonia_logistic_regression/train_final.ipynb
```

---

## Dataset utilisé

Le dataset utilisé est chargé depuis Hugging Face :

```text
PAR8/chest-xray-pneumonia
```

Ce dataset est privé. Il nécessite donc un token Hugging Face valide, stocké dans un fichier `.env`.

Le dataset contient trois splits principaux :

- `train` ;
- `validation` ;
- `test`.

Dans ce projet, les ensembles `train` et `validation` sont regroupés afin de former un **train pool** utilisé pour l’entraînement et la validation croisée.

Le jeu de test reste séparé et n’est utilisé qu’à la fin pour l’évaluation finale du modèle.

Cette séparation permet d’éviter toute fuite de données et de garantir une évaluation plus fiable sur des images jamais vues pendant l’entraînement.

---

## Structure du projet

La structure actuelle du projet est la suivante :

```text
PNEUMONIA_LOGREG/
│
├── .github/
│
├── pneumonia_efficientnet/
│
├── pneumonia_logistic_regression/
│   ├── models/
│   └── train_final.ipynb
│
├── pneumonia_svm/
│
├── result/
│
├── results/
│   ├── efficientnet/
│   └── logistic_regression/
│       ├── experiment_1/
│       ├── experiment_2/
│       ├── experiment_3/
│       ├── ...
│       ├── experiment_18/
│       ├── logreg_hyperparameter_search.csv
│       └── results_history.json
│
├── folds.json
├── cross_validation.py
├── preprocessing.ipynb
├── requirements.txt
├── .gitignore
├── .env
└── venv/
```

Le dossier important pour cette partie est :

```text
pneumonia_logistic_regression/
```

Il contient le notebook final de Logistic Regression ainsi que le dossier `models`, dans lequel les modèles entraînés sont sauvegardés.

Les résultats de la Logistic Regression sont stockés dans :

```text
results/logistic_regression/
```

---

## Fichiers importants

### `pneumonia_logistic_regression/train_final.ipynb`

Notebook principal de la partie Logistic Regression.

Il contient :

- le chargement du dataset avec le preprocessing commun ;
- l’extraction de caractéristiques adaptée à la Logistic Regression ;
- la standardisation ;
- la validation croisée ;
- la recherche d’hyperparamètres ;
- l’entraînement sans PCA ;
- l’évaluation sans PCA ;
- la feature importance ;
- l’entraînement avec PCA ;
- la matrice de confusion ;
- la comparaison finale ;
- la sauvegarde des modèles.

### `preprocessing.ipynb`

Fichier commun du groupe pour le preprocessing du dataset.

Il contient notamment :

- le chargement du dataset depuis Hugging Face ;
- la création du train pool ;
- les transformations d’évaluation ;
- les transformations avec augmentation ;
- la fonction `get_data_pipeline`.

Dans la partie Logistic Regression, le pipeline commun est utilisé avec :

```python
get_data_pipeline(augment=False)
```

### `cross_validation.py`

Fichier utilisé pour charger les folds communs du projet.

Il permet de garantir que tous les modèles utilisent les mêmes découpages pendant la validation croisée.

### `folds.json`

Fichier contenant les folds de validation croisée.

### `results/logistic_regression/`

Dossier contenant les résultats de la Logistic Regression :

- historique des expériences ;
- résultats de recherche d’hyperparamètres ;
- graphiques ;
- matrices de confusion ;
- cartes de feature importance.

### `pneumonia_logistic_regression/models/`

Dossier contenant les modèles sauvegardés avec `joblib`.

---

## Méthodologie générale

La méthodologie suivie dans le notebook est la suivante :

1. imports et authentification ;
2. chargement du dataset avec le preprocessing commun ;
3. extraction de caractéristiques depuis le preprocessing commun ;
4. standardisation des données ;
5. mise en place d’un système de sauvegarde des expériences ;
6. définition d’une fonction d’évaluation multi-classe ;
7. validation croisée commune du groupe ;
8. résumé de la validation croisée ;
9. recherche d’hyperparamètres ;
10. entraînement de la Logistic Regression sans PCA ;
11. évaluation du modèle sans PCA ;
12. visualisation des scores ;
13. sauvegarde de l’expérience sans PCA ;
14. génération des cartes de feature importance ;
15. entraînement de la Logistic Regression avec PCA ;
16. génération de la matrice de confusion avec PCA ;
17. comparaison des modèles et sauvegarde finale.

---

## Ordre des cellules du notebook

L’ordre exact des cellules du notebook est le suivant :

```text
# Cellule 1 — Imports et authentification

# Cellule 2 — Chargement du dataset avec le preprocessing commun

# Cellule 3 — Extraction de caractéristiques depuis le preprocessing commun

# Cellule 4 — Standardisation des données

# Cellule — Experiment Logger

# Cellule — Fonction d’évaluation multi-classe

# Cellule — Validation croisée commune du groupe

# Cellule — Résumé de la validation croisée

# Cellule — Recherche d’hyperparamètres Logistic Regression

# Cellule 5 — Entraînement de la Logistic Regression sans PCA

# Cellule 6 — Évaluation du modèle sans PCA

# Cellule 7 — Visualisation des scores en pourcentage

# Save experiment — Version sans PCA

# Cellule — Feature Importance

# Cellule 8 — Logistic Regression avec PCA

# Cellule — Matrice de confusion Logistic Regression + PCA

# Cellule 9 — Comparaison des modèles et sauvegarde
```

Cet ordre est important, car certaines cellules dépendent des variables créées dans les cellules précédentes.

---

## Preprocessing commun du groupe

Le notebook utilise le preprocessing commun du groupe à travers la fonction :

```python
get_data_pipeline(augment=False)
```

Cette fonction permet de charger le dataset et de récupérer les vues nécessaires :

- `train_pool_eval_view` : train pool avec preprocessing d’évaluation ;
- `test_view` : jeu de test avec preprocessing d’évaluation ;
- `train_pool_raw` : train pool brut ;
- `test_raw` : jeu de test brut.

L’option `augment=False` est utilisée pour la Logistic Regression afin de conserver une représentation stable et reproductible des images pendant la validation croisée et l’évaluation finale.

Les augmentations aléatoires sont présentes dans le preprocessing commun du groupe, mais elles sont désactivées ici car la Logistic Regression travaille sur des vecteurs numériques fixes.

---

## Extraction de caractéristiques pour la Logistic Regression

La Logistic Regression ne peut pas utiliser directement les images sous forme de tenseurs.

Une extraction de caractéristiques spécifique est donc appliquée à partir des images issues du preprocessing commun.

Les étapes sont les suivantes :

1. récupération de l’image prétraitée par le pipeline commun ;
2. vérification que l’image est bien un tenseur PyTorch ;
3. conversion en niveaux de gris ;
4. redimensionnement en **128 × 128 pixels** ;
5. aplatissement de l’image en vecteur ;
6. récupération du label associé.

Chaque radiographie est donc représentée par un vecteur de :

```text
128 × 128 = 16 384 caractéristiques
```

Cette représentation permet d’utiliser un modèle de machine learning classique comme la Logistic Regression, tout en gardant une dimension raisonnable.

---

## Standardisation des données

Les données sont standardisées avec `StandardScaler`.

La Logistic Regression est sensible à l’échelle des variables. Il est donc important de standardiser les caractéristiques afin que chaque pixel soit traité de manière homogène par le modèle.

Le scaler est entraîné uniquement sur le train pool :

```python
X_pool_s = scaler.fit_transform(X_pool)
```

Puis il est appliqué au jeu de test :

```python
X_test_s = scaler.transform(X_test)
```

Cette méthode permet d’éviter toute fuite de données, car le jeu de test n’est jamais utilisé pour calculer les paramètres de standardisation.

---

## Validation croisée commune

Le notebook utilise une validation croisée basée sur les folds communs du projet.

Les folds sont chargés avec :

```python
load_folds()
```

Cette approche permet à tous les modèles du projet d’être évalués avec les mêmes séparations entre entraînement et validation.

Pour chaque fold :

1. les indices d’entraînement et de validation sont récupérés ;
2. les données correspondantes sont extraites depuis le train pool ;
3. une Logistic Regression est entraînée ;
4. le modèle est évalué sur le fold de validation ;
5. les scores sont sauvegardés.

Les métriques utilisées sont :

- Accuracy ;
- Precision macro ;
- Recall macro ;
- F1-score macro ;
- ROC-AUC macro.

L’utilisation des moyennes macro permet de donner le même poids à chaque classe, même si le dataset est déséquilibré.

---

## Recherche d’hyperparamètres

Une recherche d’hyperparamètres a été réalisée afin de comparer plusieurs configurations de Logistic Regression.

Les paramètres testés sont :

- la représentation des images : pixels aplatis ou PCA ;
- l’utilisation ou non d’une réduction de dimension ;
- la valeur du paramètre `C` ;
- la régularisation L2.

Les configurations testées sont les suivantes :

| Config | Features | PCA   |    C | Penalty | Acc. moy. | Prec. moy. | Recall moy. | F1 moy. | ROC-AUC moy. |
| ------ | -------- | ----- | ---: | ------- | --------: | ---------: | ----------: | ------: | -----------: |
| 1      | Pixels   | False | 0.01 | L2      |    74.3 % |     73.3 % |      74.6 % |  73.9 % |       88.1 % |
| 2      | Pixels   | False | 0.10 | L2      |    72.7 % |     71.4 % |      73.1 % |  72.1 % |       87.3 % |
| 3      | PCA      | True  | 0.01 | L2      |    73.6 % |     73.0 % |      74.3 % |  73.5 % |       88.2 % |
| 4      | PCA      | True  | 0.10 | L2      |    73.0 % |     72.2 % |      73.7 % |  72.8 % |       87.8 % |

Les résultats montrent que les configurations avec `C=0.01` obtiennent de meilleures performances que celles avec `C=0.10`.

La configuration sans PCA avec `C=0.01` obtient les meilleurs résultats moyens en validation croisée sur la majorité des métriques.

Cependant, la configuration avec PCA et `C=0.01` reste très proche et obtient le meilleur ROC-AUC moyen. Elle est donc également conservée pour l’évaluation finale sur le jeu de test.

---

## Modèles entraînés

Deux versions de la Logistic Regression sont entraînées.

### 1. Logistic Regression sans PCA

Cette version utilise directement les pixels aplatis standardisés.

Configuration :

| Paramètre                  | Valeur                      |
| -------------------------- | --------------------------- |
| Modèle                     | Logistic Regression         |
| Bibliothèque               | scikit-learn                |
| Représentation             | Pixels aplatis standardisés |
| Taille utilisée            | 128 × 128                   |
| Nombre de caractéristiques | 16 384                      |
| Solver                     | saga                        |
| Régularisation             | L2                          |
| C                          | 0.01                        |
| Class weight               | balanced                    |
| PCA                        | Non                         |
| Max iter                   | 8000                        |

Cette version est entraînée puis évaluée sur le test set. Elle sert aussi à générer les cartes de feature importance, car ses coefficients correspondent directement aux pixels de l’image.

### 2. Logistic Regression avec PCA

Cette version ajoute une réduction de dimension par PCA après la standardisation.

Pipeline utilisé :

```text
Pixels aplatis standardisés → PCA → Logistic Regression
```

La PCA conserve **95 % de la variance**.

Configuration :

| Paramètre      | Valeur                            |
| -------------- | --------------------------------- |
| Modèle         | Logistic Regression               |
| Bibliothèque   | scikit-learn                      |
| Représentation | Pixels aplatis standardisés + PCA |
| PCA            | 95 % de variance conservée        |
| Solver         | saga                              |
| Régularisation | L2                                |
| C              | 0.01                              |
| Class weight   | balanced                          |
| Max iter       | 8000                              |

---

## Résultats sur le jeu de test

Les deux versions du modèle sont évaluées sur le jeu de test final.

| Modèle                       | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ---------------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression sans PCA |   62.3 % |    67.4 % | 61.8 % |   59.8 % |  84.3 % |
| Logistic Regression avec PCA |   66.7 % |    71.9 % | 67.6 % |   64.7 % |  87.2 % |

Même si la version sans PCA obtient de meilleurs résultats moyens en validation croisée, la version avec PCA généralise mieux sur le jeu de test final.

La PCA permet de réduire la dimension des données et de limiter une partie du bruit et des redondances entre pixels. Cela améliore les performances finales du modèle sur des données jamais vues.

La version retenue pour l’analyse finale est donc :

```text
Logistic Regression avec PCA, C=0.01, régularisation L2, solver saga et class_weight="balanced"
```

---

## Matrice de confusion

La matrice de confusion du modèle Logistic Regression avec PCA sur le jeu de test est la suivante :

| Classe réelle | Normal prédit | Bacteria prédit | Virus prédit | Total |
| ------------- | ------------: | --------------: | -----------: | ----: |
| Normal        |            94 |              31 |          109 |   234 |
| Bacteria      |             3 |             209 |           30 |   242 |
| Virus         |             3 |              32 |          113 |   148 |

Analyse :

- la classe **BACTERIA** est la mieux reconnue, avec 209 bonnes prédictions sur 242 ;
- la classe **VIRUS** obtient également un résultat correct, avec 113 bonnes prédictions sur 148 ;
- la classe **NORMAL** est la plus difficile à identifier, avec 94 bonnes prédictions sur 234 ;
- une partie importante des radiographies normales est confondue avec la classe Virus.

Cette confusion montre que la distinction entre radiographies normales et pneumonies virales reste difficile pour un modèle linéaire basé sur des pixels aplatis et une réduction de dimension.

---

## Feature importance

Une visualisation de **feature importance** est générée pour la Logistic Regression sans PCA.

Cette étape est placée après la sauvegarde de l’expérience sans PCA et avant l’entraînement de la version avec PCA.

La feature importance est calculée à partir des coefficients appris par la Logistic Regression.

Dans la version sans PCA, chaque coefficient correspond directement à une caractéristique d’entrée, c’est-à-dire à un pixel de l’image après extraction de caractéristiques.

Les coefficients peuvent donc être replacés sous forme d’image de taille :

```text
128 × 128
```

Une carte d’importance est générée pour chaque classe :

- **NORMAL** ;
- **BACTERIA** ;
- **VIRUS**.

Les cartes sont sauvegardées dans le dossier de l’expérience sans PCA.

Exemples de fichiers générés :

```text
feature_importance_NORMAL.png
feature_importance_BACTERIA.png
feature_importance_VIRUS.png
```

Ces visualisations permettent d’apporter une première forme d’interprétabilité au modèle.

Les zones avec des coefficients positifs indiquent les pixels qui favorisent la prédiction de la classe étudiée. Les zones avec des coefficients négatifs indiquent les pixels qui défavorisent cette classe.

Cette interprétation reste toutefois limitée. Les coefficients sont associés à des pixels ou à des informations issues des pixels, et non à des structures médicales directement identifiables. Les cartes de feature importance doivent donc être considérées comme une aide à la compréhension du comportement du modèle, et non comme une explication médicale complète.

La feature importance est générée sur la version sans PCA, car dans la version PCA les coefficients correspondent aux composantes principales et non directement aux pixels d’origine. Il serait donc plus difficile de replacer les coefficients sous forme d’image lisible.

---

## Sauvegarde des résultats

Le notebook met en place un système de suivi des expériences.

Chaque expérience est sauvegardée dans :

```text
results/logistic_regression/
```

Les éléments sauvegardés peuvent inclure :

- les scores de validation ;
- les scores de test ;
- la configuration du modèle ;
- les graphiques de performance ;
- les matrices de confusion ;
- les cartes de feature importance ;
- l’historique des expériences au format JSON.

Le fichier principal d’historique est :

```text
results/logistic_regression/results_history.json
```

La recherche d’hyperparamètres est sauvegardée sous forme de fichier CSV :

```text
results/logistic_regression/logreg_hyperparameter_search.csv
```

Les dossiers `experiment_1`, `experiment_2`, etc. contiennent les sorties générées pour chaque expérience.

---

## Sauvegarde des modèles

Les modèles entraînés sont sauvegardés avec `joblib` dans le dossier :

```text
pneumonia_logistic_regression/models/
```

Les fichiers sauvegardés sont :

```text
logreg_3classes_pixels.pkl
logreg_3classes_pca.pkl
scaler.pkl
pca.pkl
```

Ces fichiers permettent de réutiliser les modèles sans devoir relancer tout l’entraînement.

---

## Installation

### 1. Cloner le projet

```bash
git clone <url-du-repository>
cd PNEUMONIA_LOGREG
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
```

### 3. Activer l’environnement virtuel

Sur macOS ou Linux :

```bash
source venv/bin/activate
```

Sur Windows :

```bash
venv\Scripts\activate
```

### 4. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Configuration du token Hugging Face

Créer un fichier `.env` à la racine du projet :

```text
KEY_HUGGING_FACE=your_hugging_face_token
```

Le token est nécessaire pour accéder au dataset privé Hugging Face.

Le fichier `.env` ne doit pas être push sur GitHub.

---

## Exécution du notebook

Pour lancer Jupyter Notebook :

```bash
jupyter notebook
```

Puis ouvrir :

```text
pneumonia_logistic_regression/train_final.ipynb
```

L’ordre d’exécution recommandé est :

1. Cellule 1 — Imports et authentification ;
2. Cellule 2 — Chargement du dataset avec le preprocessing commun ;
3. Cellule 3 — Extraction de caractéristiques depuis le preprocessing commun ;
4. Cellule 4 — Standardisation des données ;
5. Cellule — Experiment Logger ;
6. Cellule — Fonction d’évaluation multi-classe ;
7. Cellule — Validation croisée commune du groupe ;
8. Cellule — Résumé de la validation croisée ;
9. Cellule — Recherche d’hyperparamètres Logistic Regression ;
10. Cellule 5 — Entraînement de la Logistic Regression sans PCA ;
11. Cellule 6 — Évaluation du modèle sans PCA ;
12. Cellule 7 — Visualisation des scores en pourcentage ;
13. Save experiment — Version sans PCA ;
14. Cellule — Feature Importance ;
15. Cellule 8 — Logistic Regression avec PCA ;
16. Cellule — Matrice de confusion Logistic Regression + PCA ;
17. Cellule 9 — Comparaison des modèles et sauvegarde.

---

## Temps d’exécution

Le temps d’exécution complet peut être important, car les images sont transformées en vecteurs de grande dimension.

Dans cette exécution, le temps total observé a été d’environ :

```text
463 minutes
```

Ce temps s’explique par :

- la taille du train pool ;
- le nombre de caractéristiques par image ;
- la validation croisée ;
- la recherche d’hyperparamètres ;
- l’entraînement de plusieurs configurations ;
- l’utilisation du solver `saga` ;
- l’entraînement de la version sans PCA et de la version avec PCA ;
- la génération et la sauvegarde des résultats.

---

## Limites du modèle

La Logistic Regression présente plusieurs limites dans ce projet.

Premièrement, il s’agit d’un modèle linéaire. Les différences entre les classes de radiographies peuvent être complexes et parfois non linéaires.

Deuxièmement, le modèle repose sur des pixels aplatis. Cette représentation ne conserve pas explicitement toute la structure spatiale de l’image, comme les relations entre zones voisines, les contours ou les textures.

Troisièmement, la classe NORMAL reste difficile à distinguer de la classe VIRUS. La matrice de confusion montre qu’une partie importante des radiographies normales est prédite comme virale.

Enfin, même si la PCA améliore les résultats sur le jeu de test, elle ne transforme pas les images en caractéristiques médicales ou anatomiques. Le modèle reste donc dépendant de la qualité de la représentation initiale des pixels.

---

## Conclusion

La Logistic Regression constitue une approche simple, stable et interprétable pour la classification de radiographies thoraciques.

La version sans PCA obtient les meilleurs résultats moyens en validation croisée, mais la version avec PCA obtient les meilleurs résultats sur le jeu de test final.

La configuration retenue pour l’analyse finale est donc :

```text
Logistic Regression avec PCA, C=0.01, régularisation L2, solver saga, class_weight="balanced"
```

Cette approche montre qu’un modèle linéaire associé à une réduction de dimension peut obtenir des résultats cohérents sur ce problème, tout en conservant une certaine interprétabilité.

La feature importance permet également d’apporter une première lecture du comportement de la version sans PCA, en visualisant les zones de pixels qui influencent les prédictions du modèle.

---

## Git et versioning

Les fichiers suivants ne doivent pas être push :

```text
.env
venv/
__pycache__/
.ipynb_checkpoints/
.DS_Store
```

Les fichiers importants à versionner sont :

```text
README.md
requirements.txt
preprocessing.ipynb
cross_validation.py
folds.json
pneumonia_logistic_regression/train_final.ipynb
results/logistic_regression/results_history.json
results/logistic_regression/logreg_hyperparameter_search.csv
```

Les modèles `.pkl` peuvent être versionnés si le groupe décide de sauvegarder les modèles entraînés dans le repository.

---

## Auteur

Partie Logistic Regression réalisée dans le cadre du projet Zoidberg 2.0.

Modèle développé et évalué avec :

- Python ;
- scikit-learn ;
- NumPy ;
- Matplotlib ;
- PyTorch ;
- Hugging Face Datasets ;
- PCA ;
- StandardScaler.
