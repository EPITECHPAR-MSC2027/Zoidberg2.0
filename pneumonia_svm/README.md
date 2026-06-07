# SVM — Classification de pneumonies

Modèle **Support Vector Machine** pour la classification de radiographies
thoraciques en 3 classes (`normal`, `bacteria`, `virus`).

Ce dossier reproduit **la même structure** que `pneumonia_efficientnet/` afin de
comparer équitablement les deux modèles : **mêmes données**, **mêmes folds de
cross-validation** (`results/folds.json`) et **même jeu de test final**.

## Structure

```
pneumonia_svm/
├── hyperparameter_search.py   # recherche d'hyperparamètres par cross-validation
├── train_final.ipynb          # entraînement final sur le pool complet
├── evaluate.ipynb             # évaluation sur le jeu de test final
└── README.md

results/svm/
├── hyperparameter_search/     # results.json, best_config.json
├── final_model/               # svm_pipeline.joblib + train_meta.json
└── evaluation/                # metrics.json + plots
```

## Pipeline du modèle

Le SVM n'opère pas sur les images 2D mais sur des **vecteurs de features** :

1. **Niveaux de gris + aplatissement** : chaque image 224×224 est convertie en
   niveaux de gris puis aplatie en un vecteur de 50 176 valeurs.
2. **StandardScaler** : centrage / réduction des pixels.
3. **PCA** : réduction de dimension en conservant `pca_variance` de la variance
   (ajustée sur l'entraînement uniquement, pour éviter toute fuite de données).
4. **SVC** (noyau RBF) : classification des features réduites.

Le pipeline complet (scaler + PCA + SVM) est sérialisé via `joblib` pour que
l'évaluation soit indépendante de l'entraînement.

## Workflow

```bash
# 1. Générer les folds (une seule fois, partagés par tous les modèles)
python cross_validation.py

# 2. Recherche d'hyperparamètres SVM (cross-validation)
python pneumonia_svm/hyperparameter_search.py

# 3. Entraînement final sur le pool complet (notebook)
jupyter notebook pneumonia_svm/train_final.ipynb

# 4. Évaluation sur le test set final (notebook)
jupyter notebook pneumonia_svm/evaluate.ipynb
```

## Hyperparamètres recherchés

| Hyperparamètre | Valeurs par défaut |
| -------------- | ------------------ |
| `C`            | `[1, 10]`          |
| `gamma`        | `["scale"]`        |
| `kernel`       | `["rbf"]`          |
| `pca_variance` | `[0.95]`           |

Modifier `HP_GRID` en haut de `hyperparameter_search.py` pour ajuster la grille.

Pour chaque configuration sont enregistrés : `fold_scores`, `accuracy_mean`,
`precision_mean`, `recall_mean`, `f1_mean` (+ std). La meilleure configuration
(critère : `f1_mean` macro) est extraite dans `best_config.json` et utilisée
automatiquement par `train_final.ipynb`.

## Augmentation

Le SVM s'entraîne **avec augmentation** : la constante `AUGMENT = True` en haut
de `hyperparameter_search.py` (et de `train_final.ipynb`) appelle
`get_data_pipeline(augment=True)`. Passer à `False` pour comparer sans
augmentation. Le jeu de test final n'est jamais augmenté.
