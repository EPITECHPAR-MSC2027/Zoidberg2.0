# Random Forest - Pneumonia classification

Pipeline Random Forest pour classifier les radiographies thoraciques en 3
classes : `normal`, `bacteria`, `virus`.

Ce dossier suit la meme logique que `pneumonia_efficientnet` : memes donnees,
memes folds de cross-validation, meme jeu de test final.

## Structure

```text
random_forest/
|-- hyperparameter_search.py   # recherche CV avec folds partages
|-- train_final.ipynb          # entrainement final sur train+validation
|-- evaluate.ipynb             # evaluation sur le test final
|-- random_forest.ipynb        # notebook workflow complet
|-- requirements.txt
`-- README.md
```

## Preprocessing

Le Random Forest reutilise le notebook central :

```python
from ipynb.fs.full.preprocessing import get_data_pipeline

pipeline = get_data_pipeline(augment=False)
```

`augment=False` est volontaire : la vue d'entrainement utilise uniquement le
transform d'evaluation (`resize + normalize`), sans augmentation. Les features
HOG sont extraites depuis ces images.

## Workflow

Depuis la racine du projet :

```bash
# 1. Generer les folds partages si besoin
python cross_validation.py

# 2. Recherche d'hyperparametres
python random_forest/hyperparameter_search.py

# 3. Entrainement final
jupyter notebook random_forest/train_final.ipynb

# 4. Evaluation finale
jupyter notebook random_forest/evaluate.ipynb
```

Le notebook `random_forest.ipynb` peut aussi lancer les trois etapes dans
l'ordre.

## Sorties

Dans `results/`, les donnees de resultat sont en JSON. Les images de rapport
restent en PNG.

```text
results/random_forest/
|-- hyperparameter_search/
|   |-- results.json
|   `-- best_config.json
|-- final_model/
|   `-- train_meta.json
`-- evaluation/
    |-- metrics.json
    |-- confusion_matrix.json
    |-- confusion_matrix.png
    |-- metric_scores.json
    |-- metrics_bar.png
    |-- predictions.json
    `-- predictions.png
```

Les artefacts binaires sont sauvegardes hors de `results/` :

```text
models/random_forest/
|-- feature_cache/
|   `-- train_pool_hog.joblib
`-- final_model/
    |-- random_forest_final.joblib
    `-- scaler.joblib
```

## Methode

- Train pool : split `train` + split `validation` Hugging Face.
- Test final : split `test`, jamais utilise pendant la cross-validation.
- Folds : charges via `cross_validation.load_folds()`.
- Features : HOG sur images denormalisees puis converties en grayscale.
- Normalisation features : `StandardScaler`.
- Equilibrage : `SMOTE` selon la configuration testee.
- Selection modele : meilleure configuration selon le `f1_mean` macro.

## Hyperparametres

La grille est definie dans `hyperparameter_search.py` :

```python
HP_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [None, 30],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1],
    "max_features": ["sqrt"],
    "criterion": ["gini"],
    "use_smote": [True],
    "smote_k_neighbors": [3],
}
```

Pour reduire le temps de calcul, diminuer `n_estimators` ou le nombre de
combinaisons dans `HP_GRID`.
