# Zoidberg2.0 — Pneumonia classification

Architecture pédagogique pour la classification de radiographies thoraciques
en 3 classes (`normal`, `bacteria`, `virus`).

L'architecture est pensée pour **comparer équitablement plusieurs modèles**
(CNN, SVM, Random Forest, …) sur **les mêmes données**, **les mêmes folds de
cross-validation** et **le même jeu de test final**.

## Structure du projet

```
zoidberg2.0/
├── preprocessing.ipynb              # chargement + augmentation, partagé par tous les modèles
├── cross_validation.py              # génère les folds StratifiedKFold (une seule fois)
├── pneumonia_efficientnet/          # modèle CNN
│   ├── hyperparameter_search.py
│   ├── train_final.ipynb
│   ├── evaluate.ipynb
│   └── gradcam.ipynb
└── results/
    ├── folds.json                   # folds CV partagés
    └── efficientnet/
        ├── hyperparameter_search/   # results.json, best_config.json
        ├── final_model/             # poids + meta
        ├── evaluation/              # metrics.json + plots
        └── gradcam/                 # images Grad-CAM
```

Pour ajouter un nouveau modèle (SVM, Random Forest, …) : créer un dossier
`pneumonia_<modele>/` reproduisant la même structure
(`hyperparameter_search.py`, `train_final.ipynb`, `evaluate.ipynb`) et lire
les folds partagés via `cross_validation.load_folds()`.

## Workflow

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Générer les folds (une seule fois, partagés par tous les modèles)
python cross_validation.py

# 3. Recherche d'hyperparamètres CNN
python pneumonia_efficientnet/hyperparameter_search.py

# 4. Entraînement final sur le pool complet (notebook)
jupyter notebook pneumonia_efficientnet/train_final.ipynb

# 5. Évaluation sur le test set final (notebook)
jupyter notebook pneumonia_efficientnet/evaluate.ipynb

# 6. Visualisations Grad-CAM (notebook)
jupyter notebook pneumonia_efficientnet/gradcam.ipynb
```

## Principes méthodologiques

- **Pool d'entraînement** = `train` + `validation` concaténés.
- **Jeu de test final** = split `test` HuggingFace, **jamais** utilisé en CV
  ni en recherche d'hyperparamètres.
- **Folds CV** générés une seule fois (`results/folds.json`), réutilisés par
  tous les modèles.
- **Transfer learning** : paramètre booléen unifié dans la grille, **pas** de
  duplication de code par variante.
- **Reproductibilité** : seed fixe (42), pipeline d'augmentation centralisé.

## Hyperparamètres recherchés (CNN)

| Hyperparamètre      | Valeurs par défaut |
| ------------------- | ------------------ |
| `learning_rate`     | `[1e-4, 5e-4]`     |
| `batch_size`        | `[32, 64]`         |
| `num_epochs`        | `[8]`              |
| `transfer_learning` | `[True, False]`    |

Modifier `HP_GRID` en haut de `pneumonia_efficientnet/hyperparameter_search.py`
pour ajuster la grille.

Pour chaque configuration sont enregistrés : `fold_scores`, `accuracy_mean`,
`precision_mean`, `recall_mean`, `f1_mean` (+ std). La meilleure configuration
(critère : `f1_mean` macro) est extraite dans `best_config.json` et utilisée
automatiquement par `train_final.ipynb`.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Un fichier `.env` à la racine doit contenir `KEY_HUGGING_FACE=hf_xxx` pour
l'accès au dataset.
