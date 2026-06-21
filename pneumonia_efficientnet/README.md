<<<<<<< HEAD
# Zoidberg2.0

## Pneumonia classification with EfficientNet-B0 (PyTorch)

This repository contains a small experimentation project for pneumonia detection on chest X‑ray images using **transfer learning** with **EfficientNet-B0** in **PyTorch**.

The core of the project lives in the folder `pneumonia_efficientnet/` and is organised as follows:

- `01_train_efficientnet_b0.ipynb` – training notebook
- `02_evaluate_efficientnet_b0.ipynb` – evaluation & prediction notebook
- `on_the_fly_augmentation.ipynb` (at the repo root) – data loading and medical‑oriented data augmentation

The task is to classify X‑ray images into three classes:

- `normal`
- `bacteria`
- `virus`

The dataset is loaded from Hugging Face (`PAR8/chest-xray-pneumonia`) and augmented on the fly using **Albumentations**.

---

## Environment setup

1. Create and activate a Python environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# or
.venv\\Scripts\\activate           # Windows PowerShell / CMD
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

You also need a working Jupyter environment (e.g. `jupyterlab` or the built‑in notebook support in your IDE).

---

## Notebooks overview

### 1. `on_the_fly_augmentation.ipynb`

This notebook:

- downloads the chest X‑ray dataset from Hugging Face (`PAR8/chest-xray-pneumonia`),
- defines **medically safe** augmentation pipelines using Albumentations,
- applies the transforms to the Hugging Face `datasets` objects,
- builds PyTorch `DataLoader`s:
  - `train_loader`
  - `val_loader`
  - `test_loader`

Both training and evaluation notebooks call this notebook via:

```python
%run "../on_the_fly_augmentation.ipynb"  # or %run "on_the_fly_augmentation.ipynb" from repo root
```

so that all the data loading logic stays in a single place.

### 2. `01_train_efficientnet_b0.ipynb`

This notebook focuses on **model definition and training**:

- explains the choice of tools:
  - **PyTorch** for flexible deep learning experimentation,
  - **EfficientNet-B0** from `torchvision` as a light yet powerful image classifier,
  - **transfer learning** from ImageNet weights for faster convergence.
- calls `on_the_fly_augmentation.ipynb` to obtain `train_loader` / `val_loader` / `test_loader`,
- builds an EfficientNet-B0 model with a custom final linear layer for 3 classes,
- trains the model with a classic training loop (loss + validation accuracy),
- saves the trained weights to:

```text
pneumonia_efficientnet/efficientnet_b0_pneumonia.pt
```

It also plots the training and validation losses and the validation accuracy per epoch, to quickly visualise convergence and potential overfitting.

### 3. `02_evaluate_efficientnet_b0.ipynb`

This notebook performs **evaluation and visual inspection**:

- re‑runs `on_the_fly_augmentation.ipynb` to rebuild the `test_loader`,
- reconstructs the same EfficientNet-B0 architecture and loads the saved weights,
- computes multiple metrics on the **test set**:
  - Accuracy
  - Precision (macro)
  - Recall (macro)
  - F1‑score (macro)
  - ROC‑AUC (multi‑class, one‑vs‑rest)
- prints a detailed per‑class classification report,
- plots:
  - a confusion matrix,
  - a bar chart of the global metrics (in percentage),
- shows a few example test images with ground truth and predicted labels.

---

## How to run the project

1. Start your Jupyter environment (JupyterLab / Jupyter Notebook / IDE).
2. Make sure the working directory is the root of this repository (`Zoidberg2.0`).
3. Open and run the notebooks in this order:

- `on_the_fly_augmentation.ipynb` (optional to inspect the data pipeline),
- `pneumonia_efficientnet/01_train_efficientnet_b0.ipynb` to train and save the model,
- `pneumonia_efficientnet/02_evaluate_efficientnet_b0.ipynb` to compute metrics and inspect predictions.

If you retrain the model, the evaluation notebook will automatically use the latest saved weights file.

---

## Notes

- If you have a compatible GPU, PyTorch will automatically use it (`cuda` device). Otherwise, the model will run on CPU (slower but still functional).
- The project is intended for educational / experimentation purposes and is **not** a certified medical diagnostic tool.
=======
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
>>>>>>> 9c56a4ebbd7ade2d4abcb45aaa13fa7cf580907e
