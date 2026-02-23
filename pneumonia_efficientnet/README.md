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