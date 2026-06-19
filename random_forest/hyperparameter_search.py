"""Recherche d'hyperparametres pour Random Forest.

1. chargement du preprocessing centralise avec ``augment=False``,
2. reutilisation des folds partages de ``cross_validation.py``,
3. evaluation accuracy / precision / recall / F1 macro par fold,
4. sauvegarde des resultats dans ``results/random_forest/hyperparameter_search``.

Le jeu de test final n'est jamais utilise ici.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cross_validation import load_folds  # noqa: E402
from ipynb.fs.full.preprocessing import get_data_pipeline  # noqa: E402

RESULTS_DIR = ROOT / "results" / "random_forest" / "hyperparameter_search"
RESULTS_PATH = RESULTS_DIR / "results.json"
BEST_PATH = RESULTS_DIR / "best_config.json"
CACHE_DIR = ROOT / "models" / "random_forest" / "feature_cache"

SEED = 42
LABEL_NAMES = ["normal", "bacteria", "virus"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

HP_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [30, None],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.3],
    "criterion": ["gini"],
    "class_weight": [None, "balanced_subsample"],
    "use_smote": [True, False],
    "smote_k_neighbors": [3],
}

HOG_CONFIG = {
    "orientations": 12,
    "pixels_per_cell": 8,
    "cells_per_block": 2,
    "block_norm": "L2-Hys",
}


def tensor_to_gray_image(tensor) -> np.ndarray:
    """Convertit un tensor CxHxW normalise ImageNet en image grayscale [0, 1]."""
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    arr = np.transpose(arr, (1, 2, 0))
    arr = (arr * IMAGENET_STD) + IMAGENET_MEAN
    arr = np.clip(arr, 0.0, 1.0)
    return arr.mean(axis=2)


def extract_hog_features(dataset, indices=None, desc="Extraction HOG"):
    """Extrait les features HOG depuis une vue HuggingFace avec transform."""
    if indices is None:
        indices = range(len(dataset))

    x_parts, y_parts = [], []
    for idx in tqdm(list(indices), desc=desc, leave=False):
        sample = dataset[int(idx)]
        gray = tensor_to_gray_image(sample["image"])
        features = hog(
            gray,
            orientations=HOG_CONFIG["orientations"],
            pixels_per_cell=(HOG_CONFIG["pixels_per_cell"], HOG_CONFIG["pixels_per_cell"]),
            cells_per_block=(HOG_CONFIG["cells_per_block"], HOG_CONFIG["cells_per_block"]),
            block_norm=HOG_CONFIG["block_norm"],
        )
        x_parts.append(features)
        y_parts.append(sample["label"])

    return np.asarray(x_parts, dtype=np.float32), np.asarray(y_parts, dtype=np.int64)


def load_or_build_features(pipeline):
    """Met en cache les features du train pool sans augmentation."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "train_pool_hog.joblib"
    if cache_path.exists():
        print(f"Chargement du cache HOG : {cache_path}")
        return joblib.load(cache_path)

    print("Extraction des features HOG sur le pool train+validation...")
    train_view = pipeline["train_pool_train_view"]
    x_all, y_all = extract_hog_features(train_view, desc="Train pool HOG")
    joblib.dump({"X": x_all, "y": y_all}, cache_path)
    print(f"Cache sauvegarde : {cache_path}")
    return {"X": x_all, "y": y_all}


def build_model(cfg):
    return RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        min_samples_split=cfg["min_samples_split"],
        min_samples_leaf=cfg["min_samples_leaf"],
        max_features=cfg["max_features"],
        criterion=cfg["criterion"],
        random_state=SEED,
        n_jobs=-1,
        class_weight=cfg["class_weight"],
        verbose=0,
    )


def train_and_evaluate_fold(cfg, x_all, y_all, train_idx, val_idx):
    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    if cfg["use_smote"]:
        smote = SMOTE(random_state=SEED, k_neighbors=cfg["smote_k_neighbors"])
        x_train, y_train = smote.fit_resample(x_train, y_train)

    model = build_model(cfg)
    model.fit(x_train, y_train)
    preds = model.predict(x_val)

    return {
        "accuracy": accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds, average="macro", zero_division=0),
        "recall": recall_score(y_val, preds, average="macro", zero_division=0),
        "f1": f1_score(y_val, preds, average="macro", zero_division=0),
    }


def run_search():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    print("Chargement du pipeline de donnees (augmentation OFF)...")
    pipeline = get_data_pipeline(augment=False)
    print(f"augment = {pipeline['augment']}")

    features = load_or_build_features(pipeline)
    x_all = features["X"]
    y_all = features["y"]
    print(f"Features : {x_all.shape} | distribution : {np.bincount(y_all)}")

    print("Chargement des folds CV...")
    folds_data = load_folds()
    folds = folds_data["folds"]
    print(f"  {len(folds)} folds - seed={folds_data['seed']}")

    configs = [dict(zip(HP_GRID.keys(), values)) for values in product(*HP_GRID.values())]
    print(f"Grille : {len(configs)} configurations\n")

    all_results = []
    t0 = time.time()

    for cfg_idx, cfg in enumerate(configs, 1):
        print(f"[{cfg_idx}/{len(configs)}] {cfg}")
        fold_metrics = []
        for fold in folds:
            print(f"  fold {fold['fold']}/{len(folds)}")
            metrics = train_and_evaluate_fold(
                cfg,
                x_all,
                y_all,
                np.asarray(fold["train_idx"], dtype=np.int64),
                np.asarray(fold["val_idx"], dtype=np.int64),
            )
            metrics["fold"] = fold["fold"]
            fold_metrics.append(metrics)
            print(f"    acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")
            gc.collect()

        means = {
            f"{k}_mean": float(np.mean([m[k] for m in fold_metrics]))
            for k in ("accuracy", "precision", "recall", "f1")
        }
        stds = {
            f"{k}_std": float(np.std([m[k] for m in fold_metrics]))
            for k in ("accuracy", "precision", "recall", "f1")
        }
        all_results.append({**cfg, "fold_scores": fold_metrics, **means, **stds})
        print(f"  -> acc_mean={means['accuracy_mean']:.4f} f1_mean={means['f1_mean']:.4f}\n")

    elapsed_min = (time.time() - t0) / 60
    best = max(all_results, key=lambda r: r["f1_mean"])
    best_cfg = {k: best[k] for k in HP_GRID.keys()}
    best_metrics = {
        "accuracy_mean": best["accuracy_mean"],
        "precision_mean": best["precision_mean"],
        "recall_mean": best["recall_mean"],
        "f1_mean": best["f1_mean"],
    }

    summary = {
        "model": "RandomForestClassifier",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_min": round(elapsed_min, 2),
        "k_folds": folds_data["k_folds"],
        "seed": SEED,
        "augmentation": False,
        "feature_type": "HOG",
        "hog_config": HOG_CONFIG,
        "feature_cache": str(CACHE_DIR),
        "hp_grid": HP_GRID,
        "label_names": LABEL_NAMES,
        "results": all_results,
        "best_config": best_cfg,
        "best_metrics": best_metrics,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(BEST_PATH, "w", encoding="utf-8") as f:
        json.dump({"best_config": best_cfg, "best_metrics": best_metrics}, f, indent=2)

    print(f"\nResultats : {RESULTS_PATH}")
    print(f"Meilleure config : {best_cfg}")
    print(f"  f1_mean = {best['f1_mean']:.4f}")


if __name__ == "__main__":
    run_search()
