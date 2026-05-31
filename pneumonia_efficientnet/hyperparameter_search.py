"""Recherche d'hyperparamètres pour EfficientNet-B0.

Pour chaque configuration de la grille (``learning_rate``, ``batch_size``,
``num_epochs``, ``transfer_learning``) :

1. entraînement sur chaque fold défini par ``cross_validation.py``,
2. calcul des métriques accuracy / precision / recall / F1 par fold,
3. calcul des moyennes inter-folds,
4. sauvegarde de l'intégralité des résultats dans
   ``results/efficientnet/hyperparameter_search/results.json`` et de la
   meilleure configuration dans ``best_config.json``.

Le jeu de test final n'est **jamais** utilisé ici.

Usage::

    python pneumonia_efficientnet/hyperparameter_search.py
"""
from __future__ import annotations

import gc
import json
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader, Subset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cross_validation import load_folds  # noqa: E402
from ipynb.fs.full.preprocessing import get_data_pipeline  # noqa: E402

RESULTS_DIR = ROOT / "results" / "efficientnet" / "hyperparameter_search"
RESULTS_PATH = RESULTS_DIR / "results.json"
BEST_PATH = RESULTS_DIR / "best_config.json"

# ---------------------------------------------------------------------------
# Grille d'hyperparamètres
# ---------------------------------------------------------------------------
HP_GRID = {
    "learning_rate": [1e-4, 5e-4],
    "batch_size": [32, 64],
    "num_epochs": [8],
    "transfer_learning": [True, False],
}

NUM_CLASSES = 3
NUM_WORKERS = 8
SEED = 42


# ---------------------------------------------------------------------------
# Modèle unifié (TL = True/False)
# ---------------------------------------------------------------------------
def build_model(transfer_learning: bool, device: torch.device) -> nn.Module:
    """EfficientNet-B0 — poids ImageNet si transfer_learning, sinon random init."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if transfer_learning else None
    m = efficientnet_b0(weights=weights)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    return m.to(device)


def make_loaders(pipeline, train_idx, val_idx, batch_size):
    train_view = pipeline["train_pool_train_view"]
    eval_view = pipeline["train_pool_eval_view"]
    pin = torch.cuda.is_available()
    tr = DataLoader(Subset(train_view, train_idx), batch_size=batch_size,
                    shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin,
                    persistent_workers=NUM_WORKERS > 0)
    va = DataLoader(Subset(eval_view, val_idx), batch_size=batch_size,
                    shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin,
                    persistent_workers=NUM_WORKERS > 0)
    return tr, va


def train_and_evaluate_fold(model, train_loader, val_loader, lr, num_epochs,
                            device):
    """Entraîne ``num_epochs`` et renvoie les métriques de la dernière epoch."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            preds = model(x).argmax(1)
            all_y.extend(y.cpu().tolist())
            all_p.extend(preds.cpu().tolist())

    return {
        "accuracy": accuracy_score(all_y, all_p),
        "precision": precision_score(all_y, all_p, average="macro", zero_division=0),
        "recall": recall_score(all_y, all_p, average="macro", zero_division=0),
        "f1": f1_score(all_y, all_p, average="macro", zero_division=0),
    }


def run_search():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Chargement du pipeline de données…")
    pipeline = get_data_pipeline()

    print("Chargement des folds CV…")
    folds_data = load_folds()
    folds = folds_data["folds"]
    print(f"  {len(folds)} folds — seed={folds_data['seed']}")

    configs = [
        dict(zip(HP_GRID.keys(), values))
        for values in product(*HP_GRID.values())
    ]
    print(f"Grille : {len(configs)} configurations\n")

    all_results = []
    t0 = time.time()

    for cfg_idx, cfg in enumerate(configs, 1):
        print(f"[{cfg_idx}/{len(configs)}] {cfg}")
        fold_metrics = []
        for fold in folds:
            print(f"  fold {fold['fold']}/{len(folds)}")
            train_loader, val_loader = make_loaders(
                pipeline, fold["train_idx"], fold["val_idx"], cfg["batch_size"])
            model = build_model(cfg["transfer_learning"], device)
            metrics = train_and_evaluate_fold(
                model, train_loader, val_loader,
                cfg["learning_rate"], cfg["num_epochs"], device)
            metrics["fold"] = fold["fold"]
            fold_metrics.append(metrics)
            print(f"    acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")

            del model, train_loader, val_loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        means = {
            f"{k}_mean": float(np.mean([m[k] for m in fold_metrics]))
            for k in ("accuracy", "precision", "recall", "f1")
        }
        stds = {
            f"{k}_std": float(np.std([m[k] for m in fold_metrics]))
            for k in ("accuracy", "precision", "recall", "f1")
        }

        all_results.append({
            **cfg,
            "fold_scores": fold_metrics,
            **means,
            **stds,
        })
        print(f"  -> acc_mean={means['accuracy_mean']:.4f} "
              f"f1_mean={means['f1_mean']:.4f}\n")

    elapsed_min = (time.time() - t0) / 60
    print(f"Recherche terminée en {elapsed_min:.1f} min")

    # Best config = max f1_mean (équilibré sur 3 classes déséquilibrées)
    best = max(all_results, key=lambda r: r["f1_mean"])

    summary = {
        "model": "EfficientNetB0",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_min": round(elapsed_min, 2),
        "k_folds": folds_data["k_folds"],
        "seed": SEED,
        "hp_grid": HP_GRID,
        "results": all_results,
        "best_config": {k: best[k] for k in HP_GRID.keys()},
        "best_metrics": {
            "accuracy_mean": best["accuracy_mean"],
            "precision_mean": best["precision_mean"],
            "recall_mean": best["recall_mean"],
            "f1_mean": best["f1_mean"],
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    with open(BEST_PATH, "w") as f:
        json.dump({"best_config": summary["best_config"],
                   "best_metrics": summary["best_metrics"]}, f, indent=2)

    print(f"\nRésultats : {RESULTS_PATH}")
    print(f"Meilleure config : {summary['best_config']}")
    print(f"  f1_mean = {best['f1_mean']:.4f}")


if __name__ == "__main__":
    run_search()
