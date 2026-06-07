"""Recherche d'hyperparamètres pour le SVM.

Pour chaque configuration de la grille (``C``, ``gamma``, ``kernel``,
``pca_variance``) :

1. extraction des features (niveaux de gris aplatis) sur chaque fold défini
   par ``cross_validation.py``,
2. standardisation + PCA (ajustées sur le fold d'entraînement uniquement),
3. entraînement du SVM et calcul des métriques accuracy / precision / recall /
   F1 par fold,
4. calcul des moyennes inter-folds,
5. sauvegarde de l'intégralité des résultats dans
   ``results/svm/hyperparameter_search/results.json`` et de la meilleure
   configuration dans ``best_config.json``.

Le jeu de test final n'est **jamais** utilisé ici.

Usage::

    python pneumonia_svm/hyperparameter_search.py
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
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Subset
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results" / "svm" / "hyperparameter_search"
RESULTS_PATH = RESULTS_DIR / "results.json"
BEST_PATH = RESULTS_DIR / "best_config.json"
LOG_PATH = RESULTS_DIR / "run.log"

# ---------------------------------------------------------------------------
# Grille d'hyperparamètres
# ---------------------------------------------------------------------------
HP_GRID = {
    "C": [1, 10],
    "gamma": ["scale"],
    "kernel": ["rbf"],
    "pca_variance": [0.95],
}

# Le SVM s'entraîne AVEC augmentation (vue d'entraînement augmentée du
# pipeline de preprocessing). Passer à False pour comparer sans augmentation.
AUGMENT = True

BATCH_SIZE = 32
NUM_CLASSES = 3
# On Windows multiprocessing with DataLoader can hit pickling issues for
# datasets/transforms defined in notebooks. Use 0 workers to keep runs
# stable in this automated environment.
NUM_WORKERS = 0
SEED = 42

# Normalisation ImageNet appliquée par le pipeline de preprocessing : on
# l'inverse avant la conversion en niveaux de gris.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)


# ---------------------------------------------------------------------------
# Extraction des features
# ---------------------------------------------------------------------------
def extract_features(dataset, desc="Extracting"):
    """Niveaux de gris aplatis (224*224=50176) + labels depuis un dataset.

    Le SVM opère sur des vecteurs plats, pas sur des tenseurs d'image 3D :
    on dénormalise (ImageNet -> [0, 1]), on convertit en niveaux de gris
    (les radiographies sont par nature monochromes) puis on aplatit.
    """
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS)
    all_features, all_labels = [], []
    # log batch progress so long runs show activity
    for i, batch in enumerate(loader, start=1):
        if i % 10 == 0:
            msg = f"    [{desc}] processed {i} batches"
            print(msg, flush=True)
            try:
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                with open(LOG_PATH, "a") as lf:
                    lf.write(msg + "\n")
            except Exception:
                pass
        images = batch["image"].numpy()           # (B, 3, 224, 224)
        labels = batch["label"].numpy()
        images = np.clip(images * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
        gray = (0.2989 * images[:, 0] + 0.5870 * images[:, 1]
                + 0.1140 * images[:, 2])          # (B, 224, 224)
        all_features.append(gray.reshape(gray.shape[0], -1).astype(np.float32))
        all_labels.append(labels)
    total = sum(f.shape[0] for f in all_features)
    print(f"  {desc}: {total} images", flush=True)
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a") as lf:
            lf.write(f"{desc}: {total} images\n")
    except Exception:
        pass
    return np.concatenate(all_features), np.concatenate(all_labels)


def run_search(hp_grid_override=None, folds_override=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    print(f"Augmentation (vue d'entraînement) : {AUGMENT}", flush=True)
    print("Importing data pipeline (this may take a while)...", flush=True)
    # Import lazily to avoid running heavy notebook imports at module import time
    from cross_validation import load_folds  # noqa: E402
    from ipynb.fs.full.preprocessing import get_data_pipeline  # noqa: E402

    print("Chargement du pipeline de données…", flush=True)
    pipeline = get_data_pipeline(augment=AUGMENT)
    train_view = pipeline["train_pool_train_view"]
    eval_view = pipeline["train_pool_eval_view"]

    print("Chargement des folds CV…", flush=True)
    if folds_override is None:
        folds_data = load_folds()
    else:
        folds_data = folds_override
    folds = folds_data["folds"]
    print(f"  {len(folds)} folds — seed={folds_data.get('seed', SEED)}", flush=True)

    grid = hp_grid_override if hp_grid_override is not None else HP_GRID
    configs = [
        dict(zip(grid.keys(), values))
        for values in product(*grid.values())
    ]
    print(f"Grille : {len(configs)} configurations\n", flush=True)

    # Accumulateur de métriques par configuration (index -> liste de folds).
    fold_metrics_by_cfg = {i: [] for i in range(len(configs))}
    t0 = time.time()

    # Boucle externe sur les folds : on n'extrait les features qu'une seule
    # fois par fold (étape coûteuse), puis on teste toutes les configs dessus.
    for fold in folds:
        print(f"Fold {fold['fold']}/{len(folds)}")
        X_train, y_train = extract_features(
            Subset(train_view, fold["train_idx"]), desc="train")
        X_val, y_val = extract_features(
            Subset(eval_view, fold["val_idx"]), desc="val")

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_val_s = scaler.transform(X_val)
        del X_train, X_val
        gc.collect()

        # Cache des projections PCA par valeur de variance (réutilisé entre
        # configs ne différant que par C / gamma / kernel).
        pca_cache = {}

        for cfg_idx, cfg in enumerate(configs):
            pv = cfg["pca_variance"]
            if pv not in pca_cache:
                pca = PCA(n_components=pv, random_state=SEED).fit(X_train_s)
                pca_cache[pv] = (pca.n_components_, pca.transform(X_train_s),
                                 pca.transform(X_val_s))
            n_components, X_train_p, X_val_p = pca_cache[pv]

            svm = SVC(kernel=cfg["kernel"], C=cfg["C"], gamma=cfg["gamma"],
                      probability=False, random_state=SEED)
            svm.fit(X_train_p, y_train)
            preds = svm.predict(X_val_p)

            fold_metrics_by_cfg[cfg_idx].append({
                "fold": fold["fold"],
                "pca_components": int(n_components),
                "accuracy": accuracy_score(y_val, preds),
                "precision": precision_score(y_val, preds, average="macro",
                                             zero_division=0),
                "recall": recall_score(y_val, preds, average="macro",
                                       zero_division=0),
                "f1": f1_score(y_val, preds, average="macro", zero_division=0),
            })
            print(f"  [{cfg_idx + 1}/{len(configs)}] {cfg} -> "
                f"acc={fold_metrics_by_cfg[cfg_idx][-1]['accuracy']:.4f} "
                f"f1={fold_metrics_by_cfg[cfg_idx][-1]['f1']:.4f}", flush=True)

        del X_train_s, X_val_s, pca_cache
        gc.collect()
        print()

        # Save a lightweight checkpoint after each fold so progress can be inspected
        try:
            checkpoint = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "processed_fold": fold["fold"],
                "k_folds": folds_data.get("k_folds", len(folds)),
                "fold_metrics_by_cfg": fold_metrics_by_cfg,
            }
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(RESULTS_DIR / "progress.json", "w") as cf:
                json.dump(checkpoint, cf, indent=2)
        except Exception:
            pass

    # Agrégation inter-folds par configuration.
    all_results = []
    for cfg_idx, cfg in enumerate(configs):
        fold_metrics = fold_metrics_by_cfg[cfg_idx]
        means = {
            f"{k}_mean": float(np.mean([m[k] for m in fold_metrics]))
            for k in ("accuracy", "precision", "recall", "f1")
        }
        stds = {
            f"{k}_std": float(np.std([m[k] for m in fold_metrics]))
            for k in ("accuracy", "precision", "recall", "f1")
        }
        all_results.append({**cfg, "fold_scores": fold_metrics, **means, **stds})
        print(f"{cfg} -> acc_mean={means['accuracy_mean']:.4f} "
              f"f1_mean={means['f1_mean']:.4f}", flush=True)

    elapsed_min = (time.time() - t0) / 60
    print(f"\nRecherche terminée en {elapsed_min:.1f} min")

    # Best config = max f1_mean (équilibré sur 3 classes déséquilibrées).
    best = max(all_results, key=lambda r: r["f1_mean"])

    summary = {
        "model": "SVM",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_min": round(elapsed_min, 2),
        "augment": AUGMENT,
        "k_folds": folds_data["k_folds"],
        "seed": SEED,
        "hp_grid": grid,
        "results": all_results,
        "best_config": {k: best[k] for k in grid.keys()},
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Run a quick debug search (tiny grid, single fold)")
    args = parser.parse_args()

    if args.quick:
        # Small grid and only the first fold for a fast smoke-test
        quick_grid = {
            "C": [1],
            "gamma": ["scale"],
            "kernel": ["rbf"],
            "pca_variance": [0.95],
        }
        folds_data = load_folds()
        folds_data["folds"] = folds_data["folds"][:1]
        # Disable multiprocessing workers on Windows for quick run
        NUM_WORKERS = 0
        run_search(hp_grid_override=quick_grid, folds_override=folds_data)
    else:
        run_search()
