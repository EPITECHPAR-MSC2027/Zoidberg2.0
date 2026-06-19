"""Génération des folds de cross-validation.

Génère **une seule fois** les folds StratifiedKFold sur le pool train+validation
et les sauvegarde dans ``results/folds.json``. Tous les modèles du projet
(CNN, SVM, Random Forest, …) doivent réutiliser ces folds via
``load_folds()`` pour garantir une comparaison équitable.

Le jeu de test final (split ``test`` HuggingFace) n'est **jamais** touché ici.

Usage::

    python cross_validation.py            # génère folds.json s'il n'existe pas
    python cross_validation.py --force    # régénère même si présent
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from ipynb.fs.full.preprocessing import load_raw_dataset

ROOT = Path(__file__).resolve().parent
FOLDS_PATH = ROOT / "results" / "folds.json"

K_FOLDS = 3
SEED = 42


def generate_folds(k: int = K_FOLDS, seed: int = SEED) -> dict:
    """Construit les folds StratifiedKFold sur le pool train+validation."""
    train_pool_raw, _ = load_raw_dataset()
    labels = np.array(train_pool_raw["label"])
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels)
    ):
        folds.append({
            "fold": fold_idx + 1,
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
        })

    return {
        "k_folds": k,
        "seed": seed,
        "n_samples": int(len(labels)),
        "label_distribution": np.bincount(labels).tolist(),
        "folds": folds,
    }


def load_folds() -> dict:
    """Charge folds.json. Le génère s'il n'existe pas."""
    if not FOLDS_PATH.exists():
        save_folds(generate_folds())
    with open(FOLDS_PATH) as f:
        return json.load(f)


def save_folds(data: dict) -> None:
    FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FOLDS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Régénère folds.json même s'il existe.")
    parser.add_argument("--k", type=int, default=K_FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if FOLDS_PATH.exists() and not args.force:
        print(f"folds.json déjà présent ({FOLDS_PATH}). --force pour régénérer.")
        data = load_folds()
    else:
        print(f"Génération des folds (K={args.k}, seed={args.seed})…")
        data = generate_folds(k=args.k, seed=args.seed)
        save_folds(data)
        print(f"Sauvegardé : {FOLDS_PATH}")

    print(f"  n_samples = {data['n_samples']}")
    print(f"  distribution = {data['label_distribution']}")
    for f in data["folds"]:
        print(f"  Fold {f['fold']}: train={len(f['train_idx'])} | val={len(f['val_idx'])}")


if __name__ == "__main__":
    main()
