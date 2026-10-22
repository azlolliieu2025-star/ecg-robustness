"""Shared evaluation helpers: macro-AUC with bootstrap CI, and saving predictions."""
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

SEED = 0
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
PRED_DIR = Path("data/processed/preds")


def bootstrap_auc(y, p, n=1000, seed=SEED):
    """Macro-AUC with a 95% bootstrap CI (resampling ECGs with replacement)."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if (y[idx].sum(0) == 0).any():      # skip resamples missing a class
            continue
        scores.append(roc_auc_score(y[idx], p[idx], average="macro"))
    return roc_auc_score(y, p, average="macro"), *np.percentile(scores, [2.5, 97.5])


def save_preds(name, p_val):
    """Save validation probabilities so other scripts (e.g. noise subgroups) can reuse them."""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PRED_DIR / f"{name}_val.npy", p_val)