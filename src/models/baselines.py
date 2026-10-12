"""Baselines for Table 1: prevalence, logistic regression, random forest (validation only)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
P = Path("data/processed")

labels = pd.read_csv(P / "labels.csv", index_col="ecg_id")
y_tr = labels.loc[labels["strat_fold"] <= 8, CLASSES].values
y_va = labels.loc[labels["strat_fold"] == 9, CLASSES].values
F_tr, F_va = np.load(P / "F_train.npy"), np.load(P / "F_val.npy")


def bootstrap_auc(y, p, n=1000, seed=SEED):
    """Macro-AUROC with a 95% bootstrap CI (resampling validation ECGs)."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        scores.append(roc_auc_score(y[idx], p[idx], average="macro"))
    return roc_auc_score(y, p, average="macro"), *np.percentile(scores, [2.5, 97.5])


models = {
    "Prevalence (no signal)": None,
    "Logistic regression": OneVsRestClassifier(make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"))),
    "Random forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                               class_weight="balanced", n_jobs=-1, random_state=SEED)),
}

rows = []
for name, model in models.items():
    if model is None:   # every ECG gets the training prevalence -> AUROC 0.5 by construction
        p_va = np.tile(y_tr.mean(0), (len(y_va), 1))
    else:
        model.fit(F_tr, y_tr)
        p_va = model.predict_proba(F_va)
        if isinstance(p_va, list):              # random forest returns one array per class
            p_va = np.column_stack([pc[:, 1] for pc in p_va])
    auc, lo, hi = bootstrap_auc(y_va, p_va)
    per_class = roc_auc_score(y_va, p_va, average=None)
    rows.append({"model": name, "val_macro_auroc": round(auc, 3),
                 "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                 **{f"auroc_{c}": round(a, 3) for c, a in zip(CLASSES, per_class)},
                 "test": "not yet used"})
    print(f"{name:25s} macro-AUROC {auc:.3f} (95% CI {lo:.3f}-{hi:.3f})")

Path("reports").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv("reports/table1_validation.csv", index=False)
print("\nSaved reports/table1_validation.csv")