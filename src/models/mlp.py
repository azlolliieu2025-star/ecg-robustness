"""MLP (course Ch. 7) on the 123 handcrafted features, evaluated on the validation fold."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import CLASSES, SEED, bootstrap_auc, save_preds

P = Path("data/processed")
labels = pd.read_csv(P / "labels.csv", index_col="ecg_id")
y_tr = labels.loc[labels["strat_fold"] <= 8, CLASSES].values
y_va = labels.loc[labels["strat_fold"] == 9, CLASSES].values
F_tr, F_va = np.load(P / "F_train.npy"), np.load(P / "F_val.npy")

model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-3,
                  early_stopping=True, validation_fraction=0.1, max_iter=300,
                  random_state=SEED),
)
model.fit(F_tr, y_tr)              # multi-label: one sigmoid output per class
p_va = model.predict_proba(F_va)
save_preds("mlp", p_va)

auc, lo, hi = bootstrap_auc(y_va, p_va)
per_class = roc_auc_score(y_va, p_va, average=None)
row = {"model": "MLP (128-64)", "val_macro_auroc": round(auc, 3),
       "ci_low": round(lo, 3), "ci_high": round(hi, 3),
       **{f"auroc_{c}": round(a, 3) for c, a in zip(CLASSES, per_class)},
       "test": "not yet used"}
print(f"MLP macro-AUROC {auc:.3f} (95% CI {lo:.3f}-{hi:.3f})")
print("Per class:", {c: round(float(a), 3) for c, a in zip(CLASSES, per_class)})

out = Path("reports/table2_validation.csv")
table = pd.read_csv("reports/table1_validation.csv")
table = pd.concat([table[table["model"] != row["model"]], pd.DataFrame([row])], ignore_index=True)
table.to_csv(out, index=False)
print(f"Saved {out}")