"""Clean vs. naturally noisy validation ECGs: macro-AUROC per model and subgroup."""
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import CLASSES, PRED_DIR, bootstrap_auc

labels = pd.read_csv("data/processed/labels.csv", index_col="ecg_id")
val = labels[labels["strat_fold"] == 9]
y_va = val[CLASSES].values
noisy = val["any_noise"].values == 1
print(f"Validation ECGs: {len(val)} | clean: {(~noisy).sum()} | noisy: {noisy.sum()}")

rows = []
for f in sorted(PRED_DIR.glob("*_val.npy")):
    name = f.stem.replace("_val", "")
    p = np.load(f)
    for group, mask in [("clean", ~noisy), ("noisy", noisy)]:
        auc, lo, hi = bootstrap_auc(y_va[mask], p[mask])
        rows.append({"model": name, "subgroup": group, "n": int(mask.sum()),
                     "macro_auroc": round(auc, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
        print(f"{name:8s} {group:6s} n={mask.sum():5d}  {auc:.3f} ({lo:.3f}-{hi:.3f})")

pd.DataFrame(rows).to_csv("reports/noise_subgroups_validation.csv", index=False)
print("Saved reports/noise_subgroups_validation.csv")