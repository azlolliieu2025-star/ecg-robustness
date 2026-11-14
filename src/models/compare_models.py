"""Final model comparison on one split (val or test), from the clean predictions saved by nstdb_experiment.py:
  1) macro-AUROC + 95% CI + per-class AUROC per model
  2) clean vs naturally noisy ECGs (PTB-XL noise flags)
  3) paired bootstrap: 95% CI for the DIFFERENCE in macro-AUROC between pairs of models

Usage: python src/models/compare_models.py --split val   (or --split test)
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from evaluate import CLASSES, PRED_DIR, SEED, bootstrap_auc

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["val", "test"], default="val")
parser.add_argument("--n", type=int, default=1000, help="bootstrap resamples")
args = parser.parse_args()

FOLD = {"val": 9, "test": 10}[args.split]
labels = pd.read_csv("data/processed/labels.csv", index_col="ecg_id")
part = labels[labels["strat_fold"] == FOLD]
y = part[CLASSES].values
noisy = part["any_noise"].values == 1
MODELS = ["lr", "rf", "mlp", "cnn"]
preds = {m: np.load(PRED_DIR / f"{m}_{args.split}.npy") for m in MODELS}
print(f"Split {args.split} (fold {FOLD}): n = {len(y)} | clean {(~noisy).sum()} | noisy {noisy.sum()}")

# 1) overall + per class
rows = []
for m, p in preds.items():
    auc, lo, hi = bootstrap_auc(y, p, n=args.n)
    pc = roc_auc_score(y, p, average=None)
    rows.append({"model": m, "macro_auroc": round(auc, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                 **{f"auroc_{c}": round(float(a), 3) for c, a in zip(CLASSES, pc)}})
    print(f"{m:4s} {auc:.3f} ({lo:.3f}-{hi:.3f})  " + " ".join(f"{c} {a:.3f}" for c, a in zip(CLASSES, pc)))
pd.DataFrame(rows).to_csv(f"reports/final_{args.split}_models.csv", index=False)

# 2) natural noise subgroups
rows = []
for m, p in preds.items():
    r = {"model": m}
    for g, mask in [("clean", ~noisy), ("noisy", noisy)]:
        auc, lo, hi = bootstrap_auc(y[mask], p[mask], n=args.n)
        r.update({f"{g}": round(auc, 3), f"{g}_ci_low": round(lo, 3), f"{g}_ci_high": round(hi, 3)})
    r["difference"] = round(r["noisy"] - r["clean"], 3)
    rows.append(r)
    print(f"{m:4s} clean {r['clean']:.3f}  noisy {r['noisy']:.3f}  diff {r['difference']:+.3f}")
pd.DataFrame(rows).to_csv(f"reports/final_{args.split}_natural_noise.csv", index=False)

# 3) paired bootstrap: same resampled ECGs for both models in every round
rng = np.random.default_rng(SEED)
idx_all = [rng.integers(0, len(y), len(y)) for _ in range(args.n)]
idx_all = [i for i in idx_all if (y[i].sum(0) > 0).all()]
rows = []
for a, b in combinations(["cnn", "mlp", "lr", "rf"], 2):
    diffs = np.array([roc_auc_score(y[i], preds[a][i], average="macro") -
                      roc_auc_score(y[i], preds[b][i], average="macro") for i in idx_all])
    obs = roc_auc_score(y, preds[a], average="macro") - roc_auc_score(y, preds[b], average="macro")
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    rows.append({"model_a": a, "model_b": b, "difference": round(obs, 3), "ci_low": round(lo, 3),
                 "ci_high": round(hi, 3), "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    print(f"{a} - {b}: {obs:+.3f} ({lo:+.3f} to {hi:+.3f})  {'clear difference' if lo > 0 or hi < 0 else 'no clear difference'}")
pd.DataFrame(rows).to_csv(f"reports/final_{args.split}_paired.csv", index=False)
print(f"Saved reports/final_{args.split}_models.csv, _natural_noise.csv, _paired.csv")
