"""Controlled noise experiment: add real NSTDB noise (bw, ma, em) to the ECGs of one split at fixed SNRs
and score every model (overall and per class). All models are trained on clean data only.

Usage:
  python src/noise/nstdb_experiment.py --split val --seed 0          # validation, with 95% bootstrap CIs
  python src/noise/nstdb_experiment.py --split val --seed 1 --no-ci  # extra noise draw, point estimates only
  python src/noise/nstdb_experiment.py --split test --seed 0         # FINAL test evaluation (run once)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path += [str(ROOT / "models"), str(ROOT / "features")]

import joblib
import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import resample_poly
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build import extract
from cnn_arch import build_cnn
from evaluate import CLASSES, PRED_DIR, SEED, bootstrap_auc

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["val", "test"], default="val")
parser.add_argument("--seed", type=int, default=0, help="seed for the random noise segments")
parser.add_argument("--no-ci", action="store_true", help="skip bootstrap CIs (faster)")
args = parser.parse_args()

NSTDB = Path("data/raw/nstdb")
P = Path("data/processed")
NOISE_TYPES = ["bw", "ma", "em"]
SNRS = [24, 12, 6, 0]
FOLD = {"val": 9, "test": 10}[args.split]
rng = np.random.default_rng(args.seed)
print(f"Split: {args.split} (fold {FOLD}) | noise seed: {args.seed} | CIs: {not args.no_ci}")

# --- 1. NSTDB noise: 360 Hz -> 100 Hz; only the 2nd half of each record is used for evaluation ---
noise = {}
for rec in NOISE_TYPES:
    sig, fields = wfdb.rdsamp(str(NSTDB / rec))
    assert fields["fs"] == 360, f"unexpected sampling rate in {rec}"
    sig = resample_poly(sig, up=5, down=18, axis=0)
    noise[rec] = sig[len(sig) // 2:].astype(np.float32)


def add_noise(X, rec, snr_db):
    """Random 10 s segment per ECG and lead, scaled so that SNR_dB = 10 log10(P_signal / P_noise)."""
    src = noise[rec]
    n, T, L = X.shape
    out = X.copy()
    for i in range(n):
        for l in range(L):
            start = rng.integers(0, len(src) - T)
            seg = src[start:start + T, rng.integers(0, src.shape[1])]
            seg = seg - seg.mean()
            p_sig = np.mean((X[i, :, l] - X[i, :, l].mean()) ** 2)
            p_noise = np.mean(seg ** 2) + 1e-12
            out[i, :, l] += seg * np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))
    return out


# --- 2. Data and models (trained on CLEAN training data only; fitted once, then reused) ---
labels = pd.read_csv(P / "labels.csv", index_col="ecg_id")
y_tr = labels.loc[labels["strat_fold"] <= 8, CLASSES].values
y = labels.loc[labels["strat_fold"] == FOLD, CLASSES].values
X = np.load(P / f"X_{args.split}.npy")
assert len(X) == len(y), "signals and labels do not match"

cache = Path("models/feature_models.joblib")
if cache.exists():
    feature_models = joblib.load(cache)
    print("Loaded fitted LR / RF / MLP from", cache)
else:
    F_tr = np.load(P / "F_train.npy")
    feature_models = {
        "lr": OneVsRestClassifier(make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                                LogisticRegression(max_iter=2000, class_weight="balanced"))),
        "rf": make_pipeline(SimpleImputer(strategy="median"),
                            RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                                   n_jobs=-1, random_state=SEED)),
        "mlp": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True,
                                           validation_fraction=0.1, max_iter=300, random_state=SEED)),
    }
    for name, m in feature_models.items():
        print(f"Fitting {name} on clean training features...")
        m.fit(F_tr, y_tr)
    Path("models").mkdir(exist_ok=True)
    joblib.dump(feature_models, cache)

device = "mps" if torch.backends.mps.is_available() else "cpu"
cnn = build_cnn().to(device)
cnn.load_state_dict(torch.load("models/cnn1d_best.pt", map_location=device))
cnn.eval()
stats = np.load(P / "norm_stats.npz")
mu, sd = stats["mean"].astype(np.float32), stats["std"].astype(np.float32) + 1e-6


def predict_all(Xs):
    F = extract(Xs)
    preds = {}
    for name, m in feature_models.items():
        p = m.predict_proba(F)
        preds[name] = np.column_stack([pc[:, 1] for pc in p]) if isinstance(p, list) else p
    Xc = np.ascontiguousarray(((Xs - mu) / sd).transpose(0, 2, 1), dtype=np.float32)
    with torch.no_grad():
        preds["cnn"] = np.concatenate([torch.sigmoid(cnn(torch.from_numpy(Xc[i:i + 512]).to(device))).cpu().numpy()
                                       for i in range(0, len(Xc), 512)])
    return preds


# --- 3. Run: clean + 3 noise types x 4 SNRs ---
rows = []
for rec, snr in [("none", None)] + [(r, s) for r in NOISE_TYPES for s in SNRS]:
    Xn = X if rec == "none" else add_noise(X, rec, snr)
    preds = predict_all(Xn)
    if rec == "none":                                  # keep clean predictions for compare_models.py
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        for name, p in preds.items():
            np.save(PRED_DIR / f"{name}_{args.split}.npy", p)
    for name, p in preds.items():
        row = {"split": args.split, "seed": args.seed, "noise": rec,
               "snr_db": "clean" if snr is None else snr, "model": name}
        if args.no_ci:
            row["macro_auroc"] = round(roc_auc_score(y, p, average="macro"), 4)
        else:
            auc, lo, hi = bootstrap_auc(y, p)
            row.update(macro_auroc=round(auc, 4), ci_low=round(lo, 4), ci_high=round(hi, 4))
        per_class = roc_auc_score(y, p, average=None)
        row.update({f"auroc_{c}": round(float(a), 4) for c, a in zip(CLASSES, per_class)})
        rows.append(row)
    label = "clean" if snr is None else f"{rec} {snr:>2} dB"
    print(f"{label:10s} " + "  ".join(f"{r['model']} {r['macro_auroc']:.3f}" for r in rows[-4:]))

out = Path(f"reports/nstdb_{args.split}_seed{args.seed}.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print("Saved", out)
