"""Filtering baseline: can a standard band-pass filter (0.5-40 Hz, a Fourier-domain method) remove the NSTDB
noise well enough for the feature-based models to catch up with the CNN?
For every noise type at 12, 6 and 0 dB, every model is scored on the noisy ECGs WITHOUT and WITH filtering.
All models are the same ones trained on clean (unfiltered) data; nothing is retrained.

Usage:
  python src/noise/filter_experiment.py --split val --seed 0
  python src/noise/filter_experiment.py --split test --seed 0     # run once, after the validation run
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path += [str(ROOT / "models"), str(ROOT / "features")]

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import butter, filtfilt, resample_poly
from sklearn.metrics import roc_auc_score

from build import extract
from cnn_arch import build_cnn
from evaluate import CLASSES

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["val", "test"], default="val")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

FS = 100
NSTDB = Path("data/raw/nstdb")
P = Path("data/processed")
NOISE_TYPES = ["bw", "ma", "em"]
SNRS = [12, 6, 0]
FOLD = {"val": 9, "test": 10}[args.split]
rng = np.random.default_rng(args.seed)
COLORS = {"cnn": "#2a78d6", "mlp": "#eb6834", "lr": "#1baf7a", "rf": "#eda100"}

# --- band-pass filter: 4th-order Butterworth, 0.5-40 Hz, zero-phase (filtfilt) ---
B, A = butter(4, [0.5, 40], btype="bandpass", fs=FS)

def bandpass(X):
    return filtfilt(B, A, X, axis=1).astype(np.float32)

# --- NSTDB noise (identical procedure to nstdb_experiment.py) ---
noise = {}
for rec in NOISE_TYPES:
    sig, fields = wfdb.rdsamp(str(NSTDB / rec))
    sig = resample_poly(sig, up=5, down=18, axis=0)
    noise[rec] = sig[len(sig) // 2:].astype(np.float32)

def add_noise(X, rec, snr_db):
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

# --- data and the already-trained models ---
labels = pd.read_csv(P / "labels.csv", index_col="ecg_id")
y = labels.loc[labels["strat_fold"] == FOLD, CLASSES].values
X = np.load(P / f"X_{args.split}.npy")
feature_models = joblib.load("models/feature_models.joblib")
device = "mps" if torch.backends.mps.is_available() else "cpu"
cnn = build_cnn().to(device)
cnn.load_state_dict(torch.load("models/cnn1d_best.pt", map_location=device))
cnn.eval()
stats = np.load(P / "norm_stats.npz")
mu, sd = stats["mean"].astype(np.float32), stats["std"].astype(np.float32) + 1e-6

def scores(Xs):
    F = extract(Xs)
    out = {}
    for name, m in feature_models.items():
        p = m.predict_proba(F)
        p = np.column_stack([pc[:, 1] for pc in p]) if isinstance(p, list) else p
        out[name] = roc_auc_score(y, p, average="macro")
    Xc = np.ascontiguousarray(((Xs - mu) / sd).transpose(0, 2, 1), dtype=np.float32)
    with torch.no_grad():
        p = np.concatenate([torch.sigmoid(cnn(torch.from_numpy(Xc[i:i + 512]).to(device))).cpu().numpy()
                            for i in range(0, len(Xc), 512)])
    out["cnn"] = roc_auc_score(y, p, average="macro")
    return out

# --- run ---
rows = []
for rec, snr in [("none", "clean")] + [(r, s) for r in NOISE_TYPES for s in SNRS]:
    Xn = X if rec == "none" else add_noise(X, rec, snr)
    for filt in ["raw", "filtered"]:
        s = scores(bandpass(Xn) if filt == "filtered" else Xn)
        for m, v in s.items():
            rows.append({"split": args.split, "seed": args.seed, "noise": rec, "snr_db": snr,
                         "input": filt, "model": m, "macro_auc": round(v, 4)})
        print(f"{rec:5s} {str(snr):>5s} {filt:8s} " + "  ".join(f"{m} {v:.3f}" for m, v in s.items()))

df = pd.DataFrame(rows)
Path("reports/figures").mkdir(parents=True, exist_ok=True)
df.to_csv(f"reports/filter_{args.split}_seed{args.seed}.csv", index=False)

# --- figure: macro-AUC at 0 dB, raw vs filtered, per noise type and model ---
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
models = ["cnn", "mlp", "lr", "rf"]
x = np.arange(len(models))
for ax, rec, tag in zip(axes, NOISE_TYPES, "abc"):
    d = df[(df.noise == rec) & (df.snr_db == 0)].set_index(["input", "model"])["macro_auc"]
    ax.bar(x - 0.2, [d["raw", m] for m in models], 0.4, color=[COLORS[m] for m in models], alpha=0.45)
    ax.bar(x + 0.2, [d["filtered", m] for m in models], 0.4, color=[COLORS[m] for m in models])
    ax.set_xticks(x, [m.upper() for m in models])
    ax.set_title(f"({tag}) {rec} at 0 dB", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel(f"{'Validation' if args.split == 'val' else 'Test'} macro-AUC", fontsize=11)
axes[0].set_ylim(0.7, 0.95)
handles = [plt.Rectangle((0, 0), 1, 1, color="grey", alpha=0.45), plt.Rectangle((0, 0), 1, 1, color="grey")]
fig.legend(handles, ["noisy, unfiltered", "noisy, band-pass filtered"], loc="lower center", ncol=2, fontsize=10,
           frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1))
for ext in ["pdf", "png"]:
    fig.savefig(f"reports/figures/fig_filter_{args.split}.{ext}", dpi=200)
print(f"Saved reports/filter_{args.split}_seed{args.seed}.csv and reports/figures/fig_filter_{args.split}.pdf/.png")