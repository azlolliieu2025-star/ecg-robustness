"""Controlled noise experiment: add real NSTDB noise (bw, ma, em) to the validation ECGs at
fixed SNRs and measure how each model's macro-AUROC changes. Models are trained on clean data only."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path += [str(ROOT / "models"), str(ROOT / "features")]

import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import resample_poly
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build import extract                     # same 123 features as the baselines
from cnn_arch import build_cnn
from evaluate import CLASSES, SEED, bootstrap_auc

NSTDB = Path("data/raw/nstdb")
P = Path("data/processed")
NOISE_TYPES = {"bw": "baseline wander", "ma": "muscle artefact", "em": "electrode motion"}
SNRS = [24, 12, 6, 0]                        # dB; lower = noisier
rng = np.random.default_rng(SEED)

# --- 1. Load NSTDB noise, resample 360 Hz -> 100 Hz, keep only the 2nd half for evaluation ---
#        (the 1st half is reserved for any future noise-augmented training, so no noise is reused)
noise = {}
for rec in NOISE_TYPES:
    sig, fields = wfdb.rdsamp(str(NSTDB / rec))
    assert fields["fs"] == 360, f"unexpected sampling rate in {rec}"
    sig = resample_poly(sig, up=5, down=18, axis=0)        # 360 * 5/18 = 100 Hz
    noise[rec] = sig[len(sig) // 2:].astype(np.float32)
    print(f"{rec}: {sig.shape[0]} samples at 100 Hz, using {noise[rec].shape[0]} for evaluation")


def add_noise(X, rec, snr_db):
    """Add a random 10 s noise segment to every lead of every ECG, scaled to the target SNR.
    SNR_dB = 10 log10(P_signal / P_noise), computed per ECG and per lead."""
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


# --- 2. Data and models (all trained on CLEAN training data only) ---
labels = pd.read_csv(P / "labels.csv", index_col="ecg_id")
y_tr = labels.loc[labels["strat_fold"] <= 8, CLASSES].values
y_va = labels.loc[labels["strat_fold"] == 9, CLASSES].values
X_va = np.load(P / "X_val.npy")
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

device = "mps" if torch.backends.mps.is_available() else "cpu"
cnn = build_cnn().to(device)
cnn.load_state_dict(torch.load("models/cnn1d_best.pt", map_location=device))
cnn.eval()
stats = np.load(P / "norm_stats.npz")
mu, sd = stats["mean"].astype(np.float32), stats["std"].astype(np.float32) + 1e-6


def predict_all(X):
    """Probabilities from every model for a (n, 1000, 12) signal array."""
    F = extract(X)
    preds = {}
    for name, m in feature_models.items():
        p = m.predict_proba(F)
        preds[name] = np.column_stack([pc[:, 1] for pc in p]) if isinstance(p, list) else p
    Xc = np.ascontiguousarray(((X - mu) / sd).transpose(0, 2, 1), dtype=np.float32)
    with torch.no_grad():
        preds["cnn"] = np.concatenate([torch.sigmoid(cnn(torch.from_numpy(Xc[i:i + 512]).to(device))).cpu().numpy()
                                       for i in range(0, len(Xc), 512)])
    return preds


# --- 3. Run: clean + 3 noise types x 4 SNRs ---
rows = []
conditions = [("none", None)] + [(rec, s) for rec in NOISE_TYPES for s in SNRS]
for rec, snr in conditions:
    Xn = X_va if rec == "none" else add_noise(X_va, rec, snr)
    for name, p in predict_all(Xn).items():
        auc, lo, hi = bootstrap_auc(y_va, p)
        rows.append({"noise": rec, "snr_db": "clean" if snr is None else snr, "model": name,
                     "macro_auroc": round(auc, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
    label = "clean" if snr is None else f"{rec} {snr:>2} dB"
    print(f"{label:10s} " + "  ".join(f"{r['model']} {r['macro_auroc']:.3f}" for r in rows[-4:]))

pd.DataFrame(rows).to_csv("reports/nstdb_results_validation.csv", index=False)
print("Saved reports/nstdb_results_validation.csv")