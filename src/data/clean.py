"""Clean PTB-XL: build 5-superclass labels, check signals, split by patient-wise folds."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

RAW = Path("data/raw/ptb-xl")
OUT = Path("data/processed")
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
NOISE = ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems"]
log = {}

# --- 1. Labels: SCP codes -> 5 diagnostic superclasses ---
df = pd.read_csv(RAW / "ptbxl_database.csv", index_col="ecg_id")
df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
scp = pd.read_csv(RAW / "scp_statements.csv", index_col=0)
scp = scp[scp["diagnostic"] == 1]
log["n_records_raw"] = len(df)
log["n_patients_raw"] = int(df["patient_id"].nunique())

def to_superclasses(codes):
    return sorted({scp.loc[c, "diagnostic_class"] for c in codes if c in scp.index})

df["superclasses"] = df["scp_codes"].apply(to_superclasses)
log["n_dropped_no_superclass"] = int((df["superclasses"].map(len) == 0).sum())
df = df[df["superclasses"].map(len) > 0].copy()
for c in CLASSES:
    df[c] = df["superclasses"].apply(lambda s, c=c: int(c in s))

# --- 2. Metadata fixes and flags ---
log["n_age_300"] = int((df["age"] >= 300).sum())
df.loc[df["age"] >= 300, "age"] = 90          # anonymised >89 -> "90+"
for col in NOISE:
    df[col + "_flag"] = df[col].notna().astype(int)
df["any_noise"] = df[[c + "_flag" for c in NOISE]].max(axis=1)
df["pacemaker_flag"] = df["pacemaker"].notna().astype(int)
log["n_any_noise"] = int(df["any_noise"].sum())
log["n_dropped_pacemaker"] = int(df["pacemaker_flag"].sum())
df = df[df["pacemaker_flag"] == 0].copy()     # DECISION: exclude paced ECGs

# --- 3. Signal checks ---
X = np.zeros((len(df), 1000, 12), dtype=np.float32)
bad = []
for i, (ecg_id, fname) in enumerate(df["filename_lr"].items()):
    sig, _ = wfdb.rdsamp(str(RAW / fname))
    if sig.shape != (1000, 12) or np.isnan(sig).any():
        bad.append(ecg_id)
        continue
    X[i] = sig
keep = ~df.index.isin(bad)
X, df = X[keep], df[keep].copy()
log["n_dropped_bad_signal"] = len(bad)
df["flat_lead_flag"] = (X.std(axis=1) < 1e-3).any(axis=1).astype(int)
df["extreme_amp_flag"] = (np.abs(X).max(axis=(1, 2)) > 20).astype(int)
log["n_flat_lead"] = int(df["flat_lead_flag"].sum())
log["n_extreme_amp"] = int(df["extreme_amp_flag"].sum())

# --- 4. Patient-wise split + leakage check ---
splits = {
    "train": (df["strat_fold"] <= 8).values,
    "val": (df["strat_fold"] == 9).values,
    "test": (df["strat_fold"] == 10).values,
}
pats = {k: set(df.loc[m, "patient_id"]) for k, m in splits.items()}
assert not (pats["train"] & pats["val"]), "patient leakage train/val"
assert not (pats["train"] & pats["test"]), "patient leakage train/test"
assert not (pats["val"] & pats["test"]), "patient leakage val/test"

# --- 5. Save ---
OUT.mkdir(parents=True, exist_ok=True)
for name, m in splits.items():
    np.save(OUT / f"X_{name}.npy", X[m])
    log[f"n_{name}"] = int(m.sum())
    log[f"class_counts_{name}"] = {c: int(df.loc[m, c].sum()) for c in CLASSES}
Xtr = X[splits["train"]]
np.savez(OUT / "norm_stats.npz", mean=Xtr.mean(axis=(0, 1)), std=Xtr.std(axis=(0, 1)))

flag_cols = [c + "_flag" for c in NOISE] + ["any_noise", "flat_lead_flag", "extreme_amp_flag"]
df[["patient_id", "age", "sex", "strat_fold"] + CLASSES + flag_cols].to_csv(OUT / "labels.csv")

Path("reports").mkdir(exist_ok=True)
with open("reports/cleaning_log.json", "w") as f:
    json.dump(log, f, indent=2)
print(json.dumps(log, indent=2))