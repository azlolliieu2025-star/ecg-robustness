"""Which diagnoses suffer most from noise? Drop in per-class AUROC from clean to 0 dB,
averaged over the three noise types. Usage: python src/noise/plot_perclass_noise.py --split val --seed 0"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["val", "test"], default="val")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
MODELS = {"cnn": ("1D CNN", "#2a78d6"), "mlp": ("MLP", "#eb6834"),
          "lr": ("Logistic regression", "#1baf7a"), "rf": ("Random forest", "#eda100")}
df = pd.read_csv(f"reports/nstdb_{args.split}_seed{args.seed}.csv")
df["snr_db"] = df["snr_db"].astype(str)

rows = []
for m in MODELS:
    clean = df[(df.model == m) & (df.noise == "none")][[f"auroc_{c}" for c in CLASSES]].values[0]
    zero = df[(df.model == m) & (df.snr_db == "0")][[f"auroc_{c}" for c in CLASSES]].values.mean(0)
    rows.append({"model": m, **{c: round(z - k, 4) for c, z, k in zip(CLASSES, zero, clean)}})
table = pd.DataFrame(rows)
table.to_csv(f"reports/perclass_drop_0db_{args.split}.csv", index=False)
print(table.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 3.4))
x, w = np.arange(len(CLASSES)), 0.2
for i, (m, (label, color)) in enumerate(MODELS.items()):
    ax.bar(x + (i - 1.5) * w, table.loc[table.model == m, CLASSES].values[0], w - 0.02, color=color, label=label)
ax.axhline(0, color="#52514e", lw=0.8)
ax.set_xticks(x, CLASSES)
ax.set_ylabel("Change in AUROC, clean → 0 dB", color="#52514e")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, ncol=2)
fig.tight_layout()
out = f"reports/figures/fig_perclass_noise_{args.split}"
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", dpi=250, bbox_inches="tight")
print(f"Saved {out}.pdf and .png")
