"""Macro-AUROC vs SNR for each noise type and model. Usage: python src/noise/plot_snr.py --split val --seed 0"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["val", "test"], default="val")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

df = pd.read_csv(f"reports/nstdb_{args.split}_seed{args.seed}.csv")
df["snr_db"] = df["snr_db"].astype(str)
NOISE = {"bw": "(a) Baseline wander", "ma": "(b) Muscle artefact", "em": "(c) Electrode motion"}
MODELS = {"cnn": ("1D CNN", "#2a78d6"), "mlp": ("MLP", "#eb6834"),
          "lr": ("Logistic regression", "#1baf7a"), "rf": ("Random forest", "#eda100")}
ORDER = {"clean": 0, "24": 1, "12": 2, "6": 3, "0": 4}
MUTED = "#52514e"

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for ax, (rec, title) in zip(axes, NOISE.items()):
    for m, (label, color) in MODELS.items():
        part = df[(df.model == m) & ((df.noise == rec) | (df.noise == "none"))].copy()
        part["order"] = part["snr_db"].map(ORDER)
        part = part.sort_values("order")
        ax.plot(part["order"], part["macro_auroc"], marker="o", ms=5, lw=2, color=color, label=label)
        if "ci_low" in part:
            ax.fill_between(part["order"], part["ci_low"], part["ci_high"], color=color, alpha=0.12, lw=0)
    ax.set_xticks(range(5), list(ORDER))
    ax.set_xlabel("SNR (dB)  →  noisier", color=MUTED)
    ax.set_title(title, loc="left", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
axes[0].set_ylabel(f"{'Validation' if args.split == 'val' else 'Test'} macro-AUROC", color=MUTED)
h, n = axes[0].get_legend_handles_labels()
fig.legend(h, n, loc="lower center", ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.08, 1, 1))
Path("reports/figures").mkdir(parents=True, exist_ok=True)
out = f"reports/figures/fig_snr_{args.split}"
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", dpi=250, bbox_inches="tight")
print(f"Saved {out}.pdf and .png")
