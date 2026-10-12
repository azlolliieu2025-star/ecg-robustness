"""Figure 1: class prevalence per split and diagnosis co-occurrence (training set)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SPLITS = {"Train (folds 1-8)": [1, 2, 3, 4, 5, 6, 7, 8],
          "Validation (fold 9)": [9],
          "Test (fold 10)": [10]}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]          # one colour per split
INK, MUTED = "#0b0b0b", "#52514e"

labels = pd.read_csv("data/processed/labels.csv", index_col="ecg_id")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6),
                               gridspec_kw={"width_ratios": [1.25, 1]})

# --- Panel A: % of ECGs with each diagnosis, per split ---
x = np.arange(len(CLASSES))
width = 0.26
for i, (name, folds) in enumerate(SPLITS.items()):
    part = labels[labels["strat_fold"].isin(folds)]
    pct = part[CLASSES].mean() * 100
    bars = ax1.bar(x + (i - 1) * width, pct, width - 0.03, color=COLORS[i],
                   label=f"{name}, n = {len(part):,}")
    if i == 0:   # label only the training bars to avoid clutter
        for b, v in zip(bars, pct):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.0f}%",
                     ha="center", va="bottom", fontsize=9, color=INK)
ax1.set_xticks(x, CLASSES)
ax1.set_ylabel("ECGs with diagnosis (%)", color=MUTED)
ax1.set_title("(a) Class prevalence is imbalanced but equal across splits",
              loc="left", fontsize=11, color=INK)
ax1.legend(frameon=False, fontsize=9)
ax1.set_ylim(0, labels[CLASSES].mean().max() * 100 + 10)

# --- Panel B: P(column | row) in training set ---
train = labels[labels["strat_fold"] <= 8][CLASSES]
counts = train.T.dot(train)                        # co-occurrence counts
cond = counts.div(np.diag(counts), axis=0) * 100   # row-normalised %
off = cond.values.astype(float).copy()
np.fill_diagonal(off, np.nan)                      # diagonal is always 100%: hide it
vmax = np.nanmax(off)
im = ax2.imshow(off, cmap="Blues", vmin=0, vmax=vmax)
for r in range(len(CLASSES)):
    for c in range(len(CLASSES)):
        v = off[r, c]
        txt = "–" if np.isnan(v) else f"{v:.0f}"
        ax2.text(c, r, txt, ha="center", va="center", fontsize=9,
                 color="white" if (not np.isnan(v) and v > 0.6 * vmax) else INK)
ax2.set_xticks(range(len(CLASSES)), CLASSES)
ax2.set_yticks(range(len(CLASSES)), CLASSES)
ax2.set_xlabel("…also has this diagnosis", color=MUTED)
ax2.set_ylabel("ECGs with this diagnosis…", color=MUTED)
ax2.set_title("(b) Abnormal diagnoses often co-occur; NORM rarely does",
              loc="left", fontsize=11, color=INK)
fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="%")

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)

fig.tight_layout()
Path("reports/figures").mkdir(parents=True, exist_ok=True)
fig.savefig("reports/figures/fig1_classes.png", dpi=300, bbox_inches="tight")
fig.savefig("reports/figures/fig1_classes.pdf", bbox_inches="tight")
print("Saved reports/figures/fig1_classes.png and .pdf")