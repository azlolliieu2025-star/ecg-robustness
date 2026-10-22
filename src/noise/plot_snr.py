"""Figure 3: macro-AUROC vs. SNR for each noise type and model (validation fold 9)."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("reports/nstdb_results_validation.csv")
NOISE = {"bw": "Baseline wander", "ma": "Muscle artefact", "em": "Electrode motion"}
MODELS = {"cnn": ("1D CNN", "#2a78d6"), "mlp": ("MLP", "#eb6834"),
          "lr": ("Logistic regression", "#1baf7a"), "rf": ("Random forest", "#eda100")}
X_LABELS = ["clean", "24", "12", "6", "0"]
INK, MUTED = "#0b0b0b", "#52514e"

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for ax, (rec, title) in zip(axes, NOISE.items()):
    for m, (label, color) in MODELS.items():
        clean = df[(df.noise == "none") & (df.model == m)]
        noisy = df[(df.noise == rec) & (df.model == m)].copy()
        noisy["order"] = noisy["snr_db"].astype(int).map({24: 1, 12: 2, 6: 3, 0: 4})
        noisy = noisy.sort_values("order")
        part = pd.concat([clean.assign(order=0), noisy])
        ax.plot(part["order"], part["macro_auroc"], marker="o", ms=5, lw=2, color=color, label=label)
        ax.fill_between(part["order"], part["ci_low"], part["ci_high"], color=color, alpha=0.12, lw=0)
        ax.set_ylim(0.75, 0.95)
    ax.set_xticks(range(5), X_LABELS)
    ax.set_xlabel("SNR (dB)  →  noisier", color=MUTED)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
axes[0].set_ylabel("Validation macro-AUROC", color=MUTED)
handles, names = axes[0].get_legend_handles_labels()
fig.legend(handles, names, loc="lower center", ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.08, 1, 1))
Path("reports/figures").mkdir(parents=True, exist_ok=True)
fig.savefig("reports/figures/fig3_snr.pdf", bbox_inches="tight")
fig.savefig("reports/figures/fig3_snr.png", dpi=250, bbox_inches="tight")
print("Saved reports/figures/fig3_snr.pdf and .png")