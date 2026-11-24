"""Figure: one validation ECG (lead II) clean and with each NSTDB noise type added at 6 dB and 0 dB.
Usage: python src/noise/plot_noise_examples.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wfdb
from scipy.signal import resample_poly

FS = 100
NSTDB = Path("data/raw/nstdb")
NAMES = {"bw": "baseline wander (bw)", "ma": "muscle artefact (ma)", "em": "electrode motion (em)"}
rng = np.random.default_rng(0)

X = np.load("data/processed/X_val.npy")
ecg = X[0, :, 1]                       # first validation ECG, lead II
t = np.arange(len(ecg)) / FS

def noisy(rec, snr_db):
    sig, _ = wfdb.rdsamp(str(NSTDB / rec))
    src = resample_poly(sig, up=5, down=18, axis=0)[:, 0]
    src = src[len(src) // 2:]
    start = rng.integers(0, len(src) - len(ecg))
    seg = src[start:start + len(ecg)] - src[start:start + len(ecg)].mean()
    p_sig = np.mean((ecg - ecg.mean()) ** 2)
    return ecg + seg * np.sqrt(p_sig / (np.mean(seg ** 2) * 10 ** (snr_db / 10)))

fig, axes = plt.subplots(4, 1, figsize=(7, 6), sharex=True, sharey=True)
axes[0].plot(t, ecg, color="black", lw=0.8)
axes[0].set_title("(a) clean", fontsize=10, loc="left")
for ax, rec, tag in zip(axes[1:], ["bw", "ma", "em"], "bcd"):
    ax.plot(t, noisy(rec, 0), color="#eb6834", lw=0.7, label="0 dB")
    ax.plot(t, noisy(rec, 6), color="#2a78d6", lw=0.7, label="6 dB")
    ax.set_title(f"({tag}) + {NAMES[rec]}", fontsize=10, loc="left")
for ax in axes:
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.3)
axes[1].legend(fontsize=9, loc="upper right", ncol=2)
axes[-1].set_xlabel("Time (s)", fontsize=10)
fig.supylabel("Amplitude (mV)", fontsize=10)
fig.tight_layout()
Path("reports/figures").mkdir(parents=True, exist_ok=True)
for ext in ["pdf", "png"]:
    fig.savefig(f"reports/figures/fig_noise_examples.{ext}", dpi=200)
print("Saved reports/figures/fig_noise_examples.pdf and .png")