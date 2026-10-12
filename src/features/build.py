"""Handcrafted ECG features for the baseline models (one row per ECG)."""
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

FS = 100                                   # sampling rate (Hz) of the 100 Hz records
BANDS = [(0.5, 5), (5, 15), (15, 40)]      # frequency bands (Hz) for spectral energy
IN = OUT = Path("data/processed")


def heart_rate_features(lead):
    """R-peak based rhythm features from one lead (simple peak detection)."""
    lead = lead - np.median(lead)
    peaks, _ = find_peaks(lead, distance=int(0.3 * FS), height=0.5 * np.max(lead))
    if len(peaks) < 3:
        return [np.nan, np.nan, len(peaks)]
    rr = np.diff(peaks) / FS               # RR intervals in seconds
    return [60 / rr.mean(), rr.std(), len(peaks)]   # heart rate (bpm), RR variability, beat count


def extract(X):
    """X: (n_ecgs, 1000, 12) -> feature matrix (n_ecgs, n_features)."""
    stats = [X.mean(1), X.std(1), X.min(1), X.max(1), np.ptp(X, axis=1),
             skew(X, axis=1), kurtosis(X, axis=1)]          # 7 stats x 12 leads
    spec = np.abs(np.fft.rfft(X, axis=1)) ** 2
    freqs = np.fft.rfftfreq(X.shape[1], d=1 / FS)
    total = spec.sum(1) + 1e-12
    bands = [spec[:, (freqs >= lo) & (freqs < hi)].sum(1) / total for lo, hi in BANDS]
    rhythm = np.array([heart_rate_features(x[:, 1]) for x in X])   # lead II
    return np.hstack(stats + bands + [rhythm]).astype(np.float32)


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        X = np.load(IN / f"X_{split}.npy")
        F = extract(X)
        np.save(OUT / f"F_{split}.npy", F)
        print(f"{split}: {X.shape} -> features {F.shape}")