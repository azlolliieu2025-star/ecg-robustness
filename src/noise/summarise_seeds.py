"""Combine NSTDB runs with different noise seeds: mean and std of macro-AUROC per condition and model.
Usage: python src/noise/summarise_seeds.py --split val"""
import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["val", "test"], default="val")
args = parser.parse_args()

files = sorted(Path("reports").glob(f"nstdb_{args.split}_seed*.csv"))
df = pd.concat([pd.read_csv(f) for f in files])
df["snr_db"] = df["snr_db"].astype(str)
print(f"{len(files)} seed runs found: {sorted(df['seed'].unique())}")
summ = (df.groupby(["noise", "snr_db", "model"])["macro_auroc"]
          .agg(mean="mean", std="std", n_seeds="count").round(4).reset_index())
summ.to_csv(f"reports/nstdb_{args.split}_seed_summary.csv", index=False)

clean = summ[summ.noise == "none"].set_index("model")["mean"]
print("\nDrop from clean to 0 dB (mean ± std over seeds):")
for rec in ["bw", "ma", "em"]:
    zero = summ[(summ.noise == rec) & (summ.snr_db == "0")].set_index("model")
    print(f"  {rec}: " + "  ".join(f"{m} {zero.loc[m, 'mean'] - clean[m]:+.3f}±{zero.loc[m, 'std']:.3f}"
                                   for m in ["cnn", "mlp", "lr", "rf"]))
    print(f"      CNN - MLP gap at 0 dB: {zero.loc['cnn', 'mean'] - zero.loc['mlp', 'mean']:.3f}")
print(f"Saved reports/nstdb_{args.split}_seed_summary.csv")
