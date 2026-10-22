"""1D CNN on the raw 12-lead signal (multi-label, 5 superclasses), evaluated on the validation fold."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from evaluate import CLASSES, SEED, bootstrap_auc, save_preds

EPOCHS, BATCH, LR, PATIENCE = 20, 128, 1e-3, 4
P = Path("data/processed")
torch.manual_seed(SEED)
np.random.seed(SEED)
device = "mps" if torch.backends.mps.is_available() else "cpu"   # Apple GPU if available
print("Device:", device)

# --- Data: standardise each lead with TRAINING statistics, shape (N, 12, 1000) ---
labels = pd.read_csv(P / "labels.csv", index_col="ecg_id")
train_lab = labels[labels["strat_fold"] <= 8]
inner = (train_lab["strat_fold"] == 8).values          # fold 8 = early-stopping set (patient-wise)
y_all = train_lab[CLASSES].values.astype(np.float32)
y_tr, y_es = y_all[~inner], y_all[inner]
y_va = labels.loc[labels["strat_fold"] == 9, CLASSES].values.astype(np.float32)
stats = np.load(P / "norm_stats.npz")
mu, sd = stats["mean"].astype(np.float32), stats["std"].astype(np.float32) + 1e-6


def prep(X):
    return np.ascontiguousarray(((X - mu) / sd).transpose(0, 2, 1), dtype=np.float32)


X_all = prep(np.load(P / "X_train.npy"))
X_tr, X_es = X_all[~inner], X_all[inner]
del X_all
X_va = prep(np.load(P / "X_val.npy"))
print(f"Train (folds 1-7): {len(X_tr)} | early-stopping (fold 8): {len(X_es)} | validation (fold 9): {len(X_va)}")


# --- Model: 4 conv blocks -> global average pooling -> 5 outputs ---
def block(c_in, c_out):
    return nn.Sequential(nn.Conv1d(c_in, c_out, kernel_size=7, padding=3),
                         nn.BatchNorm1d(c_out), nn.ReLU(), nn.MaxPool1d(2))


model = nn.Sequential(block(12, 32), block(32, 64), block(64, 128), block(128, 128),
                      nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3),
                      nn.Linear(128, 5)).to(device)
print("Parameters:", sum(p.numel() for p in model.parameters()))

# Class weights for the imbalance: pos_weight = negatives / positives (training set)
pos_weight = torch.tensor((1 - y_tr.mean(0)) / y_tr.mean(0), device=device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)   # sigmoid + binary cross-entropy
opt = torch.optim.Adam(model.parameters(), lr=LR)


def predict(X):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = torch.from_numpy(X[i:i + 512]).to(device)
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


# --- Training with early stopping on fold 8 (fold 9 stays untouched for reporting) ---
Path("models").mkdir(exist_ok=True)
best, wait, history = -1, 0, []
rng = np.random.default_rng(SEED)
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    order = rng.permutation(len(X_tr))
    total = 0.0
    for i in range(0, len(order), BATCH):
        idx = order[i:i + BATCH]
        xb = torch.from_numpy(X_tr[idx]).to(device)
        yb = torch.from_numpy(y_tr[idx]).to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    val_auc = roc_auc_score(y_es, predict(X_es), average="macro")
    history.append({"epoch": epoch, "train_loss": total / len(X_tr), "es_macro_auroc": val_auc})
    print(f"Epoch {epoch:2d}  loss {total / len(X_tr):.4f}  fold-8 macro-AUROC {val_auc:.3f}  ({time.time() - t0:.0f}s)")
    if val_auc > best:
        best, wait = val_auc, 0
        torch.save(model.state_dict(), "models/cnn1d_best.pt")
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early stopping.")
            break

# --- Evaluate the best epoch ---
model.load_state_dict(torch.load("models/cnn1d_best.pt"))
p_va = predict(X_va)
save_preds("cnn", p_va)
auc, lo, hi = bootstrap_auc(y_va, p_va)
per_class = roc_auc_score(y_va, p_va, average=None)
print(f"\n1D CNN macro-AUROC {auc:.3f} (95% CI {lo:.3f}-{hi:.3f})")
print("Per class:", {c: round(float(a), 3) for c, a in zip(CLASSES, per_class)})

row = {"model": "1D CNN", "val_macro_auroc": round(auc, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3),
       **{f"auroc_{c}": round(float(a), 3) for c, a in zip(CLASSES, per_class)}, "test": "not yet used"}
table = pd.read_csv("reports/table2_validation.csv")
table = pd.concat([table[table["model"] != "1D CNN"], pd.DataFrame([row])], ignore_index=True)
table.to_csv("reports/table2_validation.csv", index=False)
pd.DataFrame(history).to_csv("reports/cnn_training_history.csv", index=False)
print("Saved reports/table2_validation.csv and reports/cnn_training_history.csv")