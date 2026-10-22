"""1D CNN architecture (shared by training and the noise experiments)."""
import torch.nn as nn


def block(c_in, c_out):
    return nn.Sequential(nn.Conv1d(c_in, c_out, kernel_size=7, padding=3),
                         nn.BatchNorm1d(c_out), nn.ReLU(), nn.MaxPool1d(2))


def build_cnn():
    """4 conv blocks -> global average pooling -> dropout -> 5 outputs (same as cnn1d.py)."""
    return nn.Sequential(block(12, 32), block(32, 64), block(64, 128), block(128, 128),
                         nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3),
                         nn.Linear(128, 5))