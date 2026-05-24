"""Utilities: seed, device, focal loss, metrics, CSV logger."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


AAMI_CLASSES = ("N", "S", "V", "F", "Q")


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FocalLoss(nn.Module):
    """Multi-class focal loss: L = alpha_c * (1 - p_t)^gamma * CE.

    With alpha=None, all classes share weight 1.0 — class balance is then
    expected to come from the WeightedRandomSampler instead.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    n_classes: int = 5) -> dict:
    labels = list(range(n_classes))
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    macro_f1 = float(f1.mean()) if len(f1) else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(acc),
        "macro_f1": macro_f1,
        "per_class_precision": p.tolist(),
        "per_class_recall": r.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
        "confusion_matrix": cm.tolist(),
    }


class CSVLogger:
    """Append rows to a CSV; writes header on first row."""

    def __init__(self, path: str | Path, fieldnames: list[str]):
        self.path = Path(path)
        self.fieldnames = fieldnames
        self._fh = None
        self._writer = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
        self._writer.writeheader()
        return self

    def log(self, row: dict) -> None:
        self._writer.writerow(row)
        self._fh.flush()

    def __exit__(self, *exc):
        self._fh.close()
