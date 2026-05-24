"""1D-CNN for AAMI 5-class heartbeat classification.

Expected input  : (B, T=250, C=2) as produced by HeartbeatDataset.
Output          : (B, n_classes=5) logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                              padding=kernel // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class HeartbeatCNN(nn.Module):
    def __init__(self, in_channels: int = 2, n_classes: int = 5,
                 dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, kernel=7),          # 250 -> 125
            ConvBlock(32, 64, kernel=5),                   # 125 -> 62
            ConvBlock(64, 128, kernel=3),                  # 62  -> 31
            ConvBlock(128, 128, kernel=3, pool=False),     # 31
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HeartbeatDataset returns (B, T, C); Conv1d wants (B, C, T).
        x = x.transpose(1, 2)
        x = self.features(x)
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        return self.head(x)
