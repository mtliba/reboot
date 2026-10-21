"""SOC regression head — see CLAUDE.md §7.4. Output is in [0, 1] via sigmoid."""
from __future__ import annotations

import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:        # z: (B, in_dim)
        return self.net(z).squeeze(-1)                             # (B,)
