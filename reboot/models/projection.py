"""Projection heads g_s, g_d — see CLAUDE.md §7.2."""
from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Two-layer MLP. Used twice (once for z_s, once for z_d); weights are NOT shared."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)
