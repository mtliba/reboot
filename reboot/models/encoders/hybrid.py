"""Hybrid encoder — TCN frontend + Transformer backend. See CLAUDE.md §7.1."""
from __future__ import annotations

import torch
import torch.nn as nn


class HybridEncoder(nn.Module):
    """TCN captures local patterns; Transformer adds long-range mixing."""

    def __init__(
        self,
        in_channels: int,
        hidden: int = 128,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        ffn_dim: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden = hidden

        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_transformer_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (B, W, C)
        assert x.dim() == 3, f"expected (B, W, C), got {x.shape}"
        W = x.size(1)
        h = self.tcn(x.transpose(1, 2))                          # (B, hidden, W')
        h = h[..., :W].transpose(1, 2)                            # (B, W, hidden)
        h = self.transformer(h)
        return h.mean(dim=1)                                      # (B, hidden)
