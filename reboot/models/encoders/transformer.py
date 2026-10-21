"""Transformer encoder — see CLAUDE.md §7.1."""
from __future__ import annotations

import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    """Pure Transformer encoder over a linear projection of the input, with learned
    additive positional embeddings and mean-pool readout."""

    def __init__(
        self,
        in_channels: int,
        hidden: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 256,
        max_len: int = 512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden = hidden
        self.max_len = max_len

        self.input_proj = nn.Linear(in_channels, hidden)
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
        nn.init.trunc_normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (B, W, C)
        assert x.dim() == 3, f"expected (B, W, C), got {x.shape}"
        W = x.size(1)
        assert W <= self.max_len, (
            f"window length {W} exceeds max_len {self.max_len}; increase max_len in config"
        )
        h = self.input_proj(x) + self.pos[:, :W]                 # (B, W, hidden)
        h = self.encoder(h)
        return h.mean(dim=1)                                      # (B, hidden)
