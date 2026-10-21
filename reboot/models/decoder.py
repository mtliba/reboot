"""Reconstruction decoder — see CLAUDE.md §7.3.

Takes a single latent vector z = [z_s; z_d] and produces a (B, W, C) reconstruction
using a learned query of length W and a small Transformer decoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ReconstructionDecoder(nn.Module):
    def __init__(
        self,
        z_dim: int,
        out_channels: int,
        W: int,
        hidden: int = 128,
        nhead: int = 4,
        ffn_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.W = W
        self.out_channels = out_channels
        self.hidden = hidden

        self.query = nn.Parameter(torch.randn(1, W, hidden) * 0.02)
        self.z_proj = nn.Linear(z_dim, hidden)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden, out_channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:           # z: (B, z_dim)
        assert z.dim() == 2, f"expected (B, z_dim), got {z.shape}"
        B = z.size(0)
        memory = self.z_proj(z).unsqueeze(1)                       # (B, 1, hidden)
        q = self.query.expand(B, -1, -1)                            # (B, W, hidden)
        h = self.decoder(q, memory)                                 # (B, W, hidden)
        return self.out_proj(h)                                     # (B, W, C)
