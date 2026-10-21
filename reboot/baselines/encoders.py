"""Baseline-specific encoders. The TCN and Transformer encoders are reused from
reboot/models/encoders/; only LSTM is new here."""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """2-layer LSTM with mean-pool readout. Same interface as the spec encoders."""

    def __init__(
        self,
        in_channels: int,
        hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden = hidden
        self.bidirectional = bidirectional

        # If bidirectional, halve the per-direction hidden so the final pooled vector
        # is shape (B, hidden), matching the other encoders.
        per_dir = hidden // 2 if bidirectional else hidden
        assert per_dir > 0
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=per_dir,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (B, W, C)
        assert x.dim() == 3, f"expected (B, W, C), got {x.shape}"
        h, _ = self.lstm(x)                                       # (B, W, hidden)
        return h.mean(dim=1)                                       # (B, hidden)
