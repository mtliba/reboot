"""TCN encoder — see CLAUDE.md §7.1."""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class TCNEncoder(nn.Module):
    """Stack of dilated 1-D convolutions followed by mean-pooling over time."""

    def __init__(
        self,
        in_channels: int,
        hidden: int = 128,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden = hidden
        self.kernel_size = kernel_size
        self.dilations = list(dilations)

        layers: list[nn.Module] = []
        ch = in_channels
        for d in self.dilations:
            layers.append(
                nn.Conv1d(
                    ch, hidden, kernel_size,
                    padding=(kernel_size - 1) * d,
                    dilation=d,
                )
            )
            layers.append(nn.GELU())
            ch = hidden
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: (B, W, C)
        assert x.dim() == 3, f"expected (B, W, C), got {x.shape}"
        W = x.size(1)
        h = self.net(x.transpose(1, 2))                          # (B, hidden, W')
        h = h[..., :W]                                            # crop causal padding
        return h.mean(dim=-1)                                     # (B, hidden)
