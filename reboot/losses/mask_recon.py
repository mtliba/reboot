"""L_mask — masked L1 reconstruction. See CLAUDE.md §6.1.

L_mask = sum(|x_hat - x| * mask) / sum(mask)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MaskedReconstructionLoss(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        assert x.shape == x_hat.shape == mask.shape, (
            f"shape mismatch x={x.shape} x_hat={x_hat.shape} mask={mask.shape}"
        )
        diff = (x_hat - x).abs()                                     # L1, NOT MSE
        masked_diff = diff * mask.float()
        denom = mask.float().sum().clamp(min=1.0)
        return masked_diff.sum() / denom
