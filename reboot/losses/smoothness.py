"""L_smooth — squared L2 between z_s of consecutive windows. See CLAUDE.md §6.4."""
from __future__ import annotations

import torch
import torch.nn as nn


class SmoothnessLoss(nn.Module):
    def forward(
        self,
        z_s: torch.Tensor,                  # (B, d_s)
        consec_i: torch.Tensor,             # (P,) earlier
        consec_j: torch.Tensor,             # (P,) next
    ) -> torch.Tensor:
        if consec_i.numel() == 0:
            return z_s.new_zeros(())
        diff = z_s[consec_j] - z_s[consec_i]                         # (P, d_s)
        return diff.pow(2).sum(dim=-1).mean()
