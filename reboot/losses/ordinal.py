"""L_ord — ordinal progression hinge on phi(z_s). See CLAUDE.md §6.2.

For two windows i, j from the SAME phase segment with local_t_i < local_t_j:
    charging:    L_ij = max(0, m - (s_j - s_i))
    discharging: L_ij = max(0, m - (s_i - s_j))
where s = phi(z_s), phi: R^{d_s} -> R is a (trained) linear layer.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class OrdinalLoss(nn.Module):
    def __init__(self, d_s: int, margin: float = 0.5):
        super().__init__()
        self.phi = nn.Linear(d_s, 1)
        self.margin = float(margin)

    def forward(
        self,
        z_s: torch.Tensor,                  # (B, d_s)
        phase: torch.Tensor,                # (B,) long, 0=charge 1=discharge
        pair_idx_i: torch.Tensor,           # (P,) earlier in time
        pair_idx_j: torch.Tensor,           # (P,) later in time
    ) -> torch.Tensor:
        if pair_idx_i.numel() == 0:
            return z_s.new_zeros(())
        s = self.phi(z_s).squeeze(-1)                                # (B,)
        s_i = s[pair_idx_i]
        s_j = s[pair_idx_j]
        ph_i = phase[pair_idx_i]
        ph_j = phase[pair_idx_j]
        # Pairs must share a phase (constructed inside a single segment).
        assert torch.equal(ph_i, ph_j), "pair_idx_i and pair_idx_j must share phases"

        # Charging: want s_j > s_i; Discharging: want s_i > s_j.
        delta = torch.where(ph_i == 0, s_j - s_i, s_i - s_j)
        return torch.clamp(self.margin - delta, min=0.0).mean()
