"""L_refl — symmetric InfoNCE on z_s ONLY. See CLAUDE.md §6.3.

Critical: z_d MUST NOT enter this loss. Charge ↔ Discharge alignment lives entirely
in the shared SOC latent.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReflectiveContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.tau = float(temperature)

    def forward(
        self, z_s_c: torch.Tensor, z_s_d: torch.Tensor, neg_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """z_s_c[k] and z_s_d[k] are positive partners. All other off-diagonal
        entries in the similarity matrix are negatives.

        neg_mask: optional (N, N) bool, True = EXCLUDE this (charge i, discharge j)
        entry from the negatives, e.g. because the two windows sit at the same
        pseudo-SOC (CLAUDE.md §13: same-SOC windows are positives, not negatives).
        The diagonal (the positive) is never masked."""
        assert z_s_c.shape == z_s_d.shape, (
            f"z_s_c/z_s_d shape mismatch: {z_s_c.shape} vs {z_s_d.shape}"
        )
        N = z_s_c.size(0)
        if N < 2:
            return z_s_c.new_zeros(())

        z_c = F.normalize(z_s_c, dim=-1)
        z_d = F.normalize(z_s_d, dim=-1)
        sim = z_c @ z_d.t() / self.tau                               # (N, N)
        labels = torch.arange(N, device=sim.device)
        if neg_mask is not None:
            assert neg_mask.shape == sim.shape, f"neg_mask {tuple(neg_mask.shape)} vs sim {tuple(sim.shape)}"
            eye = torch.eye(N, dtype=torch.bool, device=sim.device)
            sim = sim.masked_fill(neg_mask & ~eye, float("-inf"))

        loss_c2d = F.cross_entropy(sim, labels)
        loss_d2c = F.cross_entropy(sim.t(), labels)
        return 0.5 * (loss_c2d + loss_d2c)
