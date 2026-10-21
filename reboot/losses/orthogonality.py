"""L_orth — Frobenius squared norm of the centered cross-covariance. See CLAUDE.md §6.5.

Forces z_s and z_d to carry independent information (decorrelated channels).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class OrthogonalityLoss(nn.Module):
    """Bounded cross-correlation penalty.

    Implementation note: the spec's raw `||z_s^T z_d||_F^2 / B^2` is *unbounded*
    when latent magnitudes drift during training, and we observed a 5000×
    instantaneous spike on real data (L_orth ~10² for one batch) which produced
    inf gradients and NaN-poisoned the model.

    To keep the same objective (decorrelate channels of z_s and z_d) without
    the unbounded magnitude term, we normalize each centered latent by its
    per-feature std before forming the cross-correlation matrix. Each entry of
    the resulting matrix is then a Pearson correlation in [-1, 1], and the
    loss is the mean squared correlation in [0, 1]. This is the Barlow-Twins
    style cross-correlation.
    """

    def forward(self, z_s: torch.Tensor, z_d: torch.Tensor) -> torch.Tensor:
        if z_s.size(0) == 0 or z_d.size(0) == 0 or z_d.size(-1) == 0:
            return z_s.new_zeros(())
        assert z_s.size(0) == z_d.size(0), "batch sizes must match"
        zs = z_s - z_s.mean(dim=0, keepdim=True)
        zd = z_d - z_d.mean(dim=0, keepdim=True)
        zs_n = zs / zs.std(dim=0, keepdim=True).clamp(min=1e-6)
        zd_n = zd / zd.std(dim=0, keepdim=True).clamp(min=1e-6)
        B = z_s.size(0)
        # cross[i,j] ≈ Pearson(z_s[:,i], z_d[:,j]) ∈ [-1, 1]
        cross = (zs_n.t() @ zd_n) / B                                # (d_s, d_d)
        d_s, d_d = cross.shape
        return cross.pow(2).sum() / (d_s * d_d)
