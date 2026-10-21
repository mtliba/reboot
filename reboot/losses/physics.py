"""L_phys — weak Coulomb-counting consistency. See CLAUDE.md §6.6.

Sign convention used here: I > 0 = charging, I < 0 = discharging.
SOC_{t+1} ≈ SOC_t + (eta * I_t * dt_t) / Q   (charging raises SOC, discharging lowers it).
Residual: y_hat_{t+1} - y_hat_t - (eta * I * dt) / Q   ≈ 0
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PhysicsLoss(nn.Module):
    """Per-battery learnable capacity (log-parameterized to stay positive) and a global
    Coulombic efficiency eta ∈ (0.9, 1.0)."""

    def __init__(self, num_batteries: int, nominal_caps_As: torch.Tensor):
        super().__init__()
        assert nominal_caps_As.dim() == 1 and nominal_caps_As.numel() == num_batteries, (
            f"nominal_caps_As must be shape ({num_batteries},), got {tuple(nominal_caps_As.shape)}"
        )
        self.num_batteries = num_batteries
        self.log_Q = nn.Parameter(torch.log(nominal_caps_As.clone().float().clamp(min=1e-6)))
        self.eta_raw = nn.Parameter(torch.zeros(1))

    def get_eta(self) -> torch.Tensor:
        # eta in (0.9, 1.0) via sigmoid + affine.
        return 0.9 + 0.1 * torch.sigmoid(self.eta_raw)

    def forward(
        self,
        y_hat_t: torch.Tensor,                # (B,)
        y_hat_tp1: torch.Tensor,              # (B,)
        I_mean: torch.Tensor,                 # (B,) mean current over window (raw, NOT scaled)
        dt_total: torch.Tensor,               # (B,) total seconds elapsed over window
        battery_idx: torch.Tensor,            # (B,) long
    ) -> torch.Tensor:
        assert y_hat_t.shape == y_hat_tp1.shape == I_mean.shape == dt_total.shape, (
            "all inputs must share shape (B,)"
        )
        Q = torch.exp(self.log_Q)[battery_idx]                       # (B,)
        eta = self.get_eta()
        predicted_delta = eta * I_mean * dt_total / Q.clamp(min=1e-6)
        residual = y_hat_tp1 - y_hat_t - predicted_delta
        return residual.abs().mean()
