"""Top-level ReBoot model — encoder + split latent + decoder + regression head.

See CLAUDE.md §7.5. The ablation flag `single_latent` collapses [z_s; z_d] into a
single z of size (d_s + d_d) for comparison; downstream code must check this flag
before routing latents to z_s-only or z_d-only losses.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from reboot.models.decoder import ReconstructionDecoder
from reboot.models.encoders import build_encoder
from reboot.models.projection import ProjectionHead
from reboot.models.regression_head import RegressionHead


class ReBootModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        W: int,
        encoder_type: str = "hybrid",
        hidden: int = 128,
        d_s: int = 64,
        d_d: int = 64,
        single_latent: bool = False,
        encoder_kwargs: dict | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.W = W
        self.hidden = hidden
        self.d_s = d_s
        self.d_d = d_d
        self.single_latent = single_latent

        self.encoder = build_encoder(
            encoder_type, in_channels=in_channels, hidden=hidden,
            **(encoder_kwargs or {}),
        )

        if single_latent:
            # One head producing a flat z of size d_s + d_d.
            self.g_s = ProjectionHead(hidden, d_s + d_d)
            self.g_d = None
        else:
            self.g_s = ProjectionHead(hidden, d_s)
            self.g_d = ProjectionHead(hidden, d_d)

        z_dim = d_s + d_d
        self.decoder = ReconstructionDecoder(z_dim=z_dim, out_channels=in_channels, W=W, hidden=hidden)
        self.regression_head = RegressionHead(z_dim, hidden=hidden)

    @property
    def z_dim(self) -> int:
        return self.d_s + self.d_d

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (z_s, z_d). When single_latent=True, z_s is the full flat z and
        z_d is a zero tensor on the same device for shape compatibility."""
        h = self.encoder(x)                                          # (B, hidden)
        if self.single_latent:
            z = self.g_s(h)                                          # (B, d_s + d_d)
            zero = torch.zeros(z.size(0), 0, device=z.device, dtype=z.dtype)
            return z, zero
        z_s = self.g_s(h)
        z_d = self.g_d(h)
        return z_s, z_d

    def reconstruct(self, z_s: torch.Tensor, z_d: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_s, z_d], dim=-1) if z_d.numel() > 0 else z_s
        return self.decoder(z)

    def predict_soc(self, z_s: torch.Tensor, z_d: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_s, z_d], dim=-1) if z_d.numel() > 0 else z_s
        return self.regression_head(z)
