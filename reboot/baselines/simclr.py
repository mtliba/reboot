"""SSL baseline #2: naive SimCLR over time series — augment-pair InfoNCE.

For each window in a batch, we generate two augmented views by combining:
  - random temporal cropping + resizing (jitter the segment boundaries)
  - random channel-/time-masking (zeroing portions of the input)
The encoder produces representations of both views; positives are the two views
of the same window, negatives are all other views in the batch.

Both augmentations are fully batched (grid_sample for the crop-resize, argsort
for the time mask), so a bs=2048 step costs a few ms instead of the ~2 s the
original per-sample Python loop needed.
"""
from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from reboot.models import ReBootModel
from reboot.training.callbacks import warmup_cosine_lambda


def batched_mask(
    B: int, W: int, C: int, time_ratio: float, channel_prob: float, device
) -> torch.Tensor:
    """(B, W, C) bool mask: ``time_ratio`` of timesteps fully masked per sample,
    plus each channel masked with prob ``channel_prob`` per sample."""
    m = torch.zeros(B, W, C, dtype=torch.bool, device=device)
    n_time = int(W * time_ratio)
    if n_time > 0:
        tidx = torch.rand(B, W, device=device).argsort(dim=1)[:, :n_time]     # (B, n_time)
        m.scatter_(1, tidx.unsqueeze(-1).expand(-1, -1, C), True)
    if channel_prob > 0:
        cm = torch.rand(B, C, device=device) < channel_prob                    # (B, C)
        m |= cm[:, None, :]
    return m


def random_crop_resize(x: torch.Tensor, crop_min: float) -> torch.Tensor:
    """Per-sample random temporal crop of length in [crop_min*W, W], linearly
    resampled back to W. x: (B, W, C) -> (B, W, C). crop_min=1 is the identity."""
    B, W, C = x.shape
    dev = x.device
    if W < 2:
        return x
    crop_len = W * (crop_min + (1.0 - crop_min) * torch.rand(B, device=dev))   # (B,)
    crop_len = crop_len.clamp(min=2.0, max=float(W))
    start = torch.rand(B, device=dev) * (W - crop_len)                         # (B,)
    t = torch.linspace(0.0, 1.0, W, device=dev)                                 # (W,)
    pos = start[:, None] + (crop_len[:, None] - 1.0) * t[None, :]              # (B, W) in [0, W-1]
    gx = pos / (W - 1) * 2.0 - 1.0
    grid = torch.stack([gx, torch.zeros_like(gx)], dim=-1).unsqueeze(1)        # (B, 1, W, 2)
    img = x.float().transpose(1, 2).unsqueeze(2)                                # (B, C, 1, W)
    v = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return v.squeeze(2).transpose(1, 2).to(x.dtype)                             # (B, W, C)


def augment_pair(
    x: torch.Tensor,
    crop_min: float = 0.6,
    mask_ratio: float = 0.20,
    channel_mask_prob: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two independently augmented views of ``x: (B, W, C)``."""
    B, W, C = x.shape
    views = []
    for _ in range(2):
        v = random_crop_resize(x, crop_min)
        m = batched_mask(B, W, C, mask_ratio, channel_mask_prob, x.device)
        views.append(v.masked_fill(m, 0.0))
    return views[0], views[1]


class SimCLRModule(pl.LightningModule):
    """SimCLR pretrain head: ReBootModel.encoder + a projection MLP + InfoNCE.

    Only ``model.encoder`` and the projector are trained. The unused ReBoot
    parts (g_s, g_d, decoder, regression_head) are frozen so vanilla DDP works
    without ``find_unused_parameters`` (same trick as PretrainModule).
    """

    def __init__(
        self,
        model: ReBootModel,
        cfg: Any,
        temperature: float = 0.1,
        crop_min: float = 0.6,
        mask_ratio: float = 0.20,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.cfg = cfg
        self.tau = float(temperature)
        self.crop_min = float(crop_min)
        self.mask_ratio = float(mask_ratio)

        for name in ("g_s", "g_d", "decoder", "regression_head"):
            mod = getattr(model, name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad_(False)

        # SimCLR projector: 2-layer MLP from encoder hidden to a contrastive dim.
        self.projector = nn.Sequential(
            nn.Linear(model.hidden, model.hidden),
            nn.GELU(),
            nn.Linear(model.hidden, model.d_s),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.model.encoder(x)
        return F.normalize(self.projector(h), dim=-1)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x = batch["x"]
        v1, v2 = augment_pair(x, crop_min=self.crop_min, mask_ratio=self.mask_ratio)
        z1 = self._encode(v1)
        z2 = self._encode(v2)
        N = z1.size(0)
        if N < 2:
            return x.new_zeros(())

        sim = (z1 @ z2.t()).float() / self.tau                       # (N, N)
        labels = torch.arange(N, device=sim.device)
        loss_12 = F.cross_entropy(sim, labels)
        loss_21 = F.cross_entropy(sim.t(), labels)
        loss = 0.5 * (loss_12 + loss_21)
        self.log_dict(
            {"simclr/loss": loss}, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            params, lr=self.cfg.pretrain.lr, weight_decay=self.cfg.pretrain.weight_decay,
        )
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=warmup_cosine_lambda(
                warmup_epochs=int(self.cfg.pretrain.get("warmup_epochs", 0)),
                max_epochs=int(self.cfg.trainer.max_epochs),
            ),
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
