"""Stage 2 LightningModule — supervised SOC regression + weak physics. See CLAUDE.md §8.2.

Two modes:
    linear_probe   : freeze encoder + g_s + g_d, train regression_head + physics params only.
    full_finetune  : unfreeze all; use parameter groups with lower lr for the encoder.

L_phys requires a "next window" prediction. MemmapBatteryDataset(mode="finetune")
provides `x_next` (the window one stride later in the same segment), a `has_next`
mask (False at segment ends), and the RAW mean current / total elapsed time over
the `stride` samples separating the two windows (`I_raw_mean`, `dt_raw_total`).
Coulomb counting over that gap predicts the change between the two window-mean
SOCs. Windows without a successor are excluded from L_phys. If the batch carries
none of these fields, L_phys = 0.
"""
from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn

from reboot.losses import PhysicsLoss
from reboot.models import ReBootModel


class FinetuneModule(pl.LightningModule):
    def __init__(
        self,
        model: ReBootModel,
        cfg: Any,
        num_batteries: int,
        nominal_caps_As: torch.Tensor,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "nominal_caps_As"])
        self.model = model
        self.cfg = cfg

        self.huber = nn.HuberLoss(delta=cfg.finetune.huber_delta)
        self.phys_loss = PhysicsLoss(num_batteries=num_batteries, nominal_caps_As=nominal_caps_As)
        # Kept (not a parameter) to report how far the learned capacity drifts.
        self.register_buffer("_nominal_caps", nominal_caps_As.clone().float(), persistent=False)

        if cfg.finetune.mode == "linear_probe":
            self._freeze_backbone()

    def _freeze_backbone(self):
        for p in self.model.encoder.parameters():
            p.requires_grad_(False)
        for p in self.model.g_s.parameters():
            p.requires_grad_(False)
        if self.model.g_d is not None:
            for p in self.model.g_d.parameters():
                p.requires_grad_(False)

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        z_s, z_d = self.model.encode(x)
        return self.model.predict_soc(z_s, z_d)                      # (B,)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x = batch["x"]
        y = batch["y"]
        y_hat = self._predict(x)
        L_sup = self.huber(y_hat, y)

        L_phys = x.new_zeros(())
        n_phys = 0
        use_phys = bool(self.cfg.finetune.losses.use_phys)
        if use_phys and "x_next" in batch and "has_next" in batch:
            sel = batch["has_next"].nonzero(as_tuple=True)[0]
            n_phys = int(sel.numel())
            if n_phys >= 1:
                y_hat_next = self._predict(batch["x_next"][sel])
                L_phys = self.phys_loss(
                    y_hat_t=y_hat[sel],
                    y_hat_tp1=y_hat_next,
                    I_mean=batch["I_raw_mean"][sel],
                    dt_total=batch["dt_raw_total"][sel],
                    battery_idx=batch["battery_idx"][sel],
                )

        total = L_sup + self.cfg.finetune.loss_weights.phys * L_phys

        self.log_dict(
            {
                "finetune/loss_sup": L_sup,
                "finetune/loss_phys": L_phys,
                "finetune/loss_total": total,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        if use_phys:
            self.log_dict(
                {
                    "finetune/phys_frac": x.new_tensor(n_phys / max(x.size(0), 1)),
                    "finetune/eta": self.phys_loss.get_eta().detach().squeeze(),
                    "finetune/Q_ratio_mean": (
                        torch.exp(self.phys_loss.log_Q).detach()
                        / self._nominal_caps.to(self.device)
                    ).mean(),
                },
                on_step=False,
                on_epoch=True,
            )
        return total

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        y = batch["y"]
        y_hat = self._predict(batch["x"])
        mae = (y_hat - y).abs().mean()
        rmse = ((y_hat - y) ** 2).mean().sqrt()
        self.log_dict(
            {"val/mae": mae, "val/rmse": rmse}, prog_bar=True, on_epoch=True,
        )
        return {"mae": mae.detach(), "rmse": rmse.detach()}

    def configure_optimizers(self):
        mode = self.cfg.finetune.mode
        lr_head = self.cfg.finetune.lr_head
        lr_enc = self.cfg.finetune.lr_encoder
        wd = self.cfg.finetune.weight_decay

        if mode == "linear_probe":
            head_params = (
                list(self.model.regression_head.parameters())
                + list(self.phys_loss.parameters())
            )
            opt = torch.optim.AdamW(head_params, lr=lr_head, weight_decay=wd)
        elif mode == "full_finetune":
            enc_params = list(self.model.encoder.parameters())
            enc_params += list(self.model.g_s.parameters())
            if self.model.g_d is not None:
                enc_params += list(self.model.g_d.parameters())
            head_params = (
                list(self.model.regression_head.parameters())
                + list(self.phys_loss.parameters())
            )
            opt = torch.optim.AdamW(
                [
                    {"params": enc_params, "lr": lr_enc},
                    {"params": head_params, "lr": lr_head},
                ],
                weight_decay=wd,
            )
        else:
            raise ValueError(f"unknown finetune.mode={mode!r}")

        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
