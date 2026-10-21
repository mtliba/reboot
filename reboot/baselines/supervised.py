"""Supervised end-to-end baselines: encoder + MLP head trained directly on SOC."""
from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn

from reboot.baselines.encoders import LSTMEncoder
from reboot.baselines.literature import LITERATURE_MODELS
from reboot.models.encoders import HybridEncoder, TCNEncoder, TransformerEncoder
from reboot.models.regression_head import RegressionHead


def build_supervised_model(
    encoder_type: str,
    in_channels: int,
    hidden: int = 128,
    encoder_kwargs: dict | None = None,
) -> nn.Module:
    """Return a single nn.Module wrapping (encoder, regression_head)."""
    encoder_kwargs = encoder_kwargs or {}
    et = encoder_type.lower()
    if et in LITERATURE_MODELS:                      # published architectures: complete models
        return LITERATURE_MODELS[et](in_channels=in_channels, **encoder_kwargs)
    if et == "lstm":
        encoder = LSTMEncoder(in_channels=in_channels, hidden=hidden, **encoder_kwargs)
    elif et == "tcn":
        encoder = TCNEncoder(in_channels=in_channels, hidden=hidden, **encoder_kwargs)
    elif et == "transformer":
        encoder = TransformerEncoder(in_channels=in_channels, hidden=hidden, **encoder_kwargs)
    elif et == "hybrid":
        encoder = HybridEncoder(in_channels=in_channels, hidden=hidden, **encoder_kwargs)
    else:
        raise ValueError(f"unknown supervised encoder_type={encoder_type!r}")

    head = RegressionHead(in_dim=hidden, hidden=hidden)

    class _Supervised(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.head = head

        def forward(self, x):
            return self.head(self.encoder(x))

    return _Supervised()


class SupervisedModule(pl.LightningModule):
    """End-to-end supervised SOC regression (no SSL pretrain)."""

    def __init__(self, model: nn.Module, cfg: Any):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.cfg = cfg
        self.huber = nn.HuberLoss(delta=cfg.finetune.huber_delta)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        y = batch["y"]
        y_hat = self.model(batch["x"])
        loss = self.huber(y_hat, y)
        self.log_dict(
            {"baseline/loss_sup": loss, "baseline/mae": (y_hat - y).abs().mean()},
            prog_bar=True, on_step=True, on_epoch=True, sync_dist=True,
        )
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        y = batch["y"]
        y_hat = self.model(batch["x"])
        mae = (y_hat - y).abs().mean()
        rmse = ((y_hat - y) ** 2).mean().sqrt()
        self.log_dict(
            {"val/mae": mae, "val/rmse": rmse}, prog_bar=True, on_epoch=True, sync_dist=True,
        )
        return {"mae": mae.detach(), "rmse": rmse.detach()}

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.finetune.lr_head,
            weight_decay=self.cfg.finetune.weight_decay,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
