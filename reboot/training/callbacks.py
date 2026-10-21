"""Common Lightning callbacks used by pretrain + finetune scripts."""
from __future__ import annotations

import math
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)


class StopOnNonFinite(pl.Callback):
    """Stop training as soon as the training loss is NaN or Inf.

    Without this, SLURM job 72178 kept training for 157 epochs on NaN weights
    after diverging at epoch 43. ModelCheckpoint keeps the best-so-far
    checkpoint, so stopping loses nothing; it just frees the GPUs.

    ``patience_batches`` consecutive non-finite losses are required so that a
    transient bad batch (whose update PretrainModule's gradient guard drops)
    does not end the run; once the weights themselves are NaN every batch is
    non-finite and the run stops within a few seconds anyway.

    The decision is taken per rank and Lightning reduces ``trainer.should_stop``
    across DDP ranks at the end of the epoch, so a NaN on any rank stops all.
    """

    def __init__(self, patience_batches: int = 50):
        super().__init__()
        self.patience_batches = int(patience_batches)
        self._bad = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is None or not torch.is_tensor(loss):
            return
        if torch.isfinite(loss).all():
            self._bad = 0
            return
        self._bad += 1
        if self._bad >= self.patience_batches:
            pl_module.print(
                f"[StopOnNonFinite] non-finite training loss at epoch "
                f"{trainer.current_epoch} step {trainer.global_step}: {loss.item()!r}. "
                "Stopping training."
            )
            trainer.should_stop = True


def warmup_cosine_lambda(warmup_epochs: int, max_epochs: int, min_factor: float = 0.0):
    """Per-epoch LR multiplier: linear warm-up from 1/warmup to 1, then cosine to
    ``min_factor`` over the remaining epochs (CLAUDE.md §9: cosine, 10-epoch warm-up).
    """
    warmup_epochs = max(int(warmup_epochs), 0)
    max_epochs = max(int(max_epochs), 1)

    def f(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        span = max(max_epochs - warmup_epochs, 1)
        progress = min(max(epoch - warmup_epochs, 0), span) / span
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return f


def build_callbacks(
    checkpoint_dir: str | Path,
    monitor: str = "val/loss_total",
    mode: str = "min",
    save_top_k: int = 3,
    stop_on_non_finite: bool = True,
) -> list[pl.Callback]:
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # Filename template intentionally omits the metric — Lightning's filename
    # formatter doesn't escape "/" cleanly when the monitored key contains one
    # (e.g. "pretrain/loss_total"), which either creates a spurious subdirectory
    # or silently no-ops the save depending on quoting. Bench logs (CSVLogger
    # under lightning_logs/) hold the real metric values per epoch, so we don't
    # lose any information by keeping the on-disk filename clean.
    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=True,
            filename="{epoch:03d}-{step}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]
    if stop_on_non_finite:
        callbacks.append(StopOnNonFinite())
    return callbacks
