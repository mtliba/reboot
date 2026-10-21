"""Logger factory: W&B when enabled, otherwise a CSVLogger INSIDE the run's own
checkpoint directory.

Returning None (Lightning's default logger) writes to a shared ./lightning_logs/
and picks the next free version number at start-up; several runs launched in
the same second on one GPU (the packed sbatch jobs) then share a version
directory and the CSV writer crashes with "dict contains fields not in
fieldnames" (job 16308). A per-run save_dir removes the race.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pytorch_lightning.loggers import CSVLogger, WandbLogger


def build_wandb_logger(cfg: Any):
    mode = getattr(cfg.wandb, "mode", "online")
    if mode == "disabled":
        save_dir = Path(getattr(cfg.paths, "checkpoint_dir", "outputs/checkpoints"))
        save_dir.mkdir(parents=True, exist_ok=True)
        return CSVLogger(save_dir=str(save_dir), name="csv_logs")
    return WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=mode,
        tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
        save_dir=cfg.paths.output_dir,
    )
