"""SSL-SimCLR pretrain baseline. Augment-pair InfoNCE on the same encoder.

Usage:
    python scripts/baseline_ssl_simclr.py
Then finetune with scripts/finetune.py on the resulting checkpoint (the encoder
weights live under `model.encoder.*` like every other ReBoot checkpoint).
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune import build_model, trainer_kwargs_from_cfg  # noqa: E402  (sibling script)

from reboot.baselines.simclr import SimCLRModule
from reboot.data.dataset import collate_pretrain
from reboot.data.memmap_dataset import MemmapBatteryDataset
from reboot.training.callbacks import build_callbacks
from reboot.utils.logging import build_wandb_logger
from reboot.utils.seed import set_global_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)
    bcfg = cfg.get("baseline", None) or {}
    print(OmegaConf.to_yaml({"baseline": bcfg, "pretrain": cfg.pretrain, "trainer": cfg.trainer}))

    dataset = MemmapBatteryDataset(
        cfg.data.processed_dir, mode="pretrain", filter_split="train",
        mask_time_ratio=cfg.pretrain.mask.time_ratio,
        mask_channel_prob=cfg.pretrain.mask.channel_prob,
        clip_value=cfg.data.get("clip_value", 10.0), input_channels=cfg.data.get("input_channels", None), stride=cfg.data.stride,
    )
    print(f"SimCLR pretrain windows: {len(dataset):,}  (in_channels={dataset.in_channels})")
    cfg.model.in_channels = dataset.in_channels

    collate = partial(
        collate_pretrain, stride=cfg.data.stride,
        pair_strategy=cfg.data.pair_strategy, K_pairs=cfg.pretrain.K_pairs,
    )
    loader = DataLoader(
        dataset, batch_size=cfg.pretrain.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
        collate_fn=collate, drop_last=True,
    )

    model = build_model(cfg, dataset.in_channels, cfg.data.window_size)
    module = SimCLRModule(
        model, cfg,
        temperature=float(bcfg.get("temperature", 0.1)),
        crop_min=float(bcfg.get("crop_min", 0.6)),
        mask_ratio=float(bcfg.get("mask_ratio", 0.20)),
    )

    kw = trainer_kwargs_from_cfg(cfg)
    kw.pop("check_val_every_n_epoch", None)
    trainer = pl.Trainer(
        callbacks=build_callbacks(
            Path(cfg.paths.checkpoint_dir) / "pretrain",     # same layout as ReBoot pretrain
            monitor="simclr/loss", mode="min",
        ),
        logger=build_wandb_logger(cfg),
        **kw,
    )
    trainer.fit(module, train_dataloaders=loader)


if __name__ == "__main__":
    main()
