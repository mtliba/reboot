"""Stage 2 finetune entrypoint. Uses the memmap-backed dataset.

Usage:
    python scripts/finetune.py "finetune.pretrained_ckpt='outputs/checkpoints/pretrain/last.ckpt'"

(Quote checkpoint paths: Lightning filenames contain "=" which Hydra's override
grammar otherwise rejects.)

Splits:
    train  = TRAIN split minus `finetune.val_fraction` of its cycles
    val    = those held-out TRAIN cycles   -> checkpoint selection (val/mae)
    test   = TEST split                    -> only touched by scripts/evaluate.py

After training, the best-on-val checkpoint is copied to <ckpt_dir>/best.ckpt.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from reboot.data.dataset import collate_finetune
from reboot.data.memmap_dataset import MemmapBatteryDataset
import numpy as np

from reboot.data.splits import (
    resolve_protocol_from_cfg, split_indices_by_cycle, subsample_indices,
)
from reboot.models import ReBootModel
from reboot.training.callbacks import build_callbacks
from reboot.training.finetune_module import FinetuneModule
from reboot.utils.logging import build_wandb_logger
from reboot.utils.seed import set_global_seed

DEFAULT_CAP_AS = 2.5 * 3600.0


def build_finetune_datasets(cfg: DictConfig):
    """Return (train_ds, val_ds) for the protocol in cfg.data.splits.strategy.

    train = protocol train pool minus `finetune.val_fraction` of its cycles
    val   = those held-out train cycles (checkpoint selection)
    test  = protocol test set — only built by `build_test_dataset` (evaluate / baselines)
    """
    clip_value = cfg.data.get("clip_value", 10.0)
    train_ds = MemmapBatteryDataset(
        cfg.data.processed_dir, mode="finetune", filter_split=None,
        clip_value=clip_value, input_channels=cfg.data.get("input_channels", None), stride=cfg.data.stride,
        scaler_path=cfg.paths.scaler_path,           # raw I/dt for L_phys
    )
    pool, test, info = resolve_protocol_from_cfg(train_ds, cfg.data.splits)
    train_ds.indices = np.intersect1d(train_ds.indices, pool)
    print(f"Protocol {info['protocol']}: train pool {len(train_ds.indices):,} labelled windows, "
          f"test {test.size:,} windows"
          + (f"; held-out cells {info['holdout_batteries']}" if "holdout_batteries" in info else ""))

    val_fraction = float(cfg.finetune.get("val_fraction", 0.0) or 0.0)
    val_ds = MemmapBatteryDataset(
        cfg.data.processed_dir, mode="eval", filter_split=None,
        clip_value=clip_value, input_channels=cfg.data.get("input_channels", None), stride=cfg.data.stride,
    )
    if val_fraction > 0.0:
        tr_idx, va_idx = split_indices_by_cycle(
            train_ds.indices, train_ds.cycle_id, val_fraction, seed=int(cfg.seed),
        )
        val_ds.indices = va_idx
        train_ds.indices = tr_idx
        print(f"Validation: {val_fraction:.0%} of train cycles held out "
              f"({len(va_idx):,} windows); test set untouched.")
    else:
        val_ds.indices = np.intersect1d(val_ds.indices, test)
        print("WARNING: finetune.val_fraction=0 -> validating on the TEST set (legacy).")

    # Optional low-label subsampling (CLAUDE.md §10 low_label protocol).
    label_fraction = float(cfg.finetune.get("label_fraction", 1.0))
    if label_fraction < 1.0:
        n_before = len(train_ds.indices)
        train_ds.indices = subsample_indices(
            train_ds.indices, label_fraction, seed=int(cfg.finetune.get("label_fraction_seed", 0)),
        )
        print(f"Low-label subsample: keep {len(train_ds.indices):,}/{n_before:,} "
              f"({label_fraction:.2%}), seed={cfg.finetune.get('label_fraction_seed', 0)}")
    return train_ds, val_ds


def build_test_dataset(cfg: DictConfig, strategy: str | None = None):
    """Protocol test set (labelled windows), plus the protocol info dict."""
    test_ds = MemmapBatteryDataset(
        cfg.data.processed_dir, mode="eval", filter_split=None,
        clip_value=cfg.data.get("clip_value", 10.0), input_channels=cfg.data.get("input_channels", None), stride=cfg.data.stride,
    )
    _, test, info = resolve_protocol_from_cfg(test_ds, cfg.data.splits, strategy)
    test_ds.indices = np.intersect1d(test_ds.indices, test)
    return test_ds, info


def build_model(cfg: DictConfig, in_channels: int, W: int) -> ReBootModel:
    encoder_kwargs = {
        k: v for k, v in OmegaConf.to_container(cfg.model, resolve=True).items()
        if k not in {"encoder_type", "in_channels", "hidden", "d_s", "d_d", "single_latent"}
    }
    return ReBootModel(
        in_channels=in_channels, W=W,
        encoder_type=cfg.model.encoder_type, hidden=cfg.model.hidden,
        d_s=cfg.model.d_s, d_d=cfg.model.d_d,
        single_latent=bool(cfg.model.single_latent),
        encoder_kwargs=encoder_kwargs,
    )


def trainer_kwargs_from_cfg(cfg: DictConfig) -> dict:
    kw = dict(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
    )
    if cfg.trainer.get("strategy", None) is not None:
        kw["strategy"] = cfg.trainer.strategy
    for k in ("limit_train_batches", "limit_val_batches"):
        v = cfg.trainer.get(k, None)
        if v is not None:
            kw[k] = v
    return kw


def export_best_checkpoint(trainer: pl.Trainer, ckpt_dir: Path) -> Path | None:
    """Copy the ModelCheckpoint's best model to <ckpt_dir>/best.ckpt (rank 0)."""
    if not trainer.is_global_zero:
        return None
    cb = trainer.checkpoint_callback
    best = getattr(cb, "best_model_path", "") or ""
    if not best or not Path(best).exists():
        print("No best checkpoint recorded; best.ckpt not written.")
        return None
    dst = ckpt_dir / "best.ckpt"
    shutil.copyfile(best, dst)
    score = getattr(cb, "best_model_score", None)
    print(f"Best checkpoint: {best} (val/mae={float(score):.5f})" if score is not None else f"Best checkpoint: {best}")
    print(f"Copied to {dst}")
    return dst


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg.trainer.max_epochs = int(cfg.finetune.get("max_epochs", 50))
    set_global_seed(cfg.seed)
    print(OmegaConf.to_yaml(cfg))

    train_ds, val_ds = build_finetune_datasets(cfg)
    print(f"Finetune train windows: {len(train_ds):,}  val windows: {len(val_ds):,}")

    in_channels = train_ds.in_channels
    cfg.model.in_channels = in_channels

    # Per-battery learnable capacity in PhysicsLoss.
    num_batteries = int(max(train_ds.battery_idx.max(), val_ds.battery_idx.max())) + 1
    nominal = torch.full((num_batteries,), DEFAULT_CAP_AS, dtype=torch.float32)
    cap_map = train_ds.extra.get("capacities_As") or {}
    for bi_str, cap_As in cap_map.items():
        nominal[int(bi_str)] = float(cap_As)

    model = build_model(cfg, in_channels, cfg.data.window_size)

    ckpt_path = cfg.finetune.pretrained_ckpt
    if ckpt_path:
        print(f"Loading pretrained weights from {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = state.get("state_dict", state)
        new_sd = {k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")}
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        print(f"  missing keys: {len(missing)}, unexpected: {len(unexpected)}")
        if "epoch" in state:
            print(f"  pretrain checkpoint epoch={state['epoch']} global_step={state.get('global_step')}")
    else:
        print("No pretrained checkpoint: training the ReBoot architecture from scratch.")

    module = FinetuneModule(model, cfg, num_batteries=num_batteries, nominal_caps_As=nominal)

    pw = cfg.data.persistent_workers and cfg.data.num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=cfg.finetune.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        persistent_workers=pw, collate_fn=collate_finetune, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.finetune.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        persistent_workers=pw, collate_fn=collate_finetune,
    )

    ckpt_dir = Path(cfg.paths.checkpoint_dir) / "finetune"
    trainer = pl.Trainer(
        callbacks=build_callbacks(ckpt_dir, monitor="val/mae", mode="min"),
        logger=build_wandb_logger(cfg),
        **trainer_kwargs_from_cfg(cfg),
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    export_best_checkpoint(trainer, ckpt_dir)


if __name__ == "__main__":
    main()
