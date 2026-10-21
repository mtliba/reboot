"""Stage 1 pretrain entrypoint. Uses the memmap-backed dataset (much faster than
the Python-object pickle) and supports multi-GPU DDP.

Quickstart:
    # one GPU
    python scripts/pretrain.py
    # 4 GPUs (must enable find_unused_parameters because regression_head is unused
    # in PretrainModule)
    CUDA_VISIBLE_DEVICES=0,2,8,9 python scripts/pretrain.py \\
        trainer.devices=4 trainer.strategy=ddp_find_unused_parameters_true
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from reboot.data.dataset import collate_pretrain
from reboot.data.memmap_dataset import MemmapBatteryDataset
from reboot.data.samplers import CycleGroupedBatchSampler
from reboot.data.splits import resolve_protocol_from_cfg, split_indices_by_cycle
from reboot.models import ReBootModel
from reboot.training.callbacks import build_callbacks
from reboot.training.pretrain_module import PretrainModule
from reboot.utils.logging import build_wandb_logger
from reboot.utils.seed import set_global_seed


def run_pretrain(cfg: DictConfig, ckpt_subdir: str = "pretrain") -> None:
    set_global_seed(cfg.seed)
    print(OmegaConf.to_yaml(cfg))

    dataset = MemmapBatteryDataset(
        cfg.data.processed_dir,
        mode="pretrain",
        filter_split="train",
        mask_time_ratio=cfg.pretrain.mask.time_ratio,
        mask_channel_prob=cfg.pretrain.mask.channel_prob,
        clip_value=cfg.data.get("clip_value", 10.0), input_channels=cfg.data.get("input_channels", None),
        stride=cfg.data.stride,
    )
    print(f"Pretrain windows: {len(dataset):,}  (in_channels={dataset.in_channels})")
    cfg.model.in_channels = dataset.in_channels

    collate = partial(
        collate_pretrain,
        stride=cfg.data.stride,
        pair_strategy=cfg.data.pair_strategy,
        K_pairs=cfg.pretrain.K_pairs,
    )
    # Protocol-aware pretrain pool: for cross_battery / cross_temperature the
    # held-out cells' signals are excluded from pretraining as well (no
    # transductive peeking), unless pretrain.exclude_holdout=false.
    strategy = str(cfg.data.splits.strategy)
    if strategy != "per_cell_random" and bool(cfg.pretrain.get("exclude_holdout", True)):
        pool, _, info = resolve_protocol_from_cfg(dataset, cfg.data.splits)
        dataset.indices = np.intersect1d(dataset.indices, pool)
        print(f"Protocol {strategy}: pretraining on {len(dataset):,} windows"
              + (f" (held-out cells excluded: {info['holdout_batteries']})" if "holdout_batteries" in info else ""))

    # Unlabeled-data scaling: keep a random fraction of the pretraining CYCLES.
    pool_fraction = float(cfg.pretrain.get("pool_fraction", 1.0))
    if pool_fraction < 1.0:
        keep, _ = split_indices_by_cycle(dataset.indices, dataset.cycle_id, 1.0 - pool_fraction, seed=int(cfg.seed))
        dataset.indices = keep
        print(f"Unlabeled pool: {pool_fraction:.0%} of cycles -> {len(dataset):,} windows")

    loader_kwargs = dict(
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
        collate_fn=collate,
    )
    batch_sampler_kind = str(cfg.pretrain.get("batch_sampler", "random"))
    use_distributed_sampler = True
    if batch_sampler_kind == "cycle":
        sampler = CycleGroupedBatchSampler(
            dataset.cycle_id[dataset.indices], batch_size=cfg.pretrain.batch_size, seed=int(cfg.seed),
        )
        loader = DataLoader(dataset, batch_sampler=sampler, **loader_kwargs)
        use_distributed_sampler = False                  # the sampler shards cycles itself
        print(f"Batch sampler: cycle-grouped ({sampler.n_cycles:,} cycles, bs={cfg.pretrain.batch_size})")
    elif batch_sampler_kind == "random":
        loader = DataLoader(
            dataset, batch_size=cfg.pretrain.batch_size, shuffle=True, drop_last=True, **loader_kwargs,
        )
    else:
        raise ValueError(f"pretrain.batch_sampler must be random|cycle, got {batch_sampler_kind!r}")

    encoder_kwargs = {
        k: v for k, v in OmegaConf.to_container(cfg.model, resolve=True).items()
        if k not in {"encoder_type", "in_channels", "hidden", "d_s", "d_d", "single_latent"}
    }
    model = ReBootModel(
        in_channels=dataset.in_channels,
        W=cfg.data.window_size,
        encoder_type=cfg.model.encoder_type,
        hidden=cfg.model.hidden,
        d_s=cfg.model.d_s,
        d_d=cfg.model.d_d,
        single_latent=bool(cfg.model.single_latent),
        encoder_kwargs=encoder_kwargs,
    )

    module = PretrainModule(model, cfg)

    trainer_kwargs = dict(
        use_distributed_sampler=use_distributed_sampler,
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        deterministic=cfg.trainer.deterministic,
        callbacks=build_callbacks(
            Path(cfg.paths.checkpoint_dir) / ckpt_subdir,
            monitor="pretrain/loss_total",
            mode="min",
        ),
        logger=build_wandb_logger(cfg),
    )
    if cfg.trainer.get("strategy", None) is not None:
        trainer_kwargs["strategy"] = cfg.trainer.strategy
    for k in ("limit_train_batches", "limit_val_batches"):
        v = cfg.trainer.get(k, None)
        if v is not None:
            trainer_kwargs[k] = v
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(module, train_dataloaders=loader)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_pretrain(cfg, ckpt_subdir="pretrain")


if __name__ == "__main__":
    main()
