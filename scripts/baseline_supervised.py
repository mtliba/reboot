"""Supervised baseline trainer: encoder + MLP head trained end-to-end on SOC labels.
No SSL pretrain. Pick the encoder via +baseline=supervised_{lstm,tcn,transformer,hybrid}.

Usage:
    python scripts/baseline_supervised.py +baseline=supervised_lstm

Uses the same memmap data, clipping, train/val(cycle)/test protocol and
label_fraction knob as scripts/finetune.py, and finishes by evaluating the
best-on-val checkpoint on the TEST split -> <ckpt_dir>/eval_metrics.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune import (                                   # noqa: E402  (sibling script)
    build_finetune_datasets, build_test_dataset, export_best_checkpoint, trainer_kwargs_from_cfg,
)

from reboot.baselines.supervised import SupervisedModule, build_supervised_model
from reboot.data.dataset import collate_finetune
from reboot.eval.runner import evaluate_and_save, print_metrics
from reboot.training.callbacks import build_callbacks
from reboot.utils.logging import build_wandb_logger
from reboot.utils.seed import set_global_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)
    assert "baseline" in cfg, "Pass +baseline=supervised_lstm (or _tcn / _transformer / _hybrid)"
    bcfg = cfg.baseline
    print(OmegaConf.to_yaml({"baseline": bcfg, "finetune": cfg.finetune, "trainer": cfg.trainer}))

    train_ds, val_ds = build_finetune_datasets(cfg)
    train_ds.with_next = False                            # no L_phys in the supervised baseline
    print(f"Supervised[{bcfg.encoder_type}] train windows: {len(train_ds):,}  val windows: {len(val_ds):,}")

    in_channels = train_ds.in_channels
    model = build_supervised_model(
        encoder_type=bcfg.encoder_type,
        in_channels=in_channels,
        hidden=bcfg.hidden,
        encoder_kwargs=OmegaConf.to_container(bcfg.encoder_kwargs, resolve=True),
    )
    module = SupervisedModule(model, cfg)

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

    ckpt_dir = Path(cfg.paths.checkpoint_dir) / f"baseline_{bcfg.encoder_type}"
    trainer = pl.Trainer(
        callbacks=build_callbacks(ckpt_dir, monitor="val/mae", mode="min"),
        logger=build_wandb_logger(cfg),
        **trainer_kwargs_from_cfg(cfg),
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best = export_best_checkpoint(trainer, ckpt_dir)

    # ---- final evaluation on the TEST split (rank 0) ----
    if not trainer.is_global_zero:
        return
    if best is not None:
        state = torch.load(best, map_location="cpu", weights_only=False)
        module.load_state_dict(state["state_dict"])
    test_ds, info = build_test_dataset(cfg)
    protocol = info["protocol"]
    limit = cfg.trainer.get("limit_val_batches", None)
    if limit is not None:                                 # smoke-test path
        test_ds.indices = test_ds.indices[: int(limit) * cfg.finetune.batch_size]
    test_loader = DataLoader(
        test_ds, batch_size=cfg.finetune.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        collate_fn=collate_finetune,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module.eval().to(device)
    suffix = "" if protocol == "per_cell_random" else f"_{protocol}"
    out_path = ckpt_dir / f"eval_metrics{suffix}.json"
    metrics = evaluate_and_save(
        module.predict, test_loader, device, out_path,
        extra={"baseline": f"supervised_{bcfg.encoder_type}",
               "label_fraction": float(cfg.finetune.get("label_fraction", 1.0)),
               "checkpoint": str(best) if best else "last",
               **info},
    )
    print_metrics(metrics, title=f"TEST metrics — supervised_{bcfg.encoder_type} [{protocol}]")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
