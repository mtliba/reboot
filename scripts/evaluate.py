"""Evaluation entrypoint — MAE / RMSE / max error / per-bin MAE on the protocol test set.

    python scripts/evaluate.py "+eval.ckpt='.../best.ckpt'" +eval.protocol=cross_battery

Protocols (CLAUDE.md §10): per_cell_random | cross_battery | cross_temperature | cross_cycle,
resolved on the memmap index space by reboot.data.splits.resolve_protocol. Writes
eval_metrics.json (per_cell_random) or eval_metrics_<protocol>.json next to the checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from reboot.data.dataset import collate_finetune
from reboot.data.memmap_dataset import MemmapBatteryDataset
from reboot.eval.runner import evaluate_and_save, print_metrics
from reboot.models import ReBootModel
from reboot.utils.seed import set_global_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune import build_test_dataset                  # noqa: E402  (sibling script)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)
    print(OmegaConf.to_yaml(cfg))

    eval_cfg = cfg.get("eval") if "eval" in cfg else None
    protocol = (eval_cfg.get("protocol") if eval_cfg is not None else None) or str(cfg.data.splits.strategy)
    test_ds, info = build_test_dataset(cfg, strategy=protocol)
    print(f"Protocol {protocol}: test windows {len(test_ds):,}"
          + (f"; held-out cells {info['holdout_batteries']}" if "holdout_batteries" in info else ""))
    limit = cfg.trainer.get("limit_val_batches", None)
    if limit is not None:                                 # smoke-test path
        test_ds.indices = test_ds.indices[: int(limit) * cfg.finetune.batch_size]

    in_channels = test_ds.in_channels
    encoder_kwargs = {
        k: v for k, v in OmegaConf.to_container(cfg.model, resolve=True).items()
        if k not in {"encoder_type", "in_channels", "hidden", "d_s", "d_d", "single_latent"}
    }
    model = ReBootModel(
        in_channels=in_channels, W=cfg.data.window_size,
        encoder_type=cfg.model.encoder_type, hidden=cfg.model.hidden,
        d_s=cfg.model.d_s, d_d=cfg.model.d_d,
        single_latent=bool(cfg.model.single_latent),
        encoder_kwargs=encoder_kwargs,
    )

    ckpt = (eval_cfg.get("ckpt") if eval_cfg is not None else None) or cfg.finetune.pretrained_ckpt
    assert ckpt is not None, "Set eval.ckpt or finetune.pretrained_ckpt to a finetuned checkpoint."
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    new_sd = {k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(new_sd, strict=False)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    loader = DataLoader(
        test_ds,
        batch_size=cfg.finetune.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_finetune,
    )

    def predict_fn(x):
        z_s, z_d = model.encode(x)
        return model.predict_soc(z_s, z_d)

    suffix = "" if protocol == "per_cell_random" else f"_{protocol}"
    out_path = Path(ckpt).parent / f"eval_metrics{suffix}.json"
    metrics = evaluate_and_save(
        predict_fn, loader, device, out_path,
        extra={"protocol": protocol, "checkpoint": str(ckpt), **{k: v for k, v in info.items() if k != "protocol"}},
    )
    print_metrics(metrics, title=f"TEST metrics — ReBoot [{protocol}]")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
