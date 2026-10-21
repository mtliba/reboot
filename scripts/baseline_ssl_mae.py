"""SSL-MAE pretrain baseline: scripts/pretrain.py with only L_mask enabled.

Usage:
    python scripts/baseline_ssl_mae.py
Then finetune with scripts/finetune.py on the resulting checkpoint.

(The SLURM file slurm/pretrain_ssl_mae.sbatch calls scripts/pretrain.py with the
same loss flags directly; this wrapper is the single-command equivalent.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, open_dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pretrain import run_pretrain                        # noqa: E402  (sibling script)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    with open_dict(cfg.pretrain.losses):
        cfg.pretrain.losses.use_mask = True
        cfg.pretrain.losses.use_ord = False
        cfg.pretrain.losses.use_refl = False
        cfg.pretrain.losses.use_smooth = False
        cfg.pretrain.losses.use_orth = False
    run_pretrain(cfg, ckpt_subdir="baseline_ssl_mae")


if __name__ == "__main__":
    main()
