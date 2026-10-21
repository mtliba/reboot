"""Baselines for the ReBoot ablation table (CLAUDE.md §12).

- supervised_lstm / supervised_tcn / supervised_transformer:
    end-to-end SOC regression with a single encoder + MLP head.
- ssl_mae:
    pretrain with L_mask only (no ord/refl/smooth/orth), then finetune.
- ssl_simclr:
    pretrain with naive InfoNCE between two augmented views of the same window,
    then finetune.
"""
from reboot.baselines.encoders import LSTMEncoder
from reboot.baselines.simclr import SimCLRModule, augment_pair
from reboot.baselines.supervised import SupervisedModule, build_supervised_model

__all__ = [
    "LSTMEncoder",
    "SimCLRModule",
    "SupervisedModule",
    "augment_pair",
    "build_supervised_model",
]
