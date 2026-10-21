"""Smoke test — one pretrain step end-to-end on synthetic data."""
from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from reboot.data.dataset import (
    BatteryDataset,
    BatteryRecord,
    collate_pretrain,
    windows_from_record,
)
from reboot.data.phase_segmentation import PHASE_CHARGE, PHASE_DISCHARGE, Segment
from reboot.models import ReBootModel
from reboot.training.pretrain_module import PretrainModule


def _fake_record(battery_idx: int, charge_phase: bool, T: int = 400) -> BatteryRecord:
    data = np.random.randn(T, 4).astype(np.float32)
    data[:, 3] = 1.0
    phase = PHASE_CHARGE if charge_phase else PHASE_DISCHARGE
    return BatteryRecord(
        battery_id=f"b{battery_idx}",
        battery_idx=battery_idx,
        cycle_id=0,
        data=data,
        segments=[Segment(0, T, phase)],
        soc=None,
    )


def _build_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        pretrain=SimpleNamespace(
            margin=0.5,
            tau=0.1,
            K_pairs=4,
            lr=1e-3,
            weight_decay=1e-4,
            loss_weights=SimpleNamespace(mask=1.0, ord=1.0, refl=0.5, smooth=0.1, orth=0.01),
            losses=SimpleNamespace(use_mask=True, use_ord=True, use_refl=True,
                                   use_smooth=True, use_orth=True),
            mask=SimpleNamespace(time_ratio=0.15, channel_prob=0.10),
        ),
        trainer=SimpleNamespace(max_epochs=1),
    )


def test_pretrain_one_step_runs():
    torch.manual_seed(0)
    np.random.seed(0)

    windows = []
    seg_id = 0
    for i in range(4):
        rec = _fake_record(battery_idx=i, charge_phase=(i % 2 == 0))
        w, seg_id = windows_from_record(rec, window_size=64, stride=16, next_segment_id=seg_id)
        windows.extend(w)
    assert len(windows) >= 16

    dataset = BatteryDataset(windows, mode="pretrain")
    collate = partial(collate_pretrain, stride=16, pair_strategy="boundary_symmetric", K_pairs=4)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate)

    model = ReBootModel(in_channels=4, W=64, encoder_type="hybrid",
                        hidden=32, d_s=16, d_d=16)
    cfg = _build_cfg()
    module = PretrainModule(model, cfg)

    batch = next(iter(loader))
    # Lightning normally moves tensors to device; here we stay on CPU.
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss).item()
    loss.backward()
