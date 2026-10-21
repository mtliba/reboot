"""Tests for the data pipeline."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from reboot.data.dataset import BatteryDataset, BatteryRecord, windows_from_record
from reboot.data.masking import generate_mask
from reboot.data.pair_sampling import (
    make_consecutive_pairs,
    make_ordinal_pairs,
    make_reflective_pairs,
)
from reboot.data.phase_segmentation import (
    PHASE_CHARGE,
    PHASE_DISCHARGE,
    Segment,
    hysteresis_filter,
    segment_phases,
)
from reboot.data.transforms import derived_channels, fit_scaler


# ---------- phase segmentation ----------

def test_segment_phases_basic():
    # 100 samples charging, 100 discharging.
    I = np.concatenate([np.full(100, 1.0), np.full(100, -1.0)])
    segs = segment_phases(I, eps=0.05, min_run=10)
    assert len(segs) == 2
    assert segs[0].phase == PHASE_CHARGE
    assert segs[1].phase == PHASE_DISCHARGE


def test_segment_phases_drops_rest():
    I = np.concatenate([np.full(50, 1.0), np.zeros(50), np.full(50, -1.0)])
    segs = segment_phases(I)
    assert all(s.phase != "rest" for s in segs)


def test_hysteresis_filter_ignores_short_spikes():
    labels = np.array([PHASE_CHARGE] * 20 + [PHASE_DISCHARGE] * 3 + [PHASE_CHARGE] * 20,
                      dtype=object)
    out = hysteresis_filter(labels, min_run=10)
    # The 3-sample dip must NOT flip the phase.
    assert all(o == PHASE_CHARGE for o in out)


# ---------- masking ----------

def test_generate_mask_shape_and_ratio():
    torch.manual_seed(0)
    m = generate_mask(W=128, C=4, time_mask_ratio=0.15, channel_mask_prob=0.0)
    assert m.shape == (128, 4)
    # At least 15% timesteps fully masked.
    fully_masked_rows = m.all(dim=1).sum().item()
    assert fully_masked_rows >= int(128 * 0.15)


# ---------- transforms ----------

def test_derived_channels_shape():
    x = np.random.randn(50, 4).astype(np.float32)
    x[:, 3] = 1.0    # dt = 1s
    out = derived_channels(x)
    assert out.shape == (50, 7)


def test_fit_scaler_transform_roundtrip():
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(200, 4)).astype(np.float32)
    arr[:, 3] = 1.0
    sc = fit_scaler([arr], has_derived=False)
    out = sc.transform(arr)
    assert out.shape == arr.shape
    # I and V columns should be ~zero-centered after robust scaling.
    assert abs(np.median(out[:, 0])) < 1e-5
    assert abs(np.median(out[:, 1])) < 1e-5


# ---------- pair sampling ----------

def test_make_consecutive_pairs():
    segment_ids = [0, 0, 0, 1]
    local_ts = [0, 32, 64, 0]
    i, j = make_consecutive_pairs(segment_ids, local_ts, stride=32)
    pairs = set(zip(i.tolist(), j.tolist()))
    assert pairs == {(0, 1), (1, 2)}


def test_make_ordinal_pairs_respects_order():
    segment_ids = [0, 0, 0]
    local_ts = [0, 32, 64]
    i, j = make_ordinal_pairs(segment_ids, local_ts, K=20)
    for ii, jj in zip(i.tolist(), j.tolist()):
        assert local_ts[ii] < local_ts[jj]


def test_make_reflective_pairs_boundary_symmetric():
    phases = [0, 0, 1, 1]
    cycle_ids = [0, 0, 0, 0]
    seg_ids = [0, 0, 1, 1]
    local_ts = [0, 50, 0, 50]
    seg_lens = [100, 100, 100, 100]
    c, d = make_reflective_pairs(
        phases, cycle_ids, seg_ids, local_ts, seg_lens, strategy="boundary_symmetric"
    )
    # charge t=0 (p=0)   should pair with discharge t=50 (p=0.5 → target 1 - 0 = 1 ⇒ farther index 1)
    # charge t=50 (p=0.5) should pair with discharge t=50 (target 0.5)
    assert len(c) == 2 and len(d) == 2


# ---------- dataset ----------

def test_battery_dataset_pretrain_rejects_soc():
    fake = BatteryRecord(
        battery_id="b", battery_idx=0, cycle_id=0,
        data=np.zeros((200, 4), dtype=np.float32),
        segments=[Segment(0, 200, PHASE_CHARGE)],
        soc=np.linspace(0, 1, 200).astype(np.float32),
    )
    windows, _ = windows_from_record(fake, window_size=128, stride=32, next_segment_id=0)
    assert all(w.soc is not None for w in windows)
    with pytest.raises(AssertionError):
        BatteryDataset(windows, mode="pretrain")


def test_battery_dataset_pretrain_item():
    fake = BatteryRecord(
        battery_id="b", battery_idx=0, cycle_id=0,
        data=np.zeros((200, 4), dtype=np.float32),
        segments=[Segment(0, 200, PHASE_CHARGE)],
        soc=None,
    )
    windows, _ = windows_from_record(fake, window_size=128, stride=32, next_segment_id=0)
    ds = BatteryDataset(windows, mode="pretrain")
    item = ds[0]
    assert item["x"].shape == (128, 4)
    assert item["mask"].shape == (128, 4)
    assert item["mask"].dtype == torch.bool
    assert item["phase"].item() == 0
