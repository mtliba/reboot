"""Eval protocols — see CLAUDE.md §10.

Each protocol partitions a flat list of `Window`s into (train, val, test) subsets.
Implementations are deliberately simple and operate on the already-windowed records,
so the same code is reused by `prepare_data.py` and `evaluate.py`.
"""
from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Literal

import numpy as np

from reboot.data.dataset import Window

Protocol = Literal["cross_battery", "cross_cycle", "cross_temperature", "low_label"]


def _split_lists(items: list[Window], train_frac: float = 0.9, rng: random.Random | None = None
                 ) -> tuple[list[Window], list[Window]]:
    rng = rng or random.Random(0)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    cut = int(len(idx) * train_frac)
    train = [items[i] for i in idx[:cut]]
    val = [items[i] for i in idx[cut:]]
    return train, val


def cross_battery(
    windows: list[Window], holdout_battery_ids: Iterable[str]
) -> tuple[list[Window], list[Window], list[Window]]:
    """Test set = all windows from `holdout_battery_ids`. Train/val from the rest."""
    holdout = set(holdout_battery_ids)
    train_pool = [w for w in windows if w.battery_id not in holdout]
    test = [w for w in windows if w.battery_id in holdout]
    train, val = _split_lists(train_pool)
    return train, val, test


def cross_cycle(
    windows: list[Window], cycle_split_at: int
) -> tuple[list[Window], list[Window], list[Window]]:
    """Train on cycles < cycle_split_at, test on the rest. Val is a slice of train."""
    train_pool = [w for w in windows if w.cycle_id < cycle_split_at]
    test = [w for w in windows if w.cycle_id >= cycle_split_at]
    train, val = _split_lists(train_pool)
    return train, val, test


def cross_temperature(
    windows: list[Window], temp_idx: int, T_threshold: float
) -> tuple[list[Window], list[Window], list[Window]]:
    """Test = windows whose mean temperature (channel index `temp_idx`) ≥ threshold.

    The threshold is on the SCALED temperature channel; the caller is responsible for
    converting the physical threshold using the active Scaler.
    """
    def _mean_temp(w: Window) -> float:
        return float(np.mean(w.data[:, temp_idx]))

    train_pool = [w for w in windows if _mean_temp(w) < T_threshold]
    test = [w for w in windows if _mean_temp(w) >= T_threshold]
    train, val = _split_lists(train_pool)
    return train, val, test


def low_label(
    pretrain_windows: list[Window],
    labeled_windows: list[Window],
    label_fraction: float,
    rng: random.Random | None = None,
) -> tuple[list[Window], list[Window], list[Window]]:
    """Pretrain on all unlabeled windows; finetune on a `label_fraction` slice of labels."""
    assert 0.0 < label_fraction <= 1.0
    rng = rng or random.Random(0)
    idx = list(range(len(labeled_windows)))
    rng.shuffle(idx)
    cut = max(1, int(len(idx) * label_fraction))
    train = [labeled_windows[i] for i in idx[:cut]]
    train, val = _split_lists(train)
    test = pretrain_windows  # caller can override with a proper held-out set
    return train, val, test


def build_split(
    windows: list[Window],
    protocol: Protocol,
    **kwargs,
) -> tuple[list[Window], list[Window], list[Window]]:
    if protocol == "cross_battery":
        return cross_battery(windows, kwargs["holdout_battery_ids"])
    if protocol == "cross_cycle":
        return cross_cycle(windows, kwargs["cycle_split_at"])
    if protocol == "cross_temperature":
        return cross_temperature(windows, kwargs["temp_idx"], kwargs["T_threshold"])
    if protocol == "low_label":
        return low_label(
            kwargs["pretrain_windows"],
            kwargs["labeled_windows"],
            kwargs["label_fraction"],
        )
    raise ValueError(f"unknown protocol: {protocol!r}")
