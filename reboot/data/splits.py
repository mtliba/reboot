"""Index-level split helpers for the memmap dataset."""
from __future__ import annotations

import numpy as np


def split_indices_by_cycle(
    indices: np.ndarray,
    cycle_id: np.ndarray,
    val_fraction: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Carve a validation subset out of ``indices`` at CYCLE granularity.

    Whole cycles (charge + discharge) move together, so no cycle is split across
    train/val and the L_refl / L_phys pairings inside a cycle stay intact. The
    held-out TEST split is never touched — this exists so that checkpoint
    selection (monitor val/mae) does not peek at the test set.

    Returns (train_indices, val_indices), both sorted, disjoint, union == indices.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if val_fraction <= 0.0 or indices.size == 0:
        return indices, indices[:0]
    assert 0.0 < val_fraction < 1.0, f"val_fraction must be in (0,1), got {val_fraction}"
    cycles = np.unique(cycle_id[indices])
    n_val = int(round(len(cycles) * val_fraction))
    n_val = min(max(n_val, 1), len(cycles) - 1) if len(cycles) > 1 else 0
    rng = np.random.default_rng(seed)
    val_cycles = rng.choice(cycles, size=n_val, replace=False)
    is_val = np.isin(cycle_id[indices], val_cycles)
    return np.sort(indices[~is_val]), np.sort(indices[is_val])


def subsample_indices(indices: np.ndarray, fraction: float, seed: int = 0) -> np.ndarray:
    """Random subset of ``indices`` (low-label protocol, CLAUDE.md §10)."""
    indices = np.asarray(indices, dtype=np.int64)
    if fraction >= 1.0:
        return indices
    rng = np.random.default_rng(seed)
    keep = max(1, int(round(indices.size * fraction)))
    return np.sort(rng.choice(indices, size=keep, replace=False))


# ----------------------------------------------------------------------------
# Evaluation protocols on the memmap index space (CLAUDE.md §10)
# ----------------------------------------------------------------------------

PROTOCOLS = ("per_cell_random", "cross_battery", "cross_temperature", "cross_cycle")


def _cell_temperature(battery_id: str) -> str | None:
    """'cell7-25' -> '25'. None if the name carries no temperature suffix."""
    if "-" in battery_id:
        return battery_id.rsplit("-", 1)[1]
    return None


def resolve_protocol(
    strategy: str,
    battery_idx: np.ndarray,
    battery_ids: list[str],
    split: np.ndarray,
    cycle_id: np.ndarray,
    *,
    test_batteries: list[str] | None = None,
    holdout_k: int = 2,
    holdout_temperature: int | str | None = None,
    cycle_split_frac: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (train_pool_indices, test_indices, info) over the FULL window array.

    per_cell_random   : the split stored at prepare_data time (80/20 of each cell's cycles).
    cross_battery     : test = every cycle of `test_batteries` (or the last `holdout_k`
                        cells); train pool = every cycle of the other cells.
    cross_temperature : test = every cycle of the cells whose name ends with
                        `holdout_temperature` (e.g. 25 -> 'cell*-25'); train = other cells.
    cross_cycle       : per cell, cycles in chronological order; the first
                        `cycle_split_frac` of them train, the rest test (ageing shift).

    The caller intersects the pools with its own filters (has_soc, label fraction)
    and carves a validation set out of the train pool by cycle.
    """
    battery_idx = np.asarray(battery_idx)
    n = battery_idx.size
    all_idx = np.arange(n, dtype=np.int64)
    info: dict = {"protocol": strategy}

    if strategy == "per_cell_random":
        split = np.asarray(split)
        return all_idx[split == 0], all_idx[split == 1], info

    if strategy in ("cross_battery", "cross_temperature"):
        if strategy == "cross_battery":
            holdout_names = list(test_batteries) if test_batteries else list(battery_ids[-int(holdout_k):])
        else:
            assert holdout_temperature is not None, "set data.splits.holdout_temperature"
            t = str(holdout_temperature)
            holdout_names = [b for b in battery_ids if _cell_temperature(b) == t]
        unknown = [b for b in holdout_names if b not in battery_ids]
        assert not unknown, f"unknown battery ids in holdout: {unknown}"
        assert 0 < len(holdout_names) < len(battery_ids), f"holdout must be a strict subset: {holdout_names}"
        hold_idx = np.array([battery_ids.index(b) for b in holdout_names])
        is_test = np.isin(battery_idx, hold_idx)
        info.update(holdout_batteries=holdout_names,
                    train_batteries=[b for b in battery_ids if b not in holdout_names])
        return all_idx[~is_test], all_idx[is_test], info

    if strategy == "cross_cycle":
        assert 0.0 < cycle_split_frac < 1.0
        cycle_id = np.asarray(cycle_id)
        is_test = np.zeros(n, dtype=bool)
        for b in np.unique(battery_idx):
            m = battery_idx == b
            cycles = np.unique(cycle_id[m])                      # chronological within a cell
            cut = int(round(len(cycles) * cycle_split_frac))
            test_cycles = cycles[cut:]
            is_test[m] = np.isin(cycle_id[m], test_cycles)
        info.update(cycle_split_frac=cycle_split_frac)
        return all_idx[~is_test], all_idx[is_test], info

    raise ValueError(f"unknown split strategy {strategy!r}; expected one of {PROTOCOLS}")


def resolve_protocol_from_cfg(ds, splits_cfg, strategy: str | None = None):
    """Convenience wrapper reading everything from a MemmapBatteryDataset + cfg.data.splits."""
    strategy = strategy or str(splits_cfg.strategy)
    tb = splits_cfg.get("test_batteries", None)
    return resolve_protocol(
        strategy, ds.battery_idx, list(ds.extra["battery_ids"]), ds.split, ds.cycle_id,
        test_batteries=list(tb) if tb else None,
        holdout_k=int(splits_cfg.get("holdout_k", 2)),
        holdout_temperature=splits_cfg.get("holdout_temperature", None),
        cycle_split_frac=float(splits_cfg.get("cycle_split_frac", 0.7)),
    )
