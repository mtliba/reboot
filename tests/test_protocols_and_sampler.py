"""Tests for memmap-level evaluation protocols and the cycle-grouped batch sampler."""
from __future__ import annotations

import numpy as np
import pytest

from reboot.data.pair_sampling import make_reflective_pairs
from reboot.data.samplers import CycleGroupedBatchSampler
from reboot.data.splits import resolve_protocol, split_indices_by_cycle

IDS = ["cell1-25", "cell2-35", "cell3-45", "cell4-25", "cell5-35", "cell6-45"]


def _meta(n_cycles_per_cell=10, windows_per_cycle=6):
    """Synthetic meta: cycle_id = cell*1_000_000 + position (like prepare_data)."""
    battery_idx, cycle_id, split = [], [], []
    for b in range(len(IDS)):
        for c in range(n_cycles_per_cell):
            for _ in range(windows_per_cycle):
                battery_idx.append(b)
                cycle_id.append(b * 1_000_000 + c)
                split.append(1 if c % 5 == 0 else 0)          # 20 % "test" cycles
    return np.array(battery_idx), np.array(cycle_id), np.array(split, dtype=np.int8)


def test_per_cell_random_uses_stored_split():
    b, c, s = _meta()
    pool, test, info = resolve_protocol("per_cell_random", b, IDS, s, c)
    assert np.array_equal(pool, np.flatnonzero(s == 0)) and np.array_equal(test, np.flatnonzero(s == 1))


def test_cross_battery_holds_out_whole_cells():
    b, c, s = _meta()
    pool, test, info = resolve_protocol("cross_battery", b, IDS, s, c, test_batteries=["cell2-35", "cell6-45"])
    assert set(b[test]) == {1, 5} and set(b[pool]) == {0, 2, 3, 4}
    assert np.intersect1d(pool, test).size == 0 and pool.size + test.size == b.size
    assert info["holdout_batteries"] == ["cell2-35", "cell6-45"]
    # stored split is ignored: every cycle of a held-out cell is test
    assert test.size == 2 * 10 * 6
    # default = last holdout_k cells
    _, test2, info2 = resolve_protocol("cross_battery", b, IDS, s, c, holdout_k=2)
    assert info2["holdout_batteries"] == ["cell5-35", "cell6-45"]
    with pytest.raises(AssertionError):
        resolve_protocol("cross_battery", b, IDS, s, c, test_batteries=["nope"])


def test_cross_temperature_by_cell_name_suffix():
    b, c, s = _meta()
    pool, test, info = resolve_protocol("cross_temperature", b, IDS, s, c, holdout_temperature=25)
    assert info["holdout_batteries"] == ["cell1-25", "cell4-25"]
    assert set(b[test]) == {0, 3} and 0 not in set(b[pool]) and 3 not in set(b[pool])


def test_cross_cycle_splits_later_cycles_per_cell():
    b, c, s = _meta(n_cycles_per_cell=10)
    pool, test, _ = resolve_protocol("cross_cycle", b, IDS, s, c, cycle_split_frac=0.7)
    for cell in range(len(IDS)):
        pos_pool = np.unique(c[pool][b[pool] == cell]) - cell * 1_000_000
        pos_test = np.unique(c[test][b[test] == cell]) - cell * 1_000_000
        assert pos_pool.tolist() == list(range(7)) and pos_test.tolist() == [7, 8, 9]


def test_unknown_protocol():
    b, c, s = _meta()
    with pytest.raises(ValueError):
        resolve_protocol("bogus", b, IDS, s, c)


# ---------------- sampler ----------------

def _cycles(n_cycles=50, lo=5, hi=12, seed=0):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(lo, hi, size=n_cycles)
    return np.repeat(np.arange(n_cycles) * 7 + 3, sizes)        # arbitrary cycle ids


def test_sampler_batches_are_exact_and_cover_epoch():
    cyc = _cycles()
    bs = 32
    smp = CycleGroupedBatchSampler(cyc, batch_size=bs, seed=1, rank=0, world_size=1)
    batches = list(smp)
    assert len(batches) == len(smp) == cyc.size // bs
    assert all(len(b) == bs for b in batches)
    flat = np.concatenate(batches)
    assert np.unique(flat).size == flat.size                     # no duplicates
    # every cycle inside a batch is (mostly) whole: number of distinct cycles per batch is small
    for b in batches:
        assert np.unique(cyc[b]).size <= bs // 5 + 2


def test_sampler_groups_whole_cycles():
    cyc = _cycles()
    smp = CycleGroupedBatchSampler(cyc, batch_size=40, seed=3, rank=0, world_size=1)
    flat = np.concatenate(list(smp))
    # in the concatenated stream every cycle appears as ONE contiguous run
    ids = cyc[flat]
    runs = 1 + np.count_nonzero(np.diff(ids))
    assert runs == np.unique(ids).size


def test_sampler_ddp_shards_are_disjoint_and_equal_length():
    cyc = _cycles(n_cycles=61)
    smps = [CycleGroupedBatchSampler(cyc, batch_size=16, seed=5, rank=r, world_size=3) for r in range(3)]
    lens = [len(s) for s in smps]
    assert len(set(lens)) == 1 and lens[0] > 0
    flats = [np.concatenate(list(s)) for s in smps]
    for i in range(3):
        for j in range(i + 1, 3):
            assert np.intersect1d(flats[i], flats[j]).size == 0
    # every rank's cycles are disjoint too
    cyc_sets = [set(cyc[f]) for f in flats]
    assert not (cyc_sets[0] & cyc_sets[1]) and not (cyc_sets[1] & cyc_sets[2])


def test_sampler_reshuffles_per_epoch():
    cyc = _cycles()
    smp = CycleGroupedBatchSampler(cyc, batch_size=32, seed=0, rank=0, world_size=1)
    a = list(smp); smp.set_epoch(1); b = list(smp); smp.set_epoch(0); c = list(smp)
    assert a != b and a == c


def test_cycle_batches_yield_many_more_reflective_pairs_than_random():
    """The reason the sampler exists: same-cycle charge/discharge pairs per batch."""
    rng = np.random.default_rng(0)
    n_cycles, per_seg, stride = 400, 10, 32
    phases, cycles, segs, local_ts, seg_lens = [], [], [], [], []
    seg = 0
    for c in range(n_cycles):
        for ph in (0, 1):
            for k in range(per_seg):
                phases.append(ph); cycles.append(c); segs.append(seg); local_ts.append(k * stride); seg_lens.append(per_seg * stride + 100)
            seg += 1
    phases, cycles, segs, local_ts, seg_lens = map(np.array, (phases, cycles, segs, local_ts, seg_lens))

    def n_pairs(idx):
        rc, rd = make_reflective_pairs(phases=phases[idx], cycle_ids=cycles[idx], segment_ids=segs[idx],
                                       local_ts=local_ts[idx], segment_lengths=seg_lens[idx],
                                       strategy="boundary_symmetric")
        return int(np.asarray(rc).size)

    bs = 256
    random_idx = rng.choice(phases.size, bs, replace=False)
    smp = CycleGroupedBatchSampler(cycles, batch_size=bs, seed=0, rank=0, world_size=1)
    grouped_idx = np.array(next(iter(smp)))
    assert n_pairs(grouped_idx) >= 5 * max(n_pairs(random_idx), 1)
    assert n_pairs(grouped_idx) >= bs // 4


def test_pool_fraction_keeps_whole_cycles():
    cycle_id = np.repeat(np.arange(40), 25); idx = np.arange(cycle_id.size)
    keep, _ = split_indices_by_cycle(idx, cycle_id, 1.0 - 0.25, seed=0)   # keep 25 % of cycles
    assert np.unique(cycle_id[keep]).size == 10
    for c in np.unique(cycle_id[keep]):
        assert (cycle_id[keep] == c).sum() == 25
