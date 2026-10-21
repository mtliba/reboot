"""Sanity tests for the synthetic battery simulator."""
from __future__ import annotations

import numpy as np
import pytest

from reboot.data.phase_segmentation import segment_phases
from reboot.data.synthetic import (
    BatteryParams,
    CycleSpec,
    make_population,
    ocv_curve,
    simulate_battery,
)


# ---------- OCV curve ----------

def test_ocv_curve_monotone_in_soc():
    soc = np.linspace(0.0, 1.0, 101)
    v = ocv_curve(soc, V_min=3.0, V_max=4.2)
    assert np.all(np.diff(v) >= 0), "OCV must be non-decreasing in SOC"
    assert v[0] >= 3.0 - 1e-3 and v[-1] <= 4.2 + 1e-3


# ---------- single-battery simulation ----------

def test_simulate_battery_shapes_match():
    p = BatteryParams()
    sim = simulate_battery(p, CycleSpec(dt_seconds=2.0), n_cycles=2, seed=0)
    keys = ("I", "V", "T", "dt", "soc", "cycle_id")
    n = sim["I"].shape[0]
    assert n > 0
    for k in keys:
        assert sim[k].shape == (n,), f"key {k} has wrong shape: {sim[k].shape} vs ({n},)"


def test_simulate_battery_soc_in_range():
    sim = simulate_battery(BatteryParams(), CycleSpec(), n_cycles=2, seed=0)
    assert sim["soc"].min() >= 0.0 - 1e-6
    assert sim["soc"].max() <= 1.0 + 1e-6


def test_simulate_battery_voltage_in_range():
    p = BatteryParams()
    sim = simulate_battery(p, CycleSpec(), n_cycles=2, seed=0)
    # Within roughly V_min ± 0.3 to V_max ± 0.3 (IR drop + hysteresis + noise)
    assert sim["V"].min() > p.V_min - 0.5
    assert sim["V"].max() < p.V_max + 0.5


def test_phase_segmentation_finds_cycles():
    sim = simulate_battery(BatteryParams(), CycleSpec(), n_cycles=3, seed=0)
    segs = segment_phases(sim["I"], eps=0.05, min_run=10)
    n_charge = sum(1 for s in segs if s.phase == "charge")
    n_disch = sum(1 for s in segs if s.phase == "discharge")
    # 3 cycles → expect 3 charge segments and 3 discharge segments
    assert n_charge == 3, f"expected 3 charge segments, got {n_charge}"
    assert n_disch == 3, f"expected 3 discharge segments, got {n_disch}"


def test_soc_monotone_within_charge_segment():
    sim = simulate_battery(BatteryParams(), CycleSpec(), n_cycles=2, seed=1)
    segs = segment_phases(sim["I"], eps=0.05, min_run=10)
    for s in segs:
        if s.phase == "charge":
            soc_seg = sim["soc"][s.start_idx : s.end_idx]
            # Up to small numerical jitter, SOC must be non-decreasing through a charge.
            diffs = np.diff(soc_seg)
            assert (diffs >= -1e-5).all(), "SOC dipped during a charge segment"
        elif s.phase == "discharge":
            soc_seg = sim["soc"][s.start_idx : s.end_idx]
            diffs = np.diff(soc_seg)
            assert (diffs <= 1e-5).all(), "SOC rose during a discharge segment"


def test_temperature_rises_under_load():
    """Joule heating: T should be higher during high-current phases than during rest."""
    p = BatteryParams(thermal_capacitance=20.0)                      # smaller → faster heating
    sim = simulate_battery(
        p,
        CycleSpec(charge_C=2.0, discharge_C=2.0, rest_seconds_after_charge=2000.0,
                  rest_seconds_after_discharge=2000.0),
        n_cycles=1,
        seed=0,
    )
    segs = segment_phases(sim["I"], eps=0.05, min_run=10)
    charge_T = np.mean([sim["T"][s.start_idx:s.end_idx].mean()
                        for s in segs if s.phase == "charge"])
    # Compare with ambient (start of simulation) — should be warmer after load.
    ambient = sim["T"][:50].mean()
    assert charge_T > ambient, f"expected charge T > ambient, got {charge_T:.2f} vs {ambient:.2f}"


# ---------- population ----------

def test_make_population_varies_capacity():
    sims = make_population(n_batteries=4, n_cycles=1, perturb_scale=0.2, seed=0)
    caps = [s["capacity_Ah"] for s in sims]
    assert len(set(caps)) == 4, "expected each battery to have a distinct capacity"
    # All capacities should remain physically plausible.
    for c in caps:
        assert 0.5 < c < 10.0


def test_population_battery_ids_unique():
    sims = make_population(n_batteries=3, n_cycles=1, seed=0)
    ids = [s["battery_id"] for s in sims]
    assert len(set(ids)) == len(ids)
