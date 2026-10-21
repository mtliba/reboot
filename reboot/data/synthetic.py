"""Synthetic battery simulator — produces realistic I/V/T/SOC trajectories.

Physics summary
---------------
- SOC evolves by Coulomb counting:   d(SOC)/dt = I / Q_As
- Open-circuit voltage:              V_OC(SOC) = sigmoid-shaped curve in [V_min, V_max]
- Terminal voltage:                  V = V_OC(SOC) + I*R_int + sign(I) * V_hyst + noise
- Temperature (lumped RC thermal):   dT/dt = (I^2 * R - (T - T_amb)/R_th) / C_th + noise
- Sign convention:                   I > 0 charging (raises SOC), I < 0 discharging

The simulator produces ONE record per (battery, cycle) call. The default cycle is a
CC-charge + rest + CC-discharge + rest sequence with configurable C-rates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BatteryParams:
    """Per-battery physical parameters. Use perturb() to generate cells that differ."""

    capacity_Ah: float = 2.5
    R_internal_ohm: float = 0.05
    V_min: float = 3.0
    V_max: float = 4.2
    V_hysteresis: float = 0.02              # half-amplitude; charge sees +, discharge sees -
    T_ambient_C: float = 25.0
    thermal_resistance: float = 5.0         # K/W — higher = less heat loss
    thermal_capacitance: float = 100.0      # J/K — higher = slower temperature change
    sigma_V: float = 0.005                  # voltage measurement noise
    sigma_I: float = 0.01                   # current measurement noise
    sigma_T: float = 0.05                   # temperature measurement noise

    def perturb(self, rng: np.random.Generator, scale: float = 0.15) -> "BatteryParams":
        """Return a new BatteryParams with each numeric field multiplied by 1 + N(0, scale).

        Use this to generate a population of batteries that share the family of physics
        but differ in nominal capacity, internal resistance, etc.
        """
        def _jitter(x: float) -> float:
            return float(x * (1.0 + rng.normal(0.0, scale)))

        return BatteryParams(
            capacity_Ah=max(_jitter(self.capacity_Ah), 0.1),
            R_internal_ohm=max(_jitter(self.R_internal_ohm), 1e-3),
            V_min=self.V_min,                # keep voltage window the same — chemistry-dependent
            V_max=self.V_max,
            V_hysteresis=max(_jitter(self.V_hysteresis), 0.0),
            T_ambient_C=self.T_ambient_C + float(rng.normal(0.0, 2.0)),
            thermal_resistance=max(_jitter(self.thermal_resistance), 0.1),
            thermal_capacitance=max(_jitter(self.thermal_capacitance), 1.0),
            sigma_V=self.sigma_V,
            sigma_I=self.sigma_I,
            sigma_T=self.sigma_T,
        )


@dataclass
class CycleSpec:
    """How to run one charge/discharge cycle."""

    charge_C: float = 1.0                   # 1.0 = full capacity per hour
    discharge_C: float = 1.0
    rest_seconds_after_charge: float = 300.0
    rest_seconds_after_discharge: float = 300.0
    soc_min: float = 0.05
    soc_max: float = 0.95
    dt_seconds: float = 1.0                 # sampling interval


def ocv_curve(soc: np.ndarray, V_min: float, V_max: float, sharpness: float = 8.0) -> np.ndarray:
    """Smooth sigmoid OCV, renormalized so OCV(0)=V_min exactly and OCV(1)=V_max exactly.

    The renormalization keeps the curve strictly inside [V_min, V_max] for any SOC ∈ [0,1].
    """
    soc_c = np.clip(soc, 0.0, 1.0)
    raw = 1.0 / (1.0 + np.exp(-sharpness * (soc_c - 0.5)))
    lo = 1.0 / (1.0 + np.exp(sharpness * 0.5))
    hi = 1.0 / (1.0 + np.exp(-sharpness * 0.5))
    norm = (raw - lo) / (hi - lo)                                    # f(0)=0, f(1)=1
    return V_min + (V_max - V_min) * norm


def _thermal_step(T_prev: float, I_prev: float, dt: float, p: BatteryParams) -> float:
    heat = I_prev * I_prev * p.R_internal_ohm
    cooling = (T_prev - p.T_ambient_C) / p.thermal_resistance
    return T_prev + dt * (heat - cooling) / p.thermal_capacitance


def _simulate_cc(
    soc_start: float,
    soc_end: float,
    I_command: float,
    dt: float,
    T_start: float,
    p: BatteryParams,
    rng: np.random.Generator,
) -> dict:
    """Constant-current phase between two SOC levels. I_command sign defines direction."""
    Q_As = p.capacity_Ah * 3600.0
    d_soc_dt = I_command / Q_As
    if abs(d_soc_dt) < 1e-12 or abs(soc_end - soc_start) < 1e-9:
        return {k: np.zeros(0, dtype=np.float32) for k in ("I", "V", "T", "soc", "dt")} | {"T_end": T_start, "soc_end": soc_start}

    duration = abs(soc_end - soc_start) / abs(d_soc_dt)
    n = max(int(duration / dt), 1)

    soc = np.clip(soc_start + np.arange(n + 1, dtype=np.float64) * dt * d_soc_dt, 0.0, 1.0)
    soc = soc[:n]                                           # (n,)
    I = np.full(n, I_command, dtype=np.float64) + rng.normal(0.0, p.sigma_I, size=n)

    is_charge = I_command > 0
    V_OC = ocv_curve(soc, p.V_min, p.V_max)
    V = V_OC + I * p.R_internal_ohm + (p.V_hysteresis if is_charge else -p.V_hysteresis)
    V += rng.normal(0.0, p.sigma_V, size=n)

    T = np.empty(n, dtype=np.float64)
    T[0] = T_start
    for k in range(1, n):
        T[k] = _thermal_step(T[k - 1], I[k - 1], dt, p)
    T_obs = T + rng.normal(0.0, p.sigma_T, size=n)

    return {
        "I": I.astype(np.float32),
        "V": V.astype(np.float32),
        "T": T_obs.astype(np.float32),
        "soc": soc.astype(np.float32),
        "dt": np.full(n, dt, dtype=np.float32),
        "T_end": float(T[-1]),
        "soc_end": float(np.clip(soc_start + n * dt * d_soc_dt, 0.0, 1.0)),
    }


def _simulate_rest(
    duration_s: float,
    dt: float,
    soc: float,
    T_start: float,
    p: BatteryParams,
    rng: np.random.Generator,
) -> dict:
    """Rest phase: I ≈ 0 (noise only), SOC constant, T relaxes toward ambient."""
    n = max(int(duration_s / dt), 1)
    I = rng.normal(0.0, p.sigma_I, size=n)
    soc_arr = np.full(n, soc, dtype=np.float64)
    V_OC = ocv_curve(soc_arr, p.V_min, p.V_max)
    V = V_OC + rng.normal(0.0, p.sigma_V, size=n)
    T = np.empty(n, dtype=np.float64)
    T[0] = T_start
    for k in range(1, n):
        T[k] = _thermal_step(T[k - 1], 0.0, dt, p)
    T_obs = T + rng.normal(0.0, p.sigma_T, size=n)

    return {
        "I": I.astype(np.float32),
        "V": V.astype(np.float32),
        "T": T_obs.astype(np.float32),
        "soc": soc_arr.astype(np.float32),
        "dt": np.full(n, dt, dtype=np.float32),
        "T_end": float(T[-1]),
        "soc_end": float(soc),
    }


def simulate_battery(
    params: BatteryParams,
    cycle_spec: CycleSpec,
    n_cycles: int,
    seed: int = 0,
) -> dict:
    """Run `n_cycles` of {charge, rest, discharge, rest} for one battery.

    Returns
    -------
    dict with:
        I, V, T, dt : (T_total,) float32
        soc         : (T_total,) float32  — ground truth
        cycle_id    : (T_total,) int32     — which cycle each sample belongs to
    """
    rng = np.random.default_rng(seed)
    soc = cycle_spec.soc_min
    T_now = params.T_ambient_C
    out_keys = ("I", "V", "T", "dt", "soc")
    chunks: dict[str, list[np.ndarray]] = {k: [] for k in out_keys}
    cycles: list[np.ndarray] = []

    Q_Ah = params.capacity_Ah
    I_ch = +cycle_spec.charge_C * Q_Ah
    I_dis = -cycle_spec.discharge_C * Q_Ah
    dt = cycle_spec.dt_seconds

    for cyc in range(n_cycles):
        # 1. CC charge
        seg = _simulate_cc(soc, cycle_spec.soc_max, I_ch, dt, T_now, params, rng)
        for k in out_keys:
            chunks[k].append(seg[k])
        cycles.append(np.full(seg["I"].shape[0], cyc, dtype=np.int32))
        T_now, soc = seg["T_end"], seg["soc_end"]

        # 2. Rest
        seg = _simulate_rest(cycle_spec.rest_seconds_after_charge, dt, soc, T_now, params, rng)
        for k in out_keys:
            chunks[k].append(seg[k])
        cycles.append(np.full(seg["I"].shape[0], cyc, dtype=np.int32))
        T_now = seg["T_end"]

        # 3. CC discharge
        seg = _simulate_cc(soc, cycle_spec.soc_min, I_dis, dt, T_now, params, rng)
        for k in out_keys:
            chunks[k].append(seg[k])
        cycles.append(np.full(seg["I"].shape[0], cyc, dtype=np.int32))
        T_now, soc = seg["T_end"], seg["soc_end"]

        # 4. Rest
        seg = _simulate_rest(cycle_spec.rest_seconds_after_discharge, dt, soc, T_now, params, rng)
        for k in out_keys:
            chunks[k].append(seg[k])
        cycles.append(np.full(seg["I"].shape[0], cyc, dtype=np.int32))
        T_now = seg["T_end"]

    out = {k: np.concatenate(chunks[k], axis=0) for k in out_keys}
    out["cycle_id"] = np.concatenate(cycles, axis=0)
    return out


def make_population(
    n_batteries: int,
    n_cycles: int,
    base_params: BatteryParams | None = None,
    cycle_spec: CycleSpec | None = None,
    perturb_scale: float = 0.15,
    seed: int = 0,
) -> list[dict]:
    """Generate a population of `n_batteries`, each running `n_cycles`.

    Each returned dict has keys: battery_id, battery_idx, capacity_Ah, plus the arrays
    produced by simulate_battery() (I, V, T, dt, soc, cycle_id).
    """
    base = base_params or BatteryParams()
    spec = cycle_spec or CycleSpec()
    rng = np.random.default_rng(seed)

    out: list[dict] = []
    for i in range(n_batteries):
        p = base.perturb(rng, scale=perturb_scale)
        sim = simulate_battery(p, spec, n_cycles=n_cycles, seed=int(rng.integers(0, 2**31 - 1)))
        sim["battery_id"] = f"synth_{i:03d}"
        sim["battery_idx"] = i
        sim["capacity_Ah"] = p.capacity_Ah
        sim["params"] = p
        out.append(sim)
    return out
