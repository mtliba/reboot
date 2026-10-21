"""Generate a synthetic ReBoot dataset and write windows.pkl directly.

The output is byte-compatible with what scripts/prepare_data.py would produce on real
data, so scripts/pretrain.py and scripts/finetune.py can consume it without changes.

Usage
-----
    # default: 6 batteries, 4 cycles each, ~3500 samples per cycle at 1 Hz with dt=1.0
    python scripts/generate_synthetic_data.py

    # smaller / quicker
    python scripts/generate_synthetic_data.py synth.n_batteries=3 synth.n_cycles=2

The script writes:
    {processed_dir}/windows.pkl
    {scaler_path}                  — Scaler fitted on the synthetic train batteries
"""
from __future__ import annotations

import pickle
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from reboot.data.dataset import build_records, windows_from_record
from reboot.data.synthetic import BatteryParams, CycleSpec, make_population
from reboot.data.transforms import derived_channels, fit_scaler
from reboot.utils.seed import set_global_seed


def _record_to_pipeline(sim: dict, has_derived: bool) -> dict:
    """Convert a simulator output dict into the shape build_records() expects."""
    I = sim["I"]
    V = sim["V"]
    T = sim["T"]
    dt = sim["dt"]
    raw = np.stack([I, V, T, dt], axis=1).astype(np.float32)         # (T, 4)
    arr = derived_channels(raw) if has_derived else raw

    # pseudo-SOC from Q_cum (label-free proxy used by L_refl soc_proxy strategy)
    q = np.cumsum(I * dt)
    if q.max() - q.min() > 1e-9:
        pseudo = ((q - q.min()) / (q.max() - q.min())).astype(np.float32)
    else:
        pseudo = np.zeros_like(q, dtype=np.float32)

    return {
        "battery_id": sim["battery_id"],
        "battery_idx": int(sim["battery_idx"]),
        "cycle_id": int(sim["cycle_id"].max()),                       # primary cycle id
        "data": arr,                                                  # will be replaced by scaled version
        "raw": arr,                                                   # kept for scaler fit
        "I_raw": I,
        "soc": sim["soc"].astype(np.float32),
        "pseudo_soc": pseudo,
        "_capacity_Ah": float(sim["capacity_Ah"]),
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Synthetic-specific knobs (registered as inline overrides since we have no group file).
    n_batteries = int(cfg.get("synth", {}).get("n_batteries", 6))
    n_cycles = int(cfg.get("synth", {}).get("n_cycles", 4))
    perturb_scale = float(cfg.get("synth", {}).get("perturb_scale", 0.15))
    dt_seconds = float(cfg.get("synth", {}).get("dt_seconds", 1.0))
    charge_C = float(cfg.get("synth", {}).get("charge_C", 1.0))
    discharge_C = float(cfg.get("synth", {}).get("discharge_C", 1.0))
    rest_seconds = float(cfg.get("synth", {}).get("rest_seconds", 300.0))

    set_global_seed(cfg.seed)
    print(OmegaConf.to_yaml({"data": cfg.data, "wandb": cfg.wandb}))

    out_dir = Path(cfg.data.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_batteries} synthetic batteries × {n_cycles} cycles...")
    cycle_spec = CycleSpec(
        charge_C=charge_C,
        discharge_C=discharge_C,
        rest_seconds_after_charge=rest_seconds,
        rest_seconds_after_discharge=rest_seconds,
        dt_seconds=dt_seconds,
    )
    sims = make_population(
        n_batteries=n_batteries,
        n_cycles=n_cycles,
        base_params=BatteryParams(),
        cycle_spec=cycle_spec,
        perturb_scale=perturb_scale,
        seed=cfg.seed,
    )

    has_derived = bool(cfg.data.derived_channels)
    per_battery = [_record_to_pipeline(s, has_derived=has_derived) for s in sims]
    battery_ids = [b["battery_id"] for b in per_battery]

    # Split: last K batteries are test.
    k = int(cfg.data.splits.holdout_k)
    train_ids = battery_ids[:-k] if k > 0 else battery_ids
    test_ids = battery_ids[-k:] if k > 0 else []
    print(f"Train batteries: {train_ids}")
    print(f"Test batteries:  {test_ids}")

    # Fit scaler on train batteries only.
    train_arrs = [b["raw"] for b in per_battery if b["battery_id"] in train_ids]
    scaler = fit_scaler(train_arrs, has_derived=has_derived)
    scaler.save(cfg.paths.scaler_path)
    print(f"Saved scaler to {cfg.paths.scaler_path}")

    # Apply scaler to all batteries.
    for b in per_battery:
        b["data"] = scaler.transform(b["raw"])
        del b["raw"]                                                  # not needed downstream

    # Build records (with phase segmentation) and window them.
    records = build_records(
        per_battery,
        has_derived=has_derived,
        phase_eps=cfg.data.phase_eps,
        phase_min_run=cfg.data.phase_min_run,
    )

    # Phase distribution diagnostic.
    n_charge = sum(1 for r in records for s in r.segments if s.phase == "charge")
    n_disch = sum(1 for r in records for s in r.segments if s.phase == "discharge")
    print(f"Charge segments: {n_charge}, Discharge segments: {n_disch}")

    all_windows = []
    seg_id = 0
    for rec in records:
        ws, seg_id = windows_from_record(
            rec,
            window_size=cfg.data.window_size,
            stride=cfg.data.stride,
            next_segment_id=seg_id,
            drop_unlabeled=False,
        )
        all_windows.extend(ws)
    n_charge_w = sum(1 for w in all_windows if w.phase == "charge")
    n_disch_w = sum(1 for w in all_windows if w.phase == "discharge")
    print(f"Total windows: {len(all_windows)}  (charge={n_charge_w}, discharge={n_disch_w})")

    # Capacities for L_phys nominal (A·s).
    capacities = {b["battery_idx"]: b["_capacity_Ah"] * 3600.0 for b in per_battery}

    out_path = out_dir / "windows.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "windows": all_windows,
                "battery_ids": battery_ids,
                "train_ids": train_ids,
                "test_ids": test_ids,
                "capacities_As": capacities,
            },
            f,
        )
    print(f"Saved {len(all_windows)} windows to {out_path}")


if __name__ == "__main__":
    main()
