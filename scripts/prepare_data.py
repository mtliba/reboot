"""Convert real_data/cell*.h5 files into a windowed dataset.

Each HDF5 file is one battery cell stored as a pandas DataFrame under key='df'.
The DataFrame has one ROW per cycle (charge + discharge bundled). Columns:

    Cycle_index : int scalar
    Time[h]     : array   — timestamps in hours within this cycle
    I[A]        : array   — current (positive = charge, negative = discharge)
    U[V]        : array   — voltage
    T1[°C]      : array   — temperature
    QC          : float scalar — total charge capacity for this cycle (Ah)
    Qd          : float scalar — total discharge capacity (Ah, negative)
    SOH         : float scalar — state of health
    SOC         : array   — state of charge (~[0, 1])

Per-column arrays sometimes differ in length by a few samples; we truncate each
cycle's columns to the common minimum length.

Pipeline:
    1. For each HDF5 file:
         a. Read DataFrame; iterate cycles; build one per-cycle record with
            [I, V, T, dt] + SOC, with globally unique cycle_id.
         b. Per-cell random 80/20 split at the cycle level — both phases of one
            cycle land in the same split.
    2. Fit Scaler on TRAIN records only.
    3. Apply Scaler everywhere; (optionally) add derived channels.
    4. Phase-segment each cycle; window inside segments.
    5. Save windows.pkl with the split attached to every Window.
"""
from __future__ import annotations

import pickle
import random
import re
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from reboot.data.dataset import build_records, windows_from_record
from reboot.data.transforms import derived_channels, fit_scaler
from reboot.utils.seed import set_global_seed

# Each cell file gets up to this many globally-unique cycle ids reserved.
# Must exceed max(len(df)) across all cells (~3000 cycles in this dataset).
_CYCLE_ID_STRIDE = 1_000_000

# Parse cell filename like "cell13-35.h5" → cell_num=13, ambient_T=35
_FNAME_RE = re.compile(r"cell(\d+)-(\d+)\.h5$")


def _parse_filename(name: str) -> tuple[int, int]:
    m = _FNAME_RE.search(name)
    if not m:
        raise ValueError(f"Unexpected cell filename: {name!r}")
    return int(m.group(1)), int(m.group(2))


def load_raw_battery(path: Path, battery_idx: int) -> list[dict]:
    """Read one HDF5 cell file and return a list of per-cycle dicts.

    Each returned dict has:
        battery_id     : str   — e.g. "cell13-35"
        battery_idx    : int   — sequential per cell file
        cycle_id       : int   — globally unique (battery_idx * STRIDE + per_cell_idx)
        ambient_T_C    : int   — parsed from filename
        nominal_cap_Ah : float — derived from QC.max() across all cycles in this cell
        SOH            : float
        raw            : (T, 4) float32  — columns [I, V, T, dt]
        soc            : (T,)  float32
    """
    df = pd.read_hdf(path, key="df")
    cell_num, ambient_T = _parse_filename(path.name)
    battery_id = f"cell{cell_num}-{ambient_T}"
    # Use QC.max() as the nominal capacity. Fall back to 2.5 Ah if absent.
    if "QC" in df.columns and len(df) > 0:
        nominal_cap_Ah = float(df["QC"].max())
        if not np.isfinite(nominal_cap_Ah) or nominal_cap_Ah <= 0:
            nominal_cap_Ah = 2.5
    else:
        nominal_cap_Ah = 2.5

    records: list[dict] = []
    base_cycle_id = battery_idx * _CYCLE_ID_STRIDE

    for cyc_pos in range(len(df)):
        t = np.asarray(df["Time[h]"].iloc[cyc_pos], dtype=np.float64)
        I = np.asarray(df["I[A]"].iloc[cyc_pos], dtype=np.float32)
        V = np.asarray(df["U[V]"].iloc[cyc_pos], dtype=np.float32)
        T = np.asarray(df["T1[°C]"].iloc[cyc_pos], dtype=np.float32)
        soc = np.asarray(df["SOC"].iloc[cyc_pos], dtype=np.float32)

        n = int(min(t.size, I.size, V.size, T.size, soc.size))
        if n < 50:                                                   # too short for any window
            continue
        t = t[:n]; I = I[:n]; V = V[:n]; T = T[:n]; soc = soc[:n]

        # dt in seconds. First sample gets the median dt as a placeholder.
        dt_h = np.diff(t, prepend=t[0])
        dt_s = dt_h.astype(np.float32) * 3600.0
        dt_s = np.where(np.isfinite(dt_s) & (dt_s > 0), dt_s, 1.0)

        soc = np.clip(soc, 0.0, 1.0).astype(np.float32)

        raw = np.stack([I, V, T, dt_s], axis=1).astype(np.float32)
        # Use positional index (0..N-1) for guaranteed uniqueness within the cell;
        # the dataset's Cycle_index can exceed our stride and collide across cells.
        records.append(
            {
                "battery_id": battery_id,
                "battery_idx": battery_idx,
                "cycle_id": base_cycle_id + cyc_pos,
                "ambient_T_C": ambient_T,
                "nominal_cap_Ah": nominal_cap_Ah,
                "SOH": float(df["SOH"].iloc[cyc_pos]) if "SOH" in df.columns else 1.0,
                "raw": raw,                                          # (T, 4)
                "soc": soc,
            }
        )
    return records


def _pseudo_soc_from_I(I: np.ndarray, dt_s: np.ndarray) -> np.ndarray:
    q = np.cumsum(I * dt_s)
    lo, hi = q.min(), q.max()
    if hi - lo < 1e-9:
        return np.zeros_like(q, dtype=np.float32)
    return ((q - lo) / (hi - lo)).astype(np.float32)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)
    raw_dir = Path(cfg.data.raw_dir)
    out_dir = Path(cfg.data.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(OmegaConf.to_yaml(cfg.data))

    raw_files = sorted(raw_dir.glob("cell*.h5"), key=lambda p: _parse_filename(p.name))
    assert raw_files, f"No cell*.h5 files found in {raw_dir}"
    print(f"Found {len(raw_files)} cell files.")

    # ---- 1. Load every cell ----
    all_cycles: list[dict] = []
    per_battery_cycles: dict[str, list[int]] = defaultdict(list)
    for idx, p in enumerate(raw_files):
        cycles = load_raw_battery(p, battery_idx=idx)
        print(f"  [{idx:2d}] {p.name:>20s}  cycles={len(cycles):4d}  "
              f"nom_cap_Ah={cycles[0]['nominal_cap_Ah']:.3f}  T_amb={cycles[0]['ambient_T_C']}°C"
              if cycles else f"  [{idx:2d}] {p.name}  EMPTY")
        all_cycles.extend(cycles)
        for c in cycles:
            per_battery_cycles[c["battery_id"]].append(c["cycle_id"])

    # ---- 2. Per-cell random 80/20 split at cycle level ----
    strategy = cfg.data.splits.get("strategy", "per_cell_random")
    train_frac = float(cfg.data.splits.get("train_fraction", 0.8))
    split_seed = int(cfg.data.splits.get("random_seed", cfg.seed))
    rng = random.Random(split_seed)

    cycle_split: dict[int, str] = {}
    if strategy == "per_cell_random":
        print(f"\nSplit strategy: per_cell_random, train_fraction={train_frac:.2f}, seed={split_seed}")
        print(f"{'cell':<14}{'n_cycles':>10}{'train':>8}{'test':>8}")
        print("-" * 40)
        for cell_id, cyc_ids in per_battery_cycles.items():
            cyc_ids = list(cyc_ids)
            rng.shuffle(cyc_ids)
            cut = max(1, int(len(cyc_ids) * train_frac))
            for cid in cyc_ids[:cut]:
                cycle_split[cid] = "train"
            for cid in cyc_ids[cut:]:
                cycle_split[cid] = "test"
            n_train = sum(1 for cid in cyc_ids if cycle_split[cid] == "train")
            n_test = sum(1 for cid in cyc_ids if cycle_split[cid] == "test")
            print(f"{cell_id:<14}{len(cyc_ids):>10}{n_train:>8}{n_test:>8}")
    else:
        raise NotImplementedError(f"split strategy {strategy!r} not supported in this loader yet.")

    # ---- 3. Optional derived channels + fit scaler on TRAIN cycles only ----
    has_derived = bool(cfg.data.derived_channels)
    train_arrays = []
    for c in all_cycles:
        arr = derived_channels(c["raw"]) if has_derived else c["raw"]
        c["_arr"] = arr
        if cycle_split[c["cycle_id"]] == "train":
            train_arrays.append(arr)
    scaler = fit_scaler(train_arrays, has_derived=has_derived)
    scaler.save(cfg.paths.scaler_path)
    print(f"\nSaved scaler to {cfg.paths.scaler_path}")

    # ---- 4. Build per-cycle records, phase-segment, window ----
    per_cycle_input = []
    for c in all_cycles:
        scaled = scaler.transform(c["_arr"])
        del c["_arr"]
        per_cycle_input.append(
            {
                "battery_id": c["battery_id"],
                "battery_idx": c["battery_idx"],
                "cycle_id": c["cycle_id"],
                "data": scaled,
                "I_raw": c["raw"][:, 0],
                "soc": c["soc"],
                "pseudo_soc": _pseudo_soc_from_I(c["raw"][:, 0], c["raw"][:, 3]),
            }
        )

    records = build_records(
        per_cycle_input,
        has_derived=has_derived,
        phase_eps=cfg.data.phase_eps,
        phase_min_run=cfg.data.phase_min_run,
    )
    # Tag each record with its split so windows_from_record propagates it.
    for rec in records:
        rec.split = cycle_split[rec.cycle_id]

    # Phase distribution diagnostic
    n_charge = sum(1 for r in records for s in r.segments if s.phase == "charge")
    n_disch = sum(1 for r in records for s in r.segments if s.phase == "discharge")
    print(f"\nSegments overall: charge={n_charge}  discharge={n_disch}")

    print(f"Windowing W={cfg.data.window_size} stride={cfg.data.stride} ...")
    all_windows = []
    seg_id = 0
    skipped_short = 0
    for rec in records:
        before = len(all_windows)
        ws, seg_id = windows_from_record(
            rec, window_size=cfg.data.window_size, stride=cfg.data.stride,
            next_segment_id=seg_id, drop_unlabeled=False,
        )
        if not ws:
            skipped_short += 1
        all_windows.extend(ws)
    print(f"Cycles producing no windows (segments too short): {skipped_short}")
    print(f"Total windows: {len(all_windows)}")
    n_tr = sum(1 for w in all_windows if w.split == "train")
    n_te = sum(1 for w in all_windows if w.split == "test")
    n_ch = sum(1 for w in all_windows if w.phase == "charge")
    n_di = sum(1 for w in all_windows if w.phase == "discharge")
    print(f"  by split: train={n_tr}  test={n_te}")
    print(f"  by phase: charge={n_ch}  discharge={n_di}")

    # ---- 5. Capacities for L_phys (A·s per battery) ----
    capacities = {}
    seen_bid = {}
    for c in all_cycles:
        seen_bid.setdefault(c["battery_idx"], c["nominal_cap_Ah"])
    capacities = {bi: cap * 3600.0 for bi, cap in seen_bid.items()}

    battery_ids = list(per_battery_cycles.keys())
    out_path = out_dir / "windows.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "windows": all_windows,
                "battery_ids": battery_ids,
                "train_ids": battery_ids,                            # every cell is in both splits
                "test_ids": battery_ids,
                "capacities_As": capacities,
                "split_strategy": strategy,
                "split_seed": split_seed,
                "train_fraction": train_frac,
            },
            f,
        )
    print(f"\nSaved {len(all_windows)} windows to {out_path}")


if __name__ == "__main__":
    main()
