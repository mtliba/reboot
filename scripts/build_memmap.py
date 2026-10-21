"""One-shot converter: windows.pkl  →  memmap-friendly numpy files.

Reads:  data/processed/windows.pkl  (slow to deserialize, full of Python objects)
Writes:
  data/processed/windows_data.npy   shape (N, W, C) float32, memmap-friendly
  data/processed/windows_meta.npz   per-window scalars
  data/processed/windows_extra.json battery_ids, train_ids, capacities_As, ...

After this runs, scripts/pretrain.py + scripts/finetune.py read the memmap
files directly via reboot.data.memmap_dataset.MemmapBatteryDataset.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

_SPLIT_TO_INT = {"train": 0, "test": 1, "val": 2}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    src = Path(cfg.data.processed_dir) / "windows.pkl"
    out_dir = src.parent
    print(f"Loading {src} ...")
    with open(src, "rb") as f:
        blob = pickle.load(f)
    windows = blob["windows"]
    N = len(windows)
    W, C = windows[0].data.shape
    print(f"  N={N:,}  W={W}  C={C}")

    out_data = out_dir / "windows_data.npy"
    out_meta = out_dir / "windows_meta.npz"
    out_extra = out_dir / "windows_extra.json"

    # Pre-allocate the big array as a memmap so we can stream-write it without
    # holding (N, W, C) in RAM twice.
    data = np.lib.format.open_memmap(
        out_data, mode="w+", dtype=np.float32, shape=(N, W, C),
    )

    phase = np.empty(N, dtype=np.int8)
    battery_idx = np.empty(N, dtype=np.int32)
    cycle_id = np.empty(N, dtype=np.int64)
    segment_id = np.empty(N, dtype=np.int32)
    segment_length = np.empty(N, dtype=np.int32)
    local_t = np.empty(N, dtype=np.int32)
    pseudo_soc = np.empty(N, dtype=np.float32)
    has_soc = np.empty(N, dtype=np.bool_)
    y = np.empty(N, dtype=np.float32)
    I_mean = np.empty(N, dtype=np.float32)
    dt_total = np.empty(N, dtype=np.float32)
    split = np.empty(N, dtype=np.int8)

    for i, w in enumerate(tqdm(windows, desc="windows → memmap")):
        data[i] = w.data                                             # copies into mmap
        phase[i] = 0 if w.phase == "charge" else 1
        battery_idx[i] = w.battery_idx
        cycle_id[i] = w.cycle_id
        segment_id[i] = w.segment_id
        segment_length[i] = w.segment_length
        local_t[i] = w.local_t
        pseudo_soc[i] = w.pseudo_soc if w.pseudo_soc is not None else 0.0
        if w.soc is not None and w.soc.size:
            has_soc[i] = True
            y[i] = float(np.mean(w.soc))
        else:
            has_soc[i] = False
            y[i] = np.nan
        I_mean[i] = float(np.mean(w.data[:, 0]))
        dt_total[i] = float(np.sum(w.data[:, 3]))
        split[i] = _SPLIT_TO_INT.get(getattr(w, "split", "train"), 0)

    data.flush()
    del data

    np.savez(
        out_meta,
        phase=phase, battery_idx=battery_idx, cycle_id=cycle_id,
        segment_id=segment_id, segment_length=segment_length, local_t=local_t,
        pseudo_soc=pseudo_soc, has_soc=has_soc, y=y, I_mean=I_mean, dt_total=dt_total,
        split=split,
    )

    extra = {
        "battery_ids": blob.get("battery_ids", []),
        "train_ids": blob.get("train_ids", []),
        "test_ids": blob.get("test_ids", []),
        "split_strategy": blob.get("split_strategy", ""),
        "split_seed": blob.get("split_seed", 0),
        "train_fraction": blob.get("train_fraction", 0.8),
        "N": N, "W": W, "C": C,
    }
    caps = blob.get("capacities_As") or {}
    extra["capacities_As"] = {str(k): float(v) for k, v in caps.items()}
    with open(out_extra, "w") as f:
        json.dump(extra, f, indent=2)

    print(f"\nWrote {out_data}        ({N * W * C * 4 / 1e9:.2f} GB)")
    print(f"Wrote {out_meta}")
    print(f"Wrote {out_extra}")
    print(f"Distribution: train={(split == 0).sum():,}  test={(split == 1).sum():,}")
    print(f"Phase:        charge={(phase == 0).sum():,}  discharge={(phase == 1).sum():,}")


if __name__ == "__main__":
    main()
