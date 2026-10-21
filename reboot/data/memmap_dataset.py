"""Memmap-backed BatteryDataset — much faster than the Python-object pickle.

Reads windows_data.npy as a read-only memmap (OS page-cached) and the metadata
arrays from windows_meta.npz. Random access is essentially RAM speed after
warm-up.

API is intentionally close to BatteryDataset so DataLoader / collate functions
stay identical.

Two additions over the original pickle dataset:

* ``clip_value`` — every scaled channel is clamped to ``[-clip, +clip]`` at load
  time. The derived channels dV/dt and dT/dt (robust-scaled by IQR) contain
  sensor / time-step glitches reaching |x| > 20 000 while 99.9 % of values sit
  inside ±250 and the bulk inside ±5. Feeding those into the L1 reconstruction
  loss produced the loss spikes (epochs 37 & 40) and the NaN divergence (epoch
  43) seen in SLURM job 72178. Clipping is applied identically in pretrain,
  finetune and eval so the model never sees a different distribution.

* finetune-mode "next window" — for L_phys (CLAUDE.md §6.6) the module needs
  ŷ on the window that follows the current one inside the same segment.
  Windows are stored in (segment, local_t) order, so the successor of index
  ``idx`` is ``idx + 1`` iff it shares the segment id and sits exactly one
  ``stride`` later. We return it as ``x_next`` with a ``has_next`` flag, plus
  the *raw* mean current and total elapsed time over the ``stride`` samples that
  separate the two windows (recovered from the scaled channels via the fitted
  Scaler). Those are the correct quantities for Coulomb counting between the two
  window means; the ``I_mean`` / ``dt_total`` stored in the meta file are scaled
  values and must not be used for physics.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from reboot.data.masking import generate_mask
from reboot.data.transforms import IDX_DT, IDX_I, Scaler

Mode = Literal["pretrain", "finetune", "eval"]
SplitName = Literal["train", "test", "val"]

_SPLIT_TO_INT = {"train": 0, "test": 1, "val": 2}

DEFAULT_CLIP_VALUE = 10.0


class MemmapBatteryDataset(Dataset):
    """Read from windows_data.npy (memmap) + windows_meta.npz."""

    def __init__(
        self,
        processed_dir: str | Path,
        mode: Mode,
        filter_split: SplitName | None = None,
        mask_time_ratio: float = 0.15,
        mask_channel_prob: float = 0.10,
        clip_value: float | None = DEFAULT_CLIP_VALUE,
        stride: int = 32,
        scaler_path: str | Path | None = None,
        with_next: bool | None = None,
        input_channels: list[int] | tuple[int, ...] | None = None,
    ):
        d = Path(processed_dir)
        self.processed_dir = d
        self.mode = mode
        self.mask_time_ratio = mask_time_ratio
        self.mask_channel_prob = mask_channel_prob
        self.clip_value = float(clip_value) if clip_value is not None else None
        self.stride = int(stride)

        self._data = None                                            # opened lazily per worker
        meta = np.load(d / "windows_meta.npz")
        self.phase = meta["phase"]
        self.battery_idx = meta["battery_idx"]
        self.cycle_id = meta["cycle_id"]
        self.segment_id = meta["segment_id"]
        self.segment_length = meta["segment_length"]
        self.local_t = meta["local_t"]
        self.pseudo_soc = meta["pseudo_soc"]
        self.has_soc = meta["has_soc"]
        self.y = meta["y"]
        self.I_mean = meta["I_mean"]                                  # SCALED — not for physics
        self.dt_total = meta["dt_total"]                              # SCALED — not for physics
        self.split = meta["split"]

        with open(d / "windows_extra.json") as f:
            self.extra = json.load(f)

        # Compute the active index set once.
        mask = np.ones(len(self.phase), dtype=np.bool_)
        if filter_split is not None:
            mask &= (self.split == _SPLIT_TO_INT[filter_split])
        if mode != "pretrain":
            mask &= self.has_soc
        self.indices = np.where(mask)[0].astype(np.int64)

        # Sanity check on the data array shape (without loading it yet).
        shape = tuple(np.load(d / "windows_data.npy", mmap_mode="r").shape)
        assert shape[0] == len(self.phase), (
            f"windows_data.npy length {shape[0]} != meta length {len(self.phase)}"
        )
        self.W = int(shape[1])
        self._C_stored = int(shape[2])
        # Optional channel subset (e.g. [0,1,2,3] = raw I,V,T,dt only, dropping the derived
        # dV/dt, dT/dt, Q_cum channels). Applied at load time. L_phys reads I (0) and dt (3)
        # from the loaded window, so any subset must start with [0,1,2,3] in finetune mode.
        self.input_channels = list(input_channels) if input_channels is not None else None
        if self.input_channels is not None:
            assert all(0 <= c < self._C_stored for c in self.input_channels), self.input_channels
            if (mode == "finetune") if with_next is None else bool(with_next):
                assert self.input_channels[:4] == [0, 1, 2, 3], "L_phys needs I,V,T,dt at positions 0-3"
        self.C = len(self.input_channels) if self.input_channels is not None else self._C_stored

        # Successor lookup: idx+1 is the next window of the same segment iff it
        # shares segment_id and is exactly one stride later.
        n = len(self.phase)
        has_next = np.zeros(n, dtype=np.bool_)
        if n > 1:
            has_next[:-1] = (
                (self.segment_id[1:] == self.segment_id[:-1])
                & (self.local_t[1:] == self.local_t[:-1] + self.stride)
            )
        self.has_next = has_next

        # Raw-unit recovery for L_phys (finetune only).
        self.with_next = (mode == "finetune") if with_next is None else bool(with_next)
        self.scaler: Scaler | None = None
        if self.with_next:
            if scaler_path is not None and Path(scaler_path).exists():
                self.scaler = Scaler.load(scaler_path)
            else:
                warnings.warn(
                    "MemmapBatteryDataset(mode='finetune') has no scaler_path; "
                    "x_next / raw I,dt will not be provided and L_phys will be 0.",
                    stacklevel=2,
                )
                self.with_next = False

    def _open_memmap(self) -> np.memmap:
        """Open the memmap lazily so each DataLoader worker has its own handle."""
        if self._data is None:
            self._data = np.load(self.processed_dir / "windows_data.npy", mmap_mode="r")
        return self._data

    @property
    def in_channels(self) -> int:
        return self.C

    def __len__(self) -> int:
        return int(len(self.indices))

    def _load_x(self, data: np.memmap, idx: int) -> torch.Tensor:
        arr = data[idx]
        if self.input_channels is not None:
            arr = arr[:, self.input_channels]
        x = torch.from_numpy(np.array(arr, copy=True)).float()  # (W, C)
        if self.clip_value is not None:
            x.clamp_(-self.clip_value, self.clip_value)
        return x

    def _raw_I_dt_over_stride(self, x_scaled: torch.Tensor) -> tuple[float, float]:
        """Mean raw current [A] and total raw elapsed time [s] over the first
        ``stride`` samples of the window — the samples that separate this window
        from its successor."""
        assert self.scaler is not None
        s = min(self.stride, x_scaled.shape[0])
        I_scaled = x_scaled[:s, IDX_I].double()
        dt_scaled = x_scaled[:s, IDX_DT].double()
        I_raw = I_scaled * max(self.scaler.I_iqr, 1e-6) + self.scaler.I_median
        dt_raw = torch.expm1(dt_scaled.clamp(min=0.0))
        return float(I_raw.mean()), float(dt_raw.sum())

    def __getitem__(self, i: int) -> dict:
        idx = int(self.indices[i])
        data = self._open_memmap()
        x = self._load_x(data, idx)

        item: dict = {
            "x": x,
            "phase": torch.tensor(int(self.phase[idx]), dtype=torch.long),
            "battery_idx": torch.tensor(int(self.battery_idx[idx]), dtype=torch.long),
            "cycle_id": torch.tensor(int(self.cycle_id[idx]), dtype=torch.long),
            "segment_id": torch.tensor(int(self.segment_id[idx]), dtype=torch.long),
            "segment_length": torch.tensor(int(self.segment_length[idx]), dtype=torch.long),
            "local_t": torch.tensor(int(self.local_t[idx]), dtype=torch.long),
            "pseudo_soc": torch.tensor(float(self.pseudo_soc[idx]), dtype=torch.float32),
        }
        if self.mode == "pretrain":
            item["mask"] = generate_mask(
                W=x.shape[0], C=x.shape[1],
                time_mask_ratio=self.mask_time_ratio,
                channel_mask_prob=self.mask_channel_prob,
            )
        if self.mode in ("finetune", "eval"):
            item["y"] = torch.tensor(float(self.y[idx]), dtype=torch.float32)
            # Kept for backward compatibility with collate_finetune; scaled units.
            item["I_mean"] = torch.tensor(float(self.I_mean[idx]), dtype=torch.float32)
            item["dt_total"] = torch.tensor(float(self.dt_total[idx]), dtype=torch.float32)
        if self.with_next:
            has_next = bool(self.has_next[idx])
            item["has_next"] = torch.tensor(has_next, dtype=torch.bool)
            item["x_next"] = self._load_x(data, idx + 1) if has_next else x.clone()
            I_raw_mean, dt_raw_total = self._raw_I_dt_over_stride(x)
            item["I_raw_mean"] = torch.tensor(I_raw_mean, dtype=torch.float32)
            item["dt_raw_total"] = torch.tensor(dt_raw_total, dtype=torch.float32)
        return item
