"""Battery dataset + Window dataclass + collate_fn — see CLAUDE.md §4.4, §4.6."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from reboot.data.masking import generate_mask
from reboot.data.pair_sampling import (
    make_consecutive_pairs,
    make_ordinal_pairs,
    make_reflective_pairs,
)
from reboot.data.phase_segmentation import Segment, segment_phases

Mode = Literal["pretrain", "finetune", "eval"]
Phase = Literal["charge", "discharge"]


@dataclass
class Window:
    """A single fixed-length window extracted from one phase segment."""

    data: np.ndarray              # (W, C) — already scaled
    phase: Phase
    battery_id: str               # used for splits
    battery_idx: int              # integer index into the battery list (for L_phys)
    cycle_id: int                 # globally unique cycle id across the dataset
    segment_id: int               # globally unique segment id across the dataset
    segment_length: int           # length of the parent segment (windows or samples)
    local_t: int                  # offset of this window inside the segment
    soc: np.ndarray | None = None # (W,) — only when mode != 'pretrain'
    pseudo_soc: float | None = None   # used by L_refl with strategy='soc_proxy'
    split: str = "train"          # "train" | "val" | "test" — assigned in prepare_data


@dataclass
class BatteryRecord:
    """Full per-battery record assembled by prepare_data.py before windowing.

    One record corresponds to ONE (battery, cycle) pair after we explode each cell
    into per-cycle records — that way the per-cell random split simply tags whole
    records and there is no chance of splitting a cycle's charge from its discharge.
    """

    battery_id: str
    battery_idx: int
    cycle_id: int                # globally unique
    data: np.ndarray             # (T, C) — scaled
    segments: list[Segment] = field(default_factory=list)
    soc: np.ndarray | None = None
    pseudo_soc: np.ndarray | None = None
    split: str = "train"         # "train" | "test" — propagated to every Window


def windows_from_record(
    record: BatteryRecord,
    window_size: int,
    stride: int,
    next_segment_id: int,
    drop_unlabeled: bool = False,
) -> tuple[list[Window], int]:
    """Yield windows from a battery record. Windows never cross phase boundaries.

    Returns (windows, next_segment_id_after) so the caller can keep segment ids unique
    across batteries.
    """
    windows: list[Window] = []
    seg_id = next_segment_id
    for segment in record.segments:
        if segment.phase not in ("charge", "discharge"):
            continue
        seg_len = segment.length
        if seg_len < window_size:
            continue
        seg_data = record.data[segment.start_idx : segment.end_idx]
        seg_soc = (
            record.soc[segment.start_idx : segment.end_idx]
            if record.soc is not None
            else None
        )
        seg_pseudo = (
            record.pseudo_soc[segment.start_idx : segment.end_idx]
            if record.pseudo_soc is not None
            else None
        )
        n_windows = (seg_len - window_size) // stride + 1
        for w_idx in range(n_windows):
            start = w_idx * stride
            end = start + window_size
            w_soc = seg_soc[start:end] if seg_soc is not None else None
            if drop_unlabeled and w_soc is None:
                continue
            w_pseudo = (
                float(seg_pseudo[start:end].mean()) if seg_pseudo is not None else None
            )
            windows.append(
                Window(
                    data=seg_data[start:end].astype(np.float32, copy=False),
                    phase=segment.phase,  # type: ignore[arg-type]
                    battery_id=record.battery_id,
                    battery_idx=record.battery_idx,
                    cycle_id=record.cycle_id,
                    segment_id=seg_id,
                    segment_length=seg_len,
                    local_t=start,
                    split=record.split,
                    soc=w_soc.astype(np.float32, copy=False) if w_soc is not None else None,
                    pseudo_soc=w_pseudo,
                )
            )
        seg_id += 1
    return windows, seg_id


def build_records(
    batteries: list[dict],
    has_derived: bool,
    phase_eps: float,
    phase_min_run: int,
) -> list[BatteryRecord]:
    """Convert a list of per-battery dicts (from prepare_data.py) into BatteryRecords.

    Each input dict must contain:
        battery_id: str
        battery_idx: int
        cycle_id: int
        data: np.ndarray (T, C)  — already scaled
        I_raw: np.ndarray (T,)   — unscaled current, used for phase segmentation
        soc: np.ndarray | None
        pseudo_soc: np.ndarray | None
    """
    records: list[BatteryRecord] = []
    for b in batteries:
        segments = segment_phases(
            b["I_raw"], eps=phase_eps, min_run=phase_min_run, keep_rest=False
        )
        records.append(
            BatteryRecord(
                battery_id=b["battery_id"],
                battery_idx=int(b["battery_idx"]),
                cycle_id=int(b["cycle_id"]),
                data=b["data"],
                segments=segments,
                soc=b.get("soc"),
                pseudo_soc=b.get("pseudo_soc"),
            )
        )
    return records


class BatteryDataset(Dataset):
    """Wraps a precomputed list of windows.

    Mode determines what each item contains:
        pretrain  →  x, phase, segment_id, local_t, battery_idx, cycle_id, segment_length, mask
        finetune  →  + y (scalar SOC), I (W,), dt (W,)
        eval      →  finetune fields, but no shuffling at the DataLoader level (left to caller)
    """

    def __init__(
        self,
        windows: list[Window],
        mode: Mode,
        mask_time_ratio: float = 0.15,
        mask_channel_prob: float = 0.10,
    ):
        assert mode in ("pretrain", "finetune", "eval")
        self.windows = windows
        self.mode = mode
        self.mask_time_ratio = mask_time_ratio
        self.mask_channel_prob = mask_channel_prob

        if mode == "pretrain":
            assert all(w.soc is None for w in windows), (
                "SOC must not be present in pretrain mode (label leakage)"
            )
        if mode in ("finetune", "eval"):
            assert all(w.soc is not None for w in windows), (
                f"mode={mode} requires SOC labels on every window"
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        w = self.windows[idx]
        x = torch.from_numpy(w.data).float()                          # (W, C)
        item: dict = {
            "x": x,
            "phase": torch.tensor(0 if w.phase == "charge" else 1, dtype=torch.long),
            "battery_idx": torch.tensor(w.battery_idx, dtype=torch.long),
            "cycle_id": torch.tensor(w.cycle_id, dtype=torch.long),
            "segment_id": torch.tensor(w.segment_id, dtype=torch.long),
            "segment_length": torch.tensor(w.segment_length, dtype=torch.long),
            "local_t": torch.tensor(w.local_t, dtype=torch.long),
            "pseudo_soc": torch.tensor(
                w.pseudo_soc if w.pseudo_soc is not None else 0.0, dtype=torch.float32
            ),
        }

        if self.mode == "pretrain":
            mask = generate_mask(
                W=x.shape[0],
                C=x.shape[1],
                time_mask_ratio=self.mask_time_ratio,
                channel_mask_prob=self.mask_channel_prob,
            )
            item["mask"] = mask

        if self.mode in ("finetune", "eval"):
            assert w.soc is not None
            item["y"] = torch.tensor(float(np.mean(w.soc)), dtype=torch.float32)
            # I and dt are in raw-channel positions 0 and 3.
            item["I_mean"] = torch.tensor(float(np.mean(w.data[:, 0])), dtype=torch.float32)
            item["dt_total"] = torch.tensor(float(np.sum(w.data[:, 3])), dtype=torch.float32)

        return item


def collate_pretrain(batch: list[dict], stride: int, pair_strategy: str, K_pairs: int = 8) -> dict:
    """Collate a batch of pretrain items and pre-compute pair indices.

    All pair indices reference positions in the FLAT batch (post-stacking).
    """
    out: dict = {}
    out["x"] = torch.stack([b["x"] for b in batch], dim=0)            # (B, W, C)
    out["mask"] = torch.stack([b["mask"] for b in batch], dim=0)       # (B, W, C)
    out["phase"] = torch.stack([b["phase"] for b in batch], dim=0)     # (B,)
    out["battery_idx"] = torch.stack([b["battery_idx"] for b in batch], dim=0)
    out["cycle_id"] = torch.stack([b["cycle_id"] for b in batch], dim=0)
    out["segment_id"] = torch.stack([b["segment_id"] for b in batch], dim=0)
    out["segment_length"] = torch.stack([b["segment_length"] for b in batch], dim=0)
    out["local_t"] = torch.stack([b["local_t"] for b in batch], dim=0)
    out["pseudo_soc"] = torch.stack([b["pseudo_soc"] for b in batch], dim=0)

    seg_ids = out["segment_id"].tolist()
    local_ts = out["local_t"].tolist()
    phases = out["phase"].tolist()
    cycle_ids = out["cycle_id"].tolist()
    seg_lens = out["segment_length"].tolist()
    pseudo_socs = out["pseudo_soc"].tolist()

    pair_i, pair_j = make_ordinal_pairs(seg_ids, local_ts, K=K_pairs)
    consec_i, consec_j = make_consecutive_pairs(seg_ids, local_ts, stride=stride)
    refl_c, refl_d = make_reflective_pairs(
        phases=phases,
        cycle_ids=cycle_ids,
        segment_ids=seg_ids,
        local_ts=local_ts,
        segment_lengths=seg_lens,
        strategy=pair_strategy,
        pseudo_socs=pseudo_socs if pair_strategy == "soc_proxy" else None,
    )

    out["pair_i"] = pair_i
    out["pair_j"] = pair_j
    out["consec_i"] = consec_i
    out["consec_j"] = consec_j
    out["refl_c"] = refl_c
    out["refl_d"] = refl_d
    return out


def collate_finetune(batch: list[dict]) -> dict:
    """Collate a batch of finetune/eval items. Pair indices are not needed."""
    keys_scalar = [
        "phase", "battery_idx", "cycle_id", "segment_id", "segment_length",
        "local_t", "y", "I_mean", "dt_total", "pseudo_soc",
    ]
    out: dict = {"x": torch.stack([b["x"] for b in batch], dim=0)}
    for k in keys_scalar:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    # Optional next-window fields for L_phys (provided by MemmapBatteryDataset
    # in finetune mode). Only stacked when every item carries them.
    for k in ("x_next", "has_next", "I_raw_mean", "dt_raw_total"):
        if all(k in b for b in batch):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
