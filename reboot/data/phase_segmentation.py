"""Phase segmentation — see CLAUDE.md §4.3.

Splits a battery time series into contiguous same-phase segments
(charge / discharge) using a hysteresis filter to avoid spurious
phase flips from noise. Rest segments are dropped.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PHASE_CHARGE = "charge"
PHASE_DISCHARGE = "discharge"
PHASE_REST = "rest"


@dataclass(frozen=True)
class Segment:
    """A contiguous same-phase slice of one battery."""

    start_idx: int
    end_idx: int          # exclusive
    phase: str            # 'charge' | 'discharge' | 'rest'

    @property
    def length(self) -> int:
        return self.end_idx - self.start_idx


def _raw_labels(I: np.ndarray, eps: float) -> np.ndarray:
    out = np.full(I.shape, PHASE_REST, dtype=object)
    out[I > eps] = PHASE_CHARGE
    out[I < -eps] = PHASE_DISCHARGE
    return out


def hysteresis_filter(labels: np.ndarray, min_run: int = 10) -> np.ndarray:
    """Keep the current phase label until `min_run` consecutive samples of a different label,
    then switch. Operates left-to-right; the first sample anchors the initial phase."""
    n = labels.shape[0]
    if n == 0:
        return labels.copy()
    out = labels.copy()
    current = labels[0]
    candidate = current
    run = 0
    for i in range(n):
        lbl = labels[i]
        if lbl == current:
            candidate = current
            run = 0
        else:
            if lbl == candidate:
                run += 1
            else:
                candidate = lbl
                run = 1
            if run >= min_run:
                current = candidate
                run = 0
        out[i] = current
    return out


def _find_runs(labels: np.ndarray) -> list[Segment]:
    if labels.shape[0] == 0:
        return []
    segs: list[Segment] = []
    start = 0
    for i in range(1, labels.shape[0]):
        if labels[i] != labels[start]:
            segs.append(Segment(start_idx=start, end_idx=i, phase=str(labels[start])))
            start = i
    segs.append(Segment(start_idx=start, end_idx=labels.shape[0], phase=str(labels[start])))
    return segs


def segment_phases(
    I: np.ndarray,
    eps: float = 0.05,
    min_run: int = 10,
    keep_rest: bool = False,
) -> list[Segment]:
    """Return contiguous same-phase segments. Rest segments are dropped by default.

    Args:
        I: current signal, shape (T,). Sign convention: I > 0 charge, I < 0 discharge.
        eps: |I| ≤ eps is considered rest.
        min_run: minimum consecutive samples of a new label before switching phases.
        keep_rest: if True, rest segments are returned as well.
    """
    assert I.ndim == 1, f"I must be 1-D, got shape {I.shape}"
    raw = _raw_labels(I, eps)
    smoothed = hysteresis_filter(raw, min_run=min_run)
    segments = _find_runs(smoothed)
    if not keep_rest:
        segments = [s for s in segments if s.phase != PHASE_REST]
    return segments
