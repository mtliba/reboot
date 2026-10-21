"""Per-channel scaling + optional derived channels — see CLAUDE.md §4.2 and §4.5.

Channel layout (raw): [I, V, T, dt]
Optional derived:     [dV/dt, dT/dt, Q_cum]  → final layout (4 + 3) = 7 channels.

Scalers are fit on training batteries only and reused everywhere.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Channel indices in the raw layout
IDX_I, IDX_V, IDX_T, IDX_DT = 0, 1, 2, 3

# Physical bounds for temperature min-max scaling
T_MIN_C = -10.0
T_MAX_C = 60.0


def derived_channels(x: np.ndarray) -> np.ndarray:
    """Compute (dV/dt, dT/dt, Q_cum) and append along last axis.

    Args:
        x: shape (T, 4) — [I, V, T, dt]
    Returns:
        shape (T, 7) — [I, V, T, dt, dV/dt, dT/dt, Q_cum]
    """
    assert x.ndim == 2 and x.shape[1] == 4, f"expected (T, 4), got {x.shape}"
    I = x[:, IDX_I]
    V = x[:, IDX_V]
    T = x[:, IDX_T]
    dt = x[:, IDX_DT]

    # Avoid div-by-zero on dt
    safe_dt = np.where(dt > 1e-9, dt, 1e-9)
    dV_dt = np.gradient(V) / safe_dt
    dT_dt = np.gradient(T) / safe_dt
    Q_cum = np.cumsum(I * dt)

    return np.concatenate(
        [x, dV_dt[:, None], dT_dt[:, None], Q_cum[:, None]], axis=1
    )


@dataclass
class Scaler:
    """Holds fitted per-channel statistics.

    - I, V: robust scaling using median / IQR.
    - T: min-max with physical bounds (-10..60 °C).
    - dt: log1p (no per-dataset stats needed).
    - Derived channels (if used): robust (median/IQR).
    """

    I_median: float
    I_iqr: float
    V_median: float
    V_iqr: float
    derived_medians: np.ndarray | None = None    # shape (3,) or None
    derived_iqrs: np.ndarray | None = None       # shape (3,) or None
    has_derived: bool = False

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply scaling. `x` must have already been augmented with derived channels
        if and only if `self.has_derived` is True."""
        expected_c = 7 if self.has_derived else 4
        assert x.ndim == 2 and x.shape[1] == expected_c, (
            f"expected (T, {expected_c}), got {x.shape}"
        )
        out = x.astype(np.float32, copy=True)
        out[:, IDX_I] = (out[:, IDX_I] - self.I_median) / max(self.I_iqr, 1e-6)
        out[:, IDX_V] = (out[:, IDX_V] - self.V_median) / max(self.V_iqr, 1e-6)
        out[:, IDX_T] = (out[:, IDX_T] - T_MIN_C) / (T_MAX_C - T_MIN_C)
        out[:, IDX_DT] = np.log1p(np.maximum(out[:, IDX_DT], 0.0))
        if self.has_derived:
            assert self.derived_medians is not None and self.derived_iqrs is not None
            for k in range(3):
                col = 4 + k
                out[:, col] = (out[:, col] - self.derived_medians[k]) / max(
                    float(self.derived_iqrs[k]), 1e-6
                )
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "Scaler":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, cls), f"expected Scaler, got {type(obj)}"
        return obj


def _iqr(a: np.ndarray) -> float:
    q1, q3 = np.percentile(a, [25.0, 75.0])
    return float(max(q3 - q1, 1e-6))


def fit_scaler(train_arrays: list[np.ndarray], has_derived: bool) -> Scaler:
    """Fit a Scaler from a list of arrays (one per training battery).

    Each array must already include derived channels if `has_derived=True`.
    """
    assert len(train_arrays) > 0, "need at least one training battery to fit scaler"
    stacked = np.concatenate(train_arrays, axis=0)
    expected_c = 7 if has_derived else 4
    assert stacked.shape[1] == expected_c, (
        f"expected last dim {expected_c}, got {stacked.shape[1]}"
    )

    I = stacked[:, IDX_I]
    V = stacked[:, IDX_V]

    derived_medians = None
    derived_iqrs = None
    if has_derived:
        derived = stacked[:, 4:7]
        derived_medians = np.median(derived, axis=0).astype(np.float64)
        derived_iqrs = np.array([_iqr(derived[:, k]) for k in range(3)], dtype=np.float64)

    return Scaler(
        I_median=float(np.median(I)),
        I_iqr=_iqr(I),
        V_median=float(np.median(V)),
        V_iqr=_iqr(V),
        derived_medians=derived_medians,
        derived_iqrs=derived_iqrs,
        has_derived=has_derived,
    )
