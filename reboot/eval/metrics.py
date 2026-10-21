"""Evaluation metrics — see CLAUDE.md §10."""
from __future__ import annotations

import numpy as np


def error_per_soc_bin(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    """Mean |error| per equal-width SOC bin in [0, 1]."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: dict[str, float] = {}
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        if k == n_bins - 1:
            sel = (y_true >= lo) & (y_true <= hi)
        else:
            sel = (y_true >= lo) & (y_true < hi)
        if not sel.any():
            out[f"mae_bin_{k}"] = float("nan")
        else:
            out[f"mae_bin_{k}"] = float(np.mean(np.abs(y_pred[sel] - y_true[sel])))
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    assert y_true.shape == y_pred.shape, (
        f"shape mismatch: {y_true.shape} vs {y_pred.shape}"
    )
    abs_err = np.abs(y_pred - y_true)
    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "max_error": float(np.max(abs_err)) if abs_err.size else float("nan"),
        "error_by_bin": error_per_soc_bin(y_true, y_pred, n_bins=10),
    }
