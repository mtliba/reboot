"""Extended evaluation: error distribution, R^2, and breakdowns by phase,
operating temperature (from the cell name suffix), SOC decile and cell.
All inputs are numpy arrays over the test windows."""
from __future__ import annotations

import numpy as np

from reboot.eval.metrics import compute_metrics


def _basic(y: np.ndarray, p: np.ndarray) -> dict:
    e = p - y
    ae = np.abs(e)
    ss_res = float((e ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n": int(y.size),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "max_error": float(ae.max()) if ae.size else float("nan"),
        "p50": float(np.percentile(ae, 50)), "p95": float(np.percentile(ae, 95)), "p99": float(np.percentile(ae, 99)),
        "bias": float(e.mean()),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "frac_within_1pct": float((ae < 0.01).mean()),
        "frac_within_2pct": float((ae < 0.02).mean()),
    }


def compute_extended_metrics(
    y: np.ndarray, p: np.ndarray, phase: np.ndarray, battery_idx: np.ndarray, battery_ids: list[str],
) -> dict:
    y = np.asarray(y).reshape(-1); p = np.asarray(p).reshape(-1)
    phase = np.asarray(phase).reshape(-1); battery_idx = np.asarray(battery_idx).reshape(-1)
    out = compute_metrics(y, p)
    out.update({k: v for k, v in _basic(y, p).items() if k not in out})
    out["by_phase"] = {name: _basic(y[phase == k], p[phase == k]) for k, name in ((0, "charge"), (1, "discharge")) if (phase == k).any()}
    temps = np.array([b.rsplit("-", 1)[1] if "-" in b else "?" for b in battery_ids])
    wt = temps[battery_idx]
    out["by_temperature"] = {t: _basic(y[wt == t], p[wt == t]) for t in sorted(set(wt)) if (wt == t).any()}
    out["by_cell"] = {battery_ids[b]: _basic(y[battery_idx == b], p[battery_idx == b]) for b in np.unique(battery_idx)}
    edges = np.linspace(0, 1, 11)
    dec = np.clip(np.digitize(y, edges[1:-1]), 0, 9)
    out["by_decile"] = {f"{int(edges[k]*100)}-{int(edges[k+1]*100)}": _basic(y[dec == k], p[dec == k]) for k in range(10) if (dec == k).any()}
    return out
