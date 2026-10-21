"""Latent-quality diagnostics described in TESTING_PLAN.txt §L4.3.

Pure compute functions — they return numpy arrays / dicts. The actual plotting
lives in scripts/diagnose.py so this module has no matplotlib dependency.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from reboot.data.dataset import BatteryDataset, Window, collate_finetune


@torch.no_grad()
def embed_windows(model, windows: list[Window], device: str = "cpu", batch_size: int = 256
                  ) -> dict[str, np.ndarray]:
    """Run the encoder over a list of windows. Returns numpy arrays for downstream plotting.

    Returns dict with keys:
        z_s, z_d         : (N, d_s), (N, d_d)
        phase, segment_id, local_t, cycle_id, battery_idx, segment_length : (N,)
        soc              : (N,) (may be NaN if labels absent)
        s_score          : (N,) — phi(z_s) for the OrdinalLoss if model has one;
                                 if no ord head is present, falls back to z_s[:, 0].
    """
    # Pretend we have labels by giving every window a placeholder SOC. We never
    # train on it; this only avoids the BatteryDataset assertion.
    has_soc = all(w.soc is not None for w in windows)
    if not has_soc:
        for w in windows:
            w.soc = np.zeros(w.data.shape[0], dtype=np.float32)
    ds = BatteryDataset(windows, mode="eval")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_finetune, num_workers=0)
    model.eval().to(device)

    zs_chunks, zd_chunks = [], []
    # BatteryDataset stores the scalar mean SOC under key `y`. Map it to `soc` here
    # so downstream diagnostics can address it consistently.
    batch_keys = ("phase", "segment_id", "local_t", "cycle_id",
                  "battery_idx", "segment_length", "y")
    meta_collect = {k: [] for k in batch_keys}
    for batch in loader:
        x = batch["x"].to(device)
        z_s, z_d = model.encode(x)
        zs_chunks.append(z_s.cpu().numpy())
        zd_chunks.append(z_d.cpu().numpy())
        for k in batch_keys:
            if k in batch:
                meta_collect[k].append(batch[k].cpu().numpy())
    z_s = np.concatenate(zs_chunks, axis=0)
    z_d = np.concatenate(zd_chunks, axis=0) if zd_chunks and zd_chunks[0].size else np.zeros((z_s.shape[0], 0))
    meta = {k: np.concatenate(v, axis=0) for k, v in meta_collect.items() if v}
    if "y" in meta:
        meta["soc"] = meta.pop("y")

    # If a trained OrdinalLoss is attached (e.g., loaded via PretrainModule), use it.
    s_score = None
    try:
        # PretrainModule holds ord_loss.phi as a Linear(d_s -> 1).
        phi = getattr(model, "phi", None)
        if phi is not None:
            s_score = phi(torch.from_numpy(z_s)).squeeze(-1).cpu().numpy()
    except Exception:
        pass
    if s_score is None:
        s_score = z_s[:, 0]                                          # fallback: first latent dim

    return {"z_s": z_s, "z_d": z_d, "s_score": s_score, **meta}


def phi_over_time(embed: dict, phase: int = 0) -> dict[int, np.ndarray]:
    """Return {segment_id -> sorted_by_local_t array of s_score} for the requested phase.
    Use this to plot progression of the SOC score within each segment.
    """
    sel = embed["phase"] == phase
    seg_ids = embed["segment_id"][sel]
    local_t = embed["local_t"][sel]
    s = embed["s_score"][sel]
    out: dict[int, np.ndarray] = {}
    for seg in np.unique(seg_ids):
        m = seg_ids == seg
        order = np.argsort(local_t[m])
        out[int(seg)] = s[m][order]
    return out


def cross_phase_alignment(embed: dict, max_per_phase: int = 200) -> np.ndarray:
    """Cosine-similarity matrix between charge z_s (rows) and discharge z_s (cols).
    A trained model should show high similarity for windows at matching SOC.

    Returns shape (n_c, n_d). Rows are sorted by mean SOC ascending (if SOC is
    available), otherwise by local_t. Same for columns.
    """
    c_mask = embed["phase"] == 0
    d_mask = embed["phase"] == 1
    c_idx = np.where(c_mask)[0]
    d_idx = np.where(d_mask)[0]
    if c_idx.size == 0 or d_idx.size == 0:
        return np.zeros((0, 0))

    soc = embed.get("soc")
    sort_c = np.argsort(soc[c_idx]) if soc is not None and not np.all(soc == 0) else np.argsort(embed["local_t"][c_idx])
    sort_d = np.argsort(soc[d_idx]) if soc is not None and not np.all(soc == 0) else np.argsort(embed["local_t"][d_idx])
    c_idx = c_idx[sort_c][:max_per_phase]
    d_idx = d_idx[sort_d][:max_per_phase]

    zs_c = embed["z_s"][c_idx]
    zs_d = embed["z_s"][d_idx]
    zs_c = zs_c / (np.linalg.norm(zs_c, axis=1, keepdims=True) + 1e-9)
    zs_d = zs_d / (np.linalg.norm(zs_d, axis=1, keepdims=True) + 1e-9)
    return zs_c @ zs_d.T


def latent_decorrelation(embed: dict) -> np.ndarray:
    """Per-sample cosine(z_s, z_d). Returns shape (N,).

    A successful L_orth keeps the distribution sharply peaked near 0.
    """
    z_s = embed["z_s"]
    z_d = embed["z_d"]
    if z_d.size == 0 or z_s.shape[1] != z_d.shape[1]:
        # Pad to compare matching slices; if dims differ, project to min.
        d = min(z_s.shape[1], z_d.shape[1]) if z_d.size else 0
        if d == 0:
            return np.zeros(z_s.shape[0])
        z_s = z_s[:, :d]; z_d = z_d[:, :d]
    cs = F.cosine_similarity(torch.from_numpy(z_s), torch.from_numpy(z_d), dim=-1).numpy()
    return cs


def soc_prediction_curve(embed: dict, predict_fn, device: str | None = None
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Run the regression head on every window. Returns (y_true, y_pred)."""
    z_s_t = torch.from_numpy(embed["z_s"])
    z_d_t = torch.from_numpy(embed["z_d"]) if embed["z_d"].size else torch.zeros(z_s_t.size(0), 0)
    if device is not None:
        z_s_t = z_s_t.to(device)
        z_d_t = z_d_t.to(device)
    with torch.no_grad():
        y_pred = predict_fn(z_s_t, z_d_t).cpu().numpy()
    y_true = embed.get("soc")
    if y_true is None:
        y_true = np.full_like(y_pred, np.nan)
    return y_true, y_pred
