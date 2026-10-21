"""Generate diagnostic plots from a trained ReBoot checkpoint.

Plots written under {paths.output_dir}/diagnostics/:
    01_phi_charge.png      — phi(z_s) over local_t inside charge segments
    02_phi_discharge.png   — phi(z_s) over local_t inside discharge segments
    03_cross_phase.png     — cosine(z_s_charge, z_s_discharge) sorted by SOC
    04_orth_hist.png       — histogram of cosine(z_s, z_d)
    05_pred_vs_true.png    — y_pred vs y_true scatter (if SOC labels available)

Usage:
    python scripts/diagnose.py "+eval.ckpt='outputs/.../best.ckpt'" [+eval.max_windows=300000]
"""
from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from reboot.data.dataset import collate_finetune
from reboot.data.memmap_dataset import MemmapBatteryDataset
from reboot.eval.diagnostics import (
    cross_phase_alignment,
    latent_decorrelation,
    phi_over_time,
    soc_prediction_curve,
)
from reboot.models import ReBootModel
from reboot.utils.seed import set_global_seed


def _subsample_whole_cycles(ds, max_windows: int, seed: int) -> None:
    """Keep whole cycles (charge + discharge, every window) until ~max_windows.

    Embedding all 2.88 M windows is unnecessary for the plots and blew the
    open-file limit in job 16192 (DataLoader worker shared-memory handles).
    Whole cycles are kept so phi-over-time and cross-phase plots see complete
    segments."""
    if max_windows <= 0 or len(ds.indices) <= max_windows:
        return
    cyc = ds.cycle_id[ds.indices]
    uniq, inv, counts = np.unique(cyc, return_inverse=True, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(uniq))
    keep_n = np.searchsorted(np.cumsum(counts[order]), max_windows) + 1
    keep_cycles = np.zeros(len(uniq), dtype=bool)
    keep_cycles[order[:keep_n]] = True
    ds.indices = ds.indices[keep_cycles[inv]]


@torch.no_grad()
def _embed_dataset(model, ds, device: str, batch_size: int = 512, num_workers: int = 4) -> dict:
    """Inline replacement for embed_windows that uses the memmap dataset directly."""
    # Worker tensors are otherwise passed as file descriptors; thousands of
    # queued batches exhaust the per-process fd limit ("Too many open files").
    torch.multiprocessing.set_sharing_strategy("file_system")
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_finetune,
    )
    zs_chunks, zd_chunks = [], []
    meta_keys = ("phase", "segment_id", "local_t", "cycle_id",
                 "battery_idx", "segment_length", "y")
    meta_collect = {k: [] for k in meta_keys}
    model.eval().to(device)
    for batch in loader:
        x = batch["x"].to(device)
        z_s, z_d = model.encode(x)
        zs_chunks.append(z_s.cpu().numpy())
        zd_chunks.append(z_d.cpu().numpy())
        for k in meta_keys:
            if k in batch:
                meta_collect[k].append(batch[k].cpu().numpy())
    z_s = np.concatenate(zs_chunks, axis=0)
    z_d = (np.concatenate(zd_chunks, axis=0)
           if zd_chunks and zd_chunks[0].size else np.zeros((z_s.shape[0], 0)))
    meta = {k: np.concatenate(v) for k, v in meta_collect.items() if v}
    if "y" in meta:
        meta["soc"] = meta.pop("y")

    # If a trained OrdinalLoss.phi is on the model, use it; otherwise fall back to z_s[:,0].
    phi = getattr(model, "phi", None)
    if phi is not None:
        dev = next(phi.parameters()).device
        s_score = phi(torch.from_numpy(z_s).to(dev)).squeeze(-1).cpu().numpy()
        print("phi(z_s): using the trained OrdinalLoss.phi from the checkpoint")
    else:
        s_score = z_s[:, 0]
        print("WARNING: no ord_loss.phi in this checkpoint (finetuned models drop it); "
              "plots 01/02 show z_s[:, 0], NOT phi(z_s). Run diagnose on the PRETRAIN ckpt.")

    return {"z_s": z_s, "z_d": z_d, "s_score": s_score, **meta}


def _load_model(cfg: DictConfig, in_channels: int) -> ReBootModel:
    encoder_kwargs = {
        k: v for k, v in OmegaConf.to_container(cfg.model, resolve=True).items()
        if k not in {"encoder_type", "in_channels", "hidden", "d_s", "d_d", "single_latent"}
    }
    model = ReBootModel(
        in_channels=in_channels, W=cfg.data.window_size,
        encoder_type=cfg.model.encoder_type, hidden=cfg.model.hidden,
        d_s=cfg.model.d_s, d_d=cfg.model.d_d,
        single_latent=bool(cfg.model.single_latent),
        encoder_kwargs=encoder_kwargs,
    )
    ckpt = cfg.get("eval", {}).get("ckpt") if "eval" in cfg else None
    assert ckpt is not None, "Set eval.ckpt=<path-to-.ckpt>."
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)

    model_sd = {k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    print(f"Loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")

    # Attach trained phi from OrdinalLoss if present in the ckpt.
    phi_key = next((k for k in sd if k.endswith("ord_loss.phi.weight")), None)
    if phi_key is not None:
        in_dim = model.d_s if not model.single_latent else model.d_s + model.d_d
        phi = torch.nn.Linear(in_dim, 1)
        phi.weight.data = sd[phi_key]
        bias_key = phi_key.replace("weight", "bias")
        if bias_key in sd:
            phi.bias.data = sd[bias_key]
        model.phi = phi
    return model


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)

    ds = MemmapBatteryDataset(
        cfg.data.processed_dir, mode="eval", filter_split=None,
        clip_value=cfg.data.get("clip_value", 10.0), input_channels=cfg.data.get("input_channels", None), stride=cfg.data.stride,
    )
    eval_cfg = cfg.get("eval", None)
    max_windows = int(eval_cfg.get("max_windows", 300_000)) if eval_cfg is not None else 300_000
    n_all = len(ds)
    _subsample_whole_cycles(ds, max_windows, seed=int(cfg.seed))
    print(f"Diagnose: embedding {len(ds):,} of {n_all:,} windows "
          f"({np.unique(ds.cycle_id[ds.indices]).size:,} whole cycles)")

    in_channels = ds.in_channels
    model = _load_model(cfg, in_channels)

    out_dir = Path(cfg.paths.output_dir) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed = _embed_dataset(
        model, ds, device=device, batch_size=cfg.finetune.batch_size,
        num_workers=int(cfg.data.num_workers),
    )
    print(f"Embedded {embed['z_s'].shape[0]:,} windows")

    # 1. phi over time, charge segments (sample up to 200 segments for plot legibility)
    fig, ax = plt.subplots(figsize=(8, 5))
    pot = phi_over_time(embed, phase=0)
    for seg_id, s in list(pot.items())[:200]:
        ax.plot(np.arange(len(s)), s, alpha=0.2)
    ax.set_xlabel("Window position inside segment")
    ax.set_ylabel("phi(z_s)")
    ax.set_title("Charge: phi(z_s) over time")
    fig.tight_layout(); fig.savefig(out_dir / "01_phi_charge.png", dpi=120); plt.close(fig)

    # 2. phi over time, discharge segments
    fig, ax = plt.subplots(figsize=(8, 5))
    pot = phi_over_time(embed, phase=1)
    for seg_id, s in list(pot.items())[:200]:
        ax.plot(np.arange(len(s)), s, alpha=0.2)
    ax.set_xlabel("Window position inside segment")
    ax.set_ylabel("phi(z_s)")
    ax.set_title("Discharge: phi(z_s) over time")
    fig.tight_layout(); fig.savefig(out_dir / "02_phi_discharge.png", dpi=120); plt.close(fig)

    # 3. cross-phase alignment heatmap
    sim = cross_phase_alignment(embed, max_per_phase=200)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim, aspect="auto", cmap="viridis", vmin=-1, vmax=1, origin="lower")
    ax.set_xlabel("Discharge windows (sorted by SOC)")
    ax.set_ylabel("Charge windows (sorted by SOC)")
    ax.set_title("Cross-phase cosine(z_s_c, z_s_d)\n(diagonal expected after L_refl convergence)")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(out_dir / "03_cross_phase.png", dpi=120); plt.close(fig)

    # 3b. same map on mean-centred z_s: the raw cosine is dominated by the shared
    #     mean direction of z_s (all values ~1), which hides the SOC structure.
    embed_c = dict(embed); embed_c["z_s"] = embed["z_s"] - embed["z_s"].mean(axis=0, keepdims=True)
    sim_c = cross_phase_alignment(embed_c, max_per_phase=200)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim_c, aspect="auto", cmap="viridis", vmin=-1, vmax=1, origin="lower")
    ax.set_xlabel("Discharge windows (sorted by SOC)")
    ax.set_ylabel("Charge windows (sorted by SOC)")
    ax.set_title("Cross-phase cosine on mean-centred z_s\n(diagonal = SOC-matched alignment)")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(out_dir / "03b_cross_phase_centred.png", dpi=120); plt.close(fig)
    d = np.diag(sim_c); off = sim_c[~np.eye(sim_c.shape[0], dtype=bool)]
    print(f"centred cross-phase cosine: diagonal mean={d.mean():.3f}  off-diagonal mean={off.mean():.3f}")

    # 4. cosine(z_s, z_d) histogram
    cs = latent_decorrelation(embed)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cs, bins=80, color="steelblue", edgecolor="white")
    ax.set_xlabel("cosine(z_s, z_d)")
    ax.set_ylabel("count")
    ax.set_title(f"Latent decorrelation (mean={cs.mean():.3f}, std={cs.std():.3f})")
    fig.tight_layout(); fig.savefig(out_dir / "04_orth_hist.png", dpi=120); plt.close(fig)

    # 5. SOC prediction scatter (only if labels are present)
    if "soc" in embed and np.isfinite(embed["soc"]).any():
        y_true, y_pred = soc_prediction_curve(
            embed, model.predict_soc, device=device,
        )
        finite = np.isfinite(y_true)
        y_true, y_pred = y_true[finite], y_pred[finite]
        # Sub-sample for plot if huge
        if y_true.size > 50000:
            sel = np.random.choice(y_true.size, 50000, replace=False)
            y_true_s, y_pred_s = y_true[sel], y_pred[sel]
        else:
            y_true_s, y_pred_s = y_true, y_pred
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true_s, y_pred_s, s=2, alpha=0.2, c="darkorange")
        lims = [0.0, 1.0]
        ax.plot(lims, lims, "k--", lw=1, label="y=x")
        ax.set_xlabel("SOC true"); ax.set_ylabel("SOC predicted")
        mae = float(np.mean(np.abs(y_pred - y_true)))
        ax.set_title(f"SOC prediction (MAE={mae:.4f}, N={y_true.size:,})")
        ax.legend(); ax.set_xlim(lims); ax.set_ylim(lims)
        fig.tight_layout()
        fig.savefig(out_dir / "05_pred_vs_true.png", dpi=120); plt.close(fig)

    print(f"Diagnostics written to {out_dir}")


if __name__ == "__main__":
    main()
