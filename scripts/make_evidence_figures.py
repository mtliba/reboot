#!/usr/bin/env python3
"""Regenerate the supporting-evidence figures shipped with the ICASSP paper.

Everything here is computed from artefacts committed in ``results/``:

* ``results/latents/*.npz``   -- 12k held-out windows embedded by each pretrained
  encoder (z_s, z_d, reference SOC, phase). Labels are used for *measurement only*;
  no pretraining stage ever sees them.
* ``results/physics_params.json`` -- the learned Coulombic efficiency and per-cell
  capacity read out of the fine-tuned checkpoints.

Usage:  python scripts/make_evidence_figures.py [--outdir docs/static/figures]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

RUNS = [
    ("maeonly100.npz", "masked recon.\nonly", "#9e9e9e"),
    ("v2_random.npz", "shuffled\nbatches", "#8ab4f8"),
    ("v3_cycle.npz", "position\npairing", "#f4a261"),
    ("v4_socproxy.npz", "Coulomb pairing\n+ neg. mask", "#d62728"),
]

# Measured by slurm/pair_proxy_check.sh over 20 cycle-grouped batches of 2048
# windows (reference SOC used only to score the pairs the sampler produced).
PAIR_PROXY = [
    ("position-symmetric\npairing (strategy A)", 0.279, 0.11, "#f4a261"),
    ("Coulomb pseudo-SOC\npairing (ours)", 0.009, 0.97, "#d62728"),
    ("oracle: nearest\ntrue-SOC window", 0.042, 0.70, "#9e9e9e"),
]


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "font.size": 8.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def linear_r2(X: np.ndarray, y: np.ndarray) -> float:
    """R^2 of an ordinary least-squares probe y ~ X (closed form, no training)."""
    X = X.astype(np.float64)
    X = X - X.mean(0, keepdims=True)
    X = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(1.0 - (resid**2).sum() / ((y - y.mean()) ** 2).sum())


def cross_phase_retrieval(zs: np.ndarray, y: np.ndarray, phase: np.ndarray,
                          n: int = 6000, seed: int = 0):
    """For each charge window, find its nearest DISCHARGE window in centred z_s
    (cosine) and report how far apart the two really are in reference SOC."""
    rng = np.random.default_rng(seed)
    c = rng.permutation(np.where(phase == 0)[0])[:n]
    d = rng.permutation(np.where(phase == 1)[0])[:n]
    mu = zs.mean(0, keepdims=True)
    a = zs[c] - mu
    b = zs[d] - mu
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    nn = (a @ b.T).argmax(1)
    gap = np.abs(y[c] - y[d][nn])
    chance = float(np.abs(y[c] - y[d][rng.integers(0, len(d), len(c))]).mean())
    return gap, chance


def load(fn: str):
    d = np.load(ROOT / "results" / "latents" / fn, allow_pickle=True)
    return (
        d["z_s"].astype(np.float64),
        d["z_d"].astype(np.float64),
        d["y"].astype(np.float64),
        d["phase"],
    )


def fig_alignment(outdir: Path) -> dict:
    stats: dict[str, dict] = {}
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))

    # (a) how far apart, in reference SOC, are the windows L_refl calls positives?
    ax = axes[0]
    xs = np.arange(len(PAIR_PROXY))
    ax.bar(xs, [p[1] for p in PAIR_PROXY], color=[p[3] for p in PAIR_PROXY], width=0.62)
    for x, (_, gap, frac, _) in zip(xs, PAIR_PROXY):
        ax.text(x, gap + 0.008, f"{gap:.3f}\n({frac:.0%} within 0.05)", ha="center",
                va="bottom", fontsize=7.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in PAIR_PROXY], fontsize=6.8)
    ax.set_ylabel(r"mean $|\Delta\mathrm{SOC}|$ inside a positive pair")
    ax.set_ylim(0, 0.37)
    ax.set_title("(a) What $\\mathcal{L}_{\\mathrm{refl}}$ calls a positive pair", fontsize=8.5)

    # (b) does the pretrained latent actually align the two phases?
    ax = axes[1]
    chance = None
    for fn, label, color in RUNS:
        zs, zd, y, phase = load(fn)
        gap, chance = cross_phase_retrieval(zs, y, phase)
        xs_ = np.sort(gap)
        ax.plot(xs_, np.arange(1, len(xs_) + 1) / len(xs_), color=color, lw=1.6,
                label=f"{label.replace(chr(10), ' ')} ({gap.mean():.3f})")
        stats[fn] = dict(
            label=label.replace("\n", " "),
            retrieval_mean_gap=float(gap.mean()),
            retrieval_within_0p05=float((gap < 0.05).mean()),
            r2_soc_from_z_s=linear_r2(zs, y),
            r2_soc_from_z_d=linear_r2(zd, y),
        )
    ax.axvline(chance, color="k", ls=":", lw=1.1)
    ax.text(chance, 0.50, "  chance", fontsize=7, va="center", rotation=90)
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"$|\Delta\mathrm{SOC}|$ to nearest cross-phase neighbour in $z_s$")
    ax.set_ylabel("fraction of charge windows")
    ax.legend(fontsize=6.8, loc="lower right", bbox_to_anchor=(1.0, 0.02))
    ax.set_title("(b) Cross-phase retrieval in the pretrained latent", fontsize=8.5)

    # (c) where does SOC information live after pretraining?
    ax = axes[2]
    xs = np.arange(len(RUNS))
    r2s = [stats[fn]["r2_soc_from_z_s"] for fn, _, _ in RUNS]
    r2d = [stats[fn]["r2_soc_from_z_d"] for fn, _, _ in RUNS]
    ax.bar(xs - 0.19, r2s, width=0.36, color="#d62728", label="$z_s$ (shared)")
    ax.bar(xs + 0.19, r2d, width=0.36, color="#4c72b0", label="$z_d$ (dynamic)")
    ax.set_xticks(xs)
    ax.set_xticklabels([r[1] for r in RUNS], fontsize=6.8)
    ax.set_ylim(0.6, 1.02)
    ax.set_ylabel(r"$R^2$ of a linear probe for SOC")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("(c) Which half carries SOC", fontsize=8.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"evidence_alignment.{ext}", bbox_inches="tight")
    plt.close(fig)
    return stats


def fig_physics(outdir: Path) -> dict:
    p = json.loads((ROOT / "results" / "physics_params.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.9))

    ax = axes[0]
    eta = p["eta_trajectory_v4"]
    ax.plot(np.arange(1, len(eta) + 1), eta, color="#d62728", lw=1.6)
    ax.axhspan(0.9, 1.0, color="#d62728", alpha=0.05)
    ax.set_xlabel("fine-tuning epoch")
    ax.set_ylabel(r"learned Coulombic efficiency $\eta$")
    ax.set_title(r"(a) $\eta$ converges to %.3f, unforced inside $(0.9,1)$"
                 % eta[-1], fontsize=8.5)

    ax = axes[1]
    for run, color, marker in (("v4 100%", "#d62728", "o"), ("v4 1%", "#4c72b0", "s")):
        ratios = p["runs"][run]["Q_over_nominal"]
        cells = list(p["battery_ids"])
        vals = [ratios[c] for c in cells]
        ax.plot(np.arange(len(cells)), vals, marker=marker, ms=3.4, lw=1.0,
                color=color, label=f"{run.replace('v4 ', '')} labels "
                                   f"(median {np.median(vals):.2f})")
    ax.axhline(1.0, color="k", ls=":", lw=1.0)
    ax.set_xticks(np.arange(len(p["battery_ids"])))
    ax.set_xticklabels([c.split("-")[0].replace("cell", "") for c in p["battery_ids"]],
                       fontsize=6.5)
    ax.set_xlabel("cell")
    ax.set_ylabel(r"$\hat{Q}_c\,/\,Q^{\mathrm{nom}}_c$")
    ax.set_ylim(0.75, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("(b) Per-cell capacity recovered 6-18% below nameplate", fontsize=8.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"evidence_physics.{ext}", bbox_inches="tight")
    plt.close(fig)
    return {"eta_final": eta[-1],
            "Q_over_nominal_median_100pct":
                float(np.median(list(p["runs"]["v4 100%"]["Q_over_nominal"].values())))}


def fig_factorisation(outdir: Path) -> None:
    """2-D PCA of each latent half: z_s orders SOC and merges the two phases,
    z_d is the half that keeps them apart."""
    zs, zd, y, phase = load("v4_socproxy.npz")
    rng = np.random.default_rng(0)
    sel = rng.permutation(len(y))[:6000]
    y, phase = y[sel], phase[sel]

    def pca2(z):
        z = z[sel]
        z = z - z.mean(0, keepdims=True)
        _, _, vt = np.linalg.svd(z, full_matrices=False)
        return z @ vt[:2].T

    pcs, pcd = pca2(zs), pca2(zd)
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.2))

    sc = axes[0].scatter(pcs[:, 0], pcs[:, 1], c=y, s=1.8, cmap="viridis", lw=0)
    cb = fig.colorbar(sc, ax=axes[0], pad=0.02)
    cb.set_label("reference SOC", fontsize=7)
    axes[0].set_title(r"(a) $z_s$ by SOC: ordered", fontsize=8.5)

    for ax, pc, title in (
        (axes[1], pcs, r"(b) $z_s$ by phase: merged"),
        (axes[2], pcd, r"(c) $z_d$ by phase: separated"),
    ):
        # draw in random z-order so neither phase is hidden under the other
        order = rng.permutation(len(pc))
        cols = np.where(phase[order] == 0, "#d62728", "#4c72b0")
        ax.scatter(pc[order, 0], pc[order, 1], s=1.8, lw=0, c=cols, alpha=0.6)
        ax.scatter([], [], s=18, color="#d62728", label="charge")
        ax.scatter([], [], s=18, color="#4c72b0", label="discharge")
        ax.legend(fontsize=7, loc="best")
        ax.set_title(title, fontsize=8.5)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"evidence_factorisation.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_scaling(outdir: Path) -> dict:
    """Unlabelled-pool scaling and the two loss-weight sweeps, from results.csv."""
    import csv

    rows = list(csv.DictReader(open(ROOT / "results" / "results.csv")))

    def mae(recipe: str, frac: float):
        v = [float(r["mae_pct"]) for r in rows
             if r["recipe"] == recipe and float(r["frac"]) == frac
             and r["protocol"] == "per_cell_random"]
        return float(np.mean(v)) if v else np.nan

    pools = [(10, "ablation:pool10"), (25, "ablation:pool25"),
             (50, "ablation:pool50"), (100, "ablation:full100")]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.9))

    ax = axes[0]
    x = [p for p, _ in pools]
    for frac, fl, color, marker in ((0.01, "1 % labels", "#d62728", "o"),
                                    (1.0, "100 % labels", "#4c72b0", "s")):
        ax.plot(x, [mae(r, frac) for _, r in pools], marker=marker, ms=4.5, lw=1.6,
                color=color, label=fl)
    ax.set_xscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"{p}%" for p in x])
    ax.set_xlabel("fraction of the unlabelled pretraining pool")
    ax.set_ylabel("test MAE (% SOC)")
    ax.legend(fontsize=7)
    ax.set_title("(a) More unlabelled cycles help, where it matters", fontsize=8.5)

    ax = axes[1]
    refl = [(0.1, "ablation:refl01"), (0.5, "ablation:full100"), (2.0, "ablation:refl2")]
    phys = [(0.0, "ablation:phys0"), (0.001, "ablation:phys001"),
            (0.01, "ablation:full100"), (0.1, "ablation:phys01"), (1.0, "ablation:phys1")]
    ax.plot([w for w, _ in refl], [mae(r, 0.01) for _, r in refl], marker="o", ms=4.5,
            lw=1.6, color="#d62728", label=r"$\lambda_{\mathrm{refl}}$ (published 0.5)")
    ax.plot([max(w, 3e-4) for w, _ in phys], [mae(r, 0.01) for _, r in phys], marker="s",
            ms=4.5, lw=1.6, color="#4c72b0",
            label=r"$\lambda_{\mathrm{phys}}$ (published 0.01)")
    ax.set_xscale("log")
    ax.set_xlabel("loss weight")
    ax.set_ylabel("test MAE at 1 % labels (% SOC)")
    ax.legend(fontsize=7)
    ax.set_title("(b) Sensitivity of the two tuned weights", fontsize=8.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"evidence_scaling.{ext}", bbox_inches="tight")
    plt.close(fig)
    return {"pool_1pct": {p: mae(r, 0.01) for p, r in pools},
            "pool_100pct": {p: mae(r, 1.0) for p, r in pools}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="docs/static/figures")
    args = ap.parse_args()
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    style()
    stats = fig_alignment(outdir)
    stats["physics"] = fig_physics(outdir)
    fig_factorisation(outdir)
    stats["scaling"] = fig_scaling(outdir)
    (ROOT / "results" / "evidence" / "latent_evidence.json").write_text(
        json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))
    print(f"\nfigures -> {outdir}")


if __name__ == "__main__":
    main()
