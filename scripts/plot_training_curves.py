"""Plot training curves from Lightning's CSVLogger output (no W&B required).

Lightning's `CSVLogger` writes:
    {save_dir}/lightning_logs/version_N/metrics.csv
with both `*_step` and `*_epoch` columns for each `self.log()` call.

This script produces, under {out_dir}/:
    losses_per_epoch.png     panel per loss + total
    losses_overlay.png       all losses overlaid, log y-axis
    lr_schedule.png          learning-rate vs epoch
    losses_per_step.png      smoothed per-step training curves
    summary.csv              minimum/final per-loss numbers

Usage
-----
    # latest run, default output dir
    python scripts/plot_training_curves.py

    # explicit metrics file
    python scripts/plot_training_curves.py --csv lightning_logs/version_15/metrics.csv

    # compare two runs in one figure
    python scripts/plot_training_curves.py \\
        --csv lightning_logs/version_14/metrics.csv lightning_logs/version_15/metrics.csv \\
        --label v14 v15 --out outputs/figures_compare

    # finetune logs (different metric prefix)
    python scripts/plot_training_curves.py --prefix finetune --csv .../metrics.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reasonable default loss names if --losses isn't given.
DEFAULT_LOSSES = ("mask", "ord", "refl", "smooth", "orth", "total")


def find_latest_metrics(root: Path) -> Path:
    """Return the metrics.csv in the highest version_N under {root}/lightning_logs."""
    logs = root / "lightning_logs"
    if not logs.exists():
        raise FileNotFoundError(f"No lightning_logs dir at {logs}")
    versions = []
    for d in logs.iterdir():
        m = re.match(r"version_(\d+)$", d.name)
        if m and d.is_dir():
            versions.append((int(m.group(1)), d))
    if not versions:
        raise FileNotFoundError(f"No version_N dirs under {logs}")
    versions.sort()
    latest = versions[-1][1]
    csv = latest / "metrics.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv} not found")
    return csv


def load_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Lightning sometimes leaves NaN-filled rows for step-only or epoch-only events;
    # we keep them and let callers select non-null subsets per column.
    return df


def _epoch_series(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return a (epoch, value) DataFrame with one row per epoch, deduplicated."""
    if col not in df.columns:
        return pd.DataFrame(columns=["epoch", col])
    s = df[["epoch", col]].dropna()
    if s.empty:
        return s
    # Last value per epoch (Lightning emits the epoch-level mean at epoch end).
    s = s.drop_duplicates(subset="epoch", keep="last").sort_values("epoch")
    return s


def _step_series(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=["step", col])
    return df[["step", col]].dropna().sort_values("step")


def plot_per_epoch_panels(
    runs: list[tuple[str, pd.DataFrame]],
    losses: tuple[str, ...],
    prefix: str,
    out_dir: Path,
) -> Path:
    n = len(losses)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for k, name in enumerate(losses):
        ax = axes[k // cols][k % cols]
        col = f"{prefix}/loss_{name}_epoch"
        any_data = False
        for label, df in runs:
            s = _epoch_series(df, col)
            if s.empty:
                continue
            any_data = True
            ax.plot(s["epoch"], s[col], label=label, alpha=0.85)
        ax.set_title(f"L_{name}")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
        if any_data and len(runs) > 1:
            ax.legend(fontsize=8)
    # Hide unused axes
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.tight_layout()
    out_path = out_dir / "losses_per_epoch.png"
    fig.savefig(out_path, dpi=120); plt.close(fig)
    return out_path


def plot_overlay_logscale(
    runs: list[tuple[str, pd.DataFrame]],
    losses: tuple[str, ...],
    prefix: str,
    out_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    # If single run: one line per loss. If multiple: line style per loss, color per run.
    color_for_run = plt.cm.tab10(np.linspace(0, 1, max(len(runs), 1)))
    for ri, (label, df) in enumerate(runs):
        for name in losses:
            col = f"{prefix}/loss_{name}_epoch"
            s = _epoch_series(df, col)
            if s.empty:
                continue
            line_label = f"{label}: {name}" if len(runs) > 1 else name
            ax.plot(
                s["epoch"], np.maximum(s[col], 1e-6),
                label=line_label,
                color=color_for_run[ri] if len(runs) > 1 else None,
                alpha=0.85,
            )
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (log scale)")
    ax.set_title("All losses, log-scale overlay")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out_path = out_dir / "losses_overlay.png"
    fig.savefig(out_path, dpi=120); plt.close(fig)
    return out_path


def plot_lr(runs: list[tuple[str, pd.DataFrame]], out_dir: Path) -> Path | None:
    fig, ax = plt.subplots(figsize=(8, 4))
    any_data = False
    for label, df in runs:
        col_candidates = [c for c in df.columns if c.startswith("lr-")]
        if not col_candidates:
            continue
        col = col_candidates[0]
        s = df[["epoch", col]].dropna().drop_duplicates("epoch").sort_values("epoch")
        if s.empty:
            continue
        any_data = True
        ax.plot(s["epoch"], s[col], label=f"{label} ({col})")
    if not any_data:
        plt.close(fig)
        return None
    ax.set_xlabel("epoch"); ax.set_ylabel("learning rate")
    ax.set_title("LR schedule")
    ax.grid(alpha=0.3)
    if len(runs) > 1:
        ax.legend()
    fig.tight_layout()
    out_path = out_dir / "lr_schedule.png"
    fig.savefig(out_path, dpi=120); plt.close(fig)
    return out_path


def plot_per_step(
    runs: list[tuple[str, pd.DataFrame]],
    losses: tuple[str, ...],
    prefix: str,
    out_dir: Path,
    window: int = 50,
) -> Path:
    fig, axes = plt.subplots(len(losses), 1, figsize=(9, 2 * len(losses)), squeeze=False)
    for k, name in enumerate(losses):
        ax = axes[k][0]
        col = f"{prefix}/loss_{name}_step"
        for label, df in runs:
            s = _step_series(df, col)
            if s.empty:
                continue
            smooth = s[col].rolling(window=window, min_periods=1).mean()
            ax.plot(s["step"], smooth, label=label, alpha=0.85)
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
        if k == 0 and len(runs) > 1:
            ax.legend(fontsize=8)
    axes[-1][0].set_xlabel("step")
    fig.suptitle(f"Per-step training loss (rolling mean, window={window})")
    fig.tight_layout()
    out_path = out_dir / "losses_per_step.png"
    fig.savefig(out_path, dpi=120); plt.close(fig)
    return out_path


def write_summary(
    runs: list[tuple[str, pd.DataFrame]],
    losses: tuple[str, ...],
    prefix: str,
    out_dir: Path,
) -> Path:
    rows = []
    for label, df in runs:
        for name in losses:
            col = f"{prefix}/loss_{name}_epoch"
            s = _epoch_series(df, col)
            if s.empty:
                rows.append({"run": label, "loss": name, "min": np.nan, "final": np.nan,
                             "min_epoch": np.nan, "n_epochs": 0})
                continue
            min_idx = int(s[col].idxmin())
            rows.append({
                "run": label, "loss": name,
                "min": float(s[col].min()),
                "min_epoch": int(s.loc[min_idx, "epoch"]),
                "final": float(s[col].iloc[-1]),
                "n_epochs": int(s["epoch"].max()) + 1,
            })
    out = pd.DataFrame(rows)
    out_path = out_dir / "summary.csv"
    out.to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", nargs="*", default=None,
                        help="path(s) to metrics.csv; if omitted, use the latest "
                             "lightning_logs/version_N/metrics.csv under --root")
    parser.add_argument("--label", nargs="*", default=None,
                        help="display label(s) matching --csv (defaults to "
                             "parent dir name)")
    parser.add_argument("--root", default="${REBOOT_ROOT:-$PWD}",
                        help="project root (used when --csv is omitted)")
    parser.add_argument("--prefix", default="pretrain",
                        help="metric prefix in CSV columns (pretrain | finetune | val)")
    parser.add_argument("--losses", nargs="*", default=list(DEFAULT_LOSSES),
                        help="loss suffixes to plot")
    parser.add_argument("--out", default=None,
                        help="output directory (default: <project>/outputs/figures/<run-label>/)")
    parser.add_argument("--smooth", type=int, default=50,
                        help="rolling-mean window for per-step plot")
    args = parser.parse_args()

    if args.csv:
        csv_paths = [Path(p) for p in args.csv]
    else:
        csv_paths = [find_latest_metrics(Path(args.root))]
    for p in csv_paths:
        if not p.exists():
            print(f"ERROR: {p} not found", file=sys.stderr)
            sys.exit(2)

    labels = args.label or [p.parent.name for p in csv_paths]
    if len(labels) != len(csv_paths):
        print("ERROR: --label count must match --csv count", file=sys.stderr)
        sys.exit(2)

    runs = [(lab, load_metrics(p)) for lab, p in zip(labels, csv_paths)]
    print(f"Loaded {len(runs)} run(s):")
    for lab, df in runs:
        print(f"  {lab}: {len(df):,} rows from {df.columns.size} columns")

    if args.out is None:
        if len(runs) == 1:
            args.out = str(Path(args.root) / "outputs" / "figures" / labels[0])
        else:
            args.out = str(Path(args.root) / "outputs" / "figures" / "compare")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    losses = tuple(args.losses)
    p1 = plot_per_epoch_panels(runs, losses, args.prefix, out_dir)
    p2 = plot_overlay_logscale(runs, losses, args.prefix, out_dir)
    p3 = plot_lr(runs, out_dir)
    p4 = plot_per_step(runs, losses, args.prefix, out_dir, window=args.smooth)
    p5 = write_summary(runs, losses, args.prefix, out_dir)

    print(f"\nWrote:")
    for p in (p1, p2, p3, p4, p5):
        if p is not None:
            print(f"  {p}")


if __name__ == "__main__":
    main()
