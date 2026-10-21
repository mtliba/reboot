"""Build the supplementary material: every result table + qualitative figures.

    python scripts/collect_results.py && python scripts/make_supplement.py
    cd paper && latexmk -pdf -interaction=nonstopmode supplement.tex

Inputs (all produced by other scripts / jobs, synced under outputs/):
  outputs/**/predictions*.npz          per-checkpoint test predictions (scripts/reevaluate.py)
  outputs/**/eval_extended*.json       extended metrics
  outputs/export/embeddings/*.npz      latents of 20k test windows + t-SNE (slurm/export_qualitative.sh)
  outputs/export/logs/pretrain_*.csv   pretrain loss curves
  outputs/export/trace_cycles.json     one mid-life test cycle per temperature
  outputs/results.csv                  all headline results (collect_results.py)
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "paper" / "figures" / "supp"
FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "paper" / "tables"

PRED = {  # label -> predictions file (per-cell random test order)
    "Supervised Hybrid, 100%": "outputs/checkpoints/baseline_hybrid_frac1.0_16208/baseline_hybrid/predictions.npz",
    "Supervised TCN, 100%": "outputs/checkpoints/baseline_tcn_frac1.0_16196/baseline_tcn/predictions.npz",
    "ReBoot v3, 100%": "outputs/finetune_16216/checkpoints/finetune/predictions.npz",
    "ReBoot v4, 100%": "outputs/finetune_16260/checkpoints/finetune/predictions.npz",
    "Supervised Hybrid, 1%": "outputs/base_lowlabel_16222/hybrid_frac_0.01/baseline_hybrid/predictions.npz",
    "ReBoot v4, 1%": "outputs/lowlabel_16263/frac_0.01/checkpoints/finetune/predictions.npz",
}
PRETRAIN_LOGS = {"v1 (bf16, diverged)": "pretrain_version_16.csv", "v2 random batches": "pretrain_version_30.csv",
                 "v3 cycle batches": "pretrain_version_35.csv", "v4 pseudo-SOC + mask": "pretrain_version_59.csv"}
EMB = {"v2 random": "v2_random", "v3 cycle": "v3_cycle", "v4 pseudo-SOC+mask": "v4_socproxy", "masked-only": "maeonly100", "v4 finetuned": "v4_finetuned"}
BATTERY_IDS = None


def load_pred(label):
    f = ROOT / PRED[label]
    if not f.exists():
        return None
    d = np.load(f)
    return {k: d[k] for k in d.files}


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf"); fig.savefig(FIG / f"{name}.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- predictions
def fig_pred_vs_true():
    labels = ["Supervised Hybrid, 100%", "ReBoot v4, 100%", "Supervised Hybrid, 1%", "ReBoot v4, 1%"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
    for ax, lab in zip(axes, labels):
        d = load_pred(lab)
        if d is None:
            ax.set_visible(False); continue
        hb = ax.hexbin(100 * d["y"], 100 * d["p"], gridsize=60, bins="log", cmap="viridis", mincnt=1)
        ax.plot([0, 100], [0, 100], "k--", lw=0.8)
        mae = 100 * np.abs(d["p"] - d["y"]).mean()
        ax.set_title(f"{lab}\nMAE {mae:.3f}%", fontsize=8); ax.set_xlabel("True SOC (%)"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    axes[0].set_ylabel("Predicted SOC (%)")
    savefig(fig, "pred_vs_true")


def fig_residuals():
    labels = ["Supervised Hybrid, 100%", "Supervised TCN, 100%", "ReBoot v3, 100%", "ReBoot v4, 100%", "Supervised Hybrid, 1%", "ReBoot v4, 1%"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
    for lab in labels:
        d = load_pred(lab)
        if d is None:
            continue
        e = 100 * (d["p"] - d["y"]); ae = np.abs(e)
        axes[0].hist(e, bins=np.linspace(-5, 5, 201), histtype="step", lw=1.0, label=lab, density=True)
        xs = np.sort(ae); axes[1].plot(xs, np.linspace(0, 1, len(xs)), lw=1.0, label=lab)
        bins = np.linspace(0, 1, 21); c = 0.5 * (bins[1:] + bins[:-1]); idx = np.clip(np.digitize(d["y"], bins[1:-1]), 0, 19)
        m = np.array([e[idx == k].mean() if (idx == k).any() else np.nan for k in range(20)])
        axes[2].plot(100 * c, m, marker="o", ms=2.5, lw=1.0, label=lab)
    axes[0].set_yscale("log"); axes[0].set_xlabel("Signed error (% SOC)"); axes[0].set_ylabel("Density"); axes[0].set_title("Residual distribution", fontsize=9)
    axes[1].set_xscale("log"); axes[1].set_xlim(1e-2, 60); axes[1].set_xlabel("|error| (% SOC)"); axes[1].set_ylabel("CDF"); axes[1].set_title("Error CDF", fontsize=9); axes[1].grid(alpha=0.3, which="both")
    axes[2].axhline(0, color="k", lw=0.6); axes[2].set_xlabel("True SOC (%)"); axes[2].set_ylabel("Mean signed error (% SOC)"); axes[2].set_title("Bias vs. SOC", fontsize=9); axes[2].grid(alpha=0.3)
    axes[1].legend(fontsize=6, frameon=False, loc="lower right")
    savefig(fig, "residuals")


def fig_error_vs_position():
    """Error as a function of window position inside the segment (start of segment = fast SOC change)."""
    tc_meta = None
    labels = ["Supervised Hybrid, 100%", "ReBoot v4, 100%", "Supervised Hybrid, 1%", "ReBoot v4, 1%"]
    # need local_t for test windows: read from memmap meta via the embeddings export? use export meta with 20k subset instead
    emb = np.load(ROOT / "outputs/export/embeddings/v4_socproxy.npz", allow_pickle=True)
    # embeddings subset positions are global indices -> can't map to test-order predictions without ds.indices; skip if unavailable
    return


def fig_per_cell_heatmap(dx: pd.DataFrame):
    d = dx[(dx.protocol == "per_cell_random") & (dx.seed == 0) & (dx.epochs == 50) & ~dx.recipe.str.startswith(("ablation", "linprobe", "v1"))]
    d = d[~d.model.isin(["supervised_chemali_dnn", "supervised_chemali_lstm", "supervised_yang_gru", "supervised_hannan_transformer", "supervised_li_cnnbilstm"])]
    rows = []
    for _, r in d.sort_values(["frac", "model", "recipe"], ascending=[False, True, True]).iterrows():
        name = (f"ReBoot {r.recipe.split()[0]}" if r.model == "reboot" else r.model.replace("supervised_", "Sup. ")) + f" @{r.frac*100:g}%"
        rows.append((name, {c: 100 * v["mae"] for c, v in r.by_cell.items()}))
    if not rows:
        return
    cells = list(rows[0][1].keys())
    M = np.array([[rr[1].get(c, np.nan) for c in cells] for rr in rows])
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(rows) + 1.2))
    im = ax.imshow(M, aspect="auto", cmap="magma_r", vmin=0, vmax=np.nanpercentile(M, 97))
    ax.set_xticks(range(len(cells))); ax.set_xticklabels(cells, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([rr[0] for rr in rows], fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=5, color="white" if M[i, j] > np.nanpercentile(M, 60) else "black")
    fig.colorbar(im, ax=ax, label="MAE (% SOC)")
    savefig(fig, "per_cell_heatmap")


def fig_group_bars(dx: pd.DataFrame):
    d = dx[(dx.protocol == "per_cell_random") & (dx.seed == 0) & (dx.frac == 1.0) & (dx.epochs == 50) & ~dx.recipe.str.startswith(("ablation", "linprobe", "v1", "v2"))]
    d = d[d.model.isin(["supervised_hybrid", "supervised_tcn", "reboot"])]
    names = [(f"ReBoot {r.recipe.split()[0]}" if r.model == "reboot" else r.model.replace("supervised_", "Sup. ")) for _, r in d.iterrows()]
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    for ax, key, groups, title in ((axes[0], "by_temperature", ["25", "35", "45"], "By cell temperature (°C)"), (axes[1], "by_phase", ["charge", "discharge"], "By phase")):
        w = 0.8 / max(len(d), 1)
        for i, (_, r) in enumerate(d.iterrows()):
            ax.bar(np.arange(len(groups)) + i * w, [100 * r[key][g]["mae"] for g in groups], width=w, label=names[i])
        ax.set_xticks(np.arange(len(groups)) + w * (len(d) - 1) / 2); ax.set_xticklabels(groups); ax.set_ylabel("MAE (% SOC)"); ax.set_title(title, fontsize=9); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=6, frameon=False)
    savefig(fig, "group_bars")


def fig_traces():
    tc = json.load(open(OUT / "export" / "trace_cycles.json"))
    labels = ["Supervised Hybrid, 100%", "ReBoot v4, 100%", "Supervised Hybrid, 1%", "ReBoot v4, 1%"]
    preds = {lab: load_pred(lab) for lab in labels}
    fig, axes = plt.subplots(len(tc), 2, figsize=(11, 2.6 * len(tc)), sharex="row")
    for row, (cell, info) in enumerate(tc.items()):
        pos = np.array(info["test_positions"]); y = np.array(info["y"]); ph = np.array(info["phase"]); seg = np.array(info["segment_id"])
        x = np.arange(len(pos))
        ax = axes[row, 0]; ax.plot(x, 100 * y, "k-", lw=1.5, label="true SOC")
        for lab, ls in zip(labels, ["-", "-", "--", "--"]):
            if preds[lab] is None: continue
            ax.plot(x, 100 * preds[lab]["p"][pos], ls, lw=1.0, label=lab)
        for k in np.flatnonzero(np.diff(seg)) + 0.5:
            ax.axvline(k, color="gray", lw=0.5, ls=":")
        ax.set_ylabel("SOC (%)"); ax.set_title(f"{cell}, test cycle {info['cycle_id'] % 1_000_000}: charge windows then discharge windows", fontsize=8)
        if row == 0: ax.legend(fontsize=6, frameon=False, ncol=3)
        ax = axes[row, 1]
        for lab, ls in zip(labels, ["-", "-", "--", "--"]):
            if preds[lab] is None: continue
            ax.plot(x, 100 * (preds[lab]["p"][pos] - y), ls, lw=1.0, label=lab)
        ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("Error (% SOC)"); ax.set_title("Signed error along the cycle", fontsize=8); ax.grid(alpha=0.3)
        for k in np.flatnonzero(np.diff(seg)) + 0.5:
            ax.axvline(k, color="gray", lw=0.5, ls=":")
    for ax in axes[-1]:
        ax.set_xlabel("Window index within cycle")
    savefig(fig, "traces")


# ----------------------------------------------------------------------------- latents
def fig_tsne():
    keys = ["v2 random", "v3 cycle", "v4 pseudo-SOC+mask", "masked-only"]
    fig, axes = plt.subplots(3, len(keys), figsize=(11, 8.2))
    for j, k in enumerate(keys):
        f = OUT / "export" / "embeddings" / f"{EMB[k]}.npz"
        if not f.exists():
            for i in range(3): axes[i, j].set_visible(False)
            continue
        d = np.load(f, allow_pickle=True); idx = d["tsne_idx"]; y = d["y"][idx]; ph = d["phase"][idx]
        ts, td = d["tsne_s"], d["tsne_d"]
        sc = axes[0, j].scatter(ts[:, 0], ts[:, 1], c=100 * y, s=2, cmap="viridis", rasterized=True); axes[0, j].set_title(f"{k}\n$z_s$ coloured by true SOC", fontsize=8)
        for p_, c, n in ((0, "tab:orange", "charge"), (1, "tab:blue", "discharge")):
            axes[1, j].scatter(ts[ph == p_, 0], ts[ph == p_, 1], c=c, s=2, label=n, rasterized=True)
        axes[1, j].set_title("$z_s$ coloured by phase", fontsize=8)
        axes[2, j].scatter(td[:, 0], td[:, 1], c=100 * y, s=2, cmap="viridis", rasterized=True); axes[2, j].set_title("$z_d$ coloured by true SOC", fontsize=8)
        for i in range(3): axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
    axes[1, 0].legend(fontsize=7, frameon=False, markerscale=4)
    fig.colorbar(sc, ax=axes[0, :].tolist(), fraction=0.02, pad=0.01, label="SOC (%)")
    savefig(fig, "tsne")


def fig_phi_vs_soc():
    keys = ["v2 random", "v3 cycle", "v4 pseudo-SOC+mask"]
    fig, axes = plt.subplots(1, len(keys), figsize=(11, 3.0), sharey=False)
    for ax, k in zip(axes, keys):
        f = OUT / "export" / "embeddings" / f"{EMB[k]}.npz"
        if not f.exists(): ax.set_visible(False); continue
        d = np.load(f, allow_pickle=True); y = d["y"]; ph = d["phase"]; s = d["phi"]
        if np.isnan(s).all(): ax.set_visible(False); continue
        rng = np.random.default_rng(0); sub = rng.choice(len(y), 6000, replace=False)
        for p_, c, n in ((0, "tab:orange", "charge"), (1, "tab:blue", "discharge")):
            m = sub[ph[sub] == p_]; ax.scatter(100 * y[m], s[m], s=2, c=c, alpha=0.4, label=n, rasterized=True)
        r = np.corrcoef(y, s)[0, 1]
        ax.set_title(f"{k}\n$\\phi(z_s)$ vs. true SOC, $\\rho$={r:.3f}", fontsize=8); ax.set_xlabel("True SOC (%)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("$\\phi(z_s)$"); axes[0].legend(fontsize=7, frameon=False, markerscale=4)
    savefig(fig, "phi_vs_soc")


def fig_retrieval():
    keys = ["v2 random", "v3 cycle", "v4 pseudo-SOC+mask", "masked-only"]
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    for k in keys:
        f = OUT / "export" / "embeddings" / f"{EMB[k]}.npz"
        if not f.exists(): continue
        d = np.load(f, allow_pickle=True); zs = d["z_s"]; y = d["y"]; ph = d["phase"]
        zc = zs - zs.mean(0); zn = zc / (np.linalg.norm(zc, axis=1, keepdims=True) + 1e-8)
        c = np.where(ph == 0)[0][:5000]; dd = np.where(ph == 1)[0]
        nn_ = dd[(zn[c] @ zn[dd].T).argmax(1)]
        gap = np.abs(y[c] - y[nn_]); xs = np.sort(gap)
        ax.plot(100 * xs, np.linspace(0, 1, len(xs)), lw=1.3, label=f"{k}: median {100*np.median(gap):.1f}%")
    rng = np.random.default_rng(0); d = np.load(OUT / "export" / "embeddings" / "v4_socproxy.npz", allow_pickle=True)
    y = d["y"]; ph = d["phase"]; c = np.where(ph == 0)[0][:5000]; dd = np.where(ph == 1)[0]
    gap = np.abs(y[c] - y[rng.choice(dd, len(c))]); xs = np.sort(gap)
    ax.plot(100 * xs, np.linspace(0, 1, len(xs)), "k:", lw=1.0, label=f"chance: median {100*np.median(gap):.1f}%")
    ax.set_xscale("log"); ax.set_xlim(0.1, 100); ax.set_xlabel("True-SOC gap to nearest discharge window in $z_s$ (%)"); ax.set_ylabel("CDF over charge windows"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=6.5, frameon=False)
    savefig(fig, "retrieval_cdf")


# ----------------------------------------------------------------------------- training curves
def fig_pretrain_curves():
    comps = ["total", "mask", "ord", "refl", "smooth", "orth"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2))
    for lab, fn in PRETRAIN_LOGS.items():
        f = OUT / "export" / "logs" / fn
        if not f.exists(): continue
        df = pd.read_csv(f); e = df[df["pretrain/loss_total_epoch"].notna()]
        for ax, c in zip(axes.flat, comps):
            ax.plot(e["epoch"], e[f"pretrain/loss_{c}_epoch"], lw=1.1, label=lab)
    for ax, c in zip(axes.flat, comps):
        ax.set_title(f"$L_{{{c}}}$" if c != "total" else "weighted total", fontsize=9); ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
        if c in ("mask", "smooth", "orth", "total"): ax.set_yscale("log")
    axes[0, 0].legend(fontsize=7, frameon=False)
    savefig(fig, "pretrain_curves")


def fig_grad_norm():
    f = OUT / "export" / "logs" / "pretrain_version_59.csv"
    if not f.exists(): return
    df = pd.read_csv(f)
    if "pretrain/grad_norm" not in df.columns: return
    g = df[df["pretrain/grad_norm"].notna()]
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    ax.plot(g["step"], g["pretrain/grad_norm"], lw=0.6, color="tab:red")
    ax.set_yscale("log"); ax.set_xlabel("optimiser step"); ax.set_ylabel("global grad norm (pre-clip)"); ax.set_title("v4 pretrain: gradient norm; 0 steps skipped by the guard", fontsize=8); ax.grid(alpha=0.3, which="both")
    savefig(fig, "grad_norm")


# ----------------------------------------------------------------------------- tables
def table_all_runs(df_all: pd.DataFrame) -> str:
    d = df_all.sort_values(["protocol", "model", "recipe", "frac", "seed", "epochs"])
    lines = [r"\begin{longtable}{llllrrrrr}", r"\caption{Every finetune/baseline run in this study (test set of the stated protocol). MAE/RMSE/max in \% SOC.}\label{tab:allruns}\\", r"\toprule",
             r"Protocol & Model & Recipe / variant & Labels & Seed & Ep. & MAE & RMSE & Max \\", r"\midrule", r"\endfirsthead", r"\toprule",
             r"Protocol & Model & Recipe / variant & Labels & Seed & Ep. & MAE & RMSE & Max \\", r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    for _, r in d.iterrows():
        rec = (r.recipe or "").replace("_", r"\_").replace("%", r"\%")
        lines.append(f"{r.protocol.replace('_', ' ')} & {r.model.replace('_', ' ')} & {rec} & {r.frac*100:g}\\% & {int(r.seed)} & {int(r.epochs)} & {100*r.mae:.3f} & {100*r.rmse:.3f} & {100*r.max_error:.1f} \\\\")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def table_per_cell(dx: pd.DataFrame) -> str:
    d = dx[(dx.protocol == "per_cell_random") & (dx.seed == 0) & (dx.epochs == 50) & ~dx.recipe.str.startswith(("ablation", "linprobe", "v1", "v2"))]
    d = d[d.model.isin(["supervised_hybrid", "supervised_tcn", "reboot"])]
    cells = list(d.iloc[0].by_cell.keys())
    cols = [((f"ReBoot {r.recipe.split()[0]}" if r.model == "reboot" else r.model.replace("supervised_", "Sup. ")) + f" {r.frac*100:g}\\%", r.by_cell) for _, r in d.sort_values(["frac", "model", "recipe"], ascending=[False, True, True]).iterrows()]
    lines = [r"\begin{table*}[t]", r"\centering", r"\caption{Test MAE (\% SOC) per cell, per-cell random split. The cell name suffix is the cycling temperature in $^\circ$C.}", r"\label{tab:percell}", r"\small",
             r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule", "Cell & " + " & ".join(c[0] for c in cols) + r" \\", r"\midrule"]
    for c in cells:
        lines.append(f"{c} & " + " & ".join(f"{100*bc[c]['mae']:.3f}" for _, bc in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def fig_qualitative_main():
    """Compact main-text figure: v3 vs v4, z_s t-SNE by phase and phi vs SOC."""
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1))
    for j, (k, tag) in enumerate((("v3 cycle", "v3"), ("v4 pseudo-SOC+mask", "v4"))):
        d = np.load(OUT / "export" / "embeddings" / f"{EMB[k]}.npz", allow_pickle=True)
        idx = d["tsne_idx"]; ph = d["phase"][idx]; ts = d["tsne_s"]
        ax = axes[j]
        for p_, c, n in ((0, "tab:orange", "charge"), (1, "tab:blue", "discharge")):
            ax.scatter(ts[ph == p_, 0], ts[ph == p_, 1], c=c, s=1.2, label=n, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{tag}: t-SNE of $z_s$", fontsize=8)
        ax = axes[2 + j]; y = d["y"]; phs = d["phase"]; sc = d["phi"]
        rng = np.random.default_rng(0); sub = rng.choice(len(y), 4000, replace=False)
        for p_, c in ((0, "tab:orange"), (1, "tab:blue")):
            m = sub[phs[sub] == p_]; ax.scatter(100 * y[m], sc[m], s=1.2, c=c, alpha=0.5, rasterized=True)
        ax.set_title(f"{tag}: $\\phi(z_s)$ vs. SOC", fontsize=8); ax.set_xlabel("True SOC (%)", fontsize=7); ax.tick_params(labelsize=6); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6, frameon=False, markerscale=6, loc="lower left")
    fig.tight_layout(); fig.savefig(ROOT / "paper" / "figures" / "qualitative_main.pdf"); fig.savefig(ROOT / "paper" / "figures" / "qualitative_main.png", dpi=160); plt.close(fig)


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cr", ROOT / "scripts" / "collect_results.py"); cr = importlib.util.module_from_spec(spec); spec.loader.exec_module(cr)
    df = cr.parse(); df_all = df.attrs["all"]; dx = cr.parse_extended()
    if "epochs" not in dx.columns: dx["epochs"] = 50
    fig_pred_vs_true(); fig_residuals(); fig_traces(); fig_qualitative_main(); fig_tsne(); fig_phi_vs_soc(); fig_retrieval(); fig_pretrain_curves(); fig_grad_norm()
    if len(dx):
        fig_per_cell_heatmap(dx); fig_group_bars(dx)
        (TAB / "per_cell.tex").write_text(table_per_cell(dx))
    (TAB / "all_runs.tex").write_text(table_all_runs(df_all))
    print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))
    print(f"all runs table: {len(df_all)} rows")


if __name__ == "__main__":
    main()
