"""Collect every eval_metrics*.json under outputs/ into one table, and emit the
LaTeX tables + figures used by paper/main.tex.

    python scripts/collect_results.py            # writes paper/tables/*.tex, paper/figures/*.png, outputs/results.csv

Run naming is inferred from the directory layout produced by the sbatch files:
  ReBoot main runs        outputs/finetune[_<proto>]_<job>/...          pretrain from run.log / .out (v2/v3/v4 by job id map below)
  ReBoot low-label        outputs/lowlabel[_<proto>][_s<seed>]_<job>/frac_<f>/...
  ReBoot ablation ft      outputs/ft_packed_<job>/<name>/...
  supervised baselines    outputs/checkpoints/baseline_<enc>_frac<f>[_<proto>]_<job>/baseline_<enc>/...
                          outputs/base_lowlabel_<job>/<enc>_frac_<f>/...
                          outputs/baseline_packed_<job>/<enc>_<proto>_frac<f>[_s<seed>]/...
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
PAPER = ROOT / "paper"

# pretrain job -> recipe label (finetune_<job> dirs point at these via the sbatch logs)
PRETRAIN_RECIPE = {
    "16191": "v1 (bf16, random batches, diverged @49)",
    "16206": "v2 random batches",
    "16215": "v3 cycle batches",
    "16259": "v4 cycle+socproxy+mask",
    "16219": "v3 recipe (cross_battery pool)",
    "16294": "v4 recipe (cross_battery pool)",
}
FINETUNE_JOB_TO_PRETRAIN = {  # from outputs/slurm/jobs_2026-09-02.txt
    "16192": "16191", "16207": "16206", "16216": "16215", "16218": "16215", "16260": "16259",
    "16220": "16219", "16295": "16294",
    "16217": "16215", "16263": "16259", "16307": "16259", "16308": "16259", "16296": "16294",
}


def _load(f: Path) -> dict:
    d = json.load(open(f))
    d["file"] = str(f.relative_to(ROOT))
    return d


def parse() -> pd.DataFrame:
    rows = []
    for f in OUT.rglob("eval_metrics*.json"):
        s = str(f.relative_to(OUT))
        if "smoke" in s:
            continue
        d = _load(f)
        proto = d.get("protocol", "per_cell_random")
        em = re.search(r"_ep(\d+)_", s) or re.search(r"ep(\d+)", s)
        row = dict(protocol=proto, mae=d["mae"], rmse=d["rmse"], max_error=d["max_error"],
                   file=d["file"], seed=0, frac=1.0, model="", recipe="", bins=d.get("error_by_bin"),
                   epochs=int(em.group(1)) if em else 50)
        if "baseline" in d:                                       # supervised
            row["model"] = "supervised_" + d["baseline"].split("_", 1)[1] + ("_raw" if "_raw/" in s or s.endswith("_raw") else "")
            row["frac"] = float(d.get("label_fraction", 1.0))
            m = re.search(r"_s(\d+)(?:/|$)", s)
            row["seed"] = int(m.group(1)) // 100 if m else 0
        else:                                                     # ReBoot
            row["model"] = "reboot"
            m = re.match(r"(?:finetune|lowlabel)(?:_[a-z_]+?)?(?:_s(\d+))?_(\d+)/", s)
            if m:
                row["seed"] = int(m.group(1)) // 100 if m.group(1) else 0
                job = m.group(2)
                row["recipe"] = PRETRAIN_RECIPE.get(FINETUNE_JOB_TO_PRETRAIN.get(job, ""), f"job {job}")
            fm = re.search(r"frac_([0-9.]+)/", s)
            if fm:
                row["frac"] = float(fm.group(1))
            am = re.match(r"ft_packed_(\d+)/(.+?)_(\d{3})(?:_ep(\d+))?/", s)
            if am:
                name, code = am.group(2), am.group(3)
                row["frac"] = {"100": 1.0, "001": 0.01, "005": 0.05, "010": 0.10, "025": 0.25}[code]
                if am.group(4):
                    row["epochs"] = int(am.group(4))
                if name == "v4":
                    row["recipe"] = PRETRAIN_RECIPE["16259"]
                elif name.startswith("lp_"):
                    row["recipe"] = "linprobe:" + name[3:]
                elif name.startswith("simclr"):
                    row["recipe"] = "ablation:simclr"
                else:
                    row["recipe"] = "ablation:" + name
        rows.append(row)
    df = pd.DataFrame(rows)
    # the 100 % row of a low-label sweep duplicates the main finetune of the same
    # pretrain/seed (identical config) -> keep one
    df = df.drop_duplicates(subset=["protocol", "model", "recipe", "frac", "seed", "mae"]).reset_index(drop=True)
    df["mae_pct"] = 100 * df["mae"]
    df = df.sort_values(["protocol", "model", "recipe", "frac", "seed"]).reset_index(drop=True)
    df.attrs["all"] = df.copy()
    out = df[(df.epochs == 50)].reset_index(drop=True)
    out.attrs["all"] = df.copy()
    return out         # generic tables: standard budget only


def fmt(x):
    return f"{100 * x:.3f}"


def tex(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def table_full_labels(df: pd.DataFrame) -> str:
    d = df[(df.protocol == "per_cell_random") & (df.frac == 1.0) & (df.seed == 0)]
    d = d[~d.recipe.str.startswith("ablation")]
    d = d[~d.model.isin(LITERATURE) & ~d.model.str.endswith("_raw")]
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{In-distribution test error with all labels (per-cell random split, 19 cells, 574\,944 test windows). MAE and RMSE in \% SOC.}",
             r"\label{tab:full}", r"\begin{tabular}{lrr}", r"\toprule", r"Model & MAE & RMSE \\", r"\midrule"]
    names = {"supervised_lstm": "Supervised LSTM", "supervised_tcn": "Supervised TCN", "supervised_transformer": "Supervised Transformer",
             "supervised_hybrid": "Supervised Hybrid (same encoder)"}
    for _, r in d[d.model != "reboot"].sort_values("mae").iterrows():
        lines.append(f"{names.get(r.model, r.model)} & {fmt(r.mae)} & {fmt(r.rmse)} \\\\")
    lines.append(r"\midrule")
    for _, r in d[d.model == "reboot"].sort_values("recipe").iterrows():
        lines.append(f"ReBoot, {tex(r.recipe)} & {fmt(r.mae)} & {fmt(r.rmse)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _agg(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(["model", "recipe", "frac"])["mae"].agg(["mean", "std", "count"]).reset_index()
    return g


def table_low_label(df: pd.DataFrame, protocol="per_cell_random") -> str:
    d = df[(df.protocol == protocol) & ~df.recipe.str.startswith("ablation")]
    d = d[(d.model == "supervised_hybrid") | (d.model == "supervised_tcn") | ((d.model == "reboot") & d.recipe.str.startswith(("v3", "v4")))]
    g = _agg(d)
    fracs = sorted(g.frac.unique())
    cols = [("supervised_tcn", "", "Sup. TCN"), ("supervised_hybrid", "", "Sup. Hybrid"),
            ("reboot", "v3", "ReBoot v3"), ("reboot", "v4", "ReBoot v4")]   # recipe matched by prefix
    cols = [c for c in cols if ((g.model == c[0]) & (g.recipe.str.startswith(c[1]))).any()]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Low-label regime, " + protocol.replace("_", r"\_") + r": test MAE (\% SOC) vs. fraction of labelled training windows. Mean$\pm$std over seeds where repeated (matched label subsets and initialisation seeds on both sides).}",
             r"\label{tab:lowlabel_" + protocol + "}", r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             "Labels & " + " & ".join(c[2] for c in cols) + r" \\", r"\midrule"]
    for f in fracs:
        cells = []
        for m, rec, _ in cols:
            sel = g[(g.model == m) & (g.frac == f) & g.recipe.str.startswith(rec)]
            if len(sel) == 0:
                cells.append("--"); continue
            r = sel.iloc[0]
            cells.append(f"{100*r['mean']:.3f}" + (f"$\\pm${100*r['std']:.3f}" if r["count"] > 1 and not np.isnan(r["std"]) else "") + (f" ({int(r['count'])})" if r["count"] > 1 else ""))
        lines.append(f"{f*100:g}\\% & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_shift(df: pd.DataFrame) -> str:
    d = df[(df.protocol != "per_cell_random") & (df.seed == 0)]
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Distribution shift: test MAE (\% SOC). Cross-battery holds out three whole cells (one per temperature), excluded from pretraining too; cross-cycle tests on the aged last 30\,\% of every cell's cycles.}",
             r"\label{tab:shift}", r"\begin{tabular}{llrr}", r"\toprule", r"Protocol & Model & Labels & MAE \\", r"\midrule"]
    for proto in ["cross_battery", "cross_cycle", "cross_temperature"]:
        dd = d[d.protocol == proto].sort_values(["frac", "model", "recipe"], ascending=[False, True, True])
        for _, r in dd.iterrows():
            name = ("ReBoot, " + tex(r.recipe)) if r.model == "reboot" else r.model.replace("supervised_", "Supervised ").replace("tcn", "TCN").replace("hybrid", "Hybrid")
            lines.append(f"{proto.replace('_', ' ')} & {name} & {r.frac*100:g}\\% & {fmt(r.mae)} \\\\")
        if len(dd):
            lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_ablation(df: pd.DataFrame) -> str:
    d = df[df.recipe.str.startswith("ablation")]
    names = {"full100": "Full model (100 ep.)", "nophys": r"$-$ $L_\mathrm{phys}$ (finetune)", "norefl": r"$-$ $L_\mathrm{refl}$",
             "noord": r"$-$ $L_\mathrm{ord}$", "single": "Single latent", "maeonly": r"$L_\mathrm{mask}$ only (MAE)"}
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Ablations (per-cell random split, 100 pretrain epochs, v4 recipe). Test MAE (\% SOC) after finetuning with 100\,\% and 1\,\% of labels.}",
             r"\label{tab:ablation}", r"\begin{tabular}{lrr}", r"\toprule", r"Variant & 100\% labels & 1\% labels \\", r"\midrule"]
    for key in ["full100", "nophys", "norefl", "noord", "single", "maeonly"]:
        a = d[(d.recipe == "ablation:" + key)]
        m100 = a[a.frac == 1.0]["mae"]; m1 = a[a.frac == 0.01]["mae"]
        if len(a) == 0:
            continue
        lines.append(f"{names[key]} & {fmt(m100.iloc[0]) if len(m100) else '--'} & {fmt(m1.iloc[0]) if len(m1) else '--'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def fig_low_label(df: pd.DataFrame, protocol="per_cell_random"):
    d = df[(df.protocol == protocol) & ~df.recipe.str.startswith("ablation")]
    g = _agg(d)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    series = [("supervised_hybrid", "", "Supervised Hybrid", "tab:gray", "s"), ("supervised_tcn", "", "Supervised TCN", "tab:olive", "^"),
              ("reboot", "v3", "ReBoot v3", "tab:blue", "o"), ("reboot", "v4", "ReBoot v4", "tab:red", "D")]
    for m, rec, label, color, marker in series:
        sel = g[(g.model == m) & g.recipe.str.startswith(rec)].sort_values("frac")
        if len(sel) == 0:
            continue
        ax.errorbar(sel.frac * 100, sel["mean"] * 100, yerr=(sel["std"].fillna(0) * 100), label=label, color=color, marker=marker, ms=4, lw=1.2, capsize=2)
    ax.set_xscale("log"); ax.set_xlabel("Labelled training windows (%)"); ax.set_ylabel("Test MAE (% SOC)")
    ax.set_xticks([1, 5, 10, 25, 100]); ax.set_xticklabels(["1", "5", "10", "25", "100"])
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(PAPER / "figures" / f"low_label_{protocol}.pdf"); fig.savefig(PAPER / "figures" / f"low_label_{protocol}.png", dpi=160); plt.close(fig)


def fig_bins(df: pd.DataFrame):
    d = df[(df.protocol == "per_cell_random") & (df.frac == 1.0) & (df.seed == 0) & ~df.recipe.str.startswith("ablation")]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for _, r in d.iterrows():
        if r.bins is None or r.model in ("supervised_lstm", "supervised_transformer") or (r.model == "reboot" and not r.recipe.startswith(("v3", "v4"))):
            continue
        vals = [100 * v for v in r.bins.values()]
        ax.plot(np.arange(10) * 10 + 5, vals, marker="o", ms=3, lw=1.1, label=(f"ReBoot {r.recipe.split()[0]}" if r.model == "reboot" else r.model.replace("supervised_", "Sup. ")))
    ax.set_xlabel("True SOC bin (%)"); ax.set_ylabel("MAE (% SOC)"); ax.grid(alpha=0.3); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(PAPER / "figures" / "mae_by_bin.pdf"); fig.savefig(PAPER / "figures" / "mae_by_bin.png", dpi=160); plt.close(fig)


def main():
    df = parse()
    (PAPER / "tables").mkdir(parents=True, exist_ok=True); (PAPER / "figures").mkdir(parents=True, exist_ok=True)
    df.drop(columns=["bins"]).to_csv(OUT / "results.csv", index=False)
    (PAPER / "tables" / "full_labels.tex").write_text(table_full_labels(df))
    (PAPER / "tables" / "low_label.tex").write_text(table_low_label(df))
    if (df.protocol == "cross_battery").any() and ((df.protocol == "cross_battery") & (df.frac < 1)).any():
        (PAPER / "tables" / "low_label_cross_battery.tex").write_text(table_low_label(df, "cross_battery"))
    (PAPER / "tables" / "shift.tex").write_text(table_shift(df))
    (PAPER / "tables" / "ablation.tex").write_text(table_ablation(df))
    (PAPER / "tables" / "lambda.tex").write_text(table_lambda(df))
    (PAPER / "tables" / "control.tex").write_text(table_control(df))
    (PAPER / "tables" / "linprobe.tex").write_text(table_linprobe(df))
    (PAPER / "tables" / "flagship.tex").write_text(table_flagship(df)); (PAPER / "tables" / "flagship_compact.tex").write_text(table_flagship(df, compact=True)); fig_flagship(df); fig_flagship_icassp(df); fig_motivation()
    (PAPER / "tables" / "ablation_compact.tex").write_text(table_ablation_compact(df)); (PAPER / "tables" / "extra_compact.tex").write_text(table_extra_compact(df)); fig_method()
    (PAPER / "tables" / "pool.tex").write_text(table_pool(df))
    (PAPER / "tables" / "ssl_methods.tex").write_text(table_ssl_methods(df))
    fig_ablation(df); fig_lambda(df); fig_physics(); fig_literature(df)
    fig_low_label(df)
    if ((df.protocol == "cross_battery") & (df.frac < 1)).any():
        fig_low_label(df, "cross_battery")
    fig_bins(df)
    pd.set_option("display.width", 200)
    print(df.drop(columns=["bins", "file"]).to_string(index=False))
    print(f"\n{len(df)} results -> outputs/results.csv, paper/tables/*.tex, paper/figures/*")




# ============================================================================
# Extended metrics (eval_extended*.json from scripts/reevaluate.py)
# ============================================================================

def parse_extended() -> pd.DataFrame:
    rows = []
    for f in OUT.rglob("eval_extended*.json"):
        s = str(f.relative_to(OUT))
        if "smoke" in s:
            continue
        d = json.load(open(f))
        row = dict(protocol=d["protocol"], kind=d["kind"], file=str(f.relative_to(ROOT)), seed=0, frac=1.0, recipe="",
                   n_params=d["n_params"], wps=d["windows_per_second"])
        for k in ("mae", "rmse", "max_error", "p50", "p95", "p99", "bias", "r2", "frac_within_1pct", "frac_within_2pct", "n"):
            row[k] = d[k]
        row["by_phase"] = d["by_phase"]; row["by_temperature"] = d["by_temperature"]; row["by_decile"] = d["by_decile"]; row["by_cell"] = d["by_cell"]
        em = re.search(r"_ep(\d+)_", s) or re.search(r"_ep(\d+)/", s)
        row["epochs"] = int(em.group(1)) if em else 50
        if d["kind"].startswith("supervised"):
            row["model"] = d["kind"] + ("_raw" if "_raw/" in s else "")
            fm = re.search(r"frac_?([0-9.]+)", s); row["frac"] = float(fm.group(1)) if fm else 1.0
            sm = re.search(r"_s(\d+)(?:/|$)", s); row["seed"] = int(sm.group(1)) // 100 if sm else 0
        else:
            row["model"] = "reboot"
            m = re.match(r"(?:finetune|lowlabel)(?:_[a-z_]+?)?(?:_s(\d+))?_(\d+)/", s)
            if m:
                row["seed"] = int(m.group(1)) // 100 if m.group(1) else 0
                row["recipe"] = PRETRAIN_RECIPE.get(FINETUNE_JOB_TO_PRETRAIN.get(m.group(2), ""), f"job {m.group(2)}")
            fm = re.search(r"frac_([0-9.]+)/", s)
            if fm:
                row["frac"] = float(fm.group(1))
            am = re.match(r"ft_packed_(\d+)/(.+?)_(\d{3})(?:_ep(\d+))?/", s)
            if am:
                name, code = am.group(2), am.group(3)
                row["frac"] = {"100": 1.0, "001": 0.01, "005": 0.05, "010": 0.10, "025": 0.25}[code]
                row["epochs"] = int(am.group(4)) if am.group(4) else 50
                row["recipe"] = PRETRAIN_RECIPE["16259"] if name == "v4" else ("linprobe:" + name[3:] if name.startswith("lp_") else "ablation:" + name)
            vm = re.match(r"ft_packed_\d+/v4_s(\d+)_frac([0-9.]+)/", s)
            if vm:
                row["recipe"] = PRETRAIN_RECIPE["16259"]; row["seed"] = int(vm.group(1)) // 100; row["frac"] = float(vm.group(2))
            if "epochs" not in row:
                row["epochs"] = 50
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "epochs" in df.columns:
        df = df[df.epochs.fillna(50) == 50]
    df = df[~df.recipe.str.startswith("linprobe")]
    return df.drop_duplicates(subset=["protocol", "model", "recipe", "frac", "seed", "mae"]).sort_values(["protocol", "model", "recipe", "frac", "seed"]).reset_index(drop=True)


MODEL_NAME = {"supervised_lstm": "Supervised LSTM", "supervised_tcn": "Supervised TCN", "supervised_transformer": "Supervised Transformer",
              "supervised_hybrid": "Supervised Hybrid",
              "supervised_chemali_dnn": r"DNN, Chemali et al.~\cite{chemali2018dnn}", "supervised_chemali_lstm": r"LSTM-RNN, Chemali et al.~\cite{chemali2018lstm}",
              "supervised_yang_gru": r"GRU-RNN, Yang et al.~\cite{yang2019gru}", "supervised_hannan_transformer": r"Transformer, Hannan et al.~\cite{hannan2021transformer}",
              "supervised_li_cnnbilstm": r"CNN-Bi-LSTM-AM, Li et al.~\cite{li2026cnnbilstm}"}
LITERATURE = tuple(k for k in MODEL_NAME if k.split("_", 1)[1] not in ("lstm", "tcn", "transformer", "hybrid"))
MODEL_NAME.update({"supervised_hybrid_raw": "Supervised Hybrid (raw inputs)", "supervised_tcn_raw": "Supervised TCN (raw inputs)"})


def _name(r) -> str:
    if r.model == "reboot":
        rec = r.recipe.replace("cycle+socproxy+mask", "cycle, pseudo-SOC").replace(" recipe (cross_battery pool)", "")
        return "ReBoot " + tex(rec)
    return MODEL_NAME.get(r.model, tex(r.model))


def table_extended_full(dx: pd.DataFrame) -> str:
    d = dx[(dx.protocol == "per_cell_random") & (dx.frac == 1.0) & (dx.seed == 0) & ~dx.recipe.str.startswith("ablation")]
    d = d[~d.recipe.str.startswith("v1")]
    d = d[~d.model.isin(LITERATURE) & ~d.model.str.endswith("_raw")]
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Full error statistics on the in-distribution test set (per-cell random split, all labels, 574\,944 windows). All errors in \% SOC; $P_{95}$/$P_{99}$ are percentiles of the absolute error; ``$<$1\%'' is the fraction of windows with absolute error below 1\,\% SOC. Throughput measured on one A100 at batch 1024, data loading included.}",
             r"\label{tab:extended}", r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
             r"Model & Params & MAE & RMSE & $P_{50}$ & $P_{95}$ & $P_{99}$ & Max & Bias & $R^2$ & $<$1\% \\", r"\midrule"]
    for _, r in d.sort_values(["model", "recipe"]).iterrows():
        lines.append(f"{_name(r)} & {r.n_params/1e3:.0f}k & {fmt(r.mae)} & {fmt(r.rmse)} & {fmt(r.p50)} & {fmt(r.p95)} & {fmt(r.p99)} & {100*r.max_error:.1f} & {100*r.bias:+.3f} & {r.r2:.4f} & {100*r.frac_within_1pct:.1f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def table_by_group(dx: pd.DataFrame, key: str, caption: str, label: str, order=None, protocol="per_cell_random", metric="mae") -> str:
    d = dx[(dx.protocol == protocol) & (dx.frac == 1.0) & (dx.seed == 0) & ~dx.recipe.str.startswith(("ablation", "v1"))]
    d = d[~d.model.isin(LITERATURE) & ~d.model.str.endswith("_raw")]
    groups = order or sorted({g for gd in d[key] for g in gd.keys()}, key=lambda s: (len(s), s))
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{tabular}{l" + "r" * len(groups) + "}", r"\toprule", "Model & " + " & ".join(tex(g) for g in groups) + r" \\", r"\midrule"]
    for _, r in d.sort_values(["model", "recipe"]).iterrows():
        cells = [fmt(r[key][g][metric]) if g in r[key] else "--" for g in groups]
        lines.append(f"{_name(r)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_low_label_extended(dx: pd.DataFrame, protocol="per_cell_random") -> str:
    d = dx[(dx.protocol == protocol) & ~dx.recipe.str.startswith(("ablation", "v1", "v2")) & ~dx.model.isin(LITERATURE)]
    g = d.groupby(["model", "recipe", "frac"]).agg(mae=("mae", "mean"), mae_sd=("mae", "std"), rmse=("rmse", "mean"), p95=("p95", "mean"), r2=("r2", "mean"), n=("mae", "size")).reset_index()
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Low-label regime, " + tex(protocol) + r": MAE (mean$\pm$std over seeds), RMSE, $P_{95}$ (\% SOC) and $R^2$ vs. fraction of labelled training windows. Label subsets and initialisation seeds are matched between ReBoot and the supervised baselines.}",
             r"\label{tab:lowlabel_ext_" + protocol + "}", r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Labels & Model & MAE & RMSE & $P_{95}$ & $R^2$ & seeds \\", r"\midrule"]
    for f in sorted(g.frac.unique()):
        gg = g[g.frac == f].sort_values(["model", "recipe"])
        for i, (_, r) in enumerate(gg.iterrows()):
            sd = f"$\\pm${100*r.mae_sd:.3f}" if r.n > 1 and not np.isnan(r.mae_sd) else ""
            lines.append(f"{f*100:g}\\% & {_name(r)} & {fmt(r.mae)}{sd} & {fmt(r.rmse)} & {fmt(r.p95)} & {r.r2:.4f} & {int(r.n)} \\\\")
        lines.append(r"\midrule")
    lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def table_literature_reimpl(df: pd.DataFrame, dx: pd.DataFrame) -> str:
    """Published architectures re-implemented and trained on OUR data, 100 % and 1 % labels."""
    d = df[(df.protocol == "per_cell_random") & (df.seed == 0)]
    ref = d[(d.model == "supervised_hybrid") | ((d.model == "reboot") & d.recipe.str.startswith("v4"))]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Published SOC architectures re-implemented from their papers and trained on our data with the same windows, splits, loss and epochs as every other model (inputs $I$, $V$, $T$ only, as in the originals). Test MAE / RMSE in \% SOC, per-cell random split.}",
             r"\label{tab:lit_reimpl}", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Architecture & \multicolumn{2}{c}{100\% labels} & \multicolumn{2}{c}{1\% labels} \\", r" & MAE & RMSE & MAE & RMSE \\", r"\midrule"]
    def cells(sel):
        r100 = sel[sel.frac == 1.0]; r1 = sel[sel.frac == 0.01]
        f = lambda r, k: fmt(r[k].iloc[0]) if len(r) else "--"
        return f"{f(r100,'mae')} & {f(r100,'rmse')} & {f(r1,'mae')} & {f(r1,'rmse')}"
    for m in LITERATURE:
        sel = d[d.model == m]
        if len(sel):
            lines.append(f"{MODEL_NAME[m]} & {cells(sel)} \\\\")
    lines.append(r"\midrule")
    lines.append(f"Supervised Hybrid (ours, from scratch) & {cells(ref[ref.model == 'supervised_hybrid'])} \\\\")
    lines.append(f"ReBoot v4 (ours, pretrained) & {cells(ref[ref.model == 'reboot'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_lambda(df: pd.DataFrame) -> str:
    """Loss-weight sensitivity: lambda_phys (finetune-only, v4 pretrain) and lambda_refl (100-ep pretrains)."""
    d = df[df.recipe.str.startswith("ablation")]
    phys = [("phys0", "0"), ("phys001", "0.001"), ("__default__", "0.01 (default)"), ("phys01", "0.1"), ("phys1", "1")]
    refl = [("norefl", "0"), ("refl01", "0.1"), ("full100", "0.5 (default)"), ("refl2", "2")]
    v4 = df[(df.model == "reboot") & df.recipe.str.startswith("v4") & (df.protocol == "per_cell_random") & (df.seed == 0)]
    def get(key, frac):
        if key == "__default__":
            r = v4[v4.frac == frac]
        else:
            r = d[(d.recipe == "ablation:" + key) & (d.frac == frac)]
        return fmt(r["mae"].iloc[0]) if len(r) else "--"
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Loss-weight sensitivity (test MAE, \% SOC, per-cell random split). $\lambda_\mathrm{phys}$ is varied at finetuning from the v4 pretrain (200 epochs); $\lambda_\mathrm{refl}$ requires a new pretrain and is varied at 100 epochs, so its rows compare with the 100-epoch full model.}",
             r"\label{tab:lambda}", r"\begin{tabular}{lrr@{\hspace{1.5em}}lrr}", r"\toprule",
             r"$\lambda_\mathrm{phys}$ & 100\% & 1\% & $\lambda_\mathrm{refl}$ & 100\% & 1\% \\", r"\midrule"]
    for i in range(max(len(phys), len(refl))):
        a = phys[i] if i < len(phys) else None; b = refl[i] if i < len(refl) else None
        left = f"{a[1]} & {get(a[0], 1.0)} & {get(a[0], 0.01)}" if a else " & & "
        right = f"{b[1]} & {get(b[0], 1.0)} & {get(b[0], 0.01)}" if b else " & & "
        lines.append(f"{left} & {right} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_linprobe(df: pd.DataFrame) -> str:
    d = df[(df.protocol == "per_cell_random") & (df.seed >= 0)]
    fracs = [0.01, 0.05, 0.10, 0.25, 1.0]
    def get(rec, f, exact=False):
        r = d[(d.recipe == rec) & (d.frac == f)] if exact else d[d.recipe.str.startswith(rec) & (d.frac == f) & (d.seed == 0) & (d.model == "reboot")]
        return fmt(r.mae.iloc[0]) if len(r) else "--"
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Frozen-encoder linear probe vs.\ full finetuning (test MAE, \% SOC, per-cell random split). Linear probe: encoder and projections frozen, regression head and physics parameters trained.}",
             r"\label{tab:linprobe}", r"\begin{tabular}{lrrrrr}", r"\toprule",
             "Model / labels & " + " & ".join(f"{f*100:g}\\%" for f in fracs) + r" \\", r"\midrule",
             "ReBoot v4, linear probe & " + " & ".join(get("linprobe:v4", f, True) for f in fracs) + r" \\",
             "ReBoot v4, full finetune & " + " & ".join(get("v4", f) for f in fracs) + r" \\",
             "$L_\\mathrm{mask}$-only, linear probe & " + " & ".join(get("linprobe:mae", f, True) for f in fracs) + r" \\",
             "$L_\\mathrm{mask}$-only, full finetune (100 ep.) & " + " & ".join(get("ablation:maeonly", f, True) for f in fracs) + r" \\",
             r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_pool(df: pd.DataFrame) -> str:
    d = df[df.protocol == "per_cell_random"]
    rows = [("10\\%", "ablation:pool10"), ("25\\%", "ablation:pool25"), ("50\\%", "ablation:pool50"), ("100\\%", "ablation:full100")]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Unlabeled-data scaling: fraction of the training cycles used for pretraining (100 epochs, v4 recipe), then finetuning with 100\,\% and 1\,\% of labels. Test MAE, \% SOC.}",
             r"\label{tab:pool}", r"\begin{tabular}{lrr}", r"\toprule", r"Unlabeled cycles & 100\% labels & 1\% labels \\", r"\midrule"]
    for name, rec in rows:
        r100 = d[(d.recipe == rec) & (d.frac == 1.0)]; r1 = d[(d.recipe == rec) & (d.frac == 0.01)]
        if len(r100) or len(r1):
            lines.append(f"{name} & {fmt(r100.mae.iloc[0]) if len(r100) else '--'} & {fmt(r1.mae.iloc[0]) if len(r1) else '--'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_ssl_methods(df: pd.DataFrame) -> str:
    d = df[(df.protocol == "per_cell_random") & (df.seed == 0)]
    rows = [("No pretraining (supervised hybrid)", d[(d.model == "supervised_hybrid")]),
            ("Masked reconstruction only (MAE)", d[d.recipe == "ablation:maeonly"]),
            ("SimCLR (augment-pair InfoNCE)", d[d.recipe == "ablation:simclr"]),
            ("ReBoot (100 ep.)", d[d.recipe == "ablation:full100"]),
            ("ReBoot v4 (200 ep.)", d[(d.model == "reboot") & d.recipe.str.startswith("v4")])]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Self-supervised methods on the same hybrid encoder, finetuned identically (test MAE, \% SOC, per-cell random split). 100-epoch pretrains unless stated.}",
             r"\label{tab:ssl}", r"\begin{tabular}{lrr}", r"\toprule", r"Pretraining & 100\% labels & 1\% labels \\", r"\midrule"]
    for name, r in rows:
        r100 = r[r.frac == 1.0]; r1 = r[r.frac == 0.01]
        if len(r100) or len(r1):
            lines.append(f"{name} & {fmt(r100.mae.iloc[0]) if len(r100) else '--'} & {fmt(r1.mae.iloc[0]) if len(r1) else '--'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_control(df: pd.DataFrame) -> str:
    a = df.attrs.get("all", df)
    a = a[(a.protocol == "per_cell_random") & (a.frac == 0.01) & (a.seed == 0)]
    rows = [("Supervised Hybrid", a[(a.model == "supervised_hybrid") & (a.epochs == 50)], 50),
            ("Supervised Hybrid", a[(a.model == "supervised_hybrid") & (a.epochs == 500)], 500),
            ("ReBoot v4", a[(a.model == "reboot") & a.recipe.str.startswith("v4") & (a.epochs == 50)], 50),
            ("ReBoot v4", a[(a.model == "reboot") & a.recipe.str.startswith("v4") & (a.epochs == 500)], 500)]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Training-budget control at 1\,\% of labels (per-cell random split). 500 finetuning epochs at 1\,\% of labels equal the number of gradient steps of 50 epochs at 10\,\%.}",
             r"\label{tab:control}", r"\begin{tabular}{lrrr}", r"\toprule", r"Model & Epochs & MAE & RMSE \\", r"\midrule"]
    for name, r, ep in rows:
        lines.append(f"{name} & {ep} & {fmt(r.mae.iloc[0]) if len(r) else '--'} & {fmt(r.rmse.iloc[0]) if len(r) else '--'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def fig_ablation(df: pd.DataFrame):
    d = df[df.recipe.str.startswith("ablation:")]
    keys = [("full100", "Full"), ("nophys", "$-L_{phys}$"), ("norefl", "$-L_{refl}$"), ("noord", "$-L_{ord}$"), ("single", "Single latent"), ("maeonly", "$L_{mask}$ only")]
    keys = [k for k in keys if (d.recipe == "ablation:" + k[0]).any()]
    if not keys:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, frac, title in ((axes[0], 0.01, "1% labels"), (axes[1], 1.0, "100% labels")):
        vals = [100 * d[(d.recipe == "ablation:" + k) & (d.frac == frac)]["mae"].iloc[0] if ((d.recipe == "ablation:" + k) & (d.frac == frac)).any() else np.nan for k, _ in keys]
        colors = ["tab:red" if k == "full100" else "tab:gray" for k, _ in keys]
        ax.bar(range(len(keys)), vals, color=colors)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels([n for _, n in keys], rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Test MAE (% SOC)"); ax.set_title(title, fontsize=9); ax.grid(axis="y", alpha=0.3)
        lo = np.nanmin(vals); ax.set_ylim(lo * 0.85, np.nanmax(vals) * 1.05)
    fig.tight_layout(); fig.savefig(PAPER / "figures" / "ablation.pdf"); fig.savefig(PAPER / "figures" / "ablation.png", dpi=160); plt.close(fig)


def fig_lambda(df: pd.DataFrame):
    d = df[df.recipe.str.startswith("ablation:")]
    v4 = df[(df.model == "reboot") & df.recipe.str.startswith("v4") & (df.protocol == "per_cell_random") & (df.seed == 0) & (df.frac == 0.01)]
    pts = [(0.0, "phys0"), (0.001, "phys001"), (0.01, None), (0.1, "phys01"), (1.0, "phys1")]
    xs, ys = [], []
    for lam, key in pts:
        r = v4 if key is None else d[(d.recipe == "ablation:" + key) & (d.frac == 0.01)]
        if len(r):
            xs.append(lam); ys.append(100 * r.mae.iloc[0])
    if len(xs) < 3:
        return
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    xplot = [1e-4 if x == 0 else x for x in xs]
    ax.plot(xplot, ys, marker="o", color="tab:red")
    ax.set_xscale("log"); ax.set_xticks(xplot); ax.set_xticklabels(["0", "0.001", "0.01", "0.1", "1"][: len(xplot)])
    ax.set_xlabel(r"$\lambda_{phys}$"); ax.set_ylabel("Test MAE (% SOC), 1% labels"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PAPER / "figures" / "lambda_phys.pdf"); fig.savefig(PAPER / "figures" / "lambda_phys.png", dpi=160); plt.close(fig)


def fig_physics():
    f = OUT / "physics_params.json"
    if not f.exists():
        return
    d = json.load(open(f)); ids = d["battery_ids"]
    runs = [("v4 100%", "tab:red"), ("v4 1%", "tab:orange"), ("v4 cross_battery 100%", "tab:blue")]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    w = 0.27
    for i, (name, c) in enumerate(runs):
        if name not in d["runs"]:
            continue
        r = d["runs"][name]["Q_over_nominal"]
        ax.bar(np.arange(len(ids)) + (i - 1) * w, [r[b] for b in ids], width=w, color=c, label=name.replace("cross_battery", "cross-battery"))
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(ids))); ax.set_xticklabels(ids, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Learned $Q_c / Q_{nom}$"); ax.set_ylim(0.75, 1.05); ax.legend(fontsize=7, frameon=False, ncol=3); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(PAPER / "figures" / "physics_Q.pdf"); fig.savefig(PAPER / "figures" / "physics_Q.png", dpi=160); plt.close(fig)


def fig_literature(df: pd.DataFrame):
    d = df[(df.protocol == "per_cell_random") & (df.seed == 0)]
    names = {"supervised_chemali_dnn": "DNN (Chemali)", "supervised_chemali_lstm": "LSTM (Chemali)", "supervised_yang_gru": "GRU (Yang)",
             "supervised_hannan_transformer": "Transformer (Hannan)", "supervised_li_cnnbilstm": "CNN-BiLSTM-AM (Li)",
             "supervised_lstm": "LSTM (ours)", "supervised_transformer": "Transformer (ours)", "supervised_tcn": "TCN (ours)", "supervised_hybrid": "Hybrid (ours)"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, frac, title in ((axes[0], 0.01, "1% labels"), (axes[1], 1.0, "100% labels")):
        items = []
        for m, n in names.items():
            r = d[(d.model == m) & (d.frac == frac)]
            if len(r):
                items.append((n, 100 * r.mae.iloc[0], "tab:gray"))
        r = d[(d.model == "reboot") & d.recipe.str.startswith("v4") & (d.frac == frac)]
        if len(r):
            items.append(("ReBoot v4 (ours)", 100 * r.mae.iloc[0], "tab:red"))
        items.sort(key=lambda t: -t[1])
        ax.barh(range(len(items)), [v for _, v, _ in items], color=[c for _, _, c in items])
        ax.set_yticks(range(len(items))); ax.set_yticklabels([n for n, _, _ in items], fontsize=7)
        ax.set_xlabel("Test MAE (% SOC)"); ax.set_title(title, fontsize=9); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(PAPER / "figures" / "literature.pdf"); fig.savefig(PAPER / "figures" / "literature.png", dpi=160); plt.close(fig)


FLAGSHIP_ROWS = [
    ("Supervised, from scratch", [("supervised_lstm", "", "LSTM~\\cite{hochreiter1997lstm}"), ("supervised_transformer", "", "Transformer~\\cite{vaswani2017attention}"),
                                  ("supervised_tcn", "", "TCN~\\cite{bai2018tcn}"), ("supervised_hybrid", "", "Hybrid (ReBoot encoder)")]),
    ("Published SOC architectures, re-implemented", [("supervised_chemali_dnn", "", "DNN~\\cite{chemali2018dnn}"), ("supervised_chemali_lstm", "", "LSTM-RNN~\\cite{chemali2018lstm}"),
                                  ("supervised_yang_gru", "", "GRU-RNN~\\cite{yang2019gru}"), ("supervised_hannan_transformer", "", "Transformer~\\cite{hannan2021transformer}"),
                                  ("supervised_li_cnnbilstm", "", "CNN-Bi-LSTM-AM~\\cite{li2026cnnbilstm}")]),
    ("Generic SSL pretraining, same encoder", [("reboot", "ablation:maeonly", "Masked reconstruction (MAE)~\\cite{he2022mae}"), ("reboot", "ablation:simclr", "SimCLR-style contrastive~\\cite{chen2020simclr}")]),
    ("ReBoot (ours)", [("reboot", "v3", "ReBoot, position-symmetric pairs"), ("reboot", "v4", "ReBoot, pseudo-SOC pairs + mask")]),
    ("Raw $I,V,T,\\Delta t$ inputs only (no derived channels)", [("supervised_hybrid_raw", "", "Hybrid, from scratch"), ("reboot", "ablation:raw", "ReBoot (100 ep.)")]),
]
FRACS = [1.0, 0.25, 0.10, 0.05, 0.01]


def _flag_cell(df, model, rec, frac, metric="mae"):
    d = df[(df.protocol == "per_cell_random") & (df.model == model) & (df.frac == frac)]
    d = d[d.recipe == rec] if rec.startswith("ablation") else (d[d.recipe.str.startswith(rec)] if rec else d)
    if len(d) == 0:
        return None
    return float(d[metric].mean()), int(len(d))


def table_flagship(df: pd.DataFrame, compact: bool = False) -> str:
    vals = {}
    ROWS = [(g, [r for r in rows if not (compact and r[0] in ('supervised_lstm', 'supervised_transformer'))]) for g, rows in FLAGSHIP_ROWS]
    for _, rows in ROWS:
        for model, rec, name in rows:
            for f in FRACS:
                vals[(name, f)] = _flag_cell(df, model, rec, f)
    raw_names = {r[2] for g, rows in ROWS if g.startswith("Raw") for r in rows}
    lit_names = {r[2] for g, rows in ROWS if g.startswith("Published") for r in rows}
    # Bold the best only among rows sharing the 7-channel input set, so that the
    # published-architecture and raw-input rows are not read as input-matched.
    best = {f: min(v[0] for (n, ff), v in vals.items()
                   if ff == f and v is not None and n not in raw_names and n not in lit_names)
            for f in FRACS}
    lines = [r"\begin{table*}[t]", r"\centering"] + ([r"\footnotesize", r"\setlength{\tabcolsep}{4pt}"] if compact else []) + [
             r"\caption{Flagship comparison: test MAE (\% SOC) on the in-distribution split (574\,944 held-out windows of all 19 cells) versus the fraction of labelled training windows. All models share the same windows, splits, loss (Huber), fine-tuning epochs (50) and label subsets; entries are averaged over seeds where repeated (count in parentheses). \textbf{Input sets differ between groups, which are therefore not interchangeable:} the first, third and fourth groups receive the 7 channels of Section~\ref{sec:setup}, the published architectures receive $I$, $V$, $T$ as in their papers, and the last group receives raw $I$, $V$, $T$, $\Delta t$ only. Bold marks the best value per column \textbf{within the 7-channel groups}, where ReBoot and the supervised hybrid share the identical encoder and the comparison is input-matched. The generic-SSL rows use 100-epoch pretraining; ReBoot at 100 epochs, which is the epoch-matched comparison, gives 0.438 and 0.781 at 100\,\% and 1\,\% of labels.}",
             r"\label{tab:flagship}", r"\begin{tabular}{l" + "r" * len(FRACS) + "}", r"\toprule",
             r"Model & " + " & ".join(f"{f*100:g}\\% labels" for f in FRACS) + r" \\", r"\midrule"]
    for group, rows in ROWS:
        if group.startswith("Raw"):
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{" + str(len(FRACS) + 1) + r"}{l}{\textit{" + group + r"}} \\")
        for model, rec, name in rows:
            cells = []
            for f in FRACS:
                v = vals[(name, f)]
                if v is None:
                    cells.append("--"); continue
                txt = f"{100*v[0]:.3f}" + (f" ({v[1]})" if v[1] > 1 else "")
                cells.append(r"\textbf{" + txt + "}" if abs(v[0] - best[f]) < 1e-9 else txt)
            lines.append(f"\\quad {name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def fig_flagship(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    styles = {"Supervised, from scratch": ("tab:gray", "-"), "Published SOC architectures, re-implemented": ("tab:olive", ":"),
              "Generic SSL pretraining, same encoder": ("tab:blue", "--"), "ReBoot (ours)": ("tab:red", "-"),
              "Raw $I,V,T,\\Delta t$ inputs only (no derived channels)": ("tab:purple", "-.")}
    markers = "osD^vP*Xh<>"
    k = 0
    for group, rows in FLAGSHIP_ROWS:
        if group not in styles:
            continue
        color, ls = styles[group]
        for model, rec, name in rows:
            xs, ys = [], []
            for f in FRACS:
                v = _flag_cell(df, model, rec, f)
                if v is not None:
                    xs.append(100 * f); ys.append(100 * v[0])
            if len(xs) >= 2:
                lw = 2.0 if "pseudo-SOC" in name else 1.1
                lab = re.sub(r"~\\cite\{[^}]*\}", "", name).replace("\\", "")
                if group.startswith("Raw"):
                    lab = lab.replace(" (100 ep.)", "") + " (raw inputs)"
                ax.plot(xs, ys, ls, color=color, marker=markers[k % len(markers)], ms=4, lw=lw, alpha=0.95 if group == "ReBoot (ours)" else 0.75, label=lab)
            k += 1
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([1, 5, 10, 25, 100]); ax.set_xticklabels(["1", "5", "10", "25", "100"])
    from matplotlib.ticker import FixedLocator, NullFormatter, FixedFormatter
    ax.yaxis.set_major_locator(FixedLocator([0.3, 0.5, 0.7, 1, 1.5, 2, 3])); ax.yaxis.set_major_formatter(FixedFormatter(["0.3", "0.5", "0.7", "1", "1.5", "2", "3"]))
    ax.yaxis.set_minor_formatter(NullFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Labelled training windows (%)"); ax.set_ylabel("Test MAE (% SOC)"); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout(); fig.savefig(PAPER / "figures" / "flagship.pdf"); fig.savefig(PAPER / "figures" / "flagship.png", dpi=160); plt.close(fig)


def table_ablation_compact(df: pd.DataFrame) -> str:
    """ICASSP: ablation + loss-weight sensitivity in one table (100-ep pretrains; test MAE % SOC)."""
    d = df[df.recipe.str.startswith("ablation")]
    v4 = df[(df.model == "reboot") & df.recipe.str.startswith("v4") & (df.protocol == "per_cell_random") & (df.seed == 0)]
    def g(key, frac):
        r = v4[v4.frac == frac] if key == "v4" else d[(d.recipe == "ablation:" + key) & (d.frac == frac)]
        return fmt(r.mae.iloc[0]) if len(r) else "--"
    rows = [(r"\textit{Full model, 100 ep.}", "full100"), (r"\quad $-\,\Lrefl$", "norefl"), (r"\quad $-\,\Lord$", "noord"),
            (r"\quad $-\,\Lphys$ (finetune)", "nophys"), (r"\quad single latent $[\zs;\zd]\to\mathbf{z}$", "single"),
            (r"\quad $\Lmask$ only (generic MAE)", "maeonly"), (r"\quad SimCLR-style contrastive", "simclr"),
            (r"\textit{Full model, 200 ep.\ (v4)}", "v4")]
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Ablation (test MAE, \% SOC; 100-epoch pretrains unless stated, per-cell random split).}",
             r"\label{tab:abl}", r"\footnotesize", r"\begin{tabular}{lrr}", r"\toprule", r"Variant & 100\% labels & 1\% labels \\", r"\midrule"]
    for name, key in rows:
        if key:
            lines.append(f"{name} & {g(key, 1.0)} & {g(key, 0.01)} \\\\")
        elif "refl" in name:
            lines.append(f"{name} & {g('refl01',1.0)} / {g('full100',1.0)} / {g('refl2',1.0)} & {g('refl01',0.01)} / {g('full100',0.01)} / {g('refl2',0.01)} \\\\")
        elif "phys" in name:
            lines.append(f"{name} & {g('phys0',1.0)} / {g('v4',1.0)} / {g('phys01',1.0)} / {g('phys1',1.0)} & {g('phys0',0.01)} / {g('v4',0.01)} / {g('phys01',0.01)} / {g('phys1',0.01)} \\\\")
        else:
            lines.append(f"{name} & {g('pool10',1.0)} / {g('pool25',1.0)} / {g('pool50',1.0)} / {g('full100',1.0)} & {g('pool10',0.01)} / {g('pool25',0.01)} / {g('pool50',0.01)} / {g('full100',0.01)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_extra_compact(df: pd.DataFrame) -> str:
    """ICASSP: shift protocols + equal-steps control + linear probe in one table."""
    a = df.attrs.get("all", df)
    def get(model, rec, proto, frac, epochs=50):
        d = a[(a.protocol == proto) & (a.frac == frac) & (a.epochs == epochs) & (a.model == model) & (a.seed == 0)]
        if rec: d = d[d.recipe.str.startswith(rec)]
        return fmt(d.mae.iloc[0]) if len(d) else "--"
    lp = a[a.recipe.str.startswith("linprobe:v4")]
    lp1 = fmt(lp[lp.frac == 0.01].mae.iloc[0]) if (lp.frac == 0.01).any() else "--"; lp100 = fmt(lp[lp.frac == 1.0].mae.iloc[0]) if (lp.frac == 1.0).any() else "--"
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Shift protocols and controls (test MAE, \% SOC). Cross-battery: three whole cells held out and excluded from pretraining; cross-cycle: aged last 30\,\% of every cell's cycles. Equal steps: 500 finetuning epochs at 1\,\% of labels. Linear probe: frozen pretrained encoder.}",
             r"\label{tab:extra}", r"\footnotesize", r"\begin{tabular}{llrr}", r"\toprule", r"Setting & Labels & Sup.\ Hybrid & ReBoot v4 \\", r"\midrule",
             f"Cross-battery & 100\\% & {get('supervised_hybrid','', 'cross_battery',1.0)} & {get('reboot','v4','cross_battery',1.0)} \\\\",
             f"Cross-battery & 1\\% & {get('supervised_hybrid','', 'cross_battery',0.01)} & {get('reboot','v4','cross_battery',0.01)} \\\\",
             f"Cross-cycle (ageing) & 100\\% & {get('supervised_hybrid','', 'cross_cycle',1.0)} & {get('reboot','v3','cross_cycle',1.0)} \\\\",
             f"Equal steps (500 ep.) & 1\\% & {get('supervised_hybrid','', 'per_cell_random',0.01,500)} & {get('reboot','v4','per_cell_random',0.01,500)} \\\\",
             f"Linear probe, frozen encoder & 1\\% / 100\\% & -- & {lp1} / {lp100} \\\\",
             r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def fig_method():
    """Block diagram of the method (matplotlib, no TikZ dependency)."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    NL = chr(10)
    fig, ax = plt.subplots(figsize=(7.4, 2.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
    def box(x, y, w, h, text, fc="#f2f2f2", fs=7.5, ec="k"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
    def arrow(x0, y0, x1, y1, col="k"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=col))
    box(1, 15, 13, 10, r"window $\mathbf{x}$" + NL + r"$(W{=}100)\times C$" + NL + r"$I,V,T,\Delta t,\dots$")
    box(19, 15, 14, 10, r"encoder $f_\theta$" + NL + "TCN + transformer" + NL + r"$\mathbf{h}\in\mathbb{R}^{128}$")
    arrow(14, 20, 19, 20)
    box(39, 26, 12, 8, r"$g_s\to\mathbf{z}_s\in\mathbb{R}^{64}$" + NL + "shared (SOC)", fc="#fde0dc")
    box(39, 6, 12, 8, r"$g_d\to\mathbf{z}_d\in\mathbb{R}^{64}$" + NL + "dynamic", fc="#dce9fd")
    arrow(33, 22, 39, 30); arrow(33, 18, 39, 10)
    box(55, 32, 25, 6.5, r"$\mathcal{L}_{ord}$: $\phi(\mathbf{z}_s)$ monotone in time", fc="white", fs=6.8)
    box(55, 24.5, 25, 6.5, r"$\mathcal{L}_{refl}$: charge $\leftrightarrow$ discharge at equal" + NL + "pseudo-SOC; near-SOC negatives masked", fc="white", fs=6.3)
    box(55, 17, 25, 6.5, r"$\mathcal{L}_{smooth}$: consecutive windows, $\mathbf{z}_s$", fc="white", fs=6.8)
    box(55, 9.5, 25, 6.5, r"$\mathcal{L}_{orth}$: decorrelate $\mathbf{z}_s$ and $\mathbf{z}_d$", fc="white", fs=6.8)
    box(55, 2, 25, 6.5, r"decoder $r_\psi([\mathbf{z}_s;\mathbf{z}_d])\to\hat{\mathbf{x}}$: $\mathcal{L}_{mask}$ (L1)", fc="white", fs=6.5)
    for y in (35, 27.5, 20):
        arrow(51, 30, 55, y, col="#c0392b")
    arrow(51, 10, 55, 12.5, col="#2e6db4"); arrow(51, 30, 55, 12.5, col="#c0392b")
    arrow(51, 10, 55, 5, col="#2e6db4"); arrow(51, 30, 55, 5, col="#c0392b")
    ax.text(67.5, 39.2, "Stage 1: pretraining, no labels", ha="center", fontsize=8, weight="bold")
    box(83, 15, 16.5, 10, "Stage 2: finetune" + NL + r"head $h_\omega([\mathbf{z}_s;\mathbf{z}_d])\to\hat y$" + NL + r"$\mathcal{L}_{sup}$ (Huber) $+\,\lambda\,\mathcal{L}_{phys}$", fc="#e8f5e9", fs=6.6)
    arrow(51, 30, 83, 21, col="#c0392b"); arrow(51, 10, 83, 19, col="#2e6db4")
    ax.text(91.2, 11.5, r"$\mathcal{L}_{phys}=|\hat y_{t+1}-\hat y_t-\eta\bar I_t\Delta T_t/Q_c|$", ha="center", fontsize=6.3)
    fig.tight_layout(); fig.savefig(PAPER / "figures" / "method.pdf"); fig.savefig(PAPER / "figures" / "method.png", dpi=170); plt.close(fig)


def fig_flagship_icassp(df: pd.DataFrame):
    """Two-panel ICASSP figure: (a) label-efficiency curves with seed error bars, (b) raw-input comparison."""
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.05), gridspec_kw={"width_ratios": [1.6, 1]})
    series = [("supervised_hybrid", "", "Hybrid, from scratch", "tab:gray", "s", "-", 1.6),
              ("supervised_tcn", "", "TCN, from scratch", "tab:gray", "D", "--", 1.0),
              ("supervised_li_cnnbilstm", "", "CNN-Bi-LSTM-AM (re-impl.)", "tab:olive", "p", ":", 1.0),
              ("supervised_hannan_transformer", "", "Transformer (re-impl.)", "tab:olive", "h", ":", 1.0),
              ("reboot", "ablation:maeonly", "Masked-only pretrain", "tab:blue", "<", "--", 1.0),
              ("reboot", "ablation:simclr", "SimCLR pretrain", "tab:blue", ">", "--", 1.0),
              ("reboot", "v4", "ReBoot (ours)", "tab:red", "o", "-", 2.0)]
    d0 = df[df.protocol == "per_cell_random"]
    for model, rec, label, color, marker, ls, lw in series:
        d = d0[d0.model == model]
        d = d[d.recipe == rec] if rec.startswith("ablation") else (d[d.recipe.str.startswith(rec)] if rec else d)
        g = d.groupby("frac")["mae"].agg(["mean", "std", "size"]).reset_index().sort_values("frac")
        if len(g) < 2:
            continue
        ax.errorbar(100 * g.frac, 100 * g["mean"], yerr=100 * g["std"].fillna(0), fmt=marker + ls, color=color, ms=4, lw=lw, capsize=2, label=label, alpha=0.95)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([1, 5, 10, 25, 100]); ax.set_xticklabels(["1", "5", "10", "25", "100"])
    ax.yaxis.set_major_locator(FixedLocator([0.3, 0.5, 0.7, 1, 1.5, 2])); ax.yaxis.set_major_formatter(FixedFormatter(["0.3", "0.5", "0.7", "1", "1.5", "2"]))
    ax.yaxis.set_minor_formatter(NullFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Labelled training windows (%)"); ax.set_ylabel("Test MAE (% SOC)"); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.24), columnspacing=0.9, handlelength=1.6); ax.set_title("(a) 7 input channels; bars = std over 3 seeds", fontsize=8)
    # (b) raw inputs
    hy = d0[d0.model == "supervised_hybrid_raw"].groupby("frac")["mae"].mean(); rb = d0[d0.recipe == "ablation:raw"].groupby("frac")["mae"].mean()
    fr = [f for f in FRACS if f in hy.index and f in rb.index]
    x = np.arange(len(fr)); w = 0.38
    bx.bar(x - w / 2, [100 * hy[f] for f in fr], w, color="tab:gray", label="Hybrid, from scratch")
    bx.bar(x + w / 2, [100 * rb[f] for f in fr], w, color="tab:red", label="ReBoot (ours)")
    for i, f in enumerate(fr):
        gain = 100 * (1 - rb[f] / hy[f]); bx.text(i, 100 * max(hy[f], rb[f]) + 0.03, f"$-${gain:.0f}%", ha="center", fontsize=6.5)
    bx.set_xticks(x); bx.set_xticklabels([f"{f*100:g}%" for f in fr]); bx.set_xlabel("Labelled training windows"); bx.set_ylabel("Test MAE (% SOC)")
    bx.set_title("(b) raw $I,V,T,\\Delta t$ inputs only", fontsize=8); bx.legend(fontsize=6, frameon=False, loc="upper left"); bx.grid(axis="y", alpha=0.3)
    bx.set_ylim(0, max(100 * hy.max(), 100 * rb.max()) * 1.35)
    fig.tight_layout(); fig.savefig(PAPER / "figures" / "flagship_icassp.pdf"); fig.savefig(PAPER / "figures" / "flagship_icassp.png", dpi=170); plt.close(fig)


def table_beyond_mae(dx: pd.DataFrame) -> str:
    """ICASSP: error statistics beyond MAE for the identical encoder, from scratch vs pretrained."""
    d = dx[(dx.protocol == "per_cell_random") & (dx.seed == 0) & (dx.epochs == 50)]
    rows = []
    for frac in (1.0, 0.01):
        for name, sel in (("Hybrid, from scratch", d[(d.model == "supervised_hybrid") & (d.frac == frac)]),
                          ("ReBoot v4", d[(d.model == "reboot") & d.recipe.str.startswith("v4") & (d.frac == frac)]),
                          ("Hybrid, raw inputs", d[(d.model == "supervised_hybrid_raw") & (d.frac == frac)]),
                          ("ReBoot, raw inputs", d[(d.recipe == "ablation:raw") & (d.frac == frac)])):
            if len(sel):
                r = sel.iloc[0]
                rows.append(f"{frac*100:g}\\% & {name} & {fmt(r.mae)} & {fmt(r.rmse)} & {fmt(r.p95)} & {fmt(r.p99)} & {r.r2:.4f} & {100*r.frac_within_1pct:.1f} \\\\")
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Beyond MAE: error statistics of the identical encoder trained from scratch and pretrained (\% SOC; $P_{95}/P_{99}$: percentiles of $|$error$|$; $<$1\%: share of test windows within 1\,\% SOC).}",
             r"\label{tab:beyond}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Labels & Model & MAE & RMSE & $P_{95}$ & $P_{99}$ & $R^2$ & $<$1\% \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def fig_trace_icassp():
    """One test cycle, 1 % labels: reference vs from-scratch vs pretrained (compact, column width)."""
    tc = json.load(open(OUT / "export" / "trace_cycles.json"))
    cell = "cell2-35"; info = tc[cell]
    pos = np.array(info["test_positions"]); y = np.array(info["y"]); seg = np.array(info["segment_id"])
    hy = np.load(ROOT / "outputs/base_lowlabel_16222/hybrid_frac_0.01/baseline_hybrid/predictions.npz")["p"][pos]
    rb = np.load(ROOT / "outputs/lowlabel_16263/frac_0.01/checkpoints/finetune/predictions.npz")["p"][pos]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.5, 1.75), gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(len(pos))
    ax.plot(x, 100 * y, "k-", lw=1.4, label="reference"); ax.plot(x, 100 * hy, "--", color="tab:gray", lw=1.0, label="from scratch"); ax.plot(x, 100 * rb, "-", color="tab:red", lw=1.0, label="ReBoot")
    for k in np.flatnonzero(np.diff(seg)) + 0.5:
        ax.axvline(k, color="gray", lw=0.5, ls=":"); bx.axvline(k, color="gray", lw=0.5, ls=":")
    ax.set_ylabel("SOC (%)", fontsize=7); ax.set_xlabel("window in cycle", fontsize=7); ax.tick_params(labelsize=6); ax.legend(fontsize=5.5, frameon=False, loc="lower center")
    bx.plot(x, 100 * (hy - y), "--", color="tab:gray", lw=1.0); bx.plot(x, 100 * (rb - y), "-", color="tab:red", lw=1.0); bx.axhline(0, color="k", lw=0.5)
    bx.set_ylabel("error (% SOC)", fontsize=7); bx.set_xlabel("window in cycle", fontsize=7); bx.tick_params(labelsize=6); bx.grid(alpha=0.3)
    ax.set_title(f"{cell}, 1% labels", fontsize=7); bx.set_title(f"MAE: {100*np.abs(hy-y).mean():.2f} vs {100*np.abs(rb-y).mean():.2f}%", fontsize=7)
    fig.tight_layout(); fig.savefig(PAPER / "figures" / "trace_icassp.pdf"); fig.savefig(PAPER / "figures" / "trace_icassp.png", dpi=170); plt.close(fig)


def fig_motivation():
    """Fig. 1 of the paper: why SOC estimation is hard, from real cells."""
    f = OUT / "export" / "motivation_cycles.json"
    if not f.exists():
        return
    d = json.load(open(f))
    fig = plt.figure(figsize=(7.2, 2.9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.35])
    ax = fig.add_subplot(gs[:, 0]); bx = fig.add_subplot(gs[:, 1])
    cx = fig.add_subplot(gs[0, 2]); dx_ = fig.add_subplot(gs[1, 2], sharex=cx)
    # (a) hysteresis: V vs SOC, charge vs discharge, one cell
    k = "cell2-35|mid"; r = d[k]
    y = 100 * np.array(r["y"]); V = np.array(r["V"]); ph = np.array(r["phase"])
    for p_, c, n in ((0, "tab:orange", "charge"), (1, "tab:blue", "discharge")):
        m = ph == p_; o = np.argsort(y[m])
        ax.plot(y[m][o], V[m][o], color=c, lw=1.6, label=n)
    lo, hi = np.interp(50, y[ph == 0][np.argsort(y[ph == 0])], V[ph == 0][np.argsort(y[ph == 0])]), np.interp(50, y[ph == 1][np.argsort(y[ph == 1])], V[ph == 1][np.argsort(y[ph == 1])])
    ax.annotate("", xy=(50, lo), xytext=(50, hi), arrowprops=dict(arrowstyle="<->", color="k", lw=0.9))
    ax.text(53, (lo + hi) / 2, f"{abs(lo-hi)*1000:.0f} mV\nat equal SOC", fontsize=6, va="center")
    ax.set_xlabel("SOC (%)", fontsize=7.5); ax.set_ylabel("Voltage (V)", fontsize=7.5); ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right"); ax.set_title("(a) charge/discharge hysteresis", fontsize=7.5); ax.grid(alpha=0.3)
    # (b) temperature and rate: discharge curves of three cells
    for k, c in (("cell1-25|mid", "tab:green"), ("cell2-35|mid", "tab:olive"), ("cell3-45|mid", "tab:red")):
        r = d[k]; y = 100 * np.array(r["y"]); V = np.array(r["V"]); ph = np.array(r["phase"]); T = np.mean(r["T"])
        m = ph == 1; o = np.argsort(y[m])
        bx.plot(y[m][o], V[m][o], color=c, lw=1.5, label=f"{k.split('|')[0]} ({T:.0f}$^\\circ$C)")
    bx.set_xlabel("SOC (%)", fontsize=7.5); bx.set_ylabel("Voltage (V)", fontsize=7.5); bx.tick_params(labelsize=6.5)
    bx.legend(fontsize=6, frameon=False, loc="lower right"); bx.set_title("(b) cell and temperature spread", fontsize=7.5); bx.grid(alpha=0.3)
    # (c) what the model sees: one cycle of I, V, T and the SOC to be inferred
    r = d["cell2-35|mid"]; n = len(r["y"]); t = np.arange(n)
    cx.plot(t, r["V"], color="tab:blue", lw=1.2, label="$V$ (V)")
    cx2 = cx.twinx(); cx2.plot(t, r["I"], color="tab:orange", lw=1.0, label="$I$ (A)"); cx2.plot(t, np.array(r["T"]) / 10, color="tab:red", lw=1.0, ls="--", label="$T$/10 ($^\\circ$C)")
    cx.set_ylabel("V", fontsize=7); cx2.set_ylabel("I, T/10", fontsize=7); cx.tick_params(labelsize=6.5, labelbottom=False); cx2.tick_params(labelsize=6.5)
    h1, l1 = cx.get_legend_handles_labels(); h2, l2 = cx2.get_legend_handles_labels()
    cx.legend(h1 + h2, l1 + l2, fontsize=5.5, frameon=False, ncol=3, loc="upper center")
    cx.set_ylim(top=max(r["V"]) + 0.34); cx2.set_ylim(top=max(max(r["I"]), max(np.array(r["T"]) / 10)) * 1.45)
    cx.set_title("(c) measured inputs and the target", fontsize=7.5); cx.grid(alpha=0.3)
    dx_.fill_between(t, 0, 100 * np.array(r["y"]), color="k", alpha=0.12)
    dx_.plot(t, 100 * np.array(r["y"]), color="k", lw=1.3)
    dx_.set_ylabel("SOC (%)", fontsize=7); dx_.set_xlabel("window index within cycle", fontsize=7); dx_.tick_params(labelsize=6.5); dx_.grid(alpha=0.3)
    dx_.text(0.02, 0.12, "not measurable: inferred from (c) top", transform=dx_.transAxes, fontsize=6)
    fig.savefig(PAPER / "figures" / "motivation.pdf"); fig.savefig(PAPER / "figures" / "motivation.png", dpi=170); plt.close(fig)


def table_cost() -> str:
    f = OUT / "sacct_times.txt"
    if not f.exists():
        return ""
    rows = []
    for line in f.read_text().splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        job, name, state, secs, tres = parts[:5]
        g = re.search(r"gres/gpu=(\d+)", tres); gpus = int(g.group(1)) if g else 0
        rows.append(dict(job=job, name=name, state=state, hours=float(secs) / 3600, gpus=gpus, gpu_hours=gpus * float(secs) / 3600))
    d = pd.DataFrame(rows)
    d = d[d.state == "COMPLETED"]
    def pick(pat):
        s = d[d.name.str.contains(pat)]
        return s.iloc[-1] if len(s) else None
    items = [("ReBoot pretrain (200 ep., 2 GPU)", pick("pretrain-v4-socproxy")), ("ReBoot finetune (50 ep., 1 GPU)", pick("finetune-eval")),
             ("Supervised Hybrid (50 ep., 1 GPU)", d[d.job == "16208"].iloc[0] if (d.job == "16208").any() else None),
             ("Supervised TCN (50 ep., 1 GPU)", d[d.job == "16196"].iloc[0] if (d.job == "16196").any() else None),
             ("Supervised Transformer (50 ep., 1 GPU)", d[d.job == "16197"].iloc[0] if (d.job == "16197").any() else None),
             ("Supervised LSTM (50 ep., 1 GPU)", d[d.job == "16195"].iloc[0] if (d.job == "16195").any() else None)]
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Training cost on NVIDIA A100 80\,GB (wall-clock includes data loading on a shared node; finetune-eval jobs include evaluation and diagnostics).}",
             r"\label{tab:cost}", r"\begin{tabular}{lrr}", r"\toprule", r"Run & Wall-clock (h) & GPU-hours \\", r"\midrule"]
    for name, r in items:
        if r is not None:
            lines.append(f"{name} & {r.hours:.2f} & {r.gpu_hours:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def fig_decile(dx: pd.DataFrame):
    d = dx[(dx.protocol == "per_cell_random") & (dx.frac == 1.0) & (dx.seed == 0) & ~dx.recipe.str.startswith(("ablation", "v1", "v2")) & ~dx.model.isin(LITERATURE)]
    if len(d) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for _, r in d.sort_values(["model", "recipe"]).iterrows():
        ks = list(r.by_decile.keys()); x = np.arange(len(ks)) * 10 + 5
        axes[0].plot(x, [100 * r.by_decile[k]["mae"] for k in ks], marker="o", ms=3, lw=1.1, label=_name(r).replace(r"\_", "_"))
        axes[1].plot(x, [100 * r.by_decile[k]["p95"] for k in ks], marker="o", ms=3, lw=1.1)
    axes[0].set_ylabel("MAE (% SOC)"); axes[1].set_ylabel("P95 |error| (% SOC)")
    for ax in axes:
        ax.set_xlabel("True SOC (%)"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(PAPER / "figures" / "decile_ext.pdf"); fig.savefig(PAPER / "figures" / "decile_ext.png", dpi=160); plt.close(fig)


def main_extended():
    dx = parse_extended()
    if len(dx) == 0:
        print("no eval_extended*.json yet"); return
    (PAPER / "tables" / "extended_full.tex").write_text(table_extended_full(dx))
    (PAPER / "tables" / "beyond_mae.tex").write_text(table_beyond_mae(dx)); fig_trace_icassp()
    (PAPER / "tables" / "by_temperature.tex").write_text(table_by_group(dx, "by_temperature", r"Test MAE (\% SOC) per operating temperature of the cell (per-cell random split, all labels).", "tab:by_temp", order=["25", "35", "45"]))
    (PAPER / "tables" / "by_phase.tex").write_text(table_by_group(dx, "by_phase", r"Test MAE (\% SOC) on charge vs.\ discharge windows (per-cell random split, all labels).", "tab:by_phase", order=["charge", "discharge"]))
    (PAPER / "tables" / "by_decile.tex").write_text(table_by_group(dx, "by_decile", r"Test MAE (\% SOC) per decile of true SOC (per-cell random split, all labels).", "tab:by_decile"))
    (PAPER / "tables" / "by_phase_p95.tex").write_text(table_by_group(dx, "by_phase", r"$P_{95}$ of the absolute error (\% SOC) on charge vs.\ discharge windows.", "tab:by_phase_p95", order=["charge", "discharge"], metric="p95"))
    (PAPER / "tables" / "low_label_ext.tex").write_text(table_low_label_extended(dx))
    if ((dx.protocol == "cross_battery") & (dx.frac < 1)).any():
        (PAPER / "tables" / "low_label_ext_cross_battery.tex").write_text(table_low_label_extended(dx, "cross_battery"))
    for proto in ("cross_battery", "cross_cycle"):
        if (dx.protocol == proto).any():
            (PAPER / "tables" / f"by_temperature_{proto}.tex").write_text(table_by_group(dx, "by_temperature", rf"Test MAE (\% SOC) per operating temperature, {tex(proto)} protocol, all labels.", f"tab:by_temp_{proto}", order=["25", "35", "45"], protocol=proto))
    df_all = parse()
    if df_all.model.isin(LITERATURE).any():
        (PAPER / "tables" / "literature_reimpl.tex").write_text(table_literature_reimpl(df_all, dx))
    cost = table_cost()
    if cost:
        (PAPER / "tables" / "cost.tex").write_text(cost)
    fig_decile(dx)
    dx.drop(columns=["by_phase", "by_temperature", "by_decile", "by_cell"]).to_csv(OUT / "results_extended.csv", index=False)
    print(f"{len(dx)} extended results -> paper/tables/{{extended_full,by_temperature,by_phase,by_decile,low_label_ext,cost}}.tex")


if __name__ == "__main__":
    main()
    main_extended()
