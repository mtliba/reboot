#!/bin/bash
# Learned physics parameters (Coulombic efficiency eta, per-cell capacity Q_c) from
# finetuned checkpoints vs. nominal capacities. CPU only.
#   bash -l slurm/run_on_cpu_node.sh slurm/physics_params.sh 15
export LC_ALL=C; cd ${REBOOT_ROOT:-$PWD}
PYTHONPATH=. ${REBOOT_PYTHON:-python} - <<'PY'
import json, glob, torch, numpy as np, pandas as pd
extra = json.load(open("data/processed/windows_extra.json"))
ids = extra["battery_ids"]; nominal = {int(k): v for k, v in extra["capacities_As"].items()}
runs = {"v4 100%": "outputs/finetune_16260/checkpoints/finetune/best.ckpt",
        "v3 100%": "outputs/finetune_16216/checkpoints/finetune/best.ckpt",
        "v4 1%":   "outputs/lowlabel_16263/frac_0.01/checkpoints/finetune/best.ckpt",
        "v4 cross_battery 100%": "outputs/finetune_cross_battery_16295/checkpoints/finetune/best.ckpt"}
out = {"battery_ids": ids, "nominal_As": {ids[k]: v for k, v in nominal.items()}, "runs": {}}
for name, ck in runs.items():
    sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
    eta = float(0.9 + 0.1 * torch.sigmoid(sd["phys_loss.eta_raw"]))
    Q = torch.exp(sd["phys_loss.log_Q"]).numpy()
    ratio = {ids[i]: float(Q[i] / nominal[i]) for i in range(len(ids)) if i in nominal}
    out["runs"][name] = {"eta": eta, "Q_As": {ids[i]: float(Q[i]) for i in range(len(ids))}, "Q_over_nominal": ratio}
    r = np.array(list(ratio.values()))
    print(f"{name:24s} eta={eta:.4f}  Q/Q_nom: mean={r.mean():.3f} min={r.min():.3f} max={r.max():.3f}")
# eta trajectory from the v4 100% finetune CSV log
for f in glob.glob("outputs/finetune_16260/checkpoints/csv_logs/version_*/metrics.csv") + glob.glob("lightning_logs/version_*/metrics.csv"):
    df = pd.read_csv(f)
    if "finetune/eta" in df.columns and df["finetune/eta"].notna().sum() >= 40:
        e = df[df["finetune/eta"].notna()]
        out["eta_trajectory_v4"] = e["finetune/eta"].round(5).tolist()
        out["Q_ratio_trajectory_v4"] = e["finetune/Q_ratio_mean"].round(5).tolist()
        print("eta trajectory:", e["finetune/eta"].iloc[[0, 9, 24, -1]].round(4).tolist(), " Q_ratio:", e["finetune/Q_ratio_mean"].iloc[[0, 9, 24, -1]].round(4).tolist())
        break
json.dump(out, open("outputs/physics_params.json", "w"), indent=1)
print("wrote outputs/physics_params.json")
PY
