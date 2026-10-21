#!/bin/bash
# Export what the supplementary figures need: latent embeddings of test windows for several
# pretrained encoders, and the pretrain/finetune CSV logs (lightning_logs is not synced).
#   bash -l slurm/run_on_cpu_node.sh slurm/export_qualitative.sh 40
export LC_ALL=C; cd ${REBOOT_ROOT:-$PWD}
mkdir -p outputs/export/logs outputs/export/embeddings
# 1) logs: pretrain runs (v2 16206 -> version_30? map by mtime is unreliable; copy all pretrain CSVs with a job tag)
for v in lightning_logs/version_*; do
  f=$v/metrics.csv; [ -f "$f" ] || continue
  if head -1 "$f" | grep -q "pretrain/loss_total_epoch"; then n=$(wc -l < "$f"); [ "$n" -gt 2000 ] && cp "$f" outputs/export/logs/pretrain_$(basename $v).csv; fi
done
for d in outputs/finetune_16260 outputs/finetune_16216 outputs/lowlabel_16263/frac_0.01 outputs/checkpoints/baseline_hybrid_frac1.0_16208 outputs/base_lowlabel_16222/hybrid_frac_0.01; do
  f=$(ls $d/checkpoints/csv_logs/version_*/metrics.csv $d/csv_logs/version_*/metrics.csv $d/baseline_hybrid/csv_logs/version_*/metrics.csv 2>/dev/null | head -1)
  [ -n "$f" ] && cp "$f" outputs/export/logs/finetune_$(echo $d | tr '/' '_').csv
done
ls outputs/export/logs
# 2) embeddings
PYTHONPATH=. ${REBOOT_PYTHON:-python} - <<'PY'
import numpy as np, torch, warnings; warnings.filterwarnings("ignore")
from reboot.data.memmap_dataset import MemmapBatteryDataset
from reboot.models import ReBootModel
torch.set_num_threads(8)
ds = MemmapBatteryDataset("data/processed", mode="eval", filter_split="test", clip_value=10.0, stride=32)
rng = np.random.default_rng(0)
sel = np.sort(rng.choice(len(ds), 20000, replace=False))
x = torch.stack([ds[int(i)]["x"] for i in sel]); g = ds.indices[sel]
meta = dict(y=ds.y[g], phase=ds.phase[g], battery_idx=ds.battery_idx[g], cycle_id=ds.cycle_id[g], local_t=ds.local_t[g], segment_length=ds.segment_length[g], battery_ids=np.array(ds.extra["battery_ids"]))
enc_kw = {"nhead": 4, "num_transformer_layers": 2, "ffn_dim": 256}
runs = {"v2_random": "outputs/checkpoints/pretrain_200_16206/pretrain/last.ckpt", "v3_cycle": "outputs/checkpoints/pretrain_200_16215/pretrain/last.ckpt",
        "v4_socproxy": "outputs/checkpoints/pretrain_200_16259/pretrain/last.ckpt", "maeonly100": "outputs/checkpoints/pretrain_200_16304/pretrain/last.ckpt",
        "v4_finetuned": "outputs/finetune_16260/checkpoints/finetune/best.ckpt"}
for name, ck in runs.items():
    sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
    m = ReBootModel(in_channels=7, W=100, encoder_type="hybrid", encoder_kwargs=enc_kw)
    m.load_state_dict({k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")}, strict=False); m.eval()
    phi_w = sd.get("ord_loss.phi.weight"); phi_b = sd.get("ord_loss.phi.bias")
    with torch.no_grad():
        zs, zd, h = [], [], []
        for i in range(0, len(x), 1000):
            hh = m.encoder(x[i:i+1000]); zs.append(m.g_s(hh)); zd.append(m.g_d(hh)); h.append(hh)
        zs = torch.cat(zs); zd = torch.cat(zd); h = torch.cat(h)
        s = (zs @ phi_w.T + phi_b).squeeze(-1).numpy() if phi_w is not None else np.full(len(zs), np.nan)
    np.savez_compressed(f"outputs/export/embeddings/{name}.npz", z_s=zs.numpy(), z_d=zd.numpy(), h=h.numpy(), phi=s, **meta)
    print(name, "saved", zs.shape)
PY
# 3) one full test cycle per temperature for trace plots: window indices in order
PYTHONPATH=. ${REBOOT_PYTHON:-python} - <<'PY'
import numpy as np, json
from reboot.data.memmap_dataset import MemmapBatteryDataset
ds = MemmapBatteryDataset("data/processed", mode="eval", filter_split="test", clip_value=10.0, stride=32)
ids = ds.extra["battery_ids"]; out = {}
for cell in ["cell1-25", "cell2-35", "cell3-45"]:
    b = ids.index(cell); idx = ds.indices[ds.battery_idx[ds.indices] == b]
    cyc = ds.cycle_id[idx]; u = np.unique(cyc); pick = u[len(u)//2]     # a mid-life test cycle
    w = idx[cyc == pick]
    out[cell] = dict(cycle_id=int(pick), test_positions=np.searchsorted(ds.indices, w).tolist(), global_idx=w.tolist(), local_t=ds.local_t[w].tolist(), phase=ds.phase[w].tolist(), y=ds.y[w].tolist(), segment_id=ds.segment_id[w].tolist())
json.dump(out, open("outputs/export/trace_cycles.json", "w"))
print({k: len(v["global_idx"]) for k, v in out.items()})
PY
du -sh outputs/export
