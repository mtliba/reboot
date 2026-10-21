#!/bin/bash
# Cross-phase retrieval diagnostic on PRETRAIN checkpoints: for each charge
# window, find the nearest DISCHARGE window in z_s (cosine) and report the true
# SOC gap. Also linear-probe R^2 of SOC from z_s. Labels used for measurement only.
export LC_ALL=C; cd ${REBOOT_ROOT:-$PWD}
PYTHONPATH=. ${REBOOT_PYTHON:-python} - <<'PY'
import numpy as np, torch, warnings; warnings.filterwarnings("ignore")
from reboot.data.memmap_dataset import MemmapBatteryDataset
from reboot.models import ReBootModel
torch.set_num_threads(8)
ds = MemmapBatteryDataset("data/processed", mode="eval", filter_split="test", clip_value=10.0, stride=32)
rng = np.random.default_rng(0)
sel = np.sort(rng.choice(len(ds), 12000, replace=False))
x = torch.stack([ds[int(i)]["x"] for i in sel]); g = ds.indices[sel]
y, ph, cyc = ds.y[g], ds.phase[g], ds.cycle_id[g]
enc_kw = {"nhead": 4, "num_transformer_layers": 2, "ffn_dim": 256}
runs = {"v2 random batches": "outputs/checkpoints/pretrain_200_16206/pretrain/last.ckpt",
        "v3 cycle batches": "outputs/checkpoints/pretrain_200_16215/pretrain/last.ckpt",
        "v4 soc_proxy+mask": "outputs/checkpoints/pretrain_200_16259/pretrain/last.ckpt"}
print(f"{'run':20s} {'|z_s|':>6s} {'ani':>5s} | {'NN dSOC c->d':>12s} {'rand dSOC':>9s} {'same-SOC@0.05':>13s} | {'R2 lin(z_s)':>11s} {'R2 lin(z_d)':>11s} {'R2 lin(z)':>9s}")
def r2(Z, y):
    Z = np.c_[Z, np.ones(len(Z))]; n = len(y)//2
    w, *_ = np.linalg.lstsq(Z[:n], y[:n], rcond=None); p = Z[n:] @ w
    return 1 - ((p - y[n:])**2).sum() / ((y[n:] - y[n:].mean())**2).sum()
for name, ck in runs.items():
    sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
    m = ReBootModel(in_channels=7, W=100, encoder_type="hybrid", encoder_kwargs=enc_kw)
    m.load_state_dict({k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")}); m.eval()
    with torch.no_grad():
        zs, zd = [], []
        for i in range(0, len(x), 1000):
            a, b = m.encode(x[i:i+1000]); zs.append(a); zd.append(b)
        zs = torch.cat(zs).numpy(); zd = torch.cat(zd).numpy()
    zc = zs - zs.mean(0); ev = np.linalg.svd(zc, compute_uv=False)**2; ani = ev[0]/ev.sum()
    c, d = np.where(ph == 0)[0], np.where(ph == 1)[0]
    zn = zc / (np.linalg.norm(zc, axis=1, keepdims=True) + 1e-8)
    sim = zn[c] @ zn[d].T
    nn_ = d[sim.argmax(1)]
    nn_gap = np.abs(y[c] - y[nn_]).mean()
    rand_gap = np.abs(y[c] - y[rng.choice(d, len(c))]).mean()
    hit = (np.abs(y[c] - y[nn_]) < 0.05).mean()
    print(f"{name:20s} {np.linalg.norm(zs,axis=1).mean():6.2f} {ani:5.2f} | {nn_gap:12.4f} {rand_gap:9.4f} {hit:13.3f} | {r2(zs,y):11.3f} {r2(zd,y):11.3f} {r2(np.c_[zs,zd],y):9.3f}")
print("\nNN dSOC: mean |true SOC gap| between a charge window and its nearest discharge window in centred z_s (lower = aligned; rand = chance).")
print("ani: fraction of z_s variance in the top principal direction. R2: linear probe of SOC (train/test halves of this sample).")
PY
