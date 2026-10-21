#!/bin/bash
# CPU diagnostic: how well does each L_refl pairing strategy match TRUE SOC?
# Labels are used for measurement only. Run: bash -l slurm/run_on_cpu_node.sh slurm/pair_proxy_check.sh
export LC_ALL=C; cd ${REBOOT_ROOT:-$PWD}
PYTHONPATH=. ${REBOOT_PYTHON:-python} - <<'PY'
import numpy as np, time
from reboot.data.memmap_dataset import MemmapBatteryDataset
from reboot.data.samplers import CycleGroupedBatchSampler
from reboot.data.pair_sampling import make_reflective_pairs
ds = MemmapBatteryDataset("data/processed", mode="eval", filter_split="train", clip_value=10.0, stride=32)
idx_all = ds.indices
cyc = ds.cycle_id[idx_all]
smp = CycleGroupedBatchSampler(cyc, batch_size=2048, seed=0, rank=0, world_size=1)
it = iter(smp)
ps = ds.pseudo_soc
# pseudo_soc range within cycles
sel = idx_all[:200000]
u, inv = np.unique(ds.cycle_id[sel], return_inverse=True)
mn = np.full(len(u), np.inf); mx = np.full(len(u), -np.inf)
np.minimum.at(mn, inv, ps[sel]); np.maximum.at(mx, inv, ps[sel])
print(f"pseudo_soc within-cycle range: min of mins={mn.min():.3f} median min={np.median(mn):.3f}  median max={np.median(mx):.3f} max of maxs={mx.max():.3f}")
# true-SOC relation
print(f"corr(pseudo_soc, true SOC) on 200k windows: {np.corrcoef(ps[sel], ds.y[sel])[0,1]:.3f}")
res = {}
for b in range(20):
    pos = np.array(next(it)); g = idx_all[pos]
    ph, cy, sg, lt, sl, pso, y = ds.phase[g], ds.cycle_id[g], ds.segment_id[g], ds.local_t[g], ds.segment_length[g], ps[g], ds.y[g]
    for strat in ("boundary_symmetric", "soc_proxy"):
        rc, rd = make_reflective_pairs(phases=ph, cycle_ids=cy, segment_ids=sg, local_ts=lt, segment_lengths=sl, strategy=strat, pseudo_socs=pso)
        rc, rd = rc.numpy(), rd.numpy()
        d = np.abs(y[rc] - y[rd])
        same = (cy[rc] == cy[rd]).mean()
        res.setdefault(strat, []).append((rc.size, d.mean(), np.median(d), (d < 0.05).mean(), same))
    # oracle: nearest TRUE-SOC discharge window in same cycle (upper bound)
    c_idx = np.where(ph == 0)[0]; d_idx = np.where(ph == 1)[0]
    dist = np.abs(y[c_idx][:, None] - y[d_idx][None, :]); dist[cy[c_idx][:, None] != cy[d_idx][None, :]] = np.inf
    best = dist.min(axis=1); best = best[np.isfinite(best)]
    res.setdefault("oracle_same_cycle", []).append((best.size, best.mean(), np.median(best), (best < 0.05).mean(), 1.0))
    # false-negative rate: fraction of other discharge windows in batch within 0.05 true SOC of a charge anchor
    fn = (np.abs(y[c_idx][:, None] - y[d_idx][None, :]) < 0.05).mean()
    res.setdefault("negatives_within_0.05_SOC", []).append((d_idx.size, fn, fn, fn, 0))
print(f"\n{'strategy':28s} {'pairs/batch':>11s} {'mean|dSOC|':>10s} {'med|dSOC|':>10s} {'frac<0.05':>9s} {'same cycle':>10s}")
for k, v in res.items():
    a = np.array(v)
    print(f"{k:28s} {a[:,0].mean():11.0f} {a[:,1].mean():10.4f} {a[:,2].mean():10.4f} {a[:,3].mean():9.3f} {a[:,4].mean():10.2f}")
PY
