#!/bin/bash
# Export raw-unit signals of a few whole test cycles for the motivation figure.
#   bash -l slurm/run_on_cpu_node.sh slurm/export_motivation.sh 20
export LC_ALL=C; cd ${REBOOT_ROOT:-$PWD}
PYTHONPATH=. ${REBOOT_PYTHON:-python} - <<'PY'
import json, numpy as np
from reboot.data.memmap_dataset import MemmapBatteryDataset
from reboot.data.transforms import Scaler, IDX_I, IDX_V, IDX_T, IDX_DT, T_MIN_C, T_MAX_C
ds = MemmapBatteryDataset("data/processed", mode="eval", filter_split="test", clip_value=None, stride=32)
sc = Scaler.load("outputs/scalers.pkl"); ids = list(ds.extra["battery_ids"])
out = {}
for cell in ["cell1-25", "cell2-35", "cell3-45"]:
    b = ids.index(cell); idx = ds.indices[ds.battery_idx[ds.indices] == b]
    cyc = ds.cycle_id[idx]; u = np.unique(cyc)
    for tag, pick in (("mid", u[len(u)//2]), ("early", u[max(1, len(u)//20)]), ("late", u[-2])):
        w = idx[cyc == pick]
        if w.size < 10: continue
        X = np.stack([ds[int(np.searchsorted(ds.indices, i))]["x"].numpy() for i in w])   # (n, W, C) scaled
        I = X[:, :, IDX_I] * max(sc.I_iqr, 1e-6) + sc.I_median
        V = X[:, :, IDX_V] * max(sc.V_iqr, 1e-6) + sc.V_median
        T = X[:, :, IDX_T] * (T_MAX_C - T_MIN_C) + T_MIN_C
        out[f"{cell}|{tag}"] = dict(cycle=int(pick), y=ds.y[w].tolist(), phase=ds.phase[w].tolist(),
                                    I=I.mean(1).tolist(), V=V.mean(1).tolist(), T=T.mean(1).tolist(),
                                    V_seq=V[:, ::10].tolist(), I_seq=I[:, ::10].tolist(), T_seq=T[:, ::10].tolist())
        print(cell, tag, "cycle", int(pick), "windows", w.size, "V", round(float(V.min()),3), round(float(V.max()),3), "I", round(float(I.min()),2), round(float(I.max()),2), "T", round(float(T.mean()),1))
json.dump(out, open("outputs/export/motivation_cycles.json", "w"))
print("wrote outputs/export/motivation_cycles.json")
PY
