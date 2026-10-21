#!/bin/bash
export LC_ALL=C; cd ${REBOOT_ROOT:-$PWD}
# finetune logs living in lightning_logs (runs before the per-run CSVLogger fix)
for v in lightning_logs/version_*; do f=$v/metrics.csv; [ -f "$f" ] || continue
  if head -1 "$f" | grep -q "val/mae" && [ "$(wc -l < "$f")" -gt 100 ]; then cp "$f" outputs/export/logs/finetune_$(basename $v).csv; fi; done
ls outputs/export/logs | wc -l
PYTHONPATH=. ${REBOOT_PYTHON:-python} - <<'PY'
import numpy as np, glob, time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
for f in sorted(glob.glob("outputs/export/embeddings/*.npz")):
    d = dict(np.load(f, allow_pickle=True))
    if "tsne_s" in d: continue
    rng = np.random.default_rng(0); sub = np.sort(rng.choice(len(d["y"]), 6000, replace=False))
    t = time.time()
    for key, name in (("z_s", "tsne_s"), ("z_d", "tsne_d")):
        X = d[key][sub]; X = PCA(n_components=min(30, X.shape[1])).fit_transform(X)
        d[name] = TSNE(n_components=2, perplexity=40, init="pca", random_state=0).fit_transform(X).astype(np.float32)
    d["tsne_idx"] = sub
    np.savez_compressed(f, **d); print(f, f"tsne done in {time.time()-t:.0f}s")
PY
