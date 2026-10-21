---
layout: default
title: Reproduction guide
description: "Every number in the paper, with the command that produced it."
---

# Reproduction guide

Every number in the paper, with the command that produced it.

All runs were executed on one node of a SLURM cluster with NVIDIA A100 80 GB GPUs
(`slurm/*.sbatch`), but nothing here requires SLURM: each job script is a thin wrapper around the
`scripts/*.py` entry points shown below. Hydra composes `configs/`, so **the committed defaults are
the published recipe** — the overrides below are only what differs from them.

---

## 0. Environment

```bash
uv venv && uv pip install -e ".[dev]"      # Python ≥ 3.10, PyTorch ≥ 2.1, Lightning ≥ 2.1
pytest -q                                   # CPU-only, needs no data
```

`pytest` runs one test per loss against the verification blocks in the specification, plus the
data-pipeline invariants (a window never crosses a phase boundary; SOC is absent from every
pretraining code path; scalers are fitted on training cells only).

## 1. Data

```bash
python scripts/prepare_data.py data.raw_dir=/path/to/raw   # segment → window → scale → split
python scripts/build_memmap.py                              # → data/processed/windows_{data,meta}
```

`prepare_data.py` phase-segments each cell by current (`|I| ≤ 0.05 A` is rest, hysteresis filter of
10 samples), drops rest segments, then takes `W = 100`, stride 32 windows **inside** segments so no
window crosses a phase boundary. Scalers (robust for `I`, `V`; physical min-max for `T`; `log1p` for
`Δt`) are fitted on training cells only and written to `outputs/scalers.pkl`. Derived channels
`dV/dt, dT/dt, Q_cum` bring `C` to 7 and are clipped to ±10 after scaling.

Splits are cycle-atomic, so both phases of a cycle always land in the same split:

| `data.splits.strategy` | train | test |
|---|---|---|
| `per_cell_random` (primary) | 80 % of each cell's cycles (10 % of those for validation) | remaining 20 % |
| `cross_battery` | all cells except 7, 11, 19 | cells 7, 11, 19 (one per temperature) |
| `cross_cycle` | first 70 % of each cell's cycles | final 30 % (later life) |
| `cross_temperature` | cells not at the held-out temperature | cells at it |

Without the raw measurements, `python scripts/generate_synthetic_data.py` writes a stand-in with the
same schema so the whole pipeline, all tests and the ablation plumbing run end to end.

## 2. Pretraining (Stage 1, no labels)

**Published model — 200 epochs, Coulomb pairing, near-state negative masking, cycle-grouped batches:**

```bash
python scripts/pretrain.py \
    trainer.max_epochs=200 trainer.devices=2 trainer.strategy=ddp trainer.precision=32 \
    pretrain.batch_size=2048 pretrain.lr=1e-3 \
    pretrain.batch_sampler=cycle \
    data.pair_strategy=soc_proxy \
    pretrain.refl_neg_mask_eps=0.05
# slurm:  sbatch --export=ALL,BATCH_SAMPLER=cycle,PAIR_STRATEGY=soc_proxy,REFL_EPS=0.05 \
#                 slurm/pretrain_200.sbatch
```

≈ 9 h on 2× A100. Effective batch 4096. Loss weights `mask 1.0 / ord 1.0 / refl 0.5 / smooth 0.1 /
orth 0.01`; margin 0.5; temperature 0.1; 15 % timestep + 10 % channel masking; AdamW `lr 1e-3`,
`wd 1e-4`, 10-epoch linear warm-up then cosine, gradient clipping 1.0.

**Variants in the paper** (each is the same command with the stated change):

| row in the paper | change |
|---|---|
| ReBoot, position-based pairing | `data.pair_strategy=boundary_symmetric pretrain.refl_neg_mask_eps=0.0` |
| Ablations and SSL baselines | `trainer.max_epochs=100` (matched budget) |
| − `L_refl` / − `L_ord` | `pretrain.losses.use_refl=false` / `pretrain.losses.use_ord=false` |
| `L_mask` only | `scripts/baseline_ssl_mae.py` (equivalently: all other `use_*=false`) |
| SimCLR-style | `scripts/baseline_ssl_simclr.py` |
| Single latent | `model.single_latent=true` |
| Raw `I, V, T, Δt` only | `data=raw4 trainer.max_epochs=100` — `input_channels=[0,1,2,3]`, so `dV/dt`, `dT/dt` and `Q_cum` are still computed for pairing but never reach the encoder |
| Cross-battery protocol | `data.splits.strategy=cross_battery` — held-out cells are dropped from the **unlabelled** pool too (`pretrain.exclude_holdout=true`), so the comparison is non-transductive |
| Pretraining-pool sweep | `pretrain.pool_fraction={0.1,0.25,0.5,1.0}` |

## 3. Fine-tuning and evaluation (Stage 2)

```bash
python scripts/finetune.py \
    "finetune.pretrained_ckpt='outputs/checkpoints/<run>/pretrain/last.ckpt'" \
    finetune.label_fraction=0.01 finetune.label_fraction_seed=0 seed=42 \
    trainer.devices=1 trainer.precision=bf16-mixed finetune.batch_size=512

python scripts/evaluate.py "+eval.ckpt='<finetune ckpt>'" +eval.protocol=per_cell_random
```

50 epochs, AdamW, parameter groups `1e-4` (encoder) / `1e-3` (head), `Huber(δ=0.05)` plus
`0.01 · L_phys`. `finetune.mode=linear_probe` freezes the encoder and both projection heads.

The quotes around `finetune.pretrained_ckpt` are required — Hydra's override grammar rejects a bare
`=` inside a path such as `epoch=032-....ckpt`.

**The whole label sweep at once** (the five fractions run concurrently on a single A100; each needs
≈ 6 GB and is data-loading bound):

```bash
sbatch --export=ALL,PRETRAIN_CKPT=<ckpt> slurm/low_label_sweep_packed.sbatch
# repeats:  add SEED_OFFSET=100 / SEED_OFFSET=200
# protocol: add PROTOCOL=cross_battery
```

For a given seed and label fraction, every method receives the **same** labelled subset: the subset
is drawn by `finetune.label_fraction_seed`, independently of the model.

**Supervised controls** — the identical encoder from random initialisation, and the five
re-implemented published SOC architectures (which use their original `I, V, T` inputs):

```bash
python scripts/baseline_supervised.py +baseline=supervised_hybrid finetune.label_fraction=0.01
python scripts/baseline_supervised.py +baseline=supervised_tcn
# published architectures, each on its original I, V, T inputs:
#   supervised_chemali_dnn   [6]   supervised_chemali_lstm [4]   supervised_yang_gru [5]
#   supervised_hannan_transformer [7]            supervised_li_cnnbilstm [8]
# slurm: sbatch slurm/baseline_packed.sbatch     (all of them, packed onto one GPU)
```

## 4. Tables and figures

```bash
python scripts/collect_results.py          # scans outputs/**/eval_metrics*.json → results/*.csv,
                                           #   paper tables (.tex) and paper figures
sbatch slurm/reevaluate_all.sbatch         # extended metrics (P50/95/99, R², by phase /
                                           #   temperature / cell / SOC decile) + saved predictions
                                           #   one checkpoint at a time:
                                           #   python scripts/reevaluate.py "+eval.ckpt='<ckpt>'" \
                                           #       "+eval.protocol=per_cell_random"
python scripts/make_evidence_figures.py    # docs/static/figures/evidence_*.{png,pdf}
```

| paper artefact | produced by |
|---|---|
| Fig. 1 (challenges) | `collect_results.py::fig_motivation` ← `results/evidence/trace_cycles.json` |
| Fig. 2 (overview) | hand-drawn schematic, visually refined (see the paper's acknowledgment) |
| Fig. 3 (label efficiency + input control) | `collect_results.py::fig_flagship_icassp` |
| Table 1 | `collect_results.py::table_flagship(compact=True)` → `results/tables/flagship_compact.tex` |
| Table 2 | `collect_results.py::table_ablation_compact` → `results/tables/ablation_compact.tex` |
| Evidence figures | `make_evidence_figures.py` → `docs/EVIDENCE.md` |

## 5. Diagnostics used in the paper and on the evidence page

```bash
bash slurm/pair_proxy_check.sh      # CPU: how far apart in true SOC are the pairs L_refl receives
bash slurm/retrieval_check.sh       # cross-phase retrieval gap + linear probes on z_s / z_d
bash slurm/physics_params.sh        # read η and per-cell Q̂ out of fine-tuned checkpoints
python scripts/diagnose.py          # φ(z_s) vs SOC, cross-phase cosine, orthogonality histogram
```

These use reference SOC **only to score** representations that were learned without it.

## 6. What to expect

Single-seed numbers move by roughly ±0.01 % SOC at 100 % labels and ±0.03 at 1 %; the principal
comparisons in Table 1 are averaged over three label subsets and initialisations where repeats
exist. Pretraining is fp32 and deterministic up to DDP reduction order and dataloader worker
scheduling; exact bit-reproduction across a different GPU count is not expected, and the
conclusions do not depend on it.

Rough budget for the full set of runs in the paper: ~9 h × 6 pretrains (2× A100) plus ~2.5 h ×
~25 packed fine-tune sweeps (1× A100).
