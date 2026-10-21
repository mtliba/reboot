# ReBoot SLURM batch jobs

These are the job scripts that produced every run reported in the paper. They are kept as they ran,
with the two site-specific paths replaced by environment variables:

```bash
export REBOOT_ROOT=/path/to/reboot          # repo checkout (default: $PWD)
export REBOOT_PYTHON=/path/to/env/bin/python # interpreter with the deps (default: python)
```

Partition names, `--gres`, CPU and memory requests are those of the cluster the work ran on
(A100 80 GB nodes, `GPU-A100-SHORT`); adjust them for yours. Log paths are relative, so submit from
the repository root. Nothing here is required to run ReBoot — each script is a thin wrapper around
the `scripts/*.py` entry points documented in `docs/REPRODUCE.md`.

## Prereqs (run once, on the login or interactive node — not in a batch job)

```bash
ENV=${REBOOT_PYTHON:-python}
cd ${REBOOT_ROOT:-$PWD}

# Parse the 19 HDF5 cells → windows.pkl (per-cell random 80/20 split)
PYTHONPATH=. ${ENV} scripts/prepare_data.py data.raw_dir=real_data

# Convert to memmap for fast training
PYTHONPATH=. ${ENV} scripts/build_memmap.py
```

You only need to redo these if `real_data/` changes or you change `configs/data/default.yaml`.

## 1. Submit the 200-epoch pretrain

```bash
sbatch slurm/pretrain_200.sbatch
# → returns a JOBID. Note it.

squeue -u $USER                 # see queue / running
tail -f outputs/slurm/pretrain_200_<JOBID>.out
```

Reserves: 4 A100s, 32 CPUs, 48 GB RAM, 6 h wall-clock. Actual time ~1.5–3.5 h (27–60 s/epoch depending on node load).

Checkpoints land in `outputs/checkpoints/pretrain_200_<JOBID>/pretrain/`.

## 2. Submit the finetune + evaluate + diagnose chain

Two ways:

```bash
# (a) After pretrain has finished, point at the saved checkpoint:
sbatch --export=ALL,PRETRAIN_CKPT=outputs/checkpoints/pretrain_200_<JOBID>/pretrain/last.ckpt \
       slurm/finetune_eval.sbatch

# (b) Or queue it with a dependency so it auto-runs when pretrain succeeds:
sbatch --dependency=afterok:<PRETRAIN_JOBID> \
       --export=ALL,PRETRAIN_JOBID=<PRETRAIN_JOBID> \
       slurm/finetune_eval.sbatch
```

Reserves: 1 A100, 12 CPUs, 64 GB RAM, 3 h wall-clock. Actual time ~1.5 h.

Outputs:
- `outputs/finetune_<JOBID>/checkpoints/finetune/best.ckpt` (+ `last.ckpt`, top-3 by val/mae)
- `outputs/finetune_<JOBID>/checkpoints/finetune/eval_metrics.json`
- `outputs/finetune_<JOBID>/diagnostics/{01_phi_charge,02_phi_discharge,03_cross_phase,04_orth_hist,05_pred_vs_true}.png`

## Conventions added 2026-09-02

- **Quote checkpoint paths inside Hydra overrides.** Lightning filenames contain `=`
  (`epoch=032-step=9273.ckpt`), which breaks Hydra's override grammar. The sbatch
  files now pass `"finetune.pretrained_ckpt='${PRETRAIN_CKPT}'"`; do the same on the CLI.
- **Validation is carved from TRAIN cycles** (`finetune.val_fraction=0.10`, cycle-atomic).
  The 20 % TEST split is only read by `scripts/evaluate.py` / the baseline's final eval.
  Checkpoint selection (`val/mae`) therefore never sees the test set.
- **`best.ckpt`** — `scripts/finetune.py` and `scripts/baseline_supervised.py` copy the
  best-on-val checkpoint to `<ckpt_dir>/best.ckpt`; `finetune_eval.sbatch` and
  `low_label_sweep.sbatch` evaluate that (falling back to `last.ckpt`).
- **Data clipping** — every scaled channel is clamped to `±data.clip_value` (10.0) at
  load time; see REPORT.txt "Post-mortem of pretrain job 72178".
- **L_phys is active** in finetune (was silently 0 before): the memmap dataset supplies
  the next window + raw I/dt over the stride; `finetune/loss_phys`, `finetune/eta`,
  `finetune/Q_ratio_mean` and `finetune/phys_frac` are logged.
- **Supervised low-label floor**: `sbatch --export=ALL,BASELINE_ENC=hybrid,LABEL_FRACTION=0.01 slurm/baseline_supervised.sbatch`
  (output dir gets a `_frac<X>` suffix; test metrics land in `eval_metrics.json`).
- **Smoke-test any script on CPU** before a long job:
  `trainer.accelerator=cpu trainer.precision=32 trainer.limit_train_batches=2 trainer.limit_val_batches=2 data.num_workers=0 finetune.max_epochs=1`
- **Pretrain stops itself on NaN** (`StopOnNonFinite`); the LR now warms up linearly for
  `pretrain.warmup_epochs` epochs before the cosine decay.

## Evaluation protocols, low-label floor, cycle-grouped batches (2026-09-03)

`data.splits.strategy` selects the protocol on the memmap index space (no data rebuild):

| strategy | train pool | test | windows (train / test) |
|---|---|---|---|
| `per_cell_random` | stored 80 % of each cell's cycles | stored 20 % | 2.30 M / 0.57 M |
| `cross_battery` | all cycles of 16 cells | all cycles of `test_batteries` (cell7-25, cell11-35, cell19-45) | 2.39 M / 0.49 M |
| `cross_temperature` | cells at 35 °C and 45 °C | the four 25 °C cells | 2.30 M / 0.58 M |
| `cross_cycle` | first 70 % of each cell's cycles | the later 30 % (aged) | 2.11 M / 0.77 M |

Validation is always carved from the train pool by cycle. Results land in
`eval_metrics_<protocol>.json` (plain `eval_metrics.json` for per_cell_random).
Every job script takes `PROTOCOL=<strategy>`:

```bash
# ReBoot on a shift protocol (pretrain must have been run with the same protocol so the
# held-out cells are excluded from the unlabeled pool — see pretrain.exclude_holdout)
sbatch --export=ALL,PRETRAIN_CKPT=<ckpt>,PROTOCOL=cross_battery slurm/finetune_eval.sbatch
sbatch --export=ALL,BASELINE_ENC=tcn,PROTOCOL=cross_battery      slurm/baseline_supervised.sbatch

# Low-label: ReBoot sweep (1/5/10/25/100 %) vs supervised floor (1/5/10/25 %), same
# label subsets (same seed per array index) so the comparison is paired
sbatch --export=ALL,PRETRAIN_CKPT=<ckpt> slurm/low_label_sweep.sbatch
sbatch --export=ALL,BASELINE_ENC=tcn     slurm/baseline_low_label.sbatch
sbatch --export=ALL,BASELINE_ENC=hybrid  slurm/baseline_low_label.sbatch

# Pretrain with whole-cycle batches (L_refl gets ~1600 same-cycle pairs per batch instead of ~28)
sbatch --export=ALL slurm/pretrain_200.sbatch   # after setting pretrain.batch_sampler=cycle in the file / CLI
```

`pretrain.batch_sampler=cycle` uses `reboot.data.samplers.CycleGroupedBatchSampler`
(DDP-sharded by cycle, equal batch counts per rank, `use_distributed_sampler=False`).

Run the test-suite / CPU smokes without touching a GPU:
`bash -l slurm/run_on_cpu_node.sh slurm/cpu_checks.sh`

## Useful SLURM commands

```bash
squeue -u $USER                      # my jobs (queued + running)
scontrol show job <JOBID>            # full job detail
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,TRESUsageInTot
scancel <JOBID>                      # kill
scancel -u $USER --name=reboot-pretrain-200   # kill by name
seff <JOBID>                         # post-mortem CPU/MEM efficiency
```

## Robust evaluation: baselines + SSL alternatives + low-label sweep

These extra sbatch files prepare the "Solid" tier described in our discussion: the
ReBoot pretrain (the main 200-epoch job) compared against supervised baselines and
two alternative SSL pretrain recipes, plus a low-label robustness sweep.

### 3. Supervised baselines (no SSL pretrain — 4 jobs, 1 GPU each)

```bash
for enc in lstm tcn transformer hybrid; do
    sbatch --export=ALL,BASELINE_ENC=$enc slurm/baseline_supervised.sbatch
done
```

Each is `Encoder + MLP head` trained directly on SOC labels for 50 epochs. ~1.5 h
each on 1 A100. If 4 GPUs are free they run fully in parallel.

### 4. SSL alternatives (each followed by the same finetune+eval chain)

```bash
# (a) Mask-only pretrain (just L_mask, no other ReBoot losses)
SSL_MAE_JOB=$(sbatch --parsable slurm/pretrain_ssl_mae.sbatch)
sbatch --dependency=afterok:${SSL_MAE_JOB} \
       --export=ALL,PRETRAIN_CKPT=outputs/checkpoints/ssl_mae_${SSL_MAE_JOB}/pretrain/last.ckpt \
       slurm/finetune_eval.sbatch

# (b) Augment-pair contrastive pretrain (naive SimCLR)
SSL_SIMCLR_JOB=$(sbatch --parsable slurm/pretrain_ssl_simclr.sbatch)
sbatch --dependency=afterok:${SSL_SIMCLR_JOB} \
       --export=ALL,PRETRAIN_CKPT=outputs/checkpoints/ssl_simclr_${SSL_SIMCLR_JOB}/pretrain/last.ckpt \
       slurm/finetune_eval.sbatch
```

Same DDP setup as the main pretrain (4 GPUs, 200 epochs, lr=2e-3, bs=2048).

### 5. Low-label sweep (CLAUDE.md §10 low_label protocol)

```bash
# Uses any pretrain checkpoint (ReBoot, SSL-MAE, SSL-SimCLR, or none).
sbatch --export=ALL,PRETRAIN_CKPT=outputs/checkpoints/pretrain_200_72178/pretrain/last.ckpt \
       slurm/low_label_sweep.sbatch
```

This submits a SLURM array of 5 tasks finetuning at label fractions
{1 %, 5 %, 10 %, 25 %, 100 %} and evaluating each on the full 20 % test set.
Each task uses 1 GPU and ~30 min. Per-task results land under
`outputs/lowlabel_<ARRAY_JOB_ID>/frac_<X>/checkpoints/finetune/eval_metrics.json`.

To compare three pretrains (ReBoot vs SSL-MAE vs SSL-SimCLR), submit the array 3×
with three different `PRETRAIN_CKPT` values.

### Comparing runs with the training-curves plotter

```bash
# Side-by-side ReBoot vs SSL-MAE vs SSL-SimCLR pretrain
python scripts/plot_training_curves.py \
    --csv lightning_logs/version_16/metrics.csv \
          lightning_logs/version_NN/metrics.csv \
          lightning_logs/version_MM/metrics.csv \
    --label reboot ssl_mae ssl_simclr \
    --out outputs/figures/compare_ssl_methods
```

## Tuning notes

- **`pretrain.lr=4e-3`** is the linear-scaling-rule LR for the 4× larger effective batch (vs spec default `lr=1e-3` for bs=256). Drop to `1e-3` if you'd rather match the spec verbatim — convergence will be slower but identical in direction.
- **`pretrain.batch_size=2048`** is per-GPU. Effective batch across 4 GPUs is 8192. Drop to 1024 if a future workflow tightens A100 memory.
- **`trainer.max_epochs=200`** is the spec default. 100 captures ~95 % of the gain (see "is 100 enough" discussion in `REPORT.txt`).
- Partition `GPU-A100-SHORT` has a 3-day max but priority is higher than `GPU-A100` (unlimited). `GPU-A100-VERYSHORT` (3 h) is too short for this job.
