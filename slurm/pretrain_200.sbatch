#!/bin/bash
# ----------------------------------------------------------------------------
#  ReBoot — 200-epoch pretrain on 2× A100 GPUs (DDP)
#
#  2026-09-03: 2 GPUs (was 4) to be a better citizen on the shared node, fp32
#  instead of bf16-mixed and lr=1e-3 (spec default, was 2e-3) after two runs
#  diverged to NaN at ~13-14k steps; plus the gradient-spike guard in
#  PretrainModule. The model is data-loading bound, so fp32 costs ~nothing.
#
#  Submit:
#      sbatch slurm/pretrain_200.sbatch
#  Check status:
#      squeue -u $USER
#      squeue -j <JOBID>
#  Stream logs:
#      tail -f outputs/slurm/pretrain_200_<JOBID>.out
#  Cancel:
#      scancel <JOBID>
#
#  Wall-clock estimate: ~10 h pretrain (at ~3 min/epoch on 4 A100 + memmap +
#  vectorized collate). 14 h limit chosen with margin for the first-epoch
#  warmup + checkpoint I/O.
# ----------------------------------------------------------------------------

#SBATCH --job-name=reboot-pretrain-200
#SBATCH --partition=GPU-A100-SHORT
#SBATCH --time=09:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2                    # ← one task PER GPU so srun launches 2 ranks
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12            # ← 12 CPUs each (24 total); data-loading bound
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:2
#SBATCH --output=outputs/slurm/pretrain_200_%j.out
#SBATCH --error=outputs/slurm/pretrain_200_%j.err

set -euo pipefail

# -------- Variant knobs (env) --------
#   BATCH_SAMPLER=random|cycle      pretrain.batch_sampler (cycle = whole-cycle batches)
#   PROTOCOL=per_cell_random|cross_battery|cross_temperature|cross_cycle
#                                   held-out cells are excluded from the unlabeled pool
#   sbatch --export=ALL,BATCH_SAMPLER=cycle,PROTOCOL=cross_battery slurm/pretrain_200.sbatch
BATCH_SAMPLER="${BATCH_SAMPLER:-random}"
PROTOCOL="${PROTOCOL:-per_cell_random}"
PAIR_STRATEGY="${PAIR_STRATEGY:-boundary_symmetric}"   # | soc_proxy
REFL_EPS="${REFL_EPS:-0.0}"                           # pretrain.refl_neg_mask_eps
EXTRA_ARGS="${EXTRA_ARGS:-}"                          # further Hydra overrides (ablations), space-separated
EPOCHS="${EPOCHS:-200}"                               # trainer.max_epochs (ablations use 100)

# -------- Environment --------
REBOT=${REBOOT_ROOT:-$PWD}
ENV=${REBOOT_PYTHON:-python}
cd "$REBOT"
mkdir -p outputs/slurm outputs/checkpoints

# -------- Banner --------
echo "============================================================"
echo "  ReBoot pretrain 200 epochs (DDP × 2 GPUs, fp32)"
echo "  Job ID         : ${SLURM_JOB_ID}"
echo "  Job name       : ${SLURM_JOB_NAME}"
echo "  Batch sampler  : ${BATCH_SAMPLER}"
echo "  Protocol       : ${PROTOCOL}"
echo "  Pair strategy  : ${PAIR_STRATEGY}  refl_neg_mask_eps=${REFL_EPS}"
echo "  Extra args     : ${EXTRA_ARGS}"
echo "  Epochs         : ${EPOCHS}"
echo "  Node           : ${SLURMD_NODENAME}"
echo "  Partition      : ${SLURM_JOB_PARTITION}"
echo "  Allocated GPUs : ${CUDA_VISIBLE_DEVICES:-(slurm-managed)}"
echo "  Allocated CPUs : ${SLURM_CPUS_PER_TASK}"
echo "  Allocated MEM  : ${SLURM_MEM_PER_NODE}"
echo "  Working dir    : $(pwd)"
echo "  Python         : ${ENV}"
echo "  Start time     : $(date)"
echo "  Git SHA        : $(git rev-parse --short HEAD 2>/dev/null || echo 'not-a-git-repo')"
echo "============================================================"
echo ""

# -------- Sanity checks --------
test -f data/processed/windows_data.npy || {
    echo "ERROR: data/processed/windows_data.npy not found."
    echo "Run: PYTHONPATH=. ${ENV} scripts/prepare_data.py data.raw_dir=real_data"
    echo "Then: PYTHONPATH=. ${ENV} scripts/build_memmap.py"
    exit 1
}
test -f data/processed/windows_meta.npz || { echo "ERROR: windows_meta.npz missing"; exit 1; }

echo "Dataset: $(ls -lh data/processed/windows_data.npy | awk '{print $5}')  memmap;  $(ls -lh data/processed/windows_meta.npz | awk '{print $5}') meta"
echo ""

# -------- Pretrain --------
# Hyperparams baked in below:
#   trainer.max_epochs=200          spec default (CLAUDE.md §9)
#   trainer.devices=2               DDP across the 2 allocated A100s (cluster
#                                   etiquette: never more than 2 GPUs per job)
#   trainer.strategy=ddp            vanilla; unused params are frozen in
#                                   PretrainModule so find_unused isn't needed
#   trainer.precision=32            bf16-mixed runs diverged twice (see REPORT)
#   pretrain.batch_size=2048        per-GPU. Effective bs = 4096.
#   pretrain.lr=1e-3                spec default; warmup_epochs=10 then cosine
#   data.num_workers=10             per rank -> 20 worker processes
srun --kill-on-bad-exit=1 \
     bash -c 'PYTHONPATH=${REBOOT_ROOT:-$PWD} '"${ENV}"' ${REBOOT_ROOT:-$PWD}/scripts/pretrain.py \
    trainer.max_epochs='"${EPOCHS}"' \
    trainer.devices=2 \
    +trainer.num_nodes=1 \
    trainer.strategy=ddp \
    trainer.precision=32 \
    pretrain.batch_size=2048 \
    pretrain.lr=1e-3 \
    pretrain.batch_sampler='"${BATCH_SAMPLER}"' \
    data.splits.strategy='"${PROTOCOL}"' \
    data.pair_strategy='"${PAIR_STRATEGY}"' \
    pretrain.refl_neg_mask_eps='"${REFL_EPS}"' \
    '"${EXTRA_ARGS}"' \
    data.num_workers=10 \
    data.persistent_workers=true \
    paths.checkpoint_dir=outputs/checkpoints/pretrain_200_'"${SLURM_JOB_ID}"' \
    wandb.mode=disabled'

# -------- Done --------
echo ""
echo "============================================================"
echo "  End time : $(date)"
echo "  Exit code: $?"
echo "  Checkpoints in: outputs/checkpoints/pretrain_200_${SLURM_JOB_ID}/pretrain/"
echo "============================================================"
