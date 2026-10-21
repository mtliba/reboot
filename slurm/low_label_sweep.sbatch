#!/bin/bash
# ----------------------------------------------------------------------------
#  Low-label evaluation sweep (CLAUDE.md §10 low_label protocol).
#  Re-finetunes the SAME pretrain checkpoint at five label fractions:
#      1 %, 5 %, 10 %, 25 %, 100 %
#  then evaluates each. This is where SSL pretrain shines — gains over a
#  supervised-only baseline are largest in the low-label regime.
#
#  Submit (you MUST provide a pretrain checkpoint):
#      sbatch --export=ALL,PRETRAIN_CKPT=outputs/checkpoints/pretrain_200_72178/pretrain/last.ckpt \
#             slurm/low_label_sweep.sbatch
#
#  The job is a SLURM array of 5 tasks, each finetuning at one fraction.
#  Wall-clock per task ~30 min (less data → fewer steps); ~2.5 h total
#  if they queue serially, or all-in-parallel if 5 GPUs are free.
# ----------------------------------------------------------------------------

#SBATCH --job-name=reboot-lowlabel
#SBATCH --partition=GPU-A100-SHORT
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-4
#SBATCH --output=outputs/slurm/lowlabel_%A_%a.out
#SBATCH --error=outputs/slurm/lowlabel_%A_%a.err

set -euo pipefail

# Array index → label fraction
FRACTIONS=(0.01 0.05 0.10 0.25 1.0)
FRAC="${FRACTIONS[$SLURM_ARRAY_TASK_ID]}"

if [[ -z "${PRETRAIN_CKPT:-}" || ! -f "$PRETRAIN_CKPT" ]]; then
    echo "ERROR: PRETRAIN_CKPT not set or not found: ${PRETRAIN_CKPT:-<unset>}"
    exit 1
fi

REBOT=${REBOOT_ROOT:-$PWD}
ENV=${REBOOT_PYTHON:-python}
cd "$REBOT"

PROTOCOL="${PROTOCOL:-per_cell_random}"
SUFFIX=""; [[ "$PROTOCOL" != "per_cell_random" ]] && SUFFIX="_${PROTOCOL}"
OUT_DIR="outputs/lowlabel${SUFFIX}_${SLURM_ARRAY_JOB_ID}/frac_${FRAC}"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "  Low-label sweep — task ${SLURM_ARRAY_TASK_ID}"
echo "  Label fraction  : ${FRAC}"
echo "  Pretrain ckpt   : ${PRETRAIN_CKPT}"
echo "  Output dir      : ${OUT_DIR}"
echo "  Start time      : $(date)"
echo "============================================================"

# --- Finetune at this fraction ---
PYTHONPATH=. "${ENV}" scripts/finetune.py \
    "finetune.pretrained_ckpt='${PRETRAIN_CKPT}'" \
    finetune.label_fraction="${FRAC}" \
    finetune.label_fraction_seed="${SLURM_ARRAY_TASK_ID}" \
    data.splits.strategy="${PROTOCOL}" \
    trainer.devices=1 \
    trainer.precision=bf16-mixed \
    finetune.batch_size=512 \
    data.num_workers=10 \
    data.persistent_workers=true \
    paths.checkpoint_dir="${OUT_DIR}/checkpoints" \
    wandb.mode=disabled

FT_CKPT="${OUT_DIR}/checkpoints/finetune/best.ckpt"
[[ -f "$FT_CKPT" ]] || FT_CKPT="${OUT_DIR}/checkpoints/finetune/last.ckpt"

# --- Evaluate on the SAME (full) test set ---
PYTHONPATH=. "${ENV}" scripts/evaluate.py \
    "+eval.ckpt='${FT_CKPT}'" \
    "+eval.protocol=${PROTOCOL}" \
    data.splits.strategy="${PROTOCOL}" \
    trainer.devices=1 \
    finetune.batch_size=512 \
    data.num_workers=10 \
    wandb.mode=disabled

echo ""
echo "Done. End: $(date)"
echo "Metrics: ${OUT_DIR}/checkpoints/finetune/eval_metrics.json"
