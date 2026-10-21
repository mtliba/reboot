#!/bin/bash
# ----------------------------------------------------------------------------
#  ReBoot — finetune + evaluate + diagnose, follow-up to pretrain_200.sbatch
#
#  NOTE: checkpoint paths are single-quoted inside the Hydra override because
#  Lightning filenames contain "=" (epoch=032-step=9273.ckpt), which otherwise
#  breaks Hydra's override grammar.
#
#  Submit AFTER the pretrain job finishes, pointing at its checkpoint:
#      sbatch --export=ALL,PRETRAIN_CKPT=outputs/checkpoints/pretrain_200_<JOBID>/pretrain/last.ckpt \
#             slurm/finetune_eval.sbatch
#
#  Or, to chain dependent on a running pretrain job:
#      sbatch --dependency=afterok:<PRETRAIN_JOBID> \
#             --export=ALL,PRETRAIN_JOBID=<PRETRAIN_JOBID> \
#             slurm/finetune_eval.sbatch
#
#  Wall-clock estimate (single GPU, memmap):
#      Finetune 50 epochs    ~1 h
#      Evaluate              ~5 min
#      Diagnose plots        ~5 min
#  ~ 1.5 h total. 3 h limit chosen with margin.
# ----------------------------------------------------------------------------

#SBATCH --job-name=reboot-finetune-eval
#SBATCH --partition=GPU-A100-SHORT
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=outputs/slurm/finetune_eval_%j.out
#SBATCH --error=outputs/slurm/finetune_eval_%j.err

set -euo pipefail

REBOT=${REBOOT_ROOT:-$PWD}
ENV=${REBOOT_PYTHON:-python}
cd "$REBOT"
mkdir -p outputs/slurm

# Resolve the pretrain checkpoint
if [[ -z "${PRETRAIN_CKPT:-}" && -n "${PRETRAIN_JOBID:-}" ]]; then
    PRETRAIN_CKPT="outputs/checkpoints/pretrain_200_${PRETRAIN_JOBID}/pretrain/last.ckpt"
fi
if [[ -z "${PRETRAIN_CKPT:-}" || ! -f "$PRETRAIN_CKPT" ]]; then
    echo "ERROR: PRETRAIN_CKPT not set or file does not exist: ${PRETRAIN_CKPT:-<unset>}"
    echo "Set via:  sbatch --export=ALL,PRETRAIN_CKPT=path/to/last.ckpt ..."
    exit 1
fi

# Evaluation protocol (CLAUDE.md §10): per_cell_random | cross_battery | cross_temperature | cross_cycle
#   sbatch --export=ALL,PRETRAIN_CKPT=...,PROTOCOL=cross_battery slurm/finetune_eval.sbatch
PROTOCOL="${PROTOCOL:-per_cell_random}"
SUFFIX=""; [[ "$PROTOCOL" != "per_cell_random" ]] && SUFFIX="_${PROTOCOL}"
OUT_DIR="outputs/finetune${SUFFIX}_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "  ReBoot finetune + evaluate + diagnose"
echo "  Job ID         : ${SLURM_JOB_ID}"
echo "  Pretrain ckpt  : ${PRETRAIN_CKPT}"
echo "  Protocol       : ${PROTOCOL}"
echo "  Output dir     : ${OUT_DIR}"
echo "  Start time     : $(date)"
echo "============================================================"
echo ""

# -------- Finetune --------
echo "--- finetune ---"
PYTHONPATH=. "${ENV}" scripts/finetune.py \
    "finetune.pretrained_ckpt='${PRETRAIN_CKPT}'" \
    data.splits.strategy="${PROTOCOL}" \
    trainer.devices=1 \
    trainer.precision=bf16-mixed \
    finetune.batch_size=512 \
    data.num_workers=10 \
    data.persistent_workers=true \
    paths.checkpoint_dir="${OUT_DIR}/checkpoints" \
    wandb.mode=disabled
# Prefer the best-on-validation checkpoint (written by scripts/finetune.py);
# fall back to last.ckpt for older runs.
FT_CKPT="${OUT_DIR}/checkpoints/finetune/best.ckpt"
[[ -f "$FT_CKPT" ]] || FT_CKPT="${OUT_DIR}/checkpoints/finetune/last.ckpt"
echo "Finetune ckpt: ${FT_CKPT}"

# -------- Evaluate --------
echo "--- evaluate ---"
PYTHONPATH=. "${ENV}" scripts/evaluate.py \
    "+eval.ckpt='${FT_CKPT}'" \
    +eval.protocol="${PROTOCOL}" \
    data.splits.strategy="${PROTOCOL}" \
    trainer.devices=1 \
    finetune.batch_size=512 \
    data.num_workers=10 \
    wandb.mode=disabled

# -------- Diagnose --------
echo "--- diagnose ---"
PYTHONPATH=. "${ENV}" scripts/diagnose.py \
    "+eval.ckpt='${FT_CKPT}'" \
    data.splits.strategy="${PROTOCOL}" \
    trainer.devices=1 \
    paths.output_dir="${OUT_DIR}/diagnostics" \
    wandb.mode=disabled

echo ""
echo "============================================================"
echo "  End time : $(date)"
echo "============================================================"
