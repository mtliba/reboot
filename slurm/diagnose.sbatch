#!/bin/bash
# ----------------------------------------------------------------------------
#  Standalone diagnostics for a finetuned (or pretrained) ReBoot checkpoint.
#      sbatch --export=ALL,CKPT=outputs/finetune_16192/checkpoints/finetune/best.ckpt,OUT_DIR=outputs/finetune_16192 \
#             slurm/diagnose.sbatch
#  ~10 min on 1 GPU (embeds a 300k-window whole-cycle subsample).
# ----------------------------------------------------------------------------
#SBATCH --job-name=reboot-diagnose
#SBATCH --partition=GPU-A100-VERYSHORT
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=outputs/slurm/diagnose_%j.out
#SBATCH --error=outputs/slurm/diagnose_%j.err
set -euo pipefail
REBOT=${REBOOT_ROOT:-$PWD}
ENV=${REBOOT_PYTHON:-python}
cd "$REBOT"
: "${CKPT:?set CKPT=path/to/ckpt}"; : "${OUT_DIR:?set OUT_DIR=dir}"
PROTOCOL="${PROTOCOL:-per_cell_random}"
echo "diagnose ckpt=${CKPT} -> ${OUT_DIR}/diagnostics  ($(date))"
PYTHONPATH=. "${ENV}" scripts/diagnose.py \
    "+eval.ckpt='${CKPT}'" \
    data.splits.strategy="${PROTOCOL}" \
    trainer.devices=1 data.num_workers=4 \
    paths.output_dir="${OUT_DIR}" \
    wandb.mode=disabled
echo "done $(date)"; ls "${OUT_DIR}/diagnostics"
