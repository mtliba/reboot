#!/bin/bash
# ----------------------------------------------------------------------------
#  Supervised low-label floor (CLAUDE.md §10 low_label protocol, baseline side).
#  Trains the supervised encoder+head from scratch on 1 / 5 / 10 / 25 % of the
#  labelled train windows and evaluates on the full protocol test set — the
#  numbers ReBoot's low_label_sweep.sbatch must beat.
#
#  Submit (1 GPU per array task; tasks run whenever a GPU is free):
#      sbatch --export=ALL,BASELINE_ENC=tcn    slurm/baseline_low_label.sbatch
#      sbatch --export=ALL,BASELINE_ENC=hybrid slurm/baseline_low_label.sbatch
#  Optional: PROTOCOL=cross_battery (default per_cell_random).
# ----------------------------------------------------------------------------

#SBATCH --job-name=reboot-base-lowlabel
#SBATCH --partition=GPU-A100-SHORT
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-3
#SBATCH --output=outputs/slurm/base_lowlabel_%A_%a.out
#SBATCH --error=outputs/slurm/base_lowlabel_%A_%a.err

set -euo pipefail

FRACTIONS=(0.01 0.05 0.10 0.25)
FRAC="${FRACTIONS[$SLURM_ARRAY_TASK_ID]}"
ENC="${BASELINE_ENC:-tcn}"
PROTOCOL="${PROTOCOL:-per_cell_random}"
SUFFIX=""; [[ "$PROTOCOL" != "per_cell_random" ]] && SUFFIX="_${PROTOCOL}"

REBOT=${REBOOT_ROOT:-$PWD}
ENV=${REBOOT_PYTHON:-python}
cd "$REBOT"

OUT_DIR="outputs/base_lowlabel${SUFFIX}_${SLURM_ARRAY_JOB_ID}/${ENC}_frac_${FRAC}"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "  Supervised low-label baseline — task ${SLURM_ARRAY_TASK_ID}"
echo "  Encoder         : ${ENC}"
echo "  Label fraction  : ${FRAC}"
echo "  Protocol        : ${PROTOCOL}"
echo "  Output dir      : ${OUT_DIR}"
echo "  Start time      : $(date)"
echo "============================================================"

# Same seed convention as low_label_sweep.sbatch so both sides see the SAME
# labelled subset at a given fraction.
PYTHONPATH=. "${ENV}" scripts/baseline_supervised.py \
    "+baseline=supervised_${ENC}" \
    finetune.label_fraction="${FRAC}" \
    finetune.label_fraction_seed="${SLURM_ARRAY_TASK_ID}" \
    data.splits.strategy="${PROTOCOL}" \
    trainer.max_epochs=50 \
    trainer.devices=1 \
    trainer.precision=bf16-mixed \
    finetune.batch_size=512 \
    data.num_workers=10 \
    data.persistent_workers=true \
    paths.checkpoint_dir="${OUT_DIR}" \
    wandb.mode=disabled

echo ""
echo "Done. End: $(date)"
echo "Metrics: ${OUT_DIR}/baseline_${ENC}/eval_metrics*.json"
