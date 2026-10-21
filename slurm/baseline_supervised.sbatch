#!/bin/bash
# ----------------------------------------------------------------------------
#  Supervised baseline: encoder + MLP head, trained directly on SOC.
#  No pretrain. Used to prove the SSL+factored-latent setup actually helps.
#
#  Pick the encoder via the BASELINE_ENC env var (lstm | tcn | transformer | hybrid).
#
#  Submit examples:
#      sbatch --export=ALL,BASELINE_ENC=lstm        slurm/baseline_supervised.sbatch
#      sbatch --export=ALL,BASELINE_ENC=tcn         slurm/baseline_supervised.sbatch
#      sbatch --export=ALL,BASELINE_ENC=transformer slurm/baseline_supervised.sbatch
#      sbatch --export=ALL,BASELINE_ENC=hybrid      slurm/baseline_supervised.sbatch
#
#  Or submit all 4 in one shot:
#      for enc in lstm tcn transformer hybrid; do
#          sbatch --export=ALL,BASELINE_ENC=$enc slurm/baseline_supervised.sbatch
#      done
#
#  Each job uses 1 A100, ~1.5 h wall-clock for 50 supervised-finetune epochs.
# ----------------------------------------------------------------------------

#SBATCH --job-name=reboot-baseline
#SBATCH --partition=GPU-A100-SHORT
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=outputs/slurm/baseline_%x_%j.out
#SBATCH --error=outputs/slurm/baseline_%x_%j.err

set -euo pipefail

ENC="${BASELINE_ENC:-hybrid}"
case "$ENC" in
    lstm|tcn|transformer|hybrid|chemali_dnn|chemali_lstm|yang_gru|hannan_transformer|li_cnnbilstm) ;;
    *)  echo "ERROR: BASELINE_ENC=$ENC not in {lstm,tcn,transformer,hybrid}"; exit 1;;
esac
# Optional low-label regime for the supervised floor (CLAUDE.md §10), e.g.
#   sbatch --export=ALL,BASELINE_ENC=hybrid,LABEL_FRACTION=0.01 slurm/baseline_supervised.sbatch
FRAC="${LABEL_FRACTION:-1.0}"
EPOCHS="${EPOCHS:-50}"          # e.g. 500 at 1 % labels = same number of gradient steps as 50 epochs at 10 %
# Evaluation protocol: per_cell_random | cross_battery | cross_temperature | cross_cycle
PROTOCOL="${PROTOCOL:-per_cell_random}"
SUFFIX=""; [[ "$PROTOCOL" != "per_cell_random" ]] && SUFFIX="_${PROTOCOL}"

REBOT=${REBOOT_ROOT:-$PWD}
ENV=${REBOOT_PYTHON:-python}
cd "$REBOT"

OUT_DIR="outputs/checkpoints/baseline_${ENC}_frac${FRAC}${SUFFIX}_ep${EPOCHS}_${SLURM_JOB_ID}"

echo "============================================================"
echo "  Supervised baseline (no SSL pretrain)"
echo "  Encoder        : ${ENC}"
echo "  Label fraction : ${FRAC}"
echo "  Protocol       : ${PROTOCOL}"
echo "  Job ID         : ${SLURM_JOB_ID}"
echo "  Node           : ${SLURMD_NODENAME}"
echo "  Output dir     : ${OUT_DIR}"
echo "  Start time     : $(date)"
echo "============================================================"

PYTHONPATH=. "${ENV}" scripts/baseline_supervised.py \
    "+baseline=supervised_${ENC}" \
    trainer.max_epochs="${EPOCHS}" \
    trainer.devices=1 \
    trainer.precision=bf16-mixed \
    finetune.batch_size=512 \
    finetune.label_fraction="${FRAC}" \
    data.splits.strategy="${PROTOCOL}" \
    data.num_workers=10 \
    data.persistent_workers=true \
    paths.checkpoint_dir="${OUT_DIR}" \
    wandb.mode=disabled

echo ""
echo "Done. End time: $(date)"
echo "Checkpoint:    ${OUT_DIR}/baseline_${ENC}/best.ckpt"
echo "Test metrics:  ${OUT_DIR}/baseline_${ENC}/eval_metrics.json"
