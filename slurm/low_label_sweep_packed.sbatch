#!/bin/bash
# ----------------------------------------------------------------------------
#  Low-label sweep (1/5/10/25/100 %) PACKED onto ONE GPU: the five finetunes run
#  concurrently as background processes on the same A100 (each needs ~5-8 GB of
#  the 80 GB and is data-loading bound), instead of reserving five GPUs.
#      sbatch --export=ALL,PRETRAIN_CKPT=<ckpt> slurm/low_label_sweep_packed.sbatch
#  Optional PROTOCOL=<strategy>. ~2.5 h wall-clock (bounded by the 100 % run).
# ----------------------------------------------------------------------------
#SBATCH --job-name=reboot-lowlabel-packed
#SBATCH --partition=GPU-A100-SHORT
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=outputs/slurm/lowlabel_packed_%j.out
#SBATCH --error=outputs/slurm/lowlabel_packed_%j.err
set -uo pipefail
: "${PRETRAIN_CKPT:?set PRETRAIN_CKPT}"
[[ -f "$PRETRAIN_CKPT" ]] || { echo "missing $PRETRAIN_CKPT"; exit 1; }
PROTOCOL="${PROTOCOL:-per_cell_random}"
SEED_OFFSET="${SEED_OFFSET:-0}"     # repeats: label_fraction_seed = i + offset, seed = 42 + offset
SUFFIX=""; [[ "$PROTOCOL" != "per_cell_random" ]] && SUFFIX="_${PROTOCOL}"
REBOT=${REBOOT_ROOT:-$PWD}
ENV=${REBOOT_PYTHON:-python}
cd "$REBOT"
ROOT="outputs/lowlabel${SUFFIX}_s${SEED_OFFSET}_${SLURM_JOB_ID}"; mkdir -p "$ROOT"
echo "packed low-label sweep on 1 GPU  ckpt=${PRETRAIN_CKPT} protocol=${PROTOCOL} seed_offset=${SEED_OFFSET}  $(date)"
FRACTIONS=(0.01 0.05 0.10 0.25 1.0)
i=0; PIDS=()
for FRAC in "${FRACTIONS[@]}"; do
  OUT="${ROOT}/frac_${FRAC}"; mkdir -p "$OUT"
  ( PYTHONPATH=. "${ENV}" scripts/finetune.py \
        "finetune.pretrained_ckpt='${PRETRAIN_CKPT}'" \
        finetune.label_fraction="${FRAC}" finetune.label_fraction_seed="$((i + SEED_OFFSET))" seed="$((42 + SEED_OFFSET))" \
        data.splits.strategy="${PROTOCOL}" \
        trainer.devices=1 trainer.precision=bf16-mixed finetune.batch_size=512 \
        data.num_workers=4 data.persistent_workers=true \
        paths.checkpoint_dir="${OUT}/checkpoints" hydra.run.dir="${OUT}/hydra" wandb.mode=disabled \
    && CK="${OUT}/checkpoints/finetune/best.ckpt" && { [[ -f "$CK" ]] || CK="${OUT}/checkpoints/finetune/last.ckpt"; } \
    && PYTHONPATH=. "${ENV}" scripts/evaluate.py "+eval.ckpt='${CK}'" "+eval.protocol=${PROTOCOL}" \
        data.splits.strategy="${PROTOCOL}" trainer.devices=1 finetune.batch_size=512 data.num_workers=4 \
        hydra.run.dir="${OUT}/hydra_eval" wandb.mode=disabled
    echo "frac ${FRAC} exit $?" ) > "${OUT}/run.log" 2>&1 &
  sleep 4
  PIDS+=($!); i=$((i+1))
done
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
for FRAC in "${FRACTIONS[@]}"; do echo "--- frac ${FRAC}: $(grep -E 'mae:|exit' "${ROOT}/frac_${FRAC}/run.log" | tr '\n' ' ')"; done
echo "done $(date) (fail=${FAIL})"; exit $FAIL
