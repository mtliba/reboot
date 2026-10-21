#!/bin/bash
# ----------------------------------------------------------------------------
#  Several ReBoot finetune+evaluate runs PACKED onto ONE GPU (ablation table).
#      sbatch --export=ALL,RUNS="name|ckpt|frac|seed|extra;name2|ckpt2|frac|seed|extra2" slurm/finetune_packed.sbatch
#  Fields: name (dir), pretrain ckpt path, label_fraction, label seed, extra Hydra
#  overrides (space separated, may be empty; applied to finetune AND evaluate,
#  e.g. model.single_latent=true or finetune.losses.use_phys=false).
#  Optional PROTOCOL. Results: outputs/ft_packed_<jobid>/<name>/checkpoints/finetune/eval_metrics*.json
# ----------------------------------------------------------------------------
#SBATCH --job-name=reboot-ft-packed
#SBATCH --partition=GPU-A100-SHORT
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=outputs/slurm/ft_packed_%j.out
#SBATCH --error=outputs/slurm/ft_packed_%j.err
set -uo pipefail
: "${RUNS:?set RUNS}"
PROTOCOL="${PROTOCOL:-per_cell_random}"
REBOT=${REBOOT_ROOT:-$PWD}
ENV=${REBOOT_PYTHON:-python}
cd "$REBOT"
ROOT="outputs/ft_packed_${SLURM_JOB_ID}"; mkdir -p "$ROOT"
echo "packed finetunes on 1 GPU, protocol=${PROTOCOL}  $(date)"
IFS=';' read -ra ITEMS <<< "$RUNS"
PIDS=(); NAMES=()
for item in "${ITEMS[@]}"; do
  IFS='|' read -r NAME CKPT FRAC LSEED EXTRA <<< "$item"
  [[ -f "$CKPT" ]] || { echo "missing ckpt for $NAME: $CKPT"; continue; }
  OUT="${ROOT}/${NAME}"; mkdir -p "$OUT"
  ( PYTHONPATH=. "${ENV}" scripts/finetune.py \
        "finetune.pretrained_ckpt='${CKPT}'" \
        finetune.label_fraction="${FRAC}" finetune.label_fraction_seed="${LSEED}" \
        data.splits.strategy="${PROTOCOL}" \
        trainer.devices=1 trainer.precision=bf16-mixed finetune.batch_size=512 \
        data.num_workers=4 data.persistent_workers=true \
        paths.checkpoint_dir="${OUT}/checkpoints" hydra.run.dir="${OUT}/hydra" wandb.mode=disabled ${EXTRA} \
    && CK="${OUT}/checkpoints/finetune/best.ckpt" && { [[ -f "$CK" ]] || CK="${OUT}/checkpoints/finetune/last.ckpt"; } \
    && PYTHONPATH=. "${ENV}" scripts/evaluate.py "+eval.ckpt='${CK}'" "+eval.protocol=${PROTOCOL}" \
        data.splits.strategy="${PROTOCOL}" trainer.devices=1 finetune.batch_size=512 data.num_workers=4 \
        hydra.run.dir="${OUT}/hydra_eval" wandb.mode=disabled ${EXTRA}
    echo "exit $?" ) > "${OUT}/run.log" 2>&1 &
  sleep 4
  PIDS+=($!); NAMES+=("$NAME")
done
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
for n in "${NAMES[@]}"; do echo "--- $n: $(grep -E '^  mae:|exit' "${ROOT}/$n/run.log" | tr '\n' ' ')"; done
echo "done $(date) (fail=${FAIL})"; exit $FAIL
