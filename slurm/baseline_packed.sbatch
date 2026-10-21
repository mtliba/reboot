#!/bin/bash
# ----------------------------------------------------------------------------
#  Several supervised baseline runs PACKED onto ONE GPU (concurrent processes;
#  each needs ~5 GB of the 80 GB and is data-loading bound).
#      sbatch --export=ALL,RUNS="hybrid:cross_battery:1.0 hybrid:cross_cycle:1.0 hybrid:cross_battery:0.01" \
#             slurm/baseline_packed.sbatch
#  Each RUNS item is  encoder:protocol:label_fraction[:label_seed] . label_seed matches
#  the ReBoot sweep (fraction index i + SEED_OFFSET); >=100 also shifts the global seed.
# ----------------------------------------------------------------------------
#SBATCH --job-name=reboot-baseline-packed
#SBATCH --partition=GPU-A100-SHORT
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=outputs/slurm/baseline_packed_%j.out
#SBATCH --error=outputs/slurm/baseline_packed_%j.err
set -uo pipefail
: "${RUNS:?set RUNS='enc:protocol:frac ...'}"
REBOT=${REBOOT_ROOT:-$PWD}
ENV=${REBOOT_PYTHON:-python}
cd "$REBOT"
ROOT="outputs/baseline_packed_${SLURM_JOB_ID}"; mkdir -p "$ROOT"
echo "packed baselines on 1 GPU: ${RUNS}   $(date)"
PIDS=(); NAMES=()
for spec in $RUNS; do
  IFS=: read -r ENC PROTO FRAC LSEED EXTRA <<< "$spec"
  LSEED="${LSEED:-0}"; EXTRA="${EXTRA:-}"          # 5th field: extra Hydra overrides, e.g. data.input_channels=[0,1,2,3]
  GSEED=42; [[ "$LSEED" -ge 100 ]] && GSEED=$((42 + LSEED - LSEED % 100))
  TAG=""; [[ -n "$EXTRA" ]] && TAG="_raw"
  OUT="${ROOT}/${ENC}_${PROTO}_frac${FRAC}_s${LSEED}${TAG}"; mkdir -p "$OUT"
  ( PYTHONPATH=. "${ENV}" scripts/baseline_supervised.py \
        "+baseline=supervised_${ENC}" \
        finetune.label_fraction="${FRAC}" finetune.label_fraction_seed="${LSEED}" seed="${GSEED}" \
        data.splits.strategy="${PROTO}" \
        trainer.max_epochs=50 trainer.devices=1 trainer.precision=bf16-mixed \
        finetune.batch_size=512 data.num_workers=4 data.persistent_workers=true \
        paths.checkpoint_dir="${OUT}" hydra.run.dir="${OUT}/hydra" wandb.mode=disabled ${EXTRA}
    echo "exit $?" ) > "${OUT}/run.log" 2>&1 &
  sleep 4
  PIDS+=($!); NAMES+=("$OUT")
done
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
for n in "${NAMES[@]}"; do echo "--- $n: $(grep -E '^  mae:|exit' "$n/run.log" | tr '\n' ' ')"; done
echo "done $(date) (fail=${FAIL})"; exit $FAIL
