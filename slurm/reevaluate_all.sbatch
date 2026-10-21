#!/bin/bash
# ----------------------------------------------------------------------------
#  Re-evaluate EVERY finished checkpoint (ReBoot finetunes + supervised baselines)
#  with the extended metrics (P95/P99, R2, per-phase / per-temperature / per-cell /
#  per-decile) and save predictions. Sequential on ONE GPU, ~1-2 min per ckpt.
#      sbatch slurm/reevaluate_all.sbatch          (FORCE=1 to redo existing)
# ----------------------------------------------------------------------------
#SBATCH --job-name=reboot-reeval
#SBATCH --partition=GPU-A100-SHORT
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=outputs/slurm/reeval_%j.out
#SBATCH --error=outputs/slurm/reeval_%j.err
set -uo pipefail
REBOT=${REBOOT_ROOT:-$PWD}
ENV=${REBOOT_PYTHON:-python}
cd "$REBOT"
FORCE="${FORCE:-0}"
proto_of() {  # infer protocol from the path
  case "$1" in *cross_battery*) echo cross_battery;; *cross_cycle*) echo cross_cycle;; *cross_temperature*) echo cross_temperature;; *) echo per_cell_random;; esac
}
n=0
# ReBoot finetunes: every best.ckpt under a 'finetune' dir (skip smoke dirs)
for CK in $(find outputs -path "*smoke*" -prune -o -path "*/finetune/best.ckpt" -print | sort); do
  P=$(proto_of "$CK"); SUF=""; [[ "$P" != per_cell_random ]] && SUF="_$P"
  [[ "$FORCE" != 1 && -f "$(dirname $CK)/eval_extended${SUF}.json" ]] && continue
  echo "=== $CK [$P]"
  PYTHONPATH=. "$ENV" scripts/reevaluate.py "+eval.ckpt='$CK'" "+eval.protocol=$P" data.splits.strategy="$P" data.num_workers=6 \
      hydra.run.dir=outputs/reeval_hydra/$SLURM_JOB_ID/$n wandb.mode=disabled 2>&1 | grep -E "^reboot|^supervised|Error|Traceback" ; n=$((n+1))
done
# supervised baselines: best.ckpt under baseline_<enc> dirs
for CK in $(find outputs -path "*smoke*" -prune -o -regex ".*/baseline_\(lstm\|tcn\|transformer\|hybrid\)/best.ckpt" -print | sort); do
  ENC=$(basename $(dirname $CK) | sed 's/baseline_//'); P=$(proto_of "$CK"); SUF=""; [[ "$P" != per_cell_random ]] && SUF="_$P"
  [[ "$FORCE" != 1 && -f "$(dirname $CK)/eval_extended${SUF}.json" ]] && continue
  echo "=== $CK [$P, $ENC]"
  PYTHONPATH=. "$ENV" scripts/reevaluate.py "+eval.ckpt='$CK'" "+eval.protocol=$P" "+baseline=supervised_$ENC" data.splits.strategy="$P" data.num_workers=6 \
      hydra.run.dir=outputs/reeval_hydra/$SLURM_JOB_ID/$n wandb.mode=disabled 2>&1 | grep -E "^reboot|^supervised|Error|Traceback" ; n=$((n+1))
done
echo "done $(date): $n checkpoints"
