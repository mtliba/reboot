#!/usr/bin/env bash
# Loop the 7 ablation rows defined in configs/sweep/ablation.yaml.
# Each row pretrains + finetunes + evaluates and writes results under
# outputs/ablation/<row_name>/.
#
# Usage:
#     bash scripts/run_ablation.sh [extra_overrides...]
#
# Example (small smoke run on synthetic data):
#     bash scripts/run_ablation.sh trainer.max_epochs=5 wandb.mode=disabled
set -euo pipefail

ENV=${ENV:-${REBOOT_PYTHON:-python}}
EXTRA="$*"

declare -A ROWS=(
  [full]=""
  [no_ord]="pretrain.losses.use_ord=false"
  [no_refl]="pretrain.losses.use_refl=false"
  [no_mask]="pretrain.losses.use_mask=false"
  [no_orth]="pretrain.losses.use_orth=false"
  [no_phys]="finetune.losses.use_phys=false"
  [single_latent]="model.single_latent=true"
)

for name in "${!ROWS[@]}"; do
  OVR="${ROWS[$name]}"
  OUT="outputs/ablation/${name}"
  echo "=========================================="
  echo "  ablation row: ${name}   ${OVR}"
  echo "=========================================="
  PYTHONPATH=. ${ENV} scripts/pretrain.py \
    paths.checkpoint_dir=${OUT}/checkpoints \
    ${OVR} ${EXTRA}
  PYTHONPATH=. ${ENV} scripts/finetune.py \
    paths.checkpoint_dir=${OUT}/checkpoints \
    finetune.pretrained_ckpt=${OUT}/checkpoints/pretrain/last.ckpt \
    ${OVR} ${EXTRA}
  PYTHONPATH=. ${ENV} scripts/evaluate.py \
    +eval.ckpt=${OUT}/checkpoints/finetune/last.ckpt \
    ${OVR} ${EXTRA} 2>&1 | tee ${OUT}/eval.log
done
