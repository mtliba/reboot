#!/bin/bash
# ----------------------------------------------------------------------------
#  SSL baseline #1 — pretrain with L_mask ONLY (MAE-style), then chain to
#  the standard finetune+eval. Lets us compare ReBoot's structured losses
#  (ord, refl, smooth, orth) against a plain masked-reconstruction pretrain.
#
#  Submit:
#      sbatch slurm/pretrain_ssl_mae.sbatch
#  Chain finetune to it:
#      sbatch --dependency=afterok:<this-jobid> \
#             --export=ALL,PRETRAIN_JOBID_LABEL=ssl_mae_<this-jobid> \
#             slurm/finetune_eval.sbatch
#  (Or set PRETRAIN_CKPT directly; see finetune_eval.sbatch.)
# ----------------------------------------------------------------------------

#SBATCH --job-name=reboot-ssl-mae
#SBATCH --partition=GPU-A100-SHORT
#SBATCH --time=09:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:2
#SBATCH --output=outputs/slurm/ssl_mae_%j.out
#SBATCH --error=outputs/slurm/ssl_mae_%j.err

set -euo pipefail
REBOT=${REBOOT_ROOT:-$PWD}
ENV=${REBOOT_PYTHON:-python}
cd "$REBOT"

echo "============================================================"
echo "  SSL-MAE pretrain (L_mask only, the others disabled)"
echo "  Job ID         : ${SLURM_JOB_ID}"
echo "  Allocated GPUs : ${CUDA_VISIBLE_DEVICES:-(slurm-managed)}"
echo "  Start time     : $(date)"
echo "============================================================"

srun --kill-on-bad-exit=1 \
     bash -c 'PYTHONPATH='"${REBOT}"' '"${ENV}"' '"${REBOT}"'/scripts/pretrain.py \
    trainer.max_epochs=200 \
    trainer.devices=2 \
    +trainer.num_nodes=1 \
    trainer.strategy=ddp \
    trainer.precision=32 \
    pretrain.batch_size=2048 \
    pretrain.lr=1e-3 \
    pretrain.losses.use_mask=true \
    pretrain.losses.use_ord=false \
    pretrain.losses.use_refl=false \
    pretrain.losses.use_smooth=false \
    pretrain.losses.use_orth=false \
    data.num_workers=10 \
    data.persistent_workers=true \
    paths.checkpoint_dir=outputs/checkpoints/ssl_mae_'"${SLURM_JOB_ID}"' \
    wandb.mode=disabled'

echo ""
echo "Done. End time: $(date)"
echo "Checkpoint:    outputs/checkpoints/ssl_mae_${SLURM_JOB_ID}/pretrain/last.ckpt"
