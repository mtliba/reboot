#!/bin/bash
# ----------------------------------------------------------------------------
#  SSL baseline #2 — pretrain with augment-pair InfoNCE (naive SimCLR on time
#  series). Lets us compare ReBoot's design (which uses the structured cross-
#  phase symmetry for positive pairs) against a generic crop-and-mask
#  augmentation contrastive baseline.
#
#  Submit:
#      sbatch slurm/pretrain_ssl_simclr.sbatch
# ----------------------------------------------------------------------------

#SBATCH --job-name=reboot-ssl-simclr
#SBATCH --partition=GPU-A100-SHORT
#SBATCH --time=09:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:2
#SBATCH --output=outputs/slurm/ssl_simclr_%j.out
#SBATCH --error=outputs/slurm/ssl_simclr_%j.err

set -euo pipefail
REBOT=${REBOOT_ROOT:-$PWD}
ENV=${REBOOT_PYTHON:-python}
cd "$REBOT"
EPOCHS="${EPOCHS:-200}"

echo "============================================================"
echo "  SSL-SimCLR pretrain (augment-pair InfoNCE on encoder hidden)"
echo "  Job ID         : ${SLURM_JOB_ID}"
echo "  Epochs         : ${EPOCHS}"
echo "  Allocated GPUs : ${CUDA_VISIBLE_DEVICES:-(slurm-managed)}"
echo "  Start time     : $(date)"
echo "============================================================"

srun --kill-on-bad-exit=1 \
     bash -c 'PYTHONPATH='"${REBOT}"' '"${ENV}"' '"${REBOT}"'/scripts/baseline_ssl_simclr.py \
    trainer.max_epochs='"${EPOCHS}"' \
    trainer.devices=2 \
    +trainer.num_nodes=1 \
    trainer.strategy=ddp \
    trainer.precision=32 \
    pretrain.batch_size=2048 \
    pretrain.lr=1e-3 \
    data.num_workers=10 \
    data.persistent_workers=true \
    paths.checkpoint_dir=outputs/checkpoints/ssl_simclr_'"${SLURM_JOB_ID}"' \
    wandb.mode=disabled'

echo ""
echo "Done. End time: $(date)"
echo "Checkpoint:    outputs/checkpoints/ssl_simclr_${SLURM_JOB_ID}/pretrain/last.ckpt"
