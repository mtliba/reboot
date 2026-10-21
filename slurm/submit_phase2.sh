#!/bin/bash
# Phase 2 submission (2026-09-03 evening): ablations @100 epochs, seed repeats,
# cross-temperature. Run once on the login node:  bash -l slurm/submit_phase2.sh
export LC_ALL=C
module load /softs/trixie/modules/x86-64/slurm/default
cd ${REBOOT_ROOT:-$PWD}
CB_PRE=16294          # cross_battery v4 pretrain (2 GPU) — ablations start after it
V4_SWEEP=16263        # v4 packed low-label sweep — seed repeats start after it
V4_CKPT=outputs/checkpoints/pretrain_200_16259/pretrain/last.ckpt
REC="BATCH_SAMPLER=cycle,PAIR_STRATEGY=soc_proxy,REFL_EPS=0.05"
ck() { echo "outputs/checkpoints/pretrain_200_$1/pretrain/last.ckpt"; }
LOG=outputs/slurm/jobs_2026-09-02.txt
echo "# phase2 $(date)" >> $LOG

# ---- ablation pretrains, 100 epochs, 2 GPU each, strictly sequential ----
A1=$(sbatch --parsable --job-name=reboot-abl-full100 --dependency=afterany:$CB_PRE --export=ALL,$REC,EPOCHS=100 slurm/pretrain_200.sbatch)
A2=$(sbatch --parsable --job-name=reboot-abl-norefl  --dependency=afterany:$A1 --export=ALL,$REC,EPOCHS=100,EXTRA_ARGS="pretrain.losses.use_refl=false" slurm/pretrain_200.sbatch)
A3=$(sbatch --parsable --job-name=reboot-abl-noord   --dependency=afterany:$A2 --export=ALL,$REC,EPOCHS=100,EXTRA_ARGS="pretrain.losses.use_ord=false" slurm/pretrain_200.sbatch)
A4=$(sbatch --parsable --job-name=reboot-abl-single  --dependency=afterany:$A3 --export=ALL,$REC,EPOCHS=100,EXTRA_ARGS="model.single_latent=true" slurm/pretrain_200.sbatch)
A5=$(sbatch --parsable --job-name=reboot-abl-maeonly --dependency=afterany:$A4 --export=ALL,$REC,EPOCHS=100,EXTRA_ARGS="pretrain.losses.use_ord=false pretrain.losses.use_refl=false pretrain.losses.use_smooth=false pretrain.losses.use_orth=false" slurm/pretrain_200.sbatch)
echo "ablation pretrains: full100=$A1 norefl=$A2 noord=$A3 single=$A4 maeonly=$A5"
# ---- ablation finetunes (100 % and 1 % labels), 6 per GPU ----
R1="full100_100|$(ck $A1)|1.0|4|;full100_001|$(ck $A1)|0.01|0|;nophys_100|$(ck $A1)|1.0|4|finetune.losses.use_phys=false;nophys_001|$(ck $A1)|0.01|0|finetune.losses.use_phys=false;norefl_100|$(ck $A2)|1.0|4|;norefl_001|$(ck $A2)|0.01|0|"
R2="noord_100|$(ck $A3)|1.0|4|;noord_001|$(ck $A3)|0.01|0|;single_100|$(ck $A4)|1.0|4|model.single_latent=true;single_001|$(ck $A4)|0.01|0|model.single_latent=true;maeonly_100|$(ck $A5)|1.0|4|;maeonly_001|$(ck $A5)|0.01|0|"
F1=$(sbatch --parsable --job-name=reboot-abl-ft1 --dependency=afterok:$A1:$A2 --export=ALL,RUNS="$R1" slurm/finetune_packed.sbatch)
F2=$(sbatch --parsable --job-name=reboot-abl-ft2 --dependency=afterok:$A3:$A4:$A5 --export=ALL,RUNS="$R2" slurm/finetune_packed.sbatch)
echo "ablation finetunes: $F1 $F2"
# ---- seed repeats (1 GPU, sequential chain) ----
S1=$(sbatch --parsable --job-name=reboot-v4-sweep-s100 --dependency=afterany:$V4_SWEEP --export=ALL,PRETRAIN_CKPT=$V4_CKPT,SEED_OFFSET=100 slurm/low_label_sweep_packed.sbatch)
S2=$(sbatch --parsable --job-name=reboot-v4-sweep-s200 --dependency=afterany:$S1 --export=ALL,PRETRAIN_CKPT=$V4_CKPT,SEED_OFFSET=200 slurm/low_label_sweep_packed.sbatch)
H1=$(sbatch --parsable --job-name=reboot-hyb-sweep-s100 --dependency=afterany:$S2 --export=ALL,RUNS="hybrid:per_cell_random:0.01:100 hybrid:per_cell_random:0.05:101 hybrid:per_cell_random:0.10:102 hybrid:per_cell_random:0.25:103 hybrid:per_cell_random:1.0:104" slurm/baseline_packed.sbatch)
H2=$(sbatch --parsable --job-name=reboot-hyb-sweep-s200 --dependency=afterany:$H1 --export=ALL,RUNS="hybrid:per_cell_random:0.01:200 hybrid:per_cell_random:0.05:201 hybrid:per_cell_random:0.10:202 hybrid:per_cell_random:0.25:203 hybrid:per_cell_random:1.0:204" slurm/baseline_packed.sbatch)
H3=$(sbatch --parsable --job-name=reboot-hyb-shift-extra --dependency=afterany:$H2 --export=ALL,RUNS="hybrid:cross_battery:0.05:1 hybrid:cross_temperature:1.0:0 hybrid:cross_temperature:0.01:0 hybrid:cross_temperature:0.05:1" slurm/baseline_packed.sbatch)
echo "seed repeats: reboot $S1 $S2 ; hybrid $H1 $H2 ; hybrid shift extras $H3"
# ---- cross-temperature (v4 recipe), after the ablation chain ----
CT=$(sbatch --parsable --job-name=reboot-pretrain-ct-v4 --dependency=afterany:$A5 --export=ALL,$REC,PROTOCOL=cross_temperature slurm/pretrain_200.sbatch)
CTFT=$(sbatch --parsable --dependency=afterok:$CT --export=ALL,PRETRAIN_JOBID=$CT,PROTOCOL=cross_temperature slurm/finetune_eval.sbatch)
CTLL=$(sbatch --parsable --dependency=afterok:$CT --export=ALL,PRETRAIN_CKPT=$(ck $CT),PROTOCOL=cross_temperature slurm/low_label_sweep_packed.sbatch)
echo "cross_temperature: pretrain $CT ft $CTFT sweep $CTLL"
echo "abl=$A1,$A2,$A3,$A4,$A5 ablft=$F1,$F2 seeds=$S1,$S2,$H1,$H2,$H3 ct=$CT,$CTFT,$CTLL" >> $LOG
echo; squeue -u $(whoami) -o "%.12i %.26j %.3t %.8M %.14b %R" | grep -E "reboot|JOBID"
