#!/bin/bash
# Run a script on a temporary CPU-node allocation (no GPU touched), e.g. the test
# suite or CPU smoke tests. Usage, from the login node:
#     bash -l slurm/run_on_cpu_node.sh path/to/script.sh [minutes]
# Uses srun inside the allocation (ssh to compute nodes needs a forwarded key).
export LC_ALL=C
module load /softs/trixie/modules/x86-64/slurm/default
SCRIPT="${1:?script path}"; MIN="${2:-45}"
OUT=$(salloc --no-shell -p COMPUTE-VERYSHORT --nodes=1 --cpus-per-task=8 --mem=24G --time="$MIN" 2>&1)
J=$(echo "$OUT" | grep -o "job allocation [0-9]*" | grep -o "[0-9]*$")
[ -z "$J" ] && { echo "$OUT"; exit 1; }
echo "cpu alloc job: $J"
for i in $(seq 1 30); do ST=$(squeue -j "$J" -h -o %t); [ "$ST" = "R" ] && break; sleep 3; done
echo "node: $(squeue -j "$J" -h -o %N) state: $ST"
srun --jobid="$J" --ntasks=1 --cpus-per-task=8 bash "$SCRIPT" 2>&1 | grep -v setlocale
scancel "$J" && echo "released $J"
