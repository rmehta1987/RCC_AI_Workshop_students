#!/usr/bin/env bash
# Grab an interactive GPU shell on Midway3. Run on a LOGIN node.
#   bash slurm/gpu_interactive.sh [n_gpus=1] [time=02:00:00] [constraint]
# Inside the shell:  source scripts/common/load_profile.sh && aimed_activate
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/scripts/common/load_profile.sh"
NGPU="${1:-1}"; TIME="${2:-02:00:00}"; CONS="${3:-}"
EXTRA=()
[ -n "$CONS" ] && EXTRA+=(--constraint="$CONS")
echo "srun GPU shell: partition=$GPU_PARTITION account=$RCC_ACCOUNT gpus=$NGPU time=$TIME ${CONS:+constraint=$CONS}"
echo "(for Evo 2 use a >=40GB card:  bash slurm/gpu_interactive.sh 1 02:00:00 'a100|a40|l40s')"
exec srun --account="$RCC_ACCOUNT" --partition="$GPU_PARTITION" \
  --gres=gpu:"$NGPU" --cpus-per-task=8 --mem=32G --time="$TIME" \
  "${EXTRA[@]}" --pty bash
