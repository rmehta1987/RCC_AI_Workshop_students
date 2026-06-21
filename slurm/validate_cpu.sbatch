#!/bin/bash
# Prove the no-GPU fallback path on the CPU partition (Goal 11/13).
#   sbatch slurm/validate_cpu.sbatch
#SBATCH --job-name=aimed-validate-cpu
#SBATCH --account=workshop-aiml
#SBATCH --partition=caslake
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/validate-cpu-%j.log
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/project/workshop-aiml/ai_medicine_genomics}"
source scripts/common/load_profile.sh
aimed_activate
export JUPYTER_PATH="$PROJECT_DIR/env/share/jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"
# CPU run proves the no-GPU FALLBACK path; skip the (slow on CPU) notebook re-exec
# — the GPU job covers headless notebook execution.
python scripts/validate_all.py --device cpu --skip-notebooks --report VALIDATION_REPORT_cpu.md
