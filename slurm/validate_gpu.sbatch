#!/bin/bash
# End-to-end validation on a GPU compute node (Goal 11). Writes VALIDATION_REPORT.md.
#   sbatch slurm/validate_gpu.sbatch
#SBATCH --job-name=aimed-validate-gpu
#SBATCH --account=workshop-aiml
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=logs/validate-gpu-%j.log
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/project/workshop-aiml/ai_medicine_genomics}"
source scripts/common/load_profile.sh
aimed_activate
# make the 'aimed' kernelspec (under env/, not env/aimed/) findable by nbconvert
export JUPYTER_PATH="$PROJECT_DIR/env/share/jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"
python scripts/validate_all.py --device cuda --report VALIDATION_REPORT.md
