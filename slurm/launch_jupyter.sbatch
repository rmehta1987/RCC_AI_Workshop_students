#!/bin/bash
# Launch Jupyter Lab on a Midway3 CPU (caslake) node with the shared 'aimed' kernel.
#   sbatch slurm/launch_jupyter.sbatch        # then read logs/jupyter-<cnetid>-<jobid>.log
#SBATCH --job-name=aimed-jupyter
#SBATCH --account=rcc-staff
#SBATCH --partition=caslake
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/jupyter-%u-%j.log
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/project/rcc/mehta5/AI_Workshop}"
source scripts/common/load_profile.sh
aimed_activate

# Make the shared 'aimed' kernelspec (registered under PROJECT_DIR/env) discoverable.
export JUPYTER_PATH="$PROJECT_DIR/env/share/jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"

PORT=$(shuf -i 8000-8999 -n 1)
NODE=$(hostname -s)
cat <<EOF

============================================================================
  AImed Jupyter Lab
  node=$NODE   port=$PORT   kernel='AImed'

  On YOUR LAPTOP, open an SSH tunnel:
      ssh -N -L ${PORT}:${NODE}:${PORT} ${USER}@midway3.rcc.uchicago.edu
  Then open the http://127.0.0.1:${PORT}/lab?token=... URL printed below
  (replace the host with 127.0.0.1) and pick the 'AImed' kernel.
============================================================================

EOF
exec jupyter lab --no-browser --ip=0.0.0.0 --port="$PORT" \
    --notebook-dir="$PROJECT_DIR"
