#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=mini-one-rec-env-setup
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=10:00:00
#SBATCH --output=slurm_output/%x-%j.out

set -euo pipefail
deactivate sidreasoner

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v module >/dev/null 2>&1; then
    unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_SHLVL
    module purge
    module load 2025 || true
    module load Anaconda3/2025.06-1

    # Restore the path to your local user binaries (~/.local/bin) where 'uv' lives
    export PATH="$HOME/.local/bin:$PATH"
fi


conda create -n MiniOneRec python=3.11 -y
source activate MiniOneRec
pip install -r requirements.txt

srun python -uc "import torch; print('GPU available?', torch.cuda.is_available())"


