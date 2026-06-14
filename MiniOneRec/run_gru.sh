#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=gru
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=slurm_output/gru_office.out

set -euo pipefail

cd ~/SIDReasoner/MiniOneRec
source ~/SIDReasoner/.venv/bin/activate

python3 sasrec.py --model GRU --data Office_Products