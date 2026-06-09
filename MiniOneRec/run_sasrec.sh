#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=mini-one-rec-env-setup
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:40:00
#SBATCH --output=slurm_output/%x-%j.out

set -euo pipefail

source activate MiniOneRec
python3 sasrec.py --model SASRec