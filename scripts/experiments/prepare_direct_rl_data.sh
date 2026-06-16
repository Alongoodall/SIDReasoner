#!/bin/bash
#SBATCH --partition=staging
#SBATCH --job-name=sid-direct-rl-data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/projects/prjs2120/groups/group_17/logs/%x-%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]]; then
    REPO_DIR="${SLURM_SUBMIT_DIR}"
else
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${REPO_DIR}"

CATEGORY="${CATEGORY:-Office_Products}"
PROJECT_ROOT="${PROJECT_ROOT:-/projects/prjs2120/groups/group_17}"
DATASET_STEM="${DATASET_STEM:-Office_Products_5_2016-10-2018-11}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/configs/ablation_data/no_think/${CATEGORY}}"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/configs"

source ./scripts/snellius_env.sh

echo "Preparing direct RL data"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Output: ${OUT_DIR}"

${PYTHON_CMD} scripts/create_direct_rl_data.py \
    --train_data "./data/Amazon/train/${DATASET_STEM}.csv" \
    --eval_data "./data/Amazon/test/${DATASET_STEM}.csv" \
    --local_dir "${OUT_DIR}" \
    --category "${CATEGORY}" \
    --category_label "office products" \
    --data_source "rec/Amazon/${CATEGORY}/direct"
