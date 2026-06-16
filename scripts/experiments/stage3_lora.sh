#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=4
#SBATCH --job-name=sid-s3-lora
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=/projects/prjs2120/groups/group_17/logs/%x-%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]]; then
    REPO_DIR="${SLURM_SUBMIT_DIR}"
else
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${REPO_DIR}"

CATEGORY="${CATEGORY:-Office_Products}"
SEED="${SEED:-42}"
PROJECT_ROOT="${PROJECT_ROOT:-/projects/prjs2120/groups/group_17}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj]}"
STAGE2_RUN_NAME="${STAGE2_RUN_NAME:-${CATEGORY}_stage2_reasoning_activation_Qwen3-1.7B}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${CATEGORY}_stage3_rl_lora-r${LORA_RANK}_Qwen3-1.7B_seed${SEED}}"

export EXPERIMENT_NAME
export STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/${STAGE2_RUN_NAME}/final_checkpoint}"
export LOG_FILE="${LOG_FILE:-${PROJECT_ROOT}/logs/${EXPERIMENT_NAME}-${SLURM_JOB_ID}.log}"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/checkpoints"

echo "Run: ${EXPERIMENT_NAME}"
echo "Seed: ${SEED}"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Stage 2 checkpoint: ${STAGE2_CHECKPOINT}"
echo "LoRA: rank=${LORA_RANK}, alpha=${LORA_ALPHA}, target=${LORA_TARGET_MODULES}"

SLURM_SUBMIT_DIR="${REPO_DIR}" bash scripts/RL_training_script.sh \
    "actor_rollout_ref.model.lora_rank=${LORA_RANK}" \
    "actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}" \
    "actor_rollout_ref.model.target_modules=${LORA_TARGET_MODULES}" \
    "actor_rollout_ref.rollout.load_format=safetensors" \
    "trainer.default_local_dir=${PROJECT_ROOT}/checkpoints/RecRL_Reasoning/${EXPERIMENT_NAME}" \
    "trainer.resume_mode=disable" \
    "+data.seed=${SEED}" \
    "$@"
