#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=4
#SBATCH --job-name=sid-s3-nothink
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
NO_THINK_BASE_STAGE="${NO_THINK_BASE_STAGE:-stage2}"

case "${NO_THINK_BASE_STAGE}" in
    stage1|1)
        DEFAULT_BASE_MODEL="${PROJECT_ROOT}/checkpoints/${CATEGORY}_stage1_sft_Qwen3-1.7B/final_checkpoint"
        DEFAULT_EXPERIMENT_NAME="${CATEGORY}_stage3_rl_no_think_Qwen3-1.7B_seed${SEED}"
        ;;
    stage2|2)
        DEFAULT_BASE_MODEL="${PROJECT_ROOT}/checkpoints/${CATEGORY}_stage2_reasoning_activation_Qwen3-1.7B/final_checkpoint"
        DEFAULT_EXPERIMENT_NAME="${CATEGORY}_stage3_rl_no_think_from_stage2_Qwen3-1.7B_seed${SEED}"
        ;;
    *)
        echo "ERROR: NO_THINK_BASE_STAGE must be 'stage1' or 'stage2'." >&2
        exit 1
        ;;
esac

EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}"
FIXED_BASE_MODEL="${DEFAULT_BASE_MODEL}_tf457"
if [[ -z "${BASE_MODEL:-}" && -d "${FIXED_BASE_MODEL}" ]]; then
    BASE_MODEL="${FIXED_BASE_MODEL}"
else
    BASE_MODEL="${BASE_MODEL:-${DEFAULT_BASE_MODEL}}"
fi
DIRECT_RL_DIR="${DIRECT_RL_DIR:-${PROJECT_ROOT}/configs/ablation_data/no_think/${CATEGORY}}"
LOG_FILE="${LOG_FILE:-${PROJECT_ROOT}/logs/${EXPERIMENT_NAME}-${SLURM_JOB_ID}.log}"

export SID_INFO_PATH="${SID_INFO_PATH:-${REPO_DIR}/data/Amazon/info/${CATEGORY}_5_2016-10-2018-11.txt}"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/checkpoints"

source ./scripts/snellius_env.sh
unset ROCR_VISIBLE_DEVICES

echo "Run: ${EXPERIMENT_NAME}"
echo "Seed: ${SEED}"
echo "Commit: $(git rev-parse --short HEAD)"
echo "No-think base stage: ${NO_THINK_BASE_STAGE}"
echo "Base model: ${BASE_MODEL}"
echo "Direct RL data: ${DIRECT_RL_DIR}"

{
${PYTHON_CMD} -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DIRECT_RL_DIR}/train.parquet" \
    data.val_files="${DIRECT_RL_DIR}/test.parquet" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${BASE_MODEL}" \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    custom_reward_function.path="./verl/utils/reward_score/direct_recommendation_no_think_Office.py" \
    custom_reward_function.name="rule_base_reward" \
    trainer.project_name='RecRL_NoThink' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${PROJECT_ROOT}/checkpoints/RecRL_NoThink/${EXPERIMENT_NAME}" \
    trainer.resume_mode=disable \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-4}" \
    trainer.nnodes="${NNODES:-1}" \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=10 \
    "+data.seed=${SEED}" \
    "$@"
} > "${LOG_FILE}" 2>&1
