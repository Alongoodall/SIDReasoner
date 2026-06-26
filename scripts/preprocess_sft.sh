#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=sid-preprocess-stage1-sft
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --output=slurm_output/%x-%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SCRIPT_DIR"

source ./scripts/snellius_env.sh

CATEGORY="${CATEGORY:-Office_Products}"
BASE_MODEL="${BASE_MODEL:-google/gemma-3-1b-it}"
TRAIN_FILE="${TRAIN_FILE:-./data/Amazon/train/Office_Products_5_2016-10-2018-11.csv}"
EVAL_FILE="${EVAL_FILE:-./data/Amazon/valid/Office_Products_5_2016-10-2018-11.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-./data/Amazon/preprocessed}"

SID_INDEX_PATH="${SID_INDEX_PATH:-./data/Amazon/index/Office_Products.index.json}"
ITEM_META_PATH="${ITEM_META_PATH:-./data/Amazon/index/Office_Products.item.json}"
LLM_GENERATED_DATA_PATH="${LLM_GENERATED_DATA_PATH:-./data/Amazon/index/Office_Products.item_enhanced_v2.json}"
LLM_GENERATED_SEQUENCE_PATH="${LLM_GENERATED_SEQUENCE_PATH:-./data/Amazon/index/Office_Products.integrated_narrative.csv}"
GENERAL_REASONING_PATH="${GENERAL_REASONING_PATH:-./data/Amazon/general/sampled_data.arrow}"

WANDB_PROJECT="${WANDB_PROJECT:-MiniOneRec}"

RUN_NAME="${RUN_NAME:-preprocess_${CATEGORY}_${BASE_MODEL##*/}}"
LOG_FILE="${LOG_FILE:-./logs/${RUN_NAME}.txt}"

mkdir -p ./logs ./data/Amazon/preprocessed

{
echo "Starting preprocessing for ${CATEGORY}"
echo "Train: ${TRAIN_FILE}"
echo "Eval:  ${EVAL_FILE}"
echo "Output: ${OUTPUT_DIR}"

${PYTHON_CMD} sft_Qwen3_preprocess.py \
    --base_model "${BASE_MODEL}" \
    --train_file "${TRAIN_FILE}" \
    --eval_file "${EVAL_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --sample -1 \
    --seed 42 \
    --category "${CATEGORY}" \
    --cutoff_len 1024 \
    --wandb_project "${WANDB_PROJECT}" \
    --train_from_scratch False \
    --sid_index_path "${SID_INDEX_PATH}" \
    --item_meta_path "${ITEM_META_PATH}" \
    --llm_generated_data_path "${LLM_GENERATED_DATA_PATH}" \
    --llm_generated_sequence_path "${LLM_GENERATED_SEQUENCE_PATH}" \
    --general_reasoning_path "${GENERAL_REASONING_PATH}" \
    --mask_assistant True \
    --train_new_token_embeddings_only False \
    "$@"

} > "${LOG_FILE}" 2>&1