#!/bin/bash

# Shared runtime setup for SIDReasoner jobs on Snellius.
# Source this file from training/evaluation scripts after `set -euo pipefail`.

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v module >/dev/null 2>&1; then
    unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_SHLVL
    module purge
    module load 2023 || true
    module load CUDA/12.4.0 || true
fi

export HF_HOME="${HF_HOME:-${PWD}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PWD}/.cache/uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${PWD}/.venv}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-1}"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"

if command -v uv >/dev/null 2>&1 && [[ -d "${UV_PROJECT_ENVIRONMENT}" ]]; then
    PYTHON_CMD="${PYTHON_CMD:-uv run --no-sync python}"
    TORCHRUN_CMD="${TORCHRUN_CMD:-uv run --no-sync torchrun}"
else
    PYTHON_CMD="${PYTHON_CMD:-python}"
    TORCHRUN_CMD="${TORCHRUN_CMD:-torchrun}"
fi

export PYTHON_CMD
export TORCHRUN_CMD
