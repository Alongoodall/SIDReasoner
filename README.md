<div align="center">
<div align="center">
  <img src="./assets/logo.svg" width="500">
</div>
<h3>Replicating and Extending SIDReasoner</h3>

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)
<a href="https://huggingface.co/Dam2/SIDReasoner_Ext/tree/main"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Checkpoints-yellow"></a>

<a href="https://huggingface.co/Dam2/SIDReasoner_Ext/tree/main">🤗 Checkpoints & Artifacts</a> | <a href="https://github.com/Alongoodall/SIDReasoner">💻 Repository</a>
</div>

---

**SIDReasoner Reproduction + Extensions** reproduces and extends the [SIDReasoner](https://github.com/HappyPointer/SIDReasoner) framework on Amazon Office Products, providing a fully local-executable pipeline spanning **SFT alignment**, **reasoning activation**, and recommendation-oriented **reinforcement learning (RL)**, along with ablations over alignment stages and cross-model comparisons.


## Environment Setup

### 1) Clone
```bash
git clone https://github.com/Alongoodall/SIDReasoner.git
cd SIDReasoner
```

### 2) Python environment (recommended: uv)

The project requires **Python 3.10** exactly (pinned in `pyproject.toml`).

```bash
# Install uv if not already available
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install all dependencies
uv sync --python 3.10
source .venv/bin/activate
```

If you prefer standard pip, a pinned `requirements.txt` is provided:
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** `flash-attn` and `flashinfer` are compiled against CUDA 12 + PyTorch 2.6. Both the `uv sync` and `pip install -r requirements.txt` paths resolve pre-built wheels automatically; building from source may require `--no-build-isolation`.

### 3) Core dependencies (installed automatically)
- torch, transformers, datasets, peft
- pandas, numpy, fire, wandb, tqdm
- accelerate, bitsandbytes
- flash-attn, flashinfer
- verl (for RL stage)

---

## How to pass parameters to scripts

All scripts under `scripts/` read configuration from **environment variables** with sensible defaults. Pass overrides as environment variable prefixes on the command line:

```bash
CATEGORY=Office_Products BASE_MODEL=Qwen/Qwen3-0.6B bash scripts/preprocess_sft.sh
```

Do **not** write `bash scripts/preprocess_sft.sh CATEGORY=... BASE_MODEL=...` — that syntax passes the strings as positional arguments to the underlying Python script, not as shell variables.

---

## Data layout (expected)

Typical paths used by scripts:
- `./data/Amazon/train/*.csv`
- `./data/Amazon/valid/*.csv`
- `./data/Amazon/test/*.csv`
- `./data/Amazon/info/*.txt`
- `./data/Amazon/index/*.index.json`
- `./data/Amazon/index/*.item.json`

---

## Reproduction pipeline

### Stage 1 — SFT preprocessing
```bash
CATEGORY=Office_Products BASE_MODEL=Qwen/Qwen3-0.6B bash scripts/preprocess_sft.sh
```

### Stage 1 — SFT training
```bash
CATEGORY=Office_Products \
BASE_MODEL=Qwen/Qwen3-0.6B \
CUDA_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
bash scripts/train_sft.sh
```

### Stage 2 — Reasoning activation
```bash
CATEGORY=Office_Products \
BASE_MODEL=./output_dir/Office_Products_stage1_sft_Qwen3-0.6B/final_checkpoint \
CUDA_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
bash scripts/sft_reasoning_activation.sh
```

### Stage 3 — RL (reasoning)
```bash
CATEGORY=Office_Products \
STAGE2_CHECKPOINT=./output_dir/Office_Products_stage2_reasoning_activation_Qwen3-1.7B/final_checkpoint \
N_GPUS_PER_NODE=4 \
NNODES=1 \
bash scripts/RL_training_script.sh
```

### Create reasoning RL data
```bash
python scripts/create_reasoning_rl_data.py \
  --category Office_Products
```

### Create direct/no-thinking RL data
```bash
python scripts/create_direct_rl_data.py \
  --category Office_Products \
  --category_label "office products"
```

### Merge FSDP checkpoint (single step)
```bash
CKPT_DIR=./checkpoints/RecRL_Reasoning/Office_Products_stage3_rl_Qwen3-1.7B/global_step_100/actor \
bash scripts/merge_fsdp_ckpt.sh
```

### Merge all periodic checkpoints
```bash
CKPT_ROOT=./checkpoints/RecRL_Reasoning/Office_Products_stage3_rl_Qwen3-1.7B \
EVAL_INTERVAL=100 \
bash scripts/merge_fsdp_ckpt_ALL.sh
```

---

## Evaluation commands

### Standard (no-think) eval
```bash
CATEGORY=Office_Products \
EXP_NAME=./output_dir/Office_Products_stage1_sft_Qwen3-1.7B/final_checkpoint \
CUDA_LIST="0 1" \
CUDA_LIST_CSV="0,1" \
bash scripts/evaluate_Qwen3.sh
```

### Think eval (batch/multi-model)
```bash
CATEGORIES=Office_Products \
EXP_LIST="./output_dir/Office_Products_stage2_reasoning_activation_Qwen3-1.7B/final_checkpoint" \
CUDA_LIST="0 1 2" \
CUDA_LIST_CSV="0,1,2" \
bash scripts/evaluate_Qwen3_think_batch.sh
```

---

## Extensions

### 1. Stage-2 alignment ablations (S1/S2/S3)

**S1** — RL from Stage 1 SFT checkpoint directly:
```bash
CATEGORY=Office_Products \
STAGE2_CHECKPOINT=./output_dir/Office_Products_stage1_sft_Qwen3-1.7B/final_checkpoint \
N_GPUS_PER_NODE=4 NNODES=1 \
bash scripts/RL_training_script.sh
```

**S2** — RL from Stage 2 reasoning-activation checkpoint:
```bash
CATEGORY=Office_Products \
STAGE2_CHECKPOINT=./output_dir/Office_Products_stage2_reasoning_activation_Qwen3-1.7B/final_checkpoint \
N_GPUS_PER_NODE=4 NNODES=1 \
bash scripts/RL_training_script.sh
```

**S3** — further reasoning-activation pass, then RL on that output:
```bash
CATEGORY=Office_Products \
BASE_MODEL=./output_dir/Office_Products_stage2_reasoning_activation_Qwen3-1.7B/final_checkpoint \
CUDA_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
bash scripts/sft_reasoning_activation.sh

CATEGORY=Office_Products \
STAGE2_CHECKPOINT=./output_dir/Office_Products_stage3_reasoning_activation_Qwen3-1.7B/final_checkpoint \
N_GPUS_PER_NODE=4 NNODES=1 \
bash scripts/RL_training_script.sh
```

---

### 2. Cross-model checks (Qwen3-0.6B vs Qwen3-1.7B)

Repeat the full pipeline substituting the target model size. Example for **Qwen3-0.6B**:

```bash
CATEGORY=Office_Products BASE_MODEL=Qwen/Qwen3-0.6B bash scripts/preprocess_sft.sh

CATEGORY=Office_Products BASE_MODEL=Qwen/Qwen3-0.6B \
CUDA_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
bash scripts/train_sft.sh

CATEGORY=Office_Products \
BASE_MODEL=./output_dir/Office_Products_stage1_sft_Qwen3-0.6B/final_checkpoint \
CUDA_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
bash scripts/sft_reasoning_activation.sh

CATEGORY=Office_Products \
STAGE2_CHECKPOINT=./output_dir/Office_Products_stage2_reasoning_activation_Qwen3-0.6B/final_checkpoint \
N_GPUS_PER_NODE=4 NNODES=1 \
bash scripts/RL_training_script.sh
```

For **Qwen3-1.7B**, substitute `Qwen/Qwen3-1.7B` and the corresponding `Qwen3-1.7B` checkpoint paths throughout.

---

## Practical notes

- RL stages are compute-sensitive; start with short runs for sanity checks.
- Keep seed fixed for comparison (`seed=42` where applicable).
- For local single-GPU debugging, reduce:
  - batch/micro-batch
  - max sequence lengths
  - beam size
- Scripts source `scripts/snellius_env.sh` which auto-detects `uv` and sets `PYTHON_CMD`/`TORCHRUN_CMD`. Works correctly both on Snellius (SLURM) and locally.

---

## Experimental context

This repo accompanies: **Replicating and Extending SIDReasoner**  
Main findings:
- broad SIDReasoner trend reproduced,
- no-think variants can remain competitive,
- RL improves top-K accuracy but can increase exposure concentration.

---
