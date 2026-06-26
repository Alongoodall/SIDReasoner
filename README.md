<div align="center">
<div align="center">
  <img src="./assets/logo.svg" width="500">
</div>
<h3>Replicating and Extending SIDReasoner</h3>

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)
<a href="https://huggingface.co/Dam2/SIDReasoner_Ext/tree/main"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Checkpoints-yellow"></a>

<a href="https://huggingface.co/Dam2/SIDReasoner_Ext/tree/main">🤗 Checkpoints & Artifacts</a> | <a href="https://github.com/Alongoodall/SIDReasoner">💻 Repository</a>
</div>

---

**SIDReasoner Reproduction + Extensions** reproduces and extends the [SIDReasoner](https://github.com/HappyPointer/SIDReasoner) framework on Amazon Office Products, providing a fully local-executable pipeline spanning **SFT alignment**, **reasoning activation**, and recommendation-oriented **reinforcement learning (RL)**, along with ablations over alignment stages, corpus composition, diversity diagnostics, and cross-model comparisons.


## Environment Setup

### 1) Clone
```bash
git clone https://github.com/Alongoodall/SIDReasoner.git
cd SIDReasoner
```

### 2) Python environment (recommended: uv)
```bash
# optional
pip install uv

# create env and install
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

If you do not use `uv`, standard `python -m venv` + `pip install -r requirements.txt` is fine.

### 3) Core dependencies
The project relies on:
- torch
- transformers
- datasets
- peft
- pandas
- numpy
- fire
- wandb
- tqdm
- accelerate
- bitsandbytes
- verl (for RL stage)

---

## Local execution policy (important)

All scripts under `scripts/` are runnable with:

```bash
bash scripts/<script_name>.sh
```

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

Adjust the model and dataset parameters to fit the respective experiments.

### Stage 1 — SFT preprocessing
```bash
bash scripts/preprocess_sft.sh \
  CATEGORY=Office_Products \
  BASE_MODEL=Qwen/Qwen3-0.6B
```

### Stage 1 — SFT training
```bash
bash scripts/train_sft.sh \
  CATEGORY=Office_Products \
  BASE_MODEL=Qwen/Qwen3-0.6B \
  CUDA_DEVICES=0,1,2,3 \
  NPROC_PER_NODE=4
```

### Stage 2 — Reasoning activation
```bash
bash scripts/sft_reasoning_activation.sh \
  CATEGORY=Office_Products \
  BASE_MODEL=./output_dir/Office_Products_stage1_sft_Qwen3-0.6B/final_checkpoint \
  CUDA_DEVICES=0,1,2,3 \
  NPROC_PER_NODE=4
```

### Stage 3 — RL (reasoning)
```bash
bash scripts/RL_training_script.sh \
  CATEGORY=Office_Products \
  STAGE2_CHECKPOINT=./output_dir/Office_Products_stage2_reasoning_activation_Qwen3-1.7B/final_checkpoint \
  N_GPUS_PER_NODE=4 \
  NNODES=1
```

### Create direct/no-thinking RL data
```bash
python scripts/create_direct_rl_data.py \
  --category Office_Products \
  --category_label "office products"
```

### Create reasoning RL data
```bash
python scripts/create_reasoning_rl_data.py \
  --category Office_Products
```

### Merge FSDP checkpoint (single step)
```bash
bash scripts/merge_fsdp_ckpt.sh \
  CKPT_DIR=./checkpoints/RecRL_Reasoning/Office_Products_stage3_rl_Qwen3-1.7B/global_step_100/actor
```

### Merge all periodic checkpoints
```bash
bash scripts/merge_fsdp_ckpt_ALL.sh \
  CKPT_ROOT=./checkpoints/RecRL_Reasoning/Office_Products_stage3_rl_Qwen3-1.7B \
  EVAL_INTERVAL=100
```

---

## Evaluation commands

### Standard (no-think) eval
```bash
bash scripts/evaluate_Qwen3.sh \
  CATEGORY=Office_Products \
  EXP_NAME=./output_dir/Office_Products_stage1_sft_Qwen3-1.7B/final_checkpoint \
  CUDA_LIST="0 1" \
  CUDA_LIST_CSV="0,1"
```

### Think eval (batch/multi-model)
```bash
bash scripts/evaluate_Qwen3_think_batch.sh \
  CATEGORIES=Office_Products \
  EXP_LIST="./output_dir/Office_Products_stage2_reasoning_activation_Qwen3-1.7B/final_checkpoint" \
  CUDA_LIST="0 1 2" \
  CUDA_LIST_CSV="0,1,2"
```

---

## Extensions

### 1. Stage-2 alignment ablations (S1/S2/S3)

**S1** — RL from Stage 1 SFT checkpoint directly:
```bash
bash scripts/RL_training_script.sh \
  CATEGORY=Office_Products \
  STAGE2_CHECKPOINT=./output_dir/Office_Products_stage1_sft_Qwen3-1.7B/final_checkpoint \
  N_GPUS_PER_NODE=4 \
  NNODES=1
```

**S2** — RL from Stage 2 reasoning-activation checkpoint:
```bash
bash scripts/RL_training_script.sh \
  CATEGORY=Office_Products \
  STAGE2_CHECKPOINT=./output_dir/Office_Products_stage2_reasoning_activation_Qwen3-1.7B/final_checkpoint \
  N_GPUS_PER_NODE=4 \
  NNODES=1
```

**S3** — further reasoning-activation pass, then RL on that output:
```bash
bash scripts/sft_reasoning_activation.sh \
  CATEGORY=Office_Products \
  BASE_MODEL=./output_dir/Office_Products_stage2_reasoning_activation_Qwen3-1.7B/final_checkpoint \
  CUDA_DEVICES=0,1,2,3 \
  NPROC_PER_NODE=4

bash scripts/RL_training_script.sh \
  CATEGORY=Office_Products \
  STAGE2_CHECKPOINT=./output_dir/Office_Products_stage3_reasoning_activation_Qwen3-1.7B/final_checkpoint \
  N_GPUS_PER_NODE=4 \
  NNODES=1
```

---

### 2. Diversity + concentration diagnostics

Computes Gini, normalized entropy, coverage, and Lorenz-curve-style interpretation over a results file:

```bash
python scripts/diversity_diagnostics.py \
  --category Office_Products \
  --results_path ./results/Office_Products_<exp_name>.json
```

---

### 3. Direct no-thinking GRPO

No `</think>` dependency in reward parsing. First create the direct RL data (see [Create direct/no-thinking RL data](#create-directno-thinking-rl-data) above), then train from either initialization:

**Initialized from SFT checkpoint:**
```bash
bash scripts/RL_training_script_direct.sh \
  CATEGORY=Office_Products \
  STAGE2_CHECKPOINT=./output_dir/Office_Products_stage1_sft_Qwen3-1.7B/final_checkpoint \
  N_GPUS_PER_NODE=4 \
  NNODES=1
```

**Initialized from reasoning-activation checkpoint:**
```bash
bash scripts/RL_training_script_direct.sh \
  CATEGORY=Office_Products \
  STAGE2_CHECKPOINT=./output_dir/Office_Products_stage2_reasoning_activation_Qwen3-1.7B/final_checkpoint \
  N_GPUS_PER_NODE=4 \
  NNODES=1
```

---

### 4. Enriched-corpus decomposition

Preprocess with each corpus variant, then run the standard SFT → Stage 2 → RL pipeline on the resulting data.

**Item-centric only:**
```bash
bash scripts/preprocess_sft.sh \
  CATEGORY=Office_Products \
  BASE_MODEL=Qwen/Qwen3-0.6B \
  CORPUS_MODE=item_only
```

**Sequence-centric only:**
```bash
bash scripts/preprocess_sft.sh \
  CATEGORY=Office_Products \
  BASE_MODEL=Qwen/Qwen3-0.6B \
  CORPUS_MODE=sequence_only
```

**Both (full enriched corpus, default):**
```bash
bash scripts/preprocess_sft.sh \
  CATEGORY=Office_Products \
  BASE_MODEL=Qwen/Qwen3-0.6B \
  CORPUS_MODE=both
```

---

### 5. Cross-model checks (Qwen3-0.6B vs Qwen3-1.7B)

Repeat the full pipeline substituting the target model size. Example for **Qwen3-0.6B**:

```bash
bash scripts/preprocess_sft.sh \
  CATEGORY=Office_Products \
  BASE_MODEL=Qwen/Qwen3-0.6B

bash scripts/train_sft.sh \
  CATEGORY=Office_Products \
  BASE_MODEL=Qwen/Qwen3-0.6B \
  CUDA_DEVICES=0,1,2,3 \
  NPROC_PER_NODE=4

bash scripts/sft_reasoning_activation.sh \
  CATEGORY=Office_Products \
  BASE_MODEL=./output_dir/Office_Products_stage1_sft_Qwen3-0.6B/final_checkpoint \
  CUDA_DEVICES=0,1,2,3 \
  NPROC_PER_NODE=4

bash scripts/RL_training_script.sh \
  CATEGORY=Office_Products \
  STAGE2_CHECKPOINT=./output_dir/Office_Products_stage2_reasoning_activation_Qwen3-0.6B/final_checkpoint \
  N_GPUS_PER_NODE=4 \
  NNODES=1
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
- If `uv` is unavailable, scripts fall back to `python`/`torchrun`.

---

## Experimental context

This repo accompanies: **Replicating and Extending SIDReasoner**  
Main findings:
- broad SIDReasoner trend reproduced,
- no-think variants can remain competitive,
- RL improves top-K accuracy but can increase exposure concentration.

---

## Citation / links

- Repository: https://github.com/Alongoodall/SIDReasoner  
- Extended artifacts/checkpoints: https://huggingface.co/Dam2/SIDReasoner_Ext/tree/main
