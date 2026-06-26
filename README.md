# SIDReasoner Reproduction + Extensions (Local-Executable)

This repository reproduces and extends SIDReasoner on Amazon Office Products, with a focus on **local reproducibility** (no SLURM required).

## Hugging Face 🤗

Extended checkpoints/artifacts are available at:  
**https://huggingface.co/Dam2/SIDReasoner_Ext/tree/main**

---

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

## Stage 1 — SFT preprocessing
```bash
bash scripts/preprocess_sft.sh \
  CATEGORY=Office_Products \
  BASE_MODEL=Qwen/Qwen3-0.6B
```

## Stage 1 — SFT training
```bash
bash scripts/train_sft.sh \
  CATEGORY=Office_Products \
  BASE_MODEL=Qwen/Qwen3-0.6B \
  CUDA_DEVICES=0,1,2,3 \
  NPROC_PER_NODE=4
```

## Stage 2 — Reasoning activation
```bash
bash scripts/sft_reasoning_activation.sh \
  CATEGORY=Office_Products \
  BASE_MODEL=./output_dir/Office_Products_stage1_sft_Qwen3-0.6B/final_checkpoint \
  CUDA_DEVICES=0,1,2,3 \
  NPROC_PER_NODE=4
```

## Stage 3 — RL (reasoning)
```bash
bash scripts/RL_training_script.sh \
  CATEGORY=Office_Products \
  STAGE2_CHECKPOINT=./output_dir/Office_Products_stage2_reasoning_activation_Qwen3-1.7B/final_checkpoint \
  N_GPUS_PER_NODE=4 \
  NNODES=1
```

## Create direct/no-thinking RL data
```bash
python scripts/create_direct_rl_data.py \
  --category Office_Products \
  --category_label "office products"
```

## Create reasoning RL data
```bash
python scripts/create_reasoning_rl_data.py \
  --category Office_Products
```

## Merge FSDP checkpoint (single step)
```bash
bash scripts/merge_fsdp_ckpt.sh \
  CKPT_DIR=./checkpoints/RecRL_Reasoning/Office_Products_stage3_rl_Qwen3-1.7B/global_step_100/actor
```

## Merge all periodic checkpoints
```bash
bash scripts/merge_fsdp_ckpt_ALL.sh \
  CKPT_ROOT=./checkpoints/RecRL_Reasoning/Office_Products_stage3_rl_Qwen3-1.7B \
  EVAL_INTERVAL=100
```

---

## Evaluation commands

## Standard (no-think) eval
```bash
bash scripts/evaluate_Qwen3.sh \
  CATEGORY=Office_Products \
  EXP_NAME=./output_dir/Office_Products_stage1_sft_Qwen3-1.7B/final_checkpoint \
  CUDA_LIST="0 1" \
  CUDA_LIST_CSV="0,1"
```

## Think eval (batch/multi-model)
```bash
bash scripts/evaluate_Qwen3_think_batch.sh \
  CATEGORIES=Office_Products \
  EXP_LIST="./output_dir/Office_Products_stage2_reasoning_activation_Qwen3-1.7B/final_checkpoint" \
  CUDA_LIST="0 1 2" \
  CUDA_LIST_CSV="0,1,2"
```

---

## Extensions included in this repo

1. **Stage-2 alignment ablations before RL** (S1/S2/S3)
2. **Diversity + concentration diagnostics**
   - Gini
   - normalized entropy
   - coverage
   - Lorenz-curve-style interpretation
3. **Direct no-thinking GRPO**
   - no `</think>` dependency in reward parsing
   - initialized from:
     - SFT checkpoint
     - reasoning-activation checkpoint
4. **Enriched-corpus decomposition**
   - item-centric only
   - sequence-centric only
   - both
5. **Cross-model checks**
   - Qwen3-1.7B / Qwen3-0.6B

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
