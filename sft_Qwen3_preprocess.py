import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import sys
from typing import List
import numpy as np
import fire
import torch
import transformers
from peft import TrainableTokensConfig, get_peft_model
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback, AutoConfig, TrainerCallback
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import torch.nn as nn
import math
import warnings
from functools import partial
import numpy as np
import fire
import transformers
from torch.optim.lr_scheduler import LambdaLR
import json
import wandb
from contextlib import contextmanager
from datasets import DatasetDict

"""
Unused imports:`
import torch.nn as nn
import bitsandbytes as bnb
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_Qwen3 import (
    SFTData,
    SidSFTDataset,
    SidItemFeatDataset,
    FusionSeqRecDataset,
    TitleHistory2SidSFTDataset,
    SidTextInterleaveDataset,
    SidTextInterleaveDataset_v2,
    SidTextInterleaveSequenceDataset,
    GeneralSFTReasonDataset,
)
import random
from datasets import Dataset as HFDataset
from torch.utils.data import ConcatDataset



def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def build_hf_dataset(dataset):
    return HFDataset.from_dict({k: [v[k] for v in dataset] for k in dataset[0].keys()})


class TokenExtender:
    def __init__(self, data_path, dataset, index_file=".index.json"):
        self.data_path = data_path
        self.dataset = dataset
        self.index_file = index_file
        self.indices = None
        self.new_tokens = None

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + self.index_file), "r") as f:
            self.indices = json.load(f)

    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens

        if self.indices is None:
            self._load_data()

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens


def _decode_tokens(tokens, tokenizer_ref):
    if not isinstance(tokens, (list, tuple)):
        return ""
    valid_ids = [tid for tid in tokens if isinstance(tid, int) and tid >= 0]
    if not valid_ids:
        return ""
    return tokenizer_ref.decode(valid_ids, skip_special_tokens=False)


def _preview_dataset(dataset, name, tokenizer_ref, max_samples=3):
    print(f"[Preview] {name}: displaying up to {max_samples} samples")
    preview_count = min(max_samples, len(dataset))
    for idx in range(preview_count):
        sample = dataset[idx]
        input_text = ""
        # label_text = ""
        if isinstance(sample, dict):
            if "input_ids" in sample:
                input_text = _decode_tokens(sample["input_ids"], tokenizer_ref)
            if "labels" in sample:
                # Filter label padding tokens (e.g., -100) before decoding for readability
                label_ids = [tid for tid in sample["labels"] if isinstance(tid, int) and tid >= 0]
                label_text = _decode_tokens(label_ids, tokenizer_ref)
        print(f"Sample {idx + 1}:")
        if input_text:
            print(f"  Input : {input_text}")
        if label_text:
            print(f"  Label : {label_text}")
            print(f"  Length: {len(label_ids)} tokens")
        print()



def preprocess(
    # model/data params
    base_model: str = "Qwen/Qwen3-1.7B",
    train_file: str = "./data/Amazon/train/Office_Products_5_2016-10-2018-11.csv",
    eval_file: str = "./data/Amazon/valid/Office_Products_5_2016-10-2018-11.csv",
    output_dir: str = "./data",
    sample: int = -1,
    seed: int = 42,
    category: str = "Office_Products",
    cutoff_len: int = 1024,
    wandb_project: str = "MiniOneRec",
    train_from_scratch: bool = False,
    sid_index_path: str = "./data/Amazon/index/Office_Products.index.json",
    item_meta_path: str = "./data/Amazon/index/Office_Products.item.json",
    llm_generated_data_path: str = "./data/Amazon/index/Office_Products.item_enhanced_v2.json",
    llm_generated_sequence_path: str = "./data/Amazon/index/Office_Products.integrated_narrative.csv",
    general_reasoning_path: str = "./data/Amazon/general/sampled_data.arrow",
    mask_assistant: bool = True,  # Whether only the target response is used for loss calculation
    train_new_token_embeddings_only: bool = False,
):

    set_seed(seed)

    os.environ["WANDB_PROJECT"] = wandb_project
    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office_Products": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books",
        "Video_Games": "video games",
    }
    print(category)
    if category in category_dict:
        category = category_dict[category]
    else:
        category = "items"

    assert base_model, "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
        )
    else:
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)
        print("Training from scratch!")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if sid_index_path and os.path.exists(sid_index_path):
        print(f"Loading index from {sid_index_path}")
        token_extender = TokenExtender(
            data_path=os.path.dirname(sid_index_path), dataset=os.path.basename(sid_index_path).split(".")[0]
        )
        new_tokens = token_extender.get_new_tokens()
        if new_tokens:
            existing_vocab = set(tokenizer.get_vocab().keys())
            tokens_to_add = [tok for tok in new_tokens if tok not in existing_vocab]
            if tokens_to_add:
                print(f"Adding {len(tokens_to_add)} new tokens to tokenizer")
                tokenizer.add_tokens(tokens_to_add)
                model.resize_token_embeddings(len(tokenizer))
                num_new_tokens = len(tokens_to_add)
            else:
                print("All candidate tokens already exist in the tokenizer; skipping addition.")
                num_new_tokens = 0
        else:
            num_new_tokens = 0
    else:
        new_tokens = []
        num_new_tokens = 0

    if train_new_token_embeddings_only:
        if num_new_tokens > 0:
            vocab_size = len(tokenizer)
            new_token_indices = list(range(vocab_size - num_new_tokens, vocab_size))
            print(f"Restricting training to new token ids.")
            peft_config = TrainableTokensConfig(
                token_indices=new_token_indices,
                target_modules=["embed_tokens"],
                init_weights=True,
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
    else:
        print("Full fine-tuning enabled: attention blocks, FFNs, and embeddings remain trainable.")

    train_datasets = []
    # train_data1 = SFTData(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    train_data1 = SidSFTDataset(
        train_file=train_file,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
        mask_assistant=mask_assistant,
    )
    train_datasets.append(train_data1)
    train_data2 = SidItemFeatDataset(
        item_file=item_meta_path,
        index_file=sid_index_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
        mask_assistant=mask_assistant,
    )
    train_datasets.append(train_data2)
    train_data3 = FusionSeqRecDataset(
        train_file=train_file,
        item_file=item_meta_path,
        index_file=sid_index_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
        mask_assistant=mask_assistant,
    )
    train_datasets.append(train_data3)
    train_data4 = SFTData(
        train_file=train_file,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
        mask_assistant=mask_assistant,
    )
    train_datasets.append(train_data4)
    train_data5 = TitleHistory2SidSFTDataset(
        train_file=train_file,
        item_file=item_meta_path,
        index_file=sid_index_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
        mask_assistant=mask_assistant,
    )
    train_datasets.append(train_data5)

    if llm_generated_data_path is not None:
        train_data7 = SidTextInterleaveDataset_v2(
            json_file=llm_generated_data_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed
        )
        train_datasets.append(train_data7)
    if llm_generated_sequence_path is not None:
        train_data8 = SidTextInterleaveSequenceDataset(
            csv_file=llm_generated_sequence_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed
        )
        train_datasets.append(train_data8)
    if general_reasoning_path is not None:
        train_data9 = GeneralSFTReasonDataset(
            train_file=general_reasoning_path, tokenizer=tokenizer, max_len=3072, sample=60000, seed=seed
        )
        train_datasets.append(train_data9)

    train_data = ConcatDataset(train_datasets)

    main_rank = int(os.environ.get("RANK", 0)) == 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0
    if main_rank:
        train_dataset_names = [
            "SidSFTDataset",
            "SidItemFeatDataset",
            "FusionSeqRecDataset",
            "SFTData",
            "TitleHistory2SidSFTDataset",
            # "SidTextInterleaveDataset",
            "SidTextInterleaveDataset_v2",
            "SidTextInterleaveSequenceDataset",
            "GeneralSFTReasonDataset",
        ]
        for ds, name in zip(train_datasets, train_dataset_names):
            _preview_dataset(ds, name, tokenizer)

    val_data_sid_prediction = SidSFTDataset(
        train_file=eval_file,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
        test=False,
        mask_assistant=True,
    )
    val_data_title2sid_translation = SidItemFeatDataset(
        item_file=item_meta_path,
        index_file=sid_index_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
        task_type="title2sid",
        test=False,
        mask_assistant=True,
    )
    val_data_sid2title_translation = SidItemFeatDataset(
        item_file=item_meta_path,
        index_file=sid_index_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
        task_type="sid2title",
        test=False,
        mask_assistant=True,
    )
    print("LOAD DATA FINISHED")

    sample_frac = 1
    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_train_dataset = hf_train_dataset.shuffle(seed=42).select(range(int(sample_frac * len(hf_train_dataset))))
    hf_val_dataset = HFDataset.from_dict(
        {k: [v[k] for v in val_data_sid_prediction] for k in val_data_sid_prediction[0].keys()}
    ).shuffle(seed=seed)
    hf_val_dataset = hf_val_dataset.shuffle(seed=42)
    # additional eval set for translation performance
    hf_eval_dataset_title2sid_translation = HFDataset.from_dict(
        {k: [v[k] for v in val_data_title2sid_translation] for k in val_data_title2sid_translation[0].keys()}
    ).shuffle(seed=seed)
    hf_eval_dataset_title2sid_translation = hf_eval_dataset_title2sid_translation.shuffle(seed=42)
    hf_eval_dataset_sid2title_translation = HFDataset.from_dict(
        {k: [v[k] for v in val_data_sid2title_translation] for k in val_data_sid2title_translation[0].keys()}
    ).shuffle(seed=seed)
    hf_eval_dataset_sid2title_translation = hf_eval_dataset_sid2title_translation.shuffle(seed=42)

    dataset_dict = DatasetDict({
    "train": hf_train_dataset,
    "val": hf_val_dataset,
    "eval_title2sid": hf_eval_dataset_title2sid_translation,
    "eval_sid2title": hf_eval_dataset_sid2title_translation,
    })

    dataset_dict.save_to_disk(os.path.join(output_dir, "preprocessed"))


if __name__ == "__main__":
    import fire
    fire.Fire(preprocess)