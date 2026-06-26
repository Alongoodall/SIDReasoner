# from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
# import transformers
# import torch
import os
import fire
import math
import json
import pandas as pd
import numpy as np

from tqdm import tqdm


def diversity_metrics(rec_counts, catalog_size, eps=1e-12):
    counts = np.array(sorted(rec_counts.values()), dtype=np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0, 0.0, 0.0

    n = len(counts)
    gini = (2.0 * np.sum(np.arange(1, n + 1) * counts) / (n * total)) - (n + 1.0) / n

    p = counts / total
    entropy = -np.sum(p * np.log2(p + eps))
    norm_entropy = entropy / math.log2(catalog_size) if catalog_size > 1 else 0.0

    coverage = len(rec_counts) / catalog_size if catalog_size > 0 else 0.0
    return gini, norm_entropy, coverage

def gao(path, item_path):
    if type(path) != list:
        path = [path]
    if item_path.endswith(".txt"):
        item_path = item_path[:-4]
    CC = 0

    f = open(f"{item_path}.txt", "r")
    items = f.readlines()
    item_names = [_.split("\t")[0].strip() for _ in items]
    item_ids = [_ for _ in range(len(item_names))]
    item_dict = dict()
    for i in range(len(item_names)):
        if item_names[i] not in item_dict:
            item_dict[item_names[i]] = [item_ids[i]]
        else:
            item_dict[item_names[i]].append(item_ids[i])

    result_dict = dict()
    topk_list = [1, 3, 5, 10, 20, 50]
    n_beam = -1
    for p in path:
        result_dict[p] = {"NDCG": [], "HR": []}
        f = open(p, "r")
        test_data = json.load(f)
        f.close()

        text = [[_.strip('"\n').strip() for _ in sample["predict"]] for sample in test_data]

        rec_counter = {}        # item -> times it appears in a top-K list
        diversity_topk = 10     # cutoff used for diversity metrics

        for index, sample in tqdm(enumerate(text)):
            if n_beam == -1:
                n_beam = len(sample)
                valid_topk = [k for k in topk_list if k <= n_beam]
                ALLNDCG = np.zeros(len(valid_topk))
                ALLHR = np.zeros(len(valid_topk))
            if type(test_data[index]["output"]) == list:
                target_item = test_data[index]["output"][0].strip('"').strip(" ")
            else:
                target_item = test_data[index]["output"].strip(' \n"')

            for pred in sample[:diversity_topk]:
                if pred in item_dict:
                    rec_counter[pred] = rec_counter.get(pred, 0) + 1

            minID = 1000000
            for i in range(len(sample)):
                if sample[i] not in item_dict:
                    CC += 1
                    print(sample[i])
                    print(target_item)
                if sample[i] == target_item:
                    minID = i
                    break

            for index, topk in enumerate(topk_list):
                if topk > n_beam:
                    continue
                if minID < topk:
                    ALLNDCG[index] = ALLNDCG[index] + (1 / math.log(minID + 2))
                    ALLHR[index] = ALLHR[index] + 1
        print(n_beam)
        valid_topk = [k for k in topk_list if k <= n_beam]
        print(valid_topk)
        print(f"NDCG:\t{ALLNDCG / len(text) / (1.0 / math.log(2))}")
        print(f"HR\t{ALLHR / len(text)}")
        print(CC)

        catalog_size = len(item_dict)
        gini, norm_entropy, coverage = diversity_metrics(rec_counter, catalog_size)
        print(f"Gini@{diversity_topk}:\t\t{gini:.4f}")
        print(f"NormEntropy@{diversity_topk}:\t{norm_entropy:.4f}")
        print(f"Coverage@{diversity_topk}:\t{coverage:.4f}")


if __name__ == "__main__":
    fire.Fire(gao)

    # # debugging
    # data_path = "./results/global_step_50__actor_merged/final_result_Video_Games.json"
    # item_path = "./data/Amazon_Games/info/Video_Games_5_2016-10-2018-11.txt"
    # gao(data_path, item_path)
