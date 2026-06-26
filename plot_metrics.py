import json
import math
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

# adjustable paths
ITEM_PATH = "data/Amazon/info/Office_Products_5_2016-10-2018-11.txt"
ITEM_META = "data/Amazon/index/Office_Products.item.json"

CONDITIONS = {
    "S3":    "results/output_dir__Office_Products_S3_full_stage2__final_checkpoint/final_result_thinking_Office_Products.json",
    "S3+RL": "results/final_sidreasoner/final_result_Office_Products.json",
}

SASREC_NPY = "MiniOneRec/result_temp/Office_Products_SASRec_emb32_bs1024_lr0.001_decay1e-05_seed1_loss_bce_dropout0.3_logits.npy"
# hard-coded results
SASREC_ACC = {"R@10": 0.1153, "N@10": 0.0886}   

BEFORE_RL = ("S3 (SFT)", "results/final_sidreasoner/stage2.json")
AFTER_RL  = ("S3 + RL",  "results/final_sidreasoner/final_result_Office_Products.json")

K = 10

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

def load_item_set(item_path):
    if item_path.endswith(".txt"):
        item_path = item_path[:-4]
    with open(f"{item_path}.txt") as f:
        return {line.split("\t")[0].strip() for line in f}


def load_item_brands(item_path, meta_path):
    if item_path.endswith(".txt"):
        item_path = item_path[:-4]
    sid_to_id, sid_to_title = {}, {}
    with open(f"{item_path}.txt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 1:
                sid = parts[0].strip()
                if len(parts) >= 2:
                    sid_to_title[sid] = parts[1].strip()
                if len(parts) >= 3:
                    sid_to_id[sid] = parts[2].strip()
    meta = json.load(open(meta_path))
    sid_to_brand = {}
    for sid in sid_to_title:
        brand = meta.get(sid_to_id.get(sid, ""), {}).get("brand", "").strip()
        if not brand:
            title = sid_to_title.get(sid, "")
            brand = title.split()[0] if title else "Unknown"
        sid_to_brand[sid] = brand
    return sid_to_brand


def load_preds(path):
    with open(path) as f:
        data = json.load(f)
    preds = [[p.strip('"\n').strip() for p in s["predict"]] for s in data]
    return preds, data


def diversity_from_counter(counter, catalog_size):
    counts = np.array(sorted(counter.values()), dtype=float)
    total = counts.sum()
    m = len(counts)
    gini = (2 * np.sum(np.arange(1, m + 1) * counts) / (m * total)) - (m + 1) / m if total else 0
    p = counts / total if total else counts
    ent = -np.sum(p * np.log2(p + 1e-12))
    norm_ent = ent / math.log2(catalog_size) if catalog_size > 1 else 0
    cov = len(counter) / catalog_size if catalog_size else 0
    return {"Gini": gini, "Entropy": norm_ent, "Coverage": cov}, counts


def llm_metrics(preds, data, item_set, k=K):
    n = len(preds)
    hr = ndcg = 0.0
    counter = Counter()
    for sample, row in zip(preds, data):
        tgt = row["output"]
        tgt = (tgt[0] if isinstance(tgt, list) else tgt).strip(' \n"')
        for pred in sample[:k]:
            if pred in item_set:
                counter[pred] += 1
        for i, pred in enumerate(sample[:k]):
            if pred == tgt:
                hr += 1
                ndcg += 1.0 / math.log2(i + 2)
                break
    div, _ = diversity_from_counter(counter, len(item_set))
    return {"R@10": hr / n, "N@10": ndcg / n, **div}, counter


def sasrec_counter(npy_path, k=K):
    logits = np.load(npy_path)
    topk = np.argpartition(-logits, kth=k, axis=1)[:, :k]  
    counter = Counter()
    for row in topk:
        counter.update(int(i) for i in row)
    return counter, logits.shape[1]            


def collect_rows(item_set):
    rows = {}
    for label, path in CONDITIONS.items():
        preds, data = load_preds(path)
        rows[label], _ = llm_metrics(preds, data, item_set)
    sc, cat = sasrec_counter(SASREC_NPY)
    sdiv, _ = diversity_from_counter(sc, cat)
    rows["SASRec"] = {**SASREC_ACC, **sdiv}
    labels = ["SASRec"] + [l for l in rows if l != "SASRec"]  
    return rows, labels


def fig_accuracy(item_set):
    rows, labels = collect_rows(item_set)
    fig, ax = plt.subplots(figsize=(5, 4))
    for metric, color in [("R@10", "tab:blue"), ("N@10", "tab:orange")]:
        ax.plot(labels, [rows[l][metric] for l in labels], "o-", label=metric, color=color)
    ax.set_title("Accuracy")
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig_accuracy.png")
    print("wrote fig_accuracy.png")


def fig_diversity(item_set):
    rows, labels = collect_rows(item_set)
    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    

    for i, (metric, color) in enumerate([("Gini", "#6a51a3"), ("Entropy", "#2d8659"), ("Coverage", "#d08a1d")]):
        bars = ax.bar(x + (i - 1) * w, [rows[l][metric] for l in labels], w, label=metric, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)  
    #ax.set_title("Diversity")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig("fig_diversity.png")
    print("wrote fig_diversity.png")


def fig4_brand_shift(item_set, sid_to_brand, top_n=12):
    predsB, dataB = load_preds(BEFORE_RL[1])
    _, cB = llm_metrics(predsB, dataB, item_set)
    predsA, dataA = load_preds(AFTER_RL[1])
    _, cA = llm_metrics(predsA, dataA, item_set)

    def by_brand(counter):
        bc = Counter()
        for sid, n in counter.items():
            bc[sid_to_brand.get(sid, "Unknown")] += n
        return bc

    bB, bA = by_brand(cB), by_brand(cA)
    top_brands = [b for b, _ in bA.most_common(top_n)]
    before = [bB.get(b, 0) for b in top_brands]
    after = [bA.get(b, 0) for b in top_brands]

    y = np.arange(len(top_brands))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(y - 0.2, before, 0.4, label=BEFORE_RL[0], color="#9bbcd9")
    ax.barh(y + 0.2, after, 0.4, label=AFTER_RL[0], color="#3b6ea5")
    ax.set_yticks(y); ax.set_yticklabels(top_brands, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Times recommended (top-10)")
    ax.set_title(f"Top-{top_n} brands: exposure before vs after RL")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig_brand_shift.png")
    print("wrote fig_brand_shift.png")


def fig5_demand_vs_supply(item_set, sid_to_brand, json_path, top_n=12,
                          out="fig_demand_supply.png"):
    preds, data = load_preds(json_path)

    gt = Counter()       
    rec = Counter()    
    for sample, row in zip(preds, data):
        tgt = row["output"]
        tgt = (tgt[0] if isinstance(tgt, list) else tgt).strip(' \n"')
        gt[sid_to_brand.get(tgt, "Unknown")] += 1
        for pred in sample[:K]:
            if pred in item_set:
                rec[sid_to_brand.get(pred, "Unknown")] += 1

    gt_total, rec_total = sum(gt.values()), sum(rec.values())
    top_brands = [b for b, _ in gt.most_common(top_n)]      # rank by true share
    demand = [gt.get(b, 0) / gt_total for b in top_brands]
    supply = [rec.get(b, 0) / rec_total for b in top_brands]

    y = np.arange(len(top_brands))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(y - 0.2, demand, 0.4, label="In data (ground truth)", color="#9bbcd9")
    ax.barh(y + 0.2, supply, 0.4, label="Recommended (top-10)", color="#3b6ea5")
    ax.set_yticks(y); ax.set_yticklabels(top_brands, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Share of occurrences")
    ax.set_title("Brand distribution: data vs. recommendations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    item_set = load_item_set(ITEM_PATH)
    brands = load_item_brands(ITEM_PATH, ITEM_META)
    fig_accuracy(item_set)
    fig_diversity(item_set)
    fig4_brand_shift(item_set, brands)
    fig5_demand_vs_supply(item_set, brands, AFTER_RL[1])