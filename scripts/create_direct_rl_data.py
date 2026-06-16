import argparse
import ast
import os

import pandas as pd


def build_rows(csv_path, split, data_source, category):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {split} CSV: {csv_path}")

    data = pd.read_csv(csv_path)
    rows = []
    for idx, row in data.iterrows():
        history = ast.literal_eval(row["history_item_sid"])
        history_str = ", ".join(history)
        target = str(row["item_sid"])
        prompt = [
            {
                "role": "system",
                "content": (
                    "Below is an instruction that describes a task, paired with an input that provides further "
                    "context. Write a response that appropriately completes the request."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"The user has interacted with items {history_str} in chronological order. "
                    f"Can you predict the next possible {category} item? Directly output the item SID."
                ),
            },
        ]
        rows.append(
            {
                "data_source": data_source,
                "prompt": prompt,
                "ability": "Recommendation",
                "reward_model": {"style": "rule", "ground_truth": target},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": target,
                    "question": prompt,
                },
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Create direct/no-thinking VERL RL data for SIDReasoner.")
    parser.add_argument("--train_data", default="./data/Amazon/train/Office_Products_5_2016-10-2018-11.csv")
    parser.add_argument("--eval_data", default="./data/Amazon/test/Office_Products_5_2016-10-2018-11.csv")
    parser.add_argument("--local_dir", default="./data/Amazon/rec_direct_verl/Office_Products")
    parser.add_argument("--category", default="Office_Products")
    parser.add_argument("--category_label", default="office products")
    parser.add_argument("--data_source", default="rec/Amazon/Office_Products/direct")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    train_path = os.path.join(args.local_dir, "train.parquet")
    test_path = os.path.join(args.local_dir, "test.parquet")

    pd.DataFrame(build_rows(args.train_data, "train", args.data_source, args.category_label)).to_parquet(
        train_path, index=False
    )
    pd.DataFrame(build_rows(args.eval_data, "test", args.data_source, args.category_label)).to_parquet(
        test_path, index=False
    )

    print(f"Saved direct/no-thinking train data to {train_path}")
    print(f"Saved direct/no-thinking test data to {test_path}")


if __name__ == "__main__":
    main()
