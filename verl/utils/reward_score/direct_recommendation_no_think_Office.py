import os
import re
from collections import defaultdict

_SOLUTION_CLIP_CHARS = 50


def extract_sid_tokens(s: str) -> list[str]:
    return re.findall(r"<[^>]+>", s)


def extract_solution(solution_str):
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if "</think>" in solution_str:
        solution_str = solution_str.split("</think>")[-1]

    answer_sids = extract_sid_tokens(solution_str)[:3]
    if len(answer_sids) == 3:
        return answer_sids
    return None


def calculate_reward(answer_sids, ground_truth_sids):
    current_score = 0.0
    if answer_sids[0] == ground_truth_sids[0]:
        current_score += 0.25
        if answer_sids[1] == ground_truth_sids[1]:
            current_score *= 2
            if answer_sids[2] == ground_truth_sids[2]:
                current_score *= 2
    return current_score


def construct_prefix_allowed_hashmap(item_info_path):
    sid_pattern = re.compile(r"<[^>]+>")
    prefix_map = defaultdict(set)

    with open(item_info_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        sid_list = sid_pattern.findall(line.split("\t")[0].strip())
        if len(sid_list) != 3:
            continue
        a, b, c = sid_list
        prefix_map[(a,)].add(b)
        prefix_map[(a, b)].add(c)

    return {prefix: list(next_tokens) for prefix, next_tokens in prefix_map.items()}


def calculate_format_reward(answer_sids, prefix_map):
    if len(answer_sids) < 3:
        return False
    a, b, c = answer_sids[:3]
    return (a,) in prefix_map and b in prefix_map[(a,)] and (a, b) in prefix_map and c in prefix_map[(a, b)]


class MyRewardComputer:
    def __init__(self):
        item_info_path = os.environ.get(
            "SID_INFO_PATH", "./data/Amazon/info/Office_Products_5_2016-10-2018-11.txt"
        )
        self.sid_hash = construct_prefix_allowed_hashmap(item_info_path)

    def compute(self, data_source: str, solution_str: str, ground_truth: str, extra_info: dict | None = None) -> float:
        answer = extract_solution(solution_str=solution_str)
        ground_truth = extract_sid_tokens(ground_truth)[:3]

        if answer is None:
            return 0
        return calculate_reward(answer, ground_truth) + 0.1 * calculate_format_reward(answer, self.sid_hash)


_reward_computer: MyRewardComputer | None = None


def _get_reward_computer() -> MyRewardComputer:
    global _reward_computer
    if _reward_computer is None:
        _reward_computer = MyRewardComputer()
    return _reward_computer


def rule_base_reward(data_source, solution_str, ground_truth, extra_info=None):
    return _get_reward_computer().compute(data_source, solution_str, ground_truth, extra_info)
