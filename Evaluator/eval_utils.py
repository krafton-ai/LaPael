import json
import os
import re
from collections import Counter, defaultdict
import pandas as pd
# import matplotlib.pyplot as plt


re_art = re.compile(r'\b(a|an|the)\b')


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    s = s.lower()
    # s = re.sub('/.+|\(.+\)', ' ', s)
    s = re.sub('[^A-Za-z0-9가-힣]', ' ', s)
    s = re_art.sub(' ', s)
    # TODO: this could almost certainly be faster with a regex \s+ -> ' '
    s = ' '.join(s.split())
    return s


def normalize_generation(gen_str):
    gen_str = gen_str.lower()
    gen_str = re.sub('[^A-Za-z0-9가-힣]', ' ', gen_str)
    gen_str = ' '.join(gen_str.split())
    return gen_str


def recall(pred_items, gold_items):
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.

    recall = num_same / len(gold_items)
    return recall

def exact_match(pred_items, gold_items):
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.

    recall = num_same == len(gold_items) and num_same == len(pred_items)
    return recall

def f1(pred_items, gold_items):
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if len(gold_items) == 0 or len(pred_items) == 0:
        return int(gold_items == pred_items)
    if num_same == 0:
        return 0.
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# save images
def open_jsonl_as_dataframe(jsonl_path):
    with open(jsonl_path, "r") as f:
        lines = f.readlines()
    jsonl_list = [json.loads(l) for l in lines]
    
    dataframe = pd.DataFrame(jsonl_list)
    return dataframe


def calculate_correctness_per_answer(dataframe):
    result = defaultdict(int)
    for i, row in dataframe.iterrows():
        if row["result"]:
            result[row["answer"]] += 1
        else:
            result[row["answer"]] += 0
    return result