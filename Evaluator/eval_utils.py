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


def get_answer_distribution_figure(title_name, fig_write_path, dataframe, n_column=3):
    # dataframe = open_jsonl_as_dataframe(jsonl_path)
    if isinstance(dataframe, list):
        dataframe = pd.DataFrame(dataframe)

    # answer_counts = pd.value_counts(dataframe["answer"].values, sort=True)
    answer_counts = dataframe["answer"].value_counts(sort=True)
    correct_counts = calculate_correctness_per_answer(dataframe)

    # Calculate number of plots (subplots) needed
    num_plots = n_column
    batch_size = len(answer_counts) // num_plots
    if batch_size == 0:
        num_plots = 1
    elif len(answer_counts) % batch_size:
        num_plots += 1

    # Create subplots
    fig, axs = plt.subplots(1, num_plots, figsize=(num_plots*6, 10))
    fig_title = title_name + "\n" + f"Result(correct/total): {dataframe['result'].sum()}/{len(dataframe)}"
    fig.suptitle(fig_title)

    # Iterate over number of plots and create barh plots
    for i in range(num_plots):
        start = i*batch_size
        end = (i+1)*batch_size
        answer_counts_slice = answer_counts.iloc[start:end]
        bars = axs[i].barh(answer_counts_slice.index, answer_counts_slice.values, color="royalblue")
        correct_bars = axs[i].barh(
            answer_counts_slice.index, [correct_counts[a] for a in answer_counts_slice.index], color="darkred")
        axs[i].tick_params(axis="y", labelsize=10)

        # Add count values to each bar
        for bar, bar2 in zip(bars, correct_bars):
            axs[i].text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                        f'{bar2.get_width():.0f}/{bar.get_width():.0f}', 
                        va='center', ha='left')

        # Set x-axis limits to be the same as the first subplot
        if i == 0:
            xlim = axs[i].get_xlim()
        else:
            axs[i].set_xlim(xlim)

    plt.tight_layout()
    # plt.show()
    plt.savefig(fig_write_path, dpi=300)
    # print(correct_counts)
    return