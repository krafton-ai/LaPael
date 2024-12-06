import string
import json
import numpy as np

import torch

from collections import defaultdict


def get_token_to_word(tokenizer, sentence):
    words = sentence.split()
    tokens = tokenizer.tokenize(sentence, add_special_tokens=False)
    token_to_word = dict()
    new_tokens = []
    for i, word in enumerate(words):
        _tokens = tokenizer.tokenize(word, add_special_tokens=False)
        for j, _token in enumerate(_tokens):
            token_idx = len(new_tokens) + j
            token_to_word[token_idx] = i
        new_tokens.extend(_tokens)
    assert tokens == new_tokens

    for k, v in token_to_word.items():
        print(f"Token {k} ({tokens[k]}) maps to Word {v} ({words[v]})")
    return token_to_word

def get_char_to_token(tokenizer, sentence, verbose=False):
    char_to_token = dict()
    tokenized = tokenizer(sentence, add_special_tokens=False, return_offsets_mapping=True)

    for i, (start, end) in enumerate(tokenized.offset_mapping):
        for _idx in range(start, end):
            char_to_token[_idx] = i
    tokens = tokenizer.convert_ids_to_tokens(tokenized.input_ids)
    if verbose:
        for k, v in char_to_token.items():
            print(f"Char {k} ({sentence[k]}) maps to Token {v} ({tokens[v]})")
    return char_to_token

def get_augmentations(aug_type, domain):
    mother_path = f"./Knowledge_Generator/openai_outputs/{domain}_oracle_sent/paraphrase_{{variant}}_c256_gen20_gpt-3.5-turbo.json"
    _path = mother_path.format(variant=aug_type)

    with open(_path, 'r') as f:
        dataset = json.load(f)

    original_to_paraphrases = defaultdict(list)
    for data in dataset:
        new_data = {"output": data["output"]}
        if "suffix" in data.keys():
            new_data["suffix"] = data["suffix"]
        original_to_paraphrases[data["data_id"]].append(new_data)
    return original_to_paraphrases

def get_augmentations_contextqa(aug_type, domain):
    try:
        mother_path = f"./Knowledge_Generator/openai_outputs/{domain}/paraphrase_{{variant}}_c256_gen10_gpt-3.5-turbo.json"
        _path = mother_path.format(variant=aug_type)
        with open(_path, 'r') as f:
            dataset = json.load(f)
    except:
        mother_path = f"./Knowledge_Generator/openai_outputs/{domain}/paraphrase_{{variant}}_c256_gen20_gpt-3.5-turbo.json"
        _path = mother_path.format(variant=aug_type)
        with open(_path, 'r') as f:
            dataset = json.load(f)

    original_to_paraphrases = defaultdict(list)
    for data in dataset:
        new_data = {"output": data["output"]}
        if "suffix" in data.keys():
            new_data["suffix"] = data["suffix"]
        original_to_paraphrases[data["data_id"]].append(new_data)
    return original_to_paraphrases


def find_sublist(main_list, target_list):
    # Length of the target list
    target_len = len(target_list)
    # Iterate over the main list
    for i in range(len(main_list) - target_len + 1):
        # Check if the slice of main_list starting at i matches target_list
        if main_list[i:i + target_len] == target_list:
            return i  # Return the starting index
    return -1  # Return -1 if no match is found

def return_suffix_embedding(model, tokenizer, sentence, suffix, device="cuda"):
    """
    Generate an embedding for a specific answer within a given sentence using a pre-trained model.

    This function takes a sentence and an answer, tokenizes the sentence, and uses the model to generate
    embeddings. It then extracts and returns the embedding corresponding to the answer's location in the sentence.

    Parameters:
    - model: The pre-trained model to use for generating embeddings.
    - tokenizer: The tokenizer corresponding to the pre-trained model.
    - sentence: A string containing the sentence to analyze.
    - answer: A string containing the answer to locate within the sentence.
    - answer_index: A pre-defined answer index 
        (in case we need to specify the correct place if there exist multiple answers in the sentence)

    Returns:
    - Tensor representing the embedding of the answer within the context of the given sentence.
    """

    if suffix not in sentence: return None # REMOVE it

    input_tokens = tokenizer(sentence)["input_ids"]
    suffix_tokens = tokenizer(suffix, add_special_tokens=False)["input_ids"]

    suffix_start_idx = find_sublist(input_tokens, suffix_tokens)

    if suffix_start_idx < 0: return None

    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    suffix_hidden_states_list = []
    for hidden_states in outputs.hidden_states:
        suffix_hidden_states = hidden_states[0, suffix_start_idx-1] # This is the hidden state for predicting the "first" token of suffix
        suffix_hidden_states_list.append(suffix_hidden_states.cpu())

    return suffix_hidden_states_list

def deduplicate_sentence(sentences):
    deduplicated_sentences = list(set(sentences))
    return deduplicated_sentences