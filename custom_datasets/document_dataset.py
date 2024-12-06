from typing import List, Dict, Union
import datasets
import itertools
import json
from pathlib import Path
from tqdm import tqdm
import copy
import random
import numpy as np
import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset


class DocumentDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split="train"):
        self.tokenizer = tokenizer
        self.context_length = dataset_config.context_length
        self.max_original_paraphrases = dataset_config.max_original_paraphrases
        self.model_max_length = self.context_length
        self.no_augment = dataset_config.no_augment
        self.repeat_dataset = dataset_config.repeat_dataset
        self.domain = dataset_config.domain

        root_path = dataset_config.dataset_path

        with open(root_path, 'r') as f:
            dataset = json.load(f)

        # dataset = self.chunk_by_toklen(book)

        tok_add_bos_origin = self.tokenizer.add_bos_token
        self.tokenizer.add_bos_token = False

        n_paraphrases = dataset_config.n_paraphrases # change to ?
        original_dataset_raw = []
        original_dataset = dict()
        for data in dataset:
            if data["data_id"] not in original_dataset.keys():
                original_dataset[data["data_id"]] = self.tokenize(data["content"])
                original_dataset_raw.append(data)

        datalen = len(original_dataset)

        if dataset_config.debug:
            key_list = list(original_dataset.keys())
            for k in key_list: 
                if k > 20: del original_dataset[k]

        data_to_paraphrase = dict()
        for data in tqdm(dataset, desc=f"{split}: Tokenize paraphrases..."):
            if data["data_id"] not in original_dataset.keys(): break
            
            tokenized_data = self.tokenize(data["output"])
            if data["data_id"] in data_to_paraphrase.keys():
                data_to_paraphrase[data["data_id"]].append(tokenized_data)
            else:
                data_to_paraphrase[data["data_id"]] = [tokenized_data]

        self.tokenized_datas = []
        for key in original_dataset.keys():
            paraphrases = data_to_paraphrase[key]

            _data = original_dataset[key]
            _data["data_id"] = key
            self.tokenized_datas.append(_data)

            if not self.no_augment:
                for i, para in enumerate(paraphrases):
                    para["data_id"] = key
                    self.tokenized_datas.append(para)
                    if i >= self.max_original_paraphrases:
                        break

        self.unique_datalen = len(original_dataset)

        self.tokenizer.add_bos_token = tok_add_bos_origin

        if self.repeat_dataset > 0:
            print(" ### BEFORE repeat: ", len(self.tokenized_datas))
            
            original_tokenized_datas = self.tokenized_datas
            new_tokenized_datas = []
            for repeat_i in range(self.repeat_dataset):
                copied_tokenized_datas = copy.deepcopy(original_tokenized_datas)
                for data in copied_tokenized_datas:
                    if repeat_i == 0:
                        data["original"] = True
                    else:
                        data["original"] = False
                new_tokenized_datas += copied_tokenized_datas
            
            self.tokenized_datas = new_tokenized_datas
            print(" ### AFTER repeat: ", len(self.tokenized_datas))

        self.input_ids_max_length = 0
        for data in self.tokenized_datas:
            if len(data["continuation_tokens"]) > self.input_ids_max_length:
                self.input_ids_max_length = len(data["continuation_tokens"])
        print("Max length for input ids: ", self.input_ids_max_length)

        datalen = len(self.tokenized_datas)
        print("### Dataset size: ", datalen)

        self.tokenized_datas = {
            # "train": tokenized_datas[:int(0.95 * datalen)],
            "train": self.tokenized_datas,
            "validation": self.tokenized_datas[int(0.95 * datalen):]
        }
        self.split = split

    def tokenize(self, text: List[str]):
        self.model_max_length = self.context_length
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.model_max_length:
            tokens = tokens[:self.model_max_length]
        return {
            "prompt_tokens": [],
            "prompt_token_length": 0,
            "continuation_tokens": tokens,
            "continuation_token_length": len(tokens),
        }

    def build_entity_mask(self, tokens, entities):
        def find_sublist_index(main_list, sublist):
            sublist_length = len(sublist)
            return next((i for i in range(len(main_list) - sublist_length + 1) 
                        if main_list[i:i+sublist_length] == sublist), -1)

        entity_mask = [0 for _ in range(len(tokens))]
        for entity in entities:
            entity_tokens = self.tokenizer.encode(entity)
            entity_index = find_sublist_index(tokens, entity_tokens)
            if entity_index < 0: continue
            for entity_idx in range(entity_index, entity_index+len(entity_tokens)):
                entity_mask[entity_idx] = 1

        return entity_mask
    
    def convert_to_features(self, batch):
        prompt_tokens = batch["prompt_tokens"]
        cont_tokens = batch["continuation_tokens"]

        input_ids = prompt_tokens + cont_tokens + [self.tokenizer.eos_token_id]
        if "label_masks" in batch.keys():
            label_masks = batch["label_masks"] + [False]
        else:
            label_masks = [True] * len(prompt_tokens) + [False] * len(cont_tokens) + [False] # 1 = -100, 0 = id
        if self.tokenizer.add_bos_token:
            input_ids = [self.tokenizer.bos_token_id] + input_ids
            label_masks = [True] + label_masks
        
        # Padding
        padding_length = self.model_max_length + 1 - len(input_ids) # considering eos
        if padding_length > 0:
            input_ids = input_ids + [0] * padding_length
            # input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            label_masks = label_masks + [True] * padding_length
        elif padding_length < 0:
            input_ids = input_ids[:self.model_max_length+1] # considering EOS
            label_masks = label_masks[:self.model_max_length+1] # considering EOS

        input_ids = torch.LongTensor(input_ids)
        label_masks = torch.BoolTensor(label_masks)
        labels = torch.masked_fill(input_ids.clone(), label_masks, -100)
        # attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        attention_mask = input_ids.ne(0).long()

        return_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return return_dict

    def __len__(self):
        return len(self.tokenized_datas[self.split])

    def __getitem__(self, index):
        sample = self.tokenized_datas[self.split][index]
        featured_sample = self.convert_to_features(sample)
        return featured_sample

    
def get_custom_dataset(dataset_config, tokenizer, split):
    dataset = DocumentDataset(
        dataset_config, tokenizer, split
    )
    return dataset