from typing import List, Dict, Union
import copy
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

class TextDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split="train"):
        self.tokenizer = tokenizer
        self.domain = dataset_config.domain
        self.context_length = dataset_config.context_length
        self.max_original_paraphrases = dataset_config.max_original_paraphrases
        self.model_max_length = self.context_length
        self.no_augment = dataset_config.no_augment
        self.use_oracle_entity = dataset_config.use_oracle_entity
        self.use_mse_loss = dataset_config.use_mse_loss
        self.embedding_type = dataset_config.embedding_type
        self.perturb_dataset_ratio = dataset_config.perturb_dataset_ratio
        self.model_name = dataset_config.model_name

        self.num_suffix_words = 5

        root_path = dataset_config.dataset_path
        with open(root_path, 'r') as f:
            dataset = json.load(f)

        # dataset = self.chunk_by_toklen(book)

        tok_add_bos_origin = self.tokenizer.add_bos_token
        self.tokenizer.add_bos_token = False

        if self.model_name == "meta-llama/Llama-2-7b-hf":
            self.model_name = "llama"
        elif self.model_name == "lmsys/vicuna-7b-v1.5":
            self.model_name = "vicuna"
        elif self.model_name == "microsoft/Phi-3-mini-4k-instruct":
            self.model_name = "phi3"
        elif self.model_name == "mistralai/Mistral-7B-Instruct-v0.2":
            self.model_name = "mistral"
        else:
            raise NotImplementedError

        self.field = "ContextQA"
        qa_path = f"./Evaluator/{self.field}/{self.domain}/test.jsonl"
        with open(qa_path, 'r') as f:
            qa_dataset = [json.loads(data.strip()) for data in f.readlines()]

        n_paraphrases = dataset_config.n_paraphrases # change to ?
        original_dataset_raw = []
        original_dataset = dict()
        for data in dataset:
            if data["data_id"] not in original_dataset.keys():
                tokenized_data = self.tokenize(data["content"], qa_dataset[data["data_id"]], verbose=True)
                if tokenized_data is None: continue
                original_dataset[data["data_id"]] = tokenized_data
                original_dataset_raw.append(data)

        datalen = len(original_dataset)

        if dataset_config.debug:
            key_list = list(original_dataset.keys())
            for k in key_list: 
                if k > 50: del original_dataset[k]
    
        data_to_paraphrase = dict()
        n_total = 0
        n_skipped = 0
        for data in tqdm(dataset, desc=f"{split}: Tokenize paraphrases..."):
            if data["data_id"] not in original_dataset.keys(): continue
            
            tokenized_data = self.tokenize(data["output"], qa_dataset[data["data_id"]])
            # tokenized_data = self.tokenize(data["output"], self.num_suffix_words)
            if tokenized_data is not None:
                if data["data_id"] in data_to_paraphrase.keys():
                    data_to_paraphrase[data["data_id"]].append(tokenized_data)
                else:
                    data_to_paraphrase[data["data_id"]] = [tokenized_data]
            else:
                n_skipped += 1
            n_total += 1

        print("### N Total", n_total)
        print("### N skipped", n_skipped)

        if self.model_name == "llama":
            target_embeddings_dir = f"./Analysis/embeddings_llama/{self.domain}_{self.embedding_type}"
        elif self.model_name == "phi3":
            target_embeddings_dir = f"./Analysis/embeddings_phi3/{self.domain}_{self.embedding_type}"
        elif self.model_name == "mistral":
            target_embeddings_dir = f"./Analysis/embeddings_mistral/{self.domain}_{self.embedding_type}"
        elif self.model_name == "vicuna":
            target_embeddings_dir = f"./Analysis/embeddings/{self.domain}_{self.embedding_type}"
        else:
            raise NotImplementedError

        print("### Target Embedding Directory: ", target_embeddings_dir)

        self.tokenized_datas = []
        # for key in tqdm(original_dataset.keys(), desc="[perturbation_dataset] load data..."):
        for key in original_dataset.keys():
            embedding_path = os.path.join(target_embeddings_dir, str(key).zfill(5) + ".pt")
            if os.path.exists(embedding_path):
                target_embedding = torch.load(embedding_path)
            else:
                target_embedding = None

            _data = original_dataset[key]
            _data["target_embedding"] = target_embedding
            _data["data_id"] = key
            self.tokenized_datas.append(_data)

            if key not in data_to_paraphrase.keys():
                continue
            paraphrases = data_to_paraphrase[key]

            if not self.no_augment:
                for i, para in enumerate(paraphrases):
                    para["target_embedding"] = target_embedding
                    para["data_id"] = key
                    self.tokenized_datas.append(para)
                    if i >= self.max_original_paraphrases:
                        break

        if self.use_oracle_entity:
            entity_path = f"./Data_Preprocessing/oracle_textbook/{self.domain}/entities_gpt-3.5-turbo.json"
            with open(entity_path, 'r') as f:
                entities = json.load(f)

            for data in self.tokenized_datas:
                data_id = data["data_id"]
                original_entities = entities[int(data_id)]["output"]
                entity_mask = self.build_entity_mask(data["continuation_tokens"], original_entities)
                data["entity_mask"] = entity_mask

        self.tokenizer.add_bos_token = tok_add_bos_origin

        print(" ### Before cleaning ", len(self.tokenized_datas))
        self.tokenized_datas = [data for data in self.tokenized_datas if data["suffix_idx"] >= 0 and data["target_embedding"] is not None]
        print(" ### After cleaning ", len(self.tokenized_datas))

        self.input_ids_max_length = 0
        for data in self.tokenized_datas:
            if len(data["continuation_tokens"]) > self.input_ids_max_length:
                self.input_ids_max_length = len(data["continuation_tokens"])
        print("Max length for input ids: ", self.input_ids_max_length)

        # Use the partial dataset
        if self.perturb_dataset_ratio < 100.0:
            train_datalen = int(len(self.tokenized_datas) * (self.perturb_dataset_ratio / 100))
            self.tokenized_datas = self.tokenized_datas[:train_datalen]

        datalen = len(self.tokenized_datas)

        print("### Number of Total Dataset: ", datalen)

        self.tokenized_datas = {
            "train": self.tokenized_datas,
            "validation": self.tokenized_datas[int(0.95 * datalen):]
        }
        self.split = split

    def add_random_word(self, sentence, word):
        words = sentence.split(" ")
        rand_idx = int(torch.poisson(torch.tensor(len(words) // 2, dtype=torch.float)).item())
        new_sent = " ".join(words[:rand_idx] + word.split(" ") + words[rand_idx:])
        return new_sent

    def tokenize(self, text: List[str], qa_data, verbose=False):
        self.model_max_length = self.context_length
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.model_max_length:
            tokens = tokens[:self.model_max_length]

        ans = qa_data["answer"]
        ans_tokens = self.tokenizer.encode(ans)
        
        def find_sublist_index(main_list, sublist):
            sublist_length = len(sublist)
            return next((i for i in range(len(main_list) - sublist_length + 1) 
                        if main_list[i:i+sublist_length] == sublist), -1)
        suffix_idx = find_sublist_index(tokens, ans_tokens)

        margin = 5
        if suffix_idx > len(tokens) - len(ans_tokens) - margin:
            return {
                "prompt_tokens": [],
                "prompt_token_length": 0,
                "continuation_tokens": tokens,
                "continuation_token_length": len(tokens),
                "suffix_idx": suffix_idx
            }
        else:
            if verbose:
                print(" ### Wrong Data")
                print(text)
                print(ans)
                print()
            return None # Not appropriate data for training perturbation

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
        label_masks = [True] * len(prompt_tokens) + [False] * len(cont_tokens) + [False] # 1 = -100, 0 = id
        if self.tokenizer.add_bos_token:
            input_ids = [self.tokenizer.bos_token_id] + input_ids
            label_masks = [True] + label_masks
        
        # Padding
        padding_length = self.model_max_length + 1 - len(input_ids) # considering eos
        if padding_length > 0:
            input_ids = input_ids + [0] * padding_length
            label_masks = label_masks + [True] * padding_length
        elif padding_length < 0:
            input_ids = input_ids[:self.model_max_length+1] # considering EOS
            label_masks = label_masks[:self.model_max_length+1] # considering EOS

        input_ids = torch.LongTensor(input_ids)
        label_masks = torch.BoolTensor(label_masks)
        labels = torch.masked_fill(input_ids.clone(), label_masks, -100)
        attention_mask = input_ids.ne(0).long()

        return_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if "target_embedding" in batch.keys():
            target_embedding = torch.FloatTensor(batch["target_embedding"])
            suffix_idx = torch.LongTensor([batch["suffix_idx"]])

            return_dict["target_embedding"] = target_embedding
            return_dict["suffix_idx"] = suffix_idx

        if "entity_mask" in batch.keys():
            entity_mask = batch["entity_mask"]
            entity_mask = entity_mask + [0.0]
            if self.tokenizer.add_bos_token:
                entity_mask = [1.0] + entity_mask
            if padding_length > 0:
                entity_mask = entity_mask + [0.0] * padding_length
            elif padding_length < 0:
                entity_mask = entity_mask[:self.model_max_length+1]
            entity_mask = torch.FloatTensor(entity_mask)
            return_dict["entity_mask"] = entity_mask

        return return_dict

    def __len__(self):
        return len(self.tokenized_datas[self.split])

    def __getitem__(self, index):
        sample = self.tokenized_datas[self.split][index]
        featured_sample = self.convert_to_features(sample)
        return featured_sample

    
def get_custom_dataset(dataset_config, tokenizer, split):
    dataset = TextDataset(
        dataset_config, tokenizer, split
    )
    return dataset