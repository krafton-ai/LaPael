import argparse
import json
# from tqdm.rich import tqdm
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import *

IDX_TO_LETTER = {1: "A", 2: "B", 3: "C", 4: "D"}

def main(args):
    if args.lm == "vicuna":
        MODEL_NAME = "lmsys/vicuna-7b-v1.5"
    elif args.lm == "llama":
        MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    elif args.lm == "phi3":
        MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
    elif args.lm == "mistral":
        MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
    else:
        raise NotImplementedError

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='auto',
        trust_remote_code=True if args.lm == "phi3" else False)
    # model.to("cuda")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    domain = args.domain
    if "streamingqa" in args.domain or "archivalqa" in args.domain or "squad" in args.domain:
        field = "ContextQA"
    else:
        field = "MedicalQA"

    augtype = args.augtype

    if args.lm == "vicuna":
        savedir = f"Analysis/embeddings/{domain}_{augtype}"
    else:
        savedir = f"Analysis/embeddings_{args.lm}/{domain}_{augtype}"
    os.makedirs(savedir, exist_ok=True)

    if augtype == "all":
        target_augs = ["easy_suffix", "medium_suffix", "hard_suffix", "qa_suffix"]
    else:
        target_augs = [f"{augtype}_suffix"]

    if field == "ContextQA":
        aug_dict = {k: get_augmentations_contextqa(k, domain) for k in target_augs}
    else:
        aug_dict = {k: get_augmentations(k, domain) for k in target_augs}

    # Load data
    with open(f"./Evaluator/{field}/{domain}/test.jsonl") as f:
        qa_dataset = [json.loads(data.strip()) for data in f.readlines()]

    with open(f"./Data_Preprocessing/oracle_textbook/{domain}/processed_sentences.json") as f:
        sample_dataset = json.load(f)
    
    id_to_reprs = defaultdict(list)
    id_to_random_reprs = defaultdict(list)
    n_layers = 33
    iterate_augmentations = True
    use_single_augmentation = False

    n_hidden_states_total = 0
    # for i, data in enumerate(tqdm(sample_dataset)):
    for i, data in enumerate(sample_dataset):
        print(f"[{i}/{len(sample_dataset)}] Embedding data...")
        savepath = os.path.join(savedir, str(i).zfill(5) + ".pt")

        if os.path.exists(savepath):
            print(f"{savepath} exist.")
            continue

        qa_data = qa_dataset[i]
        sentence = data["output"]

        if field == "MedicalQA":
            answer_key = "op" + IDX_TO_LETTER[qa_data["cop"]].lower()
            ans = qa_data[answer_key]
        else:
            ans = qa_data["answer"]

        # sentence_lower = sentence.lower()
        if ans not in sentence: 
            print(f"data idx {i}: Answer not exist in sentence")
            continue

        # hidden state of the position for the answer (first) token
        answer_hidden_states = []

        hidden_state_index = -1

        original_answer_embed = return_suffix_embedding(model, tokenizer, sentence, ans)
        if not iterate_augmentations and use_single_augmentation:
            answer_hidden_states.append(original_answer_embed) # Original

        # Also add QA
        if not iterate_augmentations:
            qa_sentence, qa_ans_idx = format_qa_data(qa_data, ans, zero_shot=True)
            answer_embed = return_suffix_embedding(model, tokenizer, qa_sentence, ans)
            answer_hidden_states.append(answer_embed) # QA

        if iterate_augmentations:
            for aug in target_augs:
                aug_outputs = aug_dict[aug][i] # List of augmentations _aug
                valid_aug_outputs = []
                for aug_output in aug_outputs:
                    if ans not in aug_output["output"]:
                        continue

                    # Should be positioned at the end of sentence
                    if ans not in " ".join(aug_output["output"].split()[-10:]):
                        continue

                    valid_aug_outputs.append(aug_output["output"])

                valid_aug_outputs = deduplicate_sentence(valid_aug_outputs)

                for aug_output in valid_aug_outputs:
                    answer_embed = return_suffix_embedding(model, tokenizer, aug_output, ans)
                    if answer_embed is not None:
                        answer_hidden_states.append(answer_embed)
                    # break # Only embed  sentence per one augmentation for test(optional)
                    if use_single_augmentation:
                        if len(answer_hidden_states) > 0:
                            break

        # Don't save anything
        if len(answer_hidden_states) == 0:
            continue
        
        print("     # Valid paraphrases: ", len(answer_hidden_states))
        n_hidden_states_total += len(answer_hidden_states)
        total_stats = []
        for layer_num in range(n_layers):
            mean = torch.mean(torch.stack([hs[layer_num] for hs in answer_hidden_states], dim=0), dim=0)
            std = torch.std(torch.stack([hs[layer_num] for hs in answer_hidden_states], dim=0), dim=0)
            if torch.isnan(std.mean()):
                std = torch.zeros_like(std, dtype=torch.bool)
            stat = torch.stack([mean, std], dim=0)
            total_stats.append(stat)
        stat_tensor = torch.stack(total_stats, dim=0)
        savepath = os.path.join(savedir, str(i).zfill(5) + ".pt")
        torch.save(stat_tensor, savepath)
        print("Save done ", savepath)
    print(n_hidden_states_total / 1000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", type=str, default="vicuna")
    parser.add_argument("--augtype", type=str, default="medium")
    parser.add_argument("--domain", type=str, default="squad_train_short")
    args = parser.parse_args()

    main(args)