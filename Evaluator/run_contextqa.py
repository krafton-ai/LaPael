import yaml
import json
from termcolor import cprint
from tqdm import tqdm, trange
from eval_utils import normalize_answer, normalize_generation, recall, exact_match, f1
import os
from datetime import datetime
import torch
import random
import fire
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import vllm
    from vllm import LLM, SamplingParams
    USE_VLLM = True
except:
    USE_VLLM = False

# write results
def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%y-%m-%d_%H:%M:%S")
    return date_of_run

def main(
    model_name="", # Path to the checkpoint of the trained model
    domain="streamingqa_train", # filename of qa data under the path of "llama-recipes-for-book/Evaluator/qa_data/"
):
    torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    random.seed(42)

    if "vicuna" in model_name.lower() or "llama" in model_name.lower():
        with open(f"ContextQA/{domain}/prompt.txt", 'r') as f:
            lines = f.readlines()
            prompt_template = ''.join(lines)
    elif "mistral" in model_name.lower():
        with open(f"ContextQA/{domain}/prompt_mistral.txt", 'r') as f:
            lines = f.readlines()
            prompt_template = ''.join(lines)
    else:
        with open(f"ContextQA/{domain}/prompt_phi.txt", 'r') as f:
            lines = f.readlines()
            prompt_template = ''.join(lines)

    MODEL_NAME = model_name
    print(MODEL_NAME)

    if USE_VLLM:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')

    if USE_VLLM:
        llm = LLM(
            model=MODEL_NAME, 
            trust_remote_code=True if "phi" in model_name.lower() else False
        )
        sampling_params = SamplingParams(
            max_tokens=32,
            temperature=0.0,
            top_p=1.00,
            n=1,
            stop=["<|end|>"]
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, 
            device_map='auto',
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )

    def run_llm(texts):
        prompts = [prompt_template.format(text=text) for text in texts]

        if USE_VLLM:
            total_outputs = llm.generate(prompts, sampling_params)
            total_preds = []
            for output in total_outputs:
                preds = []
                for pred in output.outputs:
                    pred = pred.text
                    pred = pred.split("\n")[0].strip()
                    preds.append(pred)
                total_preds.append(preds)
        else:
            # Iterate through the prompts in chunks of batch_size
            total_preds = []
            batch_size = 64
            for i in trange(0, len(prompts), batch_size, desc="Generate..."):
                batch_prompts = prompts[i:i + batch_size]

                # Tokenize the batch prompts
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)

                # Move inputs to GPU
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                input_ids = inputs["input_ids"]
                prompt_lengths = [len(ids) for ids in input_ids]

                # Generate outputs in batch
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=32,
                    temperature=0.0,    # creativity control
                    top_p=1.00,         # cumulative probability cutoff
                    num_return_sequences=1
                )

                # Decode outputs batch-wise
                for j, output in enumerate(outputs):
                    preds = []
                    pred = tokenizer.decode(output[prompt_lengths[j]:], skip_special_tokens=True)
                    pred = pred.split('\n\n')[0]
                    preds.append(pred)
                    total_preds.append(preds)
        return total_preds


    qa_path = f"ContextQA/{domain}/test.jsonl"
    if os.path.exists(qa_path):
        print(f"QA_file: {qa_path}")
        with open(qa_path, 'r') as f:
            dataset = [json.loads(i) for i in f.readlines()]
    else:
        new_qa_path = qa_path.replace(".jsonl", ".json")
        print(f"QA_file: {new_qa_path}")
        with open(new_qa_path, 'r') as f:
            dataset = json.load(f)

    # max_turn = 8
    recall_threshold = 0.7

    results = []
    em_results, f1_results, recall_results = [], [], []

    questions = [data["question"] for data in dataset]
    # questions = questions[:10]
    all_preds = run_llm(questions)

    predictions = []

    for i, preds in enumerate(tqdm(all_preds, desc="Evaluation...")):
        print(f"{sum(results)} / {i}")
        data = dataset[i]
        print(data["question"])
        answer = data["answer"]
        
        print(preds)
        potential_answers = [answer] if type(answer) != list else answer
        print(potential_answers, end='')
        predictions.append(
            {"pred": preds, "answers": potential_answers}
        )

        gens_recall = []
        gens_em = []
        gens_f1 = []
        for pred in preds:
            _r = max([recall(normalize_generation(pred).split(), normalize_answer(_answer).split()) for _answer in potential_answers])
            _em = max([exact_match(normalize_generation(pred).split(), normalize_answer(_answer).split()) for _answer in potential_answers])
            _f1 = max([f1(normalize_generation(pred).split(), normalize_answer(_answer).split()) for _answer in potential_answers])
            gens_recall.append(_r)
            gens_em.append(_em)
            gens_f1.append(_f1)

        gens_corrects = [True if r > recall_threshold else False for r in gens_recall]
        correction = sum(gens_corrects) > int(len(gens_corrects)/2)
        correction_recall = sum(gens_recall) / len(gens_recall)
        correction_f1 = sum(gens_f1) / len(gens_f1)
        correction_em = sum(gens_em) / len(gens_em)

        results.append(correction)
        em_results.append(correction_em)
        f1_results.append(correction_f1)
        recall_results.append(correction_recall)
        if correction:
            cprint("  correct", 'green')
        else:
            cprint("  wrong", 'red')

    score = {
        "total": sum(results),
        "n_corrects_recall": len(results),
        "accuracy": sum(results) / len(results) * 100,
        "exact_match": sum(em_results) / len(em_results) * 100,
        "recall": sum(recall_results) / len(recall_results) * 100,
        "f1": sum(f1_results) / len(f1_results) * 100
    }

    train_params_path = os.path.join(model_name, "train_params.yaml")

    if os.path.exists(train_params_path):
        json_data = {}
        with open(train_params_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
            json_dic = json.dumps(yaml_data)
            json_data = json.loads(json_dic)

        score["train_params"] = json_data
    
    score["results"] = results
    score["predictions"] = predictions

    os.makedirs(f"qa_generation_results/{domain}", exist_ok=True)
    subdirname = os.path.basename(MODEL_NAME.strip("/"))
    with open(os.path.join(f"qa_generation_results/{domain}/", subdirname) + ".json", 'w') as f:
        json.dump(score, f, indent=4)

    print(f"Save done! qa_generation_results/{domain}/{subdirname}.json")

    print(f"Correct {sum(results)} out of {len(results)}")
    print("done!")
    
if __name__ == "__main__":
    fire.Fire(main)