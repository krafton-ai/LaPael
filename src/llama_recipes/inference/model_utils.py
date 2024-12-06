# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaConfig
from transformers import AutoModelForCausalLM, AutoConfig

# Function to load the main model for text generation
def load_model(model_name, quantization):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True if "Phi" in model_name else False 
    )
    return model

# Function to load the PeftModel for performance optimization
def load_peft_model(model, peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model

# Loading the model from config to load FSDP checkpoints into that
def load_llama_from_config(config_path):
    # model_config = LlamaConfig.from_pretrained(config_path) 
    # model = LlamaForCausalLM(config=model_config)
    model_config = AutoConfig.from_pretrained(config_path,
        trust_remote_code=True if "Phi" in config_path else False)
    model = AutoModelForCausalLM.from_config(config=model_config,
        trust_remote_code=True if "Phi" in config_path else False)
    return model
    
    