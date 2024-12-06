# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from pkg_resources import packaging

import fire
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import math
from peft import get_peft_model, prepare_model_for_kbit_training
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    MistralForCausalLM,
    MistralConfig,
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from llama_recipes.utils.phi3.modeling_phi3 import Phi3DecoderLayer

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.configs import load_yaml_config, load_perturb_config_fromdict
from llama_recipes.data.concatenator import ConcatDataset
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset

from llama_recipes.utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
    print_trainable_parameters
)
from llama_recipes.utils.train_perturbation_utils import(
    train_perturbation
)


def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    if train_config.lm_type == "llama":
        if train_config.use_noisy_embedding:
            from llama_recipes.utils.model_utils import LlamaForCausalLMWrapper
            model_cls = LlamaForCausalLMWrapper
        elif train_config.train_perturbation or train_config.use_perturbation:
            from llama_recipes.utils.model_perturbation_utils import CustomLlamaForCausalLM
            model_cls = CustomLlamaForCausalLM
        elif train_config.do_adversarial_training:
            from llama_recipes.utils.model_adversarial_utils import LlamaForCausalLMWrapper
            model_cls = LlamaForCausalLMWrapper
        else:
            model_cls = LlamaForCausalLM
        config_cls = LlamaConfig
        tokenizer_cls = LlamaTokenizer
    elif train_config.lm_type == "mistral":
        if train_config.train_perturbation or train_config.use_perturbation:
            from llama_recipes.utils.model_perturbation_utils import CustomMistralForCausalLM
            model_cls = CustomMistralForCausalLM
        elif train_config.use_noisy_embedding:
            from llama_recipes.utils.model_utils import MistralForCausalLMWrapper
            model_cls = MistralForCausalLMWrapper
        elif train_config.do_adversarial_training:
            from llama_recipes.utils.model_adversarial_utils import MistralForCausalLMWrapper
            model_cls = MistralForCausalLMWrapper
        else:
            model_cls = MistralForCausalLM
        config_cls = MistralConfig
        tokenizer_cls = AutoTokenizer
    elif train_config.lm_type == "phi3":
        if train_config.train_perturbation or train_config.use_perturbation:
            from llama_recipes.utils.model_perturbation_utils import CustomPhi3ForCausalLM
            model_cls = CustomPhi3ForCausalLM
        elif train_config.use_noisy_embedding:
            from llama_recipes.utils.model_utils import Phi3ForCausalLMWrapper
            model_cls = Phi3ForCausalLMWrapper
        elif train_config.do_adversarial_training:
            from llama_recipes.utils.model_adversarial_utils import Phi3ForCausalLMWrapper
            model_cls = Phi3ForCausalLMWrapper
        else:
            model_cls = AutoModelForCausalLM
        config_cls = AutoConfig
        tokenizer_cls = AutoTokenizer
    else:
        raise NotImplementedError

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        if rank == 0:
            model = model_cls.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
                attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            )
        else:
            llama_config = config_cls.from_pretrained(train_config.model_name)
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                model = model_cls(llama_config)
    else:
        model = model_cls.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            trust_remote_code=True if train_config.lm_type in ["phi3"] else False
        )
    if train_config.use_noisy_embedding:
        model.use_noisy_embedding = True

    # Load the tokenizer and add special tokens
    tokenizer = tokenizer_cls.from_pretrained(train_config.model_name)
    # tokenizer.pad_token_id = tokenizer.eos_token_id # minki: To handle ShareGPT :(
    tokenizer.add_special_tokens({"pad_token": "<PAD>",})
    # tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load perturbation model
    if train_config.train_perturbation or train_config.use_perturbation:
        from llama_recipes.utils.perturb_utils import PerturbModel
        if train_config.use_perturbation:
            # Load config first
            model_dir = os.path.dirname(train_config.perturb_model_path)
            config_path = os.path.join(model_dir, "train_params.yaml")
            perturb_config = load_perturb_config_fromdict(config_path)
            model.config.perturb_config = perturb_config

            # Then, initialize and load model
            model.model.perturb_model = PerturbModel(model.config)
            state_dict = torch.load(train_config.perturb_model_path)
            model.model.perturb_model.load_state_dict(state_dict)
        else:
            # Load config from the config path
            perturb_config = load_yaml_config(train_config.perturb_config_path)
            model.config.perturb_config = perturb_config
            model.model.perturb_model = PerturbModel(model.config)
            train_config.perturb_config = perturb_config

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if len(train_config.update_parameters) > 0:
        for k, v in model.named_parameters():
            v.requires_grad = False
        
        for k, v in model.named_parameters():
            for update_k in train_config.update_parameters:
                if update_k in k:
                    v.requires_grad = True

    use_orig_params = True

    if train_config.train_perturbation:
        for k, v in model.named_parameters(): # Train perturbation
            if "perturb_model" in k:
                v.requires_grad = True
            else:
                v.requires_grad = False
    elif train_config.use_perturbation: # Use Perturbation
        for k, v in model.named_parameters():
            if "perturb_model" in k:
                v.requires_grad = False

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            freeze_transformer_layers(model, train_config.num_non_freeze_layers)

        fsdp_config.lm_type = train_config.lm_type
        fsdp_config.use_perturbation = train_config.use_perturbation or train_config.train_perturbation

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        if train_config.lm_type == "llama":
            if train_config.train_perturbation:
                from llama_recipes.utils.model_perturbation_utils import CustomLlamaDecoderLayer
                target_layer = CustomLlamaDecoderLayer
            else:
                target_layer = LlamaDecoderLayer
        elif train_config.lm_type == "mistral":
            if train_config.train_perturbation:
                from llama_recipes.utils.model_perturbation_utils import CustomMistralDecoderLayer
                target_layer = CustomMistralDecoderLayer
            else:
                target_layer = MistralDecoderLayer
        elif train_config.lm_type == "phi3":
            if train_config.train_perturbation:
                from llama_recipes.utils.model_perturbation_utils import CustomPhi3DecoderLayer
                target_layer = CustomPhi3DecoderLayer
            else:
                target_layer = Phi3DecoderLayer

        if train_config.use_peft and train_config.use_perturbation:
            remove_perturb_model = True
        else:
            remove_perturb_model = False
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, target_layer, remove_perturb_model=remove_perturb_model)
        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft or train_config.train_perturbation else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
            use_orig_params=use_orig_params,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")

    print_trainable_parameters(model)

    ### Dataset part 1
    dataset_config = generate_dataset_config(train_config, kwargs)
    dataset_config.model_name = train_config.model_name
    if train_config.train_perturbation:
        dataset_config.embedding_type = perturb_config.embedding_type
        dataset_config.use_mse_loss = perturb_config.use_mse_loss
        if getattr(perturb_config, "train_entity_mask", False):
            dataset_config.use_oracle_entity = True

    # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    if train_config.run_validation:
        dataset_val = get_preprocessed_dataset(
            tokenizer,
            dataset_config,
            split="test",
        )
        if not train_config.enable_fsdp or rank == 0:
                print(f"--> Validation Set Length = {len(dataset_val)}")

    ### Dataset Part 2
    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length)

    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=False,
        **train_dl_kwargs,
    )

    eval_dataloader = None
    if train_config.run_validation:
        if train_config.batching_strategy == "packing":
            dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length)

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )

    parameters = [v for k, v in model.named_parameters() if v.requires_grad]
    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            parameters,
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            parameters,
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )

    if train_config.use_cosine_scheduler:
        total_length = len(train_dataloader)//train_config.gradient_accumulation_steps
        total_steps = total_length * train_config.num_epochs
        lr_lambda = lambda step: 0.1 + (1 - 0.1) * (1 + math.cos(math.pi * step / total_steps)) / 2
        # Create the LambdaLR scheduler
        scheduler = LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    train_config.dataset_config = dataset_config
    if train_config.do_adversarial_training:
        from llama_recipes.utils.train_adversarial_utils import train_adv
        train_fn = train_adv

        # Start the training process
        results = train_fn(
            model,
            train_dataloader,
            eval_dataloader,
            tokenizer,
            optimizer,
            scheduler,
            train_config.gradient_accumulation_steps,
            train_config,
            fsdp_config if train_config.enable_fsdp else None,
            local_rank if train_config.enable_fsdp else None,
            rank if train_config.enable_fsdp else None,
        )
        if not train_config.enable_fsdp or rank==0:
            [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

    elif train_config.train_perturbation:
        train_fn = train_perturbation

        noise_parameters = [v for k, v in model.named_parameters() if "perturb_model" in k]
        optimizer = optim.AdamW(
            noise_parameters,
            lr=train_config.noise_lr,
            weight_decay=train_config.weight_decay,
        )

        # Start the training process
        results = train_fn(
            model,
            train_dataloader,
            eval_dataloader,
            tokenizer,
            optimizer,
            scheduler,
            train_config.gradient_accumulation_steps,
            train_config,
            fsdp_config if train_config.enable_fsdp else None,
            local_rank if train_config.enable_fsdp else None,
            rank if train_config.enable_fsdp else None,
        )
        if not train_config.enable_fsdp or rank==0:
            [print(f'Key: {k}, Value: {v}') for k, v in results.items()]
    else:
        train_fn = train

        # Start the training process
        results = train_fn(
            model,
            train_dataloader,
            eval_dataloader,
            tokenizer,
            optimizer,
            scheduler,
            train_config.gradient_accumulation_steps,
            train_config,
            fsdp_config if train_config.enable_fsdp else None,
            local_rank if train_config.enable_fsdp else None,
            rank if train_config.enable_fsdp else None,
        )
        if not train_config.enable_fsdp or rank==0:
            [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    fire.Fire(main)
