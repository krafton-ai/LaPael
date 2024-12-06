from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str="PATH/to/LLAMA/7B"
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=4
    batching_strategy: str="padding" #alternative: padding, packing
    context_length: int=4096
    gradient_accumulation_steps: int=1
    num_epochs: int=3
    num_workers_dataloader: int=1
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "samsum_dataset"
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_non_freeze_layers: str = "" # Indicates the range of non-frozen layers
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    update_parameters: str = "" # Indicate the parameters to be fine-tuned
    no_shuffle: bool = False
    writer_config_path: str = "aiflow.yaml"
    lm_type: str="llama"
    use_noisy_embedding: bool = False
    no_kl: bool = False
    use_cosine_scheduler: bool = False
    noise_lr: float=1e-3
    use_adversarial: bool = False
    do_adversarial_training: bool = False
    train_perturbation: bool = False
    use_perturbation: bool = False
    perturb_model_path: str = "PATH/to/save/PERTURB/model"
    perturb_config_path: str = ""
    n_augments: int = 1
    use_neftune: bool = False