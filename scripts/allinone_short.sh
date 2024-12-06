#!/bin/sh

set -e
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3

config=$1
dataset=${2:-"archivalqa"}
epochs=${3:-10}
seed=${4:-42}
modelname=${5:-"vicuna"}

lr=5e-5
if [ "vicuna" = $modelname ]; then
	modelpath="lmsys/vicuna-7b-v1.5"
	modelshort="vicuna-7b-v1.5"
	modeltype="llama"
elif [ "orca" = $modelname ]; then
	modelpath="microsoft/Orca-2-7b"
	modelshort="Orca-2-7b"
	modeltype="llama"
elif [ "llama" = $modelname ]; then
	modelpath="meta-llama/Llama-2-7b-hf"
	modelshort="Llama-2-7b-hf"
	modeltype="llama"
elif [ "phi3" = $modelname ]; then
	modelpath="microsoft/Phi-3-mini-4k-instruct"
	modelshort="Phi-3-mini-4k-instruct"
	modeltype="phi3"
elif [ "mistral" = $modelname ]; then
	modelpath="mistralai/Mistral-7B-Instruct-v0.2"
	modelshort="Mistral-7B-Instruct-v0.2"
	modeltype="mistral"
	lr=1e-5
else
	echo "No path exists"
fi

lastpart="${modelpath##*/}"
currentdate=$(date +%m%d)
seed=$seed
epochs_var=$(expr $epochs - 1)
configlastpart="${config##*/}"

perturbsavepath=$currentdate-$modelname-epoch=$epochs-config=$configlastpart-short-seed${seed}

# Perturbation model training
torchrun --nnodes 1 --nproc_per_node 4 examples/finetuning.py \
	--enable_fsdp --fsdp_config.pure_bf16 \
	--model_name $modelpath --seed $seed --lm_type $modeltype \
	--dataset "custom_dataset" --custom_dataset.file custom_datasets/perturbation_dataset.py \
	--dist_checkpoint_root_folder model_checkpoints \
	--dist_checkpoint_folder $perturbsavepath \
	--batch_size_training 4 --num_epochs $epochs --custom_dataset.context_length 192 \
	--writer_config_path aiflow.yaml \
	--custom_dataset.dataset_path Data_Preprocessing/oracle_textbook/${dataset}_train_short/processed_sentences.json \
	--custom_dataset.no_augment True --custom_dataset.domain ${dataset}_train_short --custom_dataset.use_oracle_entity True \
    --noise_lr 1e-3 \
	--run_validation False --train_perturbation True --use_cosine_scheduler True --use_perturbation False \
	--perturb_config_path "$config.yaml"

mkdir -p ./outputs_perturb/${dataset}_train_short/
cp -r model_checkpoints/$perturbsavepath-$lastpart ./outputs_perturb/${dataset}_train_short/$perturbsavepath
rm -r "model_checkpoints/$perturbsavepath-$lastpart"

savepath=$currentdate-$perturbsavepath-training-3epoch-mlponly-augment4-repeat4-freeze04-lr${lr}

# LM Fine-tuning (For test set)
torchrun --nnodes 1 --nproc_per_node 4 examples/finetuning.py \
	--enable_fsdp --fsdp_config.pure_bf16 \
	--model_name $modelpath --seed $seed --lm_type $modeltype \
	--dataset "custom_dataset" --custom_dataset.file custom_datasets/document_dataset.py \
	--dist_checkpoint_root_folder model_checkpoints \
	--dist_checkpoint_folder $savepath \
	--batch_size_training 4 --num_epochs 3 --use_importance False \
	--use_noisy_embedding False --custom_dataset.context_length 128 \
	--writer_config_path None \
	--custom_dataset.dataset_path Data_Preprocessing/oracle_textbook/${dataset}_test_short/processed_sentences.json \
    --lr $lr --update_parameters gate_proj,up_proj,down_proj,gate_up_proj \
	--freeze_layers True --num_non_freeze_layers 5,33 \
	--custom_dataset.no_augment True --run_validation False \
	--custom_dataset.repeat_dataset 4 --n_augments 4 \
	--use_perturbation True --perturb_model_path ./outputs_perturb/${dataset}_train_short/$perturbsavepath/${modelshort}-${epochs_var}.pt

sh convert_fsdp_checkpoint.sh model_checkpoints/$savepath-$lastpart ./outputs/${dataset}_test_short/$savepath $modelname
rm -r "model_checkpoints/$savepath-$lastpart"

# Evaluation (For test set)
cd Evaluator
python run_contextqa.py --domain ${dataset}_test --model_name ../outputs/${dataset}_test_short/$savepath

cd ..
rm -r "./outputs/${dataset}_test_short/$savepath"

echo "### End Pipeline."