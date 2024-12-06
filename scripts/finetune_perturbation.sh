#!/bin/sh

set -e
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3

# dataset is target dataset
src_dataset=${1:-"streamingqa"}
tgt_dataset=${2:-"squad"}
perturbpath=$3
epoch=${4:-3}
seed=${5:-42}
modelname=${6:-"vicuna"}

if [ "vicuna" = $modelname ]; then
	modelpath="lmsys/vicuna-7b-v1.5"
elif [ "orca" = $modelname ]; then
	modelpath="microsoft/Orca-2-7b"
elif [ "llama" = $modelname ]; then
	modelpath="meta-llama/Llama-2-7b-hf"
else
	echo "No path exists"
fi

lastpart="${modelpath##*/}"
currentdate=$(date +%m%d)

savepath=$perturbpath-training-seed=$seed-epoch=$epoch-mlponly-naugments=4-repeat=4-short

torchrun --nnodes 1 --nproc_per_node 4 examples/finetuning.py \
	--enable_fsdp --fsdp_config.pure_bf16 \
	--model_name $modelpath --seed $seed \
	--dataset "custom_dataset" --custom_dataset.file custom_datasets/document_dataset.py \
	--dist_checkpoint_root_folder model_checkpoints \
	--dist_checkpoint_folder $savepath \
	--batch_size_training 4 --num_epochs $epoch --use_importance False \
	--use_noisy_embedding False --custom_dataset.context_length 192 \
	--writer_config_path None \
	--custom_dataset.dataset_path Data_Preprocessing/oracle_textbook/${tgt_dataset}_test_short/processed_sentences.json \
    --lr 5e-5 --update_parameters gate_proj,up_proj,down_proj \
	--freeze_layers True --num_non_freeze_layers 5,33 \
	--custom_dataset.no_augment True --run_validation False --use_fast_kernels True \
	--use_perturbation True --n_augments 4 --custom_dataset.repeat_dataset 4 \
	--perturb_model_path ./outputs_perturb/${src_dataset}_train_short/$perturbpath/vicuna-7b-v1.5-9.pt

sh convert_fsdp_checkpoint.sh model_checkpoints/$savepath-$lastpart ./outputs/${tgt_dataset}_test/$savepath
rm -r "model_checkpoints/$savepath-$lastpart"

# Evaluation
cd Evaluator
python run_contextqa.py --domain ${tgt_dataset}_test --model_name ../outputs/${tgt_dataset}_test/$savepath

cd ..
rm -r ./outputs/${tgt_dataset}_test/$savepath

echo "### End Pipeline."