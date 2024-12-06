#!/bin/sh

set -e
set -x

fsdp=$1
output=$2
modelname=${3:-"vicuna"}

if [ "vicuna" = $modelname ]; then
	DEFAULTMODEL="lmsys/vicuna-7b-v1.5"
elif [ "orca" = $modelname ]; then
	DEFAULTMODEL="microsoft/Orca-2-7b"
elif [ "llama" = $modelname ]; then
	DEFAULTMODEL="meta-llama/Llama-2-7b-hf"
elif [ "phi3" = $modelname ]; then
	DEFAULTMODEL="microsoft/Phi-3-mini-4k-instruct"
elif [ "mistral" = $modelname ]; then
	DEFAULTMODEL="mistralai/Mistral-7B-Instruct-v0.2"
else
	echo "No path exists"
fi


# hfmodel="${1:-$DEFAULTMODEL}"
hfmodel=$DEFAULTMODEL

python src/llama_recipes/inference/checkpoint_converter_fsdp_hf.py \
    --fsdp_checkpoint_path $fsdp --consolidated_model_path $output --HF_model_path_or_name $hfmodel

cp $fsdp/train_params.yaml $output/train_params.yaml