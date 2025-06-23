#!/usr/bin/env bash

source scripts/setup/setup.sh

MODEL=${1:-"NousResearch/Meta-Llama-3.1-8B"}
SPARSITY_CONFIG=${2:-"llama3.1-8b-codefeedback.yaml"}

#* Check Predictors:
bash scripts/setup/predictor.sh $MODEL configs/sparsity/$SPARSITY_CONFIG


# Shift first 2 arguments so $@ contains only extras
shift 2

SEED=42
CHECKPOINT_PATH=checkpoints/$MODEL/codefeedback

#* Check if "enable-unsloth is in @"
if [[ "$@" == *"--enable-unsloth"* ]]; then
    echo "Enabling unsloth"
    NPROC=1
else
    NPROC=8
fi

torchrun --nproc_per_node=8 \
    spft/train/main.py \
    --report_to wandb \
    --output_dir $CHECKPOINT_PATH \
    --seed $SEED \
    --model_name_or_path $MODEL \
    --num-train-epochs 1 \
    --config configs/train/codefeedback_train.yaml \
    --spft configs/sparsity/$SPARSITY_CONFIG \
    "$@"  # <<-- forward all extra args

torchrun --nproc_per_node=8 \
    spft/train/main.py \
    --report_to none \
    --output_dir $CHECKPOINT_PATH \
    --seed $SEED \
    --model_name_or_path $MODEL \
    --num-train-epochs 1 \
    --config configs/train/codefeedback_train.yaml \
    --spft configs/sparsity/$SPARSITY_CONFIG \
    --eval_only True \
    "$@"  # <<-- forward all extra args
