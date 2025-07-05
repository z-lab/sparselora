#!/usr/bin/env bash

source scripts/setup/setup.sh

MODEL=${1:-"barc0/Llama-3.1-ARC-Potpourri-Transduction-8B"}
SPARSITY_CONFIG=${2:-"llama3.1-8b-arc-agi.yaml"}

#* Check SVD Estimator:
bash scripts/setup/svd_estimator.sh $MODEL configs/sparsity/$SPARSITY_CONFIG

# Shift first 2 arguments so $@ contains only extras
shift 2

SEED=42
CHECKPOINT_PATH=checkpoints/$MODEL/arc_agi

#* Check if "enable-unsloth is in @"
if [[ "$@" == *"--enable-unsloth"* ]]; then
    echo "Enabling unsloth"
    NPROC=1
else
    NPROC=8
fi

torchrun --nproc_per_node=$NPROC \
    spft/train/main.py \
    --report_to wandb \
    --output_dir $CHECKPOINT_PATH \
    --seed $SEED \
    --model_name_or_path $MODEL \
    --model_short_name $MODEL_NAME \
    --config configs/train/arc_agi_train.yaml \
    --spft configs/sparsity/$SPARSITY_CONFIG \
    "$@"  # <<-- forward all extra args

python \
    spft/train/main.py \
    --report_to none \
    --output_dir $CHECKPOINT_PATH \
    --seed $SEED \
    --model_name_or_path $MODEL \
    --config configs/train/arc_agi_train.yaml \
    --spft configs/sparsity/$SPARSITY_CONFIG \
    --eval_only True \
    "$@"  # <<-- forward all extra args