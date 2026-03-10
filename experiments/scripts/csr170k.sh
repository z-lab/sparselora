#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=8

MODEL_PATH=${1:-"NousResearch/Meta-Llama-3-8B-Instruct"}
SPARSELORA_PATH=${2:-"z-lab/Meta-Llama-3-8B-Instruct-SparseLoRA"}
SEED=42

torchrun --nproc_per_node=gpu experiments/train.py \
    --model_name_or_path $MODEL_PATH \
    --dataset datasets/csr170k.json \
    --sparselora path=$SPARSELORA_PATH,mode=o1,start_step=0.05 \
    --output_dir checkpoints/csr170k \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --learning_rate 3e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.04 \
    --seed $SEED \
    --bf16 true \
    --logging_steps 1 \
    --save_strategy no \
    --report_to none \
    --ddp_find_unused_parameters false

torchrun --nproc_per_node=gpu experiments/eval.py \
    --model_name_or_path checkpoints/csr170k \
    --dataset boolq+piqa+social-iqa+hellaswag+winogrande+arc-easy+arc-challenge+openbookqa
