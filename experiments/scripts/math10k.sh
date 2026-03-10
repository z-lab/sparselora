#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=8

MODEL_PATH=${1:-"NousResearch/Meta-Llama-3-8B-Instruct"}
SPARSELORA_PATH=${2:-"z-lab/Meta-Llama-3-8B-Instruct-SparseLoRA"}
SEED=42

torchrun --nproc_per_node=gpu experiments/train.py \
    --model_name_or_path $MODEL_PATH \
    --dataset datasets/math10k.json \
    --sparselora path=$SPARSELORA_PATH,mode=o2,start_step=0.05 \
    --output_dir checkpoints/math10k \
    --num_train_epochs 3 \
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
    --model_name_or_path checkpoints/math10k \
    --dataset gsm8k+svamp+mawps \
    --max_new_tokens 256
