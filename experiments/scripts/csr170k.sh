#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=8

mkdir -p datasets
if [ ! -f datasets/csr170k.json ]; then
    echo "Downloading csr170k dataset..."
    curl -sL -o datasets/csr170k.json "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/refs/heads/main/ft-training_set/commonsense_170k.json"
fi

MODEL_PATH=${1:-"NousResearch/Meta-Llama-3-8B-Instruct"}
SPARSELORA_PATH=${2:-"z-lab/Meta-Llama-3-8B-Instruct-SparseLoRA"}
SPARSELORA_MODE=${3:-"o1"}
SEED=42
NPROC=8

torchrun --nproc_per_node=$NPROC experiments/train.py \
    --output_dir checkpoints/$MODEL_PATH/csr170k \
    --model_name_or_path $MODEL_PATH \
    --sparselora path=$SPARSELORA_PATH,mode=$SPARSELORA_MODE,start_step=0.01 \
    --dataset datasets/csr170k.json \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --learning_rate 3e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.04 \
    --seed $SEED \
    --bf16 true \
    --logging_steps 1 \
    --save_strategy no \
    --report_to none \
    --ddp_find_unused_parameters false

if [ -d "datasets/boolq" ] && [ -d "datasets/arc-challenge" ]; then
    torchrun --nproc_per_node=$NPROC experiments/evaluate.py \
        --model_name_or_path checkpoints/$MODEL_PATH/csr170k \
        --dataset boolq+piqa+social-iqa+hellaswag+winogrande+arc-easy+arc-challenge+openbookqa
fi
