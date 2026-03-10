#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=8

mkdir -p datasets
if [ ! -f datasets/math10k.json ]; then
    echo "Downloading math10k dataset..."
    curl -sL -o datasets/math10k.json "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/refs/heads/main/ft-training_set/math_10k.json"
fi

MODEL=${1:-"NousResearch/Meta-Llama-3-8B-Instruct"}
MODE=${2:-"o2"}
shift 2

SPARSELORA_PATH="models/${MODEL}-SparseLoRA"
SEED=42
NPROC=8

torchrun --nproc_per_node=$NPROC experiments/train.py \
    --output_dir checkpoints/$MODEL/math10k \
    --model_name_or_path $MODEL \
    --sparselora path=$SPARSELORA_PATH,mode=$MODE \
    --dataset datasets/math10k.json \
    --per_device_train_batch_size 8 \
    --num_train_epochs 3 \
    --learning_rate 3e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.04 \
    --seed $SEED \
    --bf16 true \
    --logging_steps 1 \
    --save_strategy no \
    --report_to none \
    --ddp_find_unused_parameters false \
    "$@"

if [ -d "datasets/gsm8k" ] && [ -d "datasets/svamp" ] && [ -d "datasets/mawps" ]; then
    torchrun --nproc_per_node=$NPROC experiments/evaluate.py \
        --model_name_or_path checkpoints/$MODEL/math10k \
        --dataset gsm8k+svamp+mawps \
        "$@"
fi
