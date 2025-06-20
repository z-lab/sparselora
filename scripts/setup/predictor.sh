#!/bin/bash

# Fail on error
set -e

# Parse arguments
HF_MODEL_ID=${1:-"NousResearch/Meta-Llama-3-8B-Instruct"}
SPARSITY_CONFIG=${2:-"configs/sparsity/llama3-8b-math10k.yaml"}

# Check arguments
if [ -z "$HF_MODEL_ID" ] || [ -z "$SPARSITY_CONFIG" ]; then
    echo "Usage: bash scripts/setup/predictor.sh {HF_MODEL_ID} {SPARSITY_CONFIG}"
    exit 1
fi

# Run the SVD predictor generation
CUDA_VISIBLE_DEVICES=0 python tools/svd.py \
    --model_id "$HF_MODEL_ID" \
    --sparsity_config "$SPARSITY_CONFIG"
