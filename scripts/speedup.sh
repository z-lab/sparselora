#!/usr/bin/env bash

source scripts/setup/setup.sh

#* If dataset folder is not present, download it
if [ ! -d "datasets" ]; then
    bash scripts/setup/data.sh
fi


MODEL=${1:-"NousResearch/Meta-Llama-3-8B-Instruct"}
SPARSITY_CONFIG=${2:-llama3-8b-math10k.yaml}
DATASET=${3:-"math10k"}

#* Check Predictors:
bash scripts/setup/predictor.sh $MODEL configs/sparsity/$SPARSITY_CONFIG

# Shift first 2 arguments so $@ contains only extras
shift 2

SEED=42
CHECKPOINT_PATH=checkpoints/$MODEL/speedup

#* Check if "enable-unsloth is in @"
if [[ "$@" == *"--enable-unsloth"* ]]; then
    echo "Enabling unsloth"
    NPROC=1
else
    NPROC=8
fi

#* Use Dataset to Pick configs
if [[ "$DATASET" == "math10k" ]]; then
    DATA_CONFIG_FILE="configs/train/math10k_train.yaml"
elif [[ "$DATASET" == "commonsense_170k" ]]; then
    DATA_CONFIG_FILE="configs/train/csr170k_train.yaml"
elif [[ "$DATASET" == "WizardLMTeam/WizardLM_evol_instruct_70k" ]]; then
    DATA_CONFIG_FILE="configs/train/wizardlm_train.yaml"
elif [[ "$DATASET" == "m-a-p/CodeFeedback-Filtered-Instruction" ]]; then
    DATA_CONFIG_FILE="configs/train/codefeedback_train.yaml"
else
    echo "Unknown dataset: $DATASET"
    echo "Defaulting to math10k"
    DATA_CONFIG_FILE="configs/train/math10k_train.yaml"
    DATASET="math10k"
fi


N=5
times=()

echo ""

for i in $(seq 1 $N); do
    echo "▶️  Iteration $i"
    
    TIME_PER_STEP=$( 
        CUDA_VISIBLE_DEVICES=1 python -u spft/train/speedup.py \
            --report_to none \
            --logging_strategy no \
            --output_dir "$CHECKPOINT_PATH" \
            --seed "$SEED" \
            --model_name_or_path "$MODEL" \
            --spft "configs/sparsity/$SPARSITY_CONFIG" \
            --spft_benchmark True \
            --benchmark True \
            --spft_start_step 0 \
            --config "$DATA_CONFIG_FILE" \
            "$@" \
        2>/dev/null | tee /dev/tty | grep "\[Local\] Time (per step):" | awk '{print $5}' | sed 's/ms\/step//'
    )
    echo "⏱️ Time per step: $TIME_PER_STEP ms"
    times+=("$TIME_PER_STEP")        
done

# Compute mean
sum=0
for t in "${times[@]}"; do
    sum=$(echo "$sum + $t" | bc)
done
mean=$(echo "scale=2; $sum / $N" | bc)

# Compute standard deviation
sum_sq=0
for t in "${times[@]}"; do
    diff=$(echo "$t - $mean" | bc)
    sq=$(echo "$diff * $diff" | bc)
    sum_sq=$(echo "$sum_sq + $sq" | bc)
done
std=$(echo "scale=2; sqrt($sum_sq / $N)" | bc -l)

echo ""
echo "✅ Average time per step over $N runs: $mean ms ± $std ms"