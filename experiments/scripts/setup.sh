#!/usr/bin/env bash
set -e

pip install -r experiments/requirements.txt

mkdir -p datasets

echo "Downloading training datasets..."
curl -sL -o datasets/csr170k.json \
    "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/refs/heads/main/ft-training_set/commonsense_170k.json"
curl -sL -o datasets/math10k.json \
    "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/refs/heads/main/ft-training_set/math_10k.json"

echo "Downloading evaluation datasets..."
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/AGI-Edgerunners/LLM-Adapters.git _tmp_adapters
cd _tmp_adapters
git sparse-checkout set dataset
cp -r dataset/* ../datasets/
cd ..
rm -rf _tmp_adapters

mv datasets/ARC-Challenge datasets/arc-challenge
mv datasets/ARC-Easy     datasets/arc-easy
mv datasets/SVAMP        datasets/svamp

echo "Done. Evaluation datasets saved to datasets/."
