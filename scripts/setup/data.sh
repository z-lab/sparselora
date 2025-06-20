echo "Downloading CSR170K & Math10k datasets..."
echo "Current working directory:"
pwd
echo "Downloading datasets..."
mkdir -p datasets
curl -o datasets/math_10k.json https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/refs/heads/main/ft-training_set/math_10k.json
curl -o datasets/commonsense_170k.json https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/refs/heads/main/ft-training_set/commonsense_170k.json
git clone --depth 1 --filter=blob:none --sparse https://github.com/AGI-Edgerunners/LLM-Adapters.git temp_llm_adapters
cd temp_llm_adapters
git sparse-checkout set dataset
cp -r dataset/* ../datasets/
mv ../datasets/ARC-Challenge ../datasets/arc-challenge
mv ../datasets/ARC-Easy ../datasets/arc-easy
mv ../datasets/SVAMP ../datasets/svamp
cd ..
rm -rf temp_llm_adapters