# SparseLoRA

**Accelerating LLM Fine-Tuning with Contextual Sparsity**

[![Paper](https://img.shields.io/badge/arXiv-2506.16500-b31b1b)](https://arxiv.org/abs/2506.16500)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://z-lab.ai/projects/sparselora/)
[![Slides](https://img.shields.io/badge/ICML%202025-Slides-green)](https://icml.cc/media/icml-2025/Slides/43493.pdf)

Drop-in acceleration for LoRA fine-tuning. SparseLoRA predicts and skips redundant neurons at each training step using lightweight SVD predictors, reducing compute by up to **2.2x** and wall-clock time by **1.6x** with no loss in accuracy.

## Installation

```bash
pip install git+https://github.com/z-lab/sparselora.git
```

## Quick Start

Add three lines to any LoRA training script:

```python
from sparselora import SparseLoRAConfig, apply_sparselora

config = SparseLoRAConfig.from_pretrained("z-lab/Meta-Llama-3-8B-Instruct-SparseLoRA", mode="o1")
model = apply_sparselora(model, config)
```

Full example:

```python
from transformers import AutoModelForCausalLM, Trainer
from peft import get_peft_model, LoraConfig
from sparselora import SparseLoRAConfig, apply_sparselora

model = AutoModelForCausalLM.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")
model = get_peft_model(model, LoraConfig(r=32, target_modules="all-linear"))

config = SparseLoRAConfig.from_pretrained("z-lab/Meta-Llama-3-8B-Instruct-SparseLoRA", mode="o1")
model = apply_sparselora(model, config)

trainer = Trainer(model=model, ...)
trainer.train()
```

## Pre-built Models

| Model | Path |
|-------|------|
| Llama 2 7B | `z-lab/Llama-2-7b-hf-SparseLoRA` |
| Llama 2 13B | `z-lab/Llama-2-13b-hf-SparseLoRA` |
| Llama 3 8B | `z-lab/Meta-Llama-3-8B-Instruct-SparseLoRA` |

Each directory contains `config.json` (per-layer sparsity for each mode) and `model.safetensors` (SVD predictor weights). Two modes are available: `o1` (conservative) and `o2` (aggressive).

## Reproducing Experiments

```bash
bash experiments/scripts/setup.sh

bash experiments/scripts/csr170k.sh
bash experiments/scripts/math10k.sh
```

## Citation

```bibtex
@inproceedings{khaki2025sparselora,
  title     = {{SparseLoRA: Accelerating LLM Fine-Tuning with Contextual Sparsity}},
  author    = {Khaki, Samir and Li, Xiuyu and Guo, Junxian and Zhu, Ligeng and Plataniotis, Konstantinos N. and Yazdanbakhsh, Amir and Keutzer, Kurt and Han, Song and Liu, Zhijian},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025}
}
```
