"""SparseLoRA training script."""

from ast import literal_eval
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, Optional

import torch
import transformers
from datasets import load_dataset
from liger_kernel.transformers import apply_liger_kernel_to_llama
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed

from sparselora import SparseLoRAConfig, apply_sparselora


def _parse_val(v):
    try:
        return literal_eval(v)
    except (ValueError, SyntaxError):
        return v


# -- Args -------------------------------------------------------------------


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    peft: Optional[str] = field(default="lora")
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=64)
    lora_dropout: Optional[float] = field(default=0)
    lora_target_modules: Optional[str] = field(default="q_proj,k_proj,v_proj,o_proj")
    sparselora: Optional[str] = field(default=None)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    chat_template_format: Optional[str] = field(default="none")


@dataclass
class DataArguments:
    dataset: Optional[str] = field(default=None)
    model_max_length: int = field(default=512)
    max_seq_length: int = field(default=512)


# -- Data -------------------------------------------------------------------


def _tokenize_and_mask(tokenizer, data_point, response_key="output"):
    instruction, response = data_point["instruction"], data_point[response_key]
    full = tokenizer(f"{instruction}\n{response}", truncation=True, max_length=256, padding=True, return_tensors=None)
    if full["input_ids"][-1] != tokenizer.eos_token_id and len(full["input_ids"]) < 256:
        full["input_ids"].append(tokenizer.eos_token_id)
    prefix = tokenizer(f"{instruction}\n", truncation=True, max_length=256, padding=True, return_tensors=None)
    labels = list(full["input_ids"])
    labels[: len(prefix["input_ids"])] = [-100] * len(prefix["input_ids"])
    return {"input_ids": torch.as_tensor(full["input_ids"]), "labels": torch.as_tensor(labels)}


DATASETS = {
    "math10k": "output",
    "csr170k": "answer",
}


def load_train_dataset(data_args, tokenizer) -> Dict[str, Any]:
    name = data_args.dataset.split("/")[-1].split(".")[0]
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Supported: {list(DATASETS.keys())}")
    ds = load_dataset("json", data_files=data_args.dataset)["train"]
    return ds.map(partial(_tokenize_and_mask, tokenizer, response_key=DATASETS[name]))


# -- Model ------------------------------------------------------------------


def create_model_and_tokenizer(model_args, data_args, training_args):
    apply_liger_kernel_to_llama(
        rope=True, swiglu=False, cross_entropy=True, fused_linear_cross_entropy=False, rms_norm=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=data_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    if len(tokenizer) > 32000:
        tokenizer.pad_token = "<|reserved_special_token_0|>"
        tokenizer.pad_token_id = 128002
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if model_args.chat_template_format != "none":
        tokenizer.chat_template = model_args.chat_template_format

    if training_args.peft is not None:
        if training_args.peft != "lora":
            raise ValueError(f"Unsupported PEFT: '{training_args.peft}'")
        model = get_peft_model(
            model,
            LoraConfig(
                r=training_args.lora_r,
                target_modules=training_args.lora_target_modules.split(","),
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )

    return model, tokenizer


# -- Main -------------------------------------------------------------------


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    set_seed(training_args.seed)

    assert training_args.sparselora, "Use --sparselora path=<dir>,mode=<mode>"
    config = SparseLoRAConfig.from_pretrained(
        **{k: _parse_val(v) for k, v in (kv.split("=", 1) for kv in training_args.sparselora.split(","))}
    )

    ds = data_args.dataset.split("/")[-1].split(".")[0]
    training_args.run_name = (
        f"{ds}_b{training_args.per_device_train_batch_size}_ep{training_args.num_train_epochs}"
        f"_lr{training_args.learning_rate}_peft-{training_args.peft}_{training_args.sparselora}"
    )
    peft = training_args.peft or "none"
    training_args.output_dir = f"{training_args.output_dir}/{peft}/{training_args.run_name}/{training_args.seed}/"
    config.save_pretrained(training_args.output_dir)

    model, tokenizer = create_model_and_tokenizer(model_args, data_args, training_args)
    model = apply_sparselora(model, config).to(torch.bfloat16)
    model.print_trainable_parameters()

    assert data_args.model_max_length == data_args.max_seq_length
    train_dataset = load_train_dataset(data_args, tokenizer)

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=data_args.max_seq_length,
            return_tensors="pt",
            padding=True,
        ),
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(output_dir=training_args.output_dir)
    trainer.save_state()


if __name__ == "__main__":
    main()
