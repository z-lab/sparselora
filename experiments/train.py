"""SparseLoRA training script."""

from ast import literal_eval
from dataclasses import dataclass, field
from functools import partial

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


@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": "HF model name or local path"})
    dataset: str = field(metadata={"help": "Path to training JSON file"})
    sparselora: str = field(metadata={"help": "Comma-separated key=value pairs, e.g. path=z-lab/...,mode=o1"})
    max_seq_length: int = field(default=512)
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.0)
    lora_target_modules: str = field(default="q_proj,k_proj,v_proj,o_proj")


def tokenize_and_mask(tokenizer, max_len, data_point):
    instruction = data_point["instruction"]
    response = data_point.get("output", data_point.get("answer", ""))
    full = tokenizer(f"{instruction}\n{response}", truncation=True, max_length=max_len)
    if full["input_ids"][-1] != tokenizer.eos_token_id and len(full["input_ids"]) < max_len:
        full["input_ids"].append(tokenizer.eos_token_id)
    prefix = tokenizer(f"{instruction}\n", truncation=True, max_length=max_len)
    labels = list(full["input_ids"])
    labels[: len(prefix["input_ids"])] = [-100] * len(prefix["input_ids"])
    return {"input_ids": full["input_ids"], "labels": labels}


def main():
    parser = HfArgumentParser((ScriptArguments, transformers.TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    config = SparseLoRAConfig.from_pretrained(
        **{k: _parse_val(v) for k, v in (kv.split("=", 1) for kv in args.sparselora.split(","))}
    )

    apply_liger_kernel_to_llama(
        rope=True,
        swiglu=False,
        cross_entropy=True,
        fused_linear_cross_entropy=False,
        rms_norm=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_seq_length,
        padding_side="left",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(","),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model = apply_sparselora(model, config)
    model.print_trainable_parameters()

    train_dataset = load_dataset("json", data_files=args.dataset)["train"]
    train_dataset = train_dataset.map(partial(tokenize_and_mask, tokenizer, args.max_seq_length))

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=args.max_seq_length,
            return_tensors="pt",
            padding=True,
        ),
    )
    trainer.train()
    trainer.save_model()
    config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
