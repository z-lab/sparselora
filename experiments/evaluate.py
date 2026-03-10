"""Evaluation script for SparseLoRA-trained models."""

import argparse
import json
import os
import re

import torch
from peft import AutoPeftModelForCausalLM
from tabulate import tabulate
from torch import distributed as dist
from tqdm import trange
from transformers import AutoTokenizer, GenerationConfig

MATH_DATASETS = {"gsm8k", "mawps", "svamp"}

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)

ANSWER_PATTERNS = {
    "boolq": (r"true|false", ""),
    "piqa": (r"1|2", "solution"),
    "social-iqa": (r"1|2|3|4|5", "answer"),
    "arc-challenge": (r"1|2|3|4|5", "answer"),
    "arc-easy": (r"1|2|3|4|5", "answer"),
    "openbookqa": (r"1|2|3|4|5", "answer"),
    "hellaswag": (r"1|2|3|4", "ending"),
    "winogrande": (r"1|2", "option"),
}


def extract_answer(response: str, dataset: str):
    response = response.strip().lower()
    if dataset in MATH_DATASETS:
        nums = re.findall(r"-?\d+\.?\d*", response.replace(",", ""))
        return float(nums[-1]) if nums else float("inf")
    pattern, prefix = ANSWER_PATTERNS[dataset]
    m = re.findall(pattern, response)
    return prefix + m[0] if m else ""


def match(pred, target, dataset: str) -> bool:
    if dataset in MATH_DATASETS:
        return abs(float(target) - pred) <= 0.001
    return pred == target


def rank():
    return int(os.environ.get("RANK", 0))


def world_size():
    return int(os.environ.get("WORLD_SIZE", 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    args = parser.parse_args()

    is_math = bool(MATH_DATASETS & set(args.dataset.split("+")))
    if is_math:
        args.max_new_tokens = 256

    if "RANK" in os.environ:
        dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="sdpa",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model.peft_config["default"].base_model_name_or_path,
        model_max_length=512,
        padding_side="left",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen_cfg = GenerationConfig(max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id)

    metrics = {}
    for ds in args.dataset.split("+"):
        with open(os.path.join("datasets", ds, "test.json")) as f:
            instances = json.load(f)[rank() :: world_size()]

        correct, total = 0, 0
        for k in trange(0, len(instances), args.batch_size, disable=rank() != 0, desc=ds):
            batch = instances[k : k + args.batch_size]
            prompts = [PROMPT_TEMPLATE.format(instruction=b["instruction"]) for b in batch]
            input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids.cuda()

            with torch.inference_mode():
                out_ids = model.generate(input_ids, generation_config=gen_cfg)
            responses = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

            for resp, b in zip(responses, batch):
                pred = extract_answer(resp.split("### Response:")[-1], ds)
                correct += match(pred, b["answer"], ds)
            total += len(batch)

        gathered = [None] * world_size()
        dist.all_gather_object(gathered, (correct, total))
        metrics[ds] = sum(c for c, _ in gathered) / sum(t for _, t in gathered)

    if rank() == 0:
        print(tabulate(metrics.items(), headers=["Dataset", "Accuracy"], tablefmt="simple_outline"))
        out_path = os.path.join(args.model_name_or_path, "metrics.json")
        if os.path.isdir(args.model_name_or_path):
            with open(out_path, "w") as f:
                json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
