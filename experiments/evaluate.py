"""Evaluation script for SparseLoRA-trained models."""

import argparse
import json
import os
import re
import warnings
from datetime import timedelta
from typing import Any, List

import torch
from peft import AutoPeftModelForCausalLM
from tabulate import tabulate
from torch import distributed as dist
from torch.distributed.constants import default_pg_timeout
from tqdm import trange
from transformers import AutoTokenizer, GenerationConfig


# -- Distributed helpers ----------------------------------------------------

def dist_init(timeout: timedelta = default_pg_timeout):
    if "RANK" not in os.environ:
        warnings.warn("RANK not set, skipping distributed init.")
        return
    dist.init_process_group(backend="nccl", init_method="env://", timeout=timeout)


def dist_rank() -> int:
    return int(os.environ.get("RANK", 0))


def dist_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def dist_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def dist_local_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", 1))


def dist_is_main() -> bool:
    return dist_rank() == 0


def dist_all_gather(obj: Any) -> List[Any]:
    objs = [None for _ in range(dist_size())]
    dist.all_gather_object(objs, obj)
    return objs


# -- Answer extraction ------------------------------------------------------

def generate_prompt(instruction: str) -> str:
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n### Response:\n"
    )


def extract_answer(response: str, dataset: str):
    response = response.strip().lower()
    if dataset == "boolq":
        m = re.findall(r"true|false", response)
        return m[0] if m else ""
    elif dataset == "piqa":
        m = re.findall(r"1|2", response)
        return "solution" + m[0] if m else ""
    elif dataset in ("social-iqa", "arc-challenge", "arc-easy", "openbookqa"):
        m = re.findall(r"1|2|3|4|5", response)
        return "answer" + m[0] if m else ""
    elif dataset == "hellaswag":
        m = re.findall(r"1|2|3|4", response)
        return "ending" + m[0] if m else ""
    elif dataset == "winogrande":
        m = re.findall(r"1|2", response)
        return "option" + m[0] if m else ""
    elif dataset in ("gsm8k", "mawps", "svamp"):
        response = response.replace(",", "")
        m = re.findall(r"-?\d+\.?\d*", response)
        return float(m[-1]) if m else float("inf")
    raise ValueError(f"Unsupported dataset: '{dataset}'")


# -- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    args = parser.parse_args()

    if any(d in args.dataset for d in ("gsm8k", "mawps", "svamp")):
        args.max_new_tokens = 256

    dist_init()
    devices = range(dist_local_rank(), torch.cuda.device_count(), dist_local_size())
    torch.cuda.set_device(devices[0])
    max_memory = {d: torch.cuda.get_device_properties(d).total_memory for d in devices}

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_name_or_path, attn_implementation="sdpa",
        torch_dtype=torch.float16, device_map="auto", max_memory=max_memory,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model.peft_config["default"].base_model_name_or_path,
        model_max_length=512, padding_side="left", use_fast=False,
    )
    if len(tokenizer) > 32000:
        tokenizer.pad_token = "<|reserved_special_token_0|>"
        tokenizer.pad_token_id = 128002

    gen_config = GenerationConfig(max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id)

    metrics = {}
    for dataset in args.dataset.split("+"):
        with open(os.path.join("datasets", dataset, "test.json")) as f:
            instances = json.load(f)
        instances = instances[dist_rank()::dist_size()]

        correct, total = 0, 0
        for k in trange(0, len(instances), args.batch_size, disable=not dist_is_main(), desc=dataset):
            batch = instances[k: k + args.batch_size]
            targets = [b["answer"] for b in batch]
            inputs = [generate_prompt(b["instruction"]) for b in batch]
            input_ids = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True).input_ids.cuda()

            with torch.inference_mode():
                output_ids = model.generate(input_ids, generation_config=gen_config)
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            outputs = [extract_answer(o.split("### Response:")[1].strip(), dataset) for o in outputs]

            if dataset in ("gsm8k", "mawps", "svamp"):
                correct += sum(abs(float(t) - o) <= 0.001 for o, t in zip(outputs, targets))
            else:
                correct += sum(o == t for o, t in zip(outputs, targets))
            total += len(targets)

        metrics[dataset] = sum(dist_all_gather(correct)) / sum(dist_all_gather(total))

    if dist_is_main():
        print(tabulate(metrics.items(), headers=["Dataset", "Accuracy"], tablefmt="simple_outline"))
        if os.path.exists(args.model_name_or_path):
            with open(os.path.join(args.model_name_or_path, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
