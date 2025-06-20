import argparse
import json
import os
import re

import torch
from peft import AutoPeftModelForCausalLM
from tabulate import tabulate
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from spft.utils import distributed as dist


def generate_prompt(instruction):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def extract_answer(response: str, dataset) -> str:
    response = response.strip().lower()
    print(response)
    if dataset == "boolq":
        answers = re.findall(r"true|false", response)
        if not answers:
            return ""
        return answers[0]
    elif dataset == "piqa":
        answers = re.findall(r"1|2", response)
        if not answers:
            return ""
        return "solution" + answers[0]
    elif dataset in ["social-iqa", "arc-challenge", "arc-easy", "openbookqa"]:
        answers = re.findall(r"1|2|3|4|5", response)
        if not answers:
            return ""
        return "answer" + answers[0]
    elif dataset == "hellaswag":
        answers = re.findall(r"1|2|3|4", response)
        if not answers:
            return ""
        return "ending" + answers[0]
    elif dataset == "winogrande":
        answers = re.findall(r"1|2", response)
        if not answers:
            return ""
        return "option" + answers[0]
    elif dataset in ["gsm8k", "mawps", "svamp"]:
        response = response.replace(',', '')
        answers = [s for s in re.findall(r'-?\d+\.?\d*', response)]
        if not answers:
            return float('inf')
        return float(answers[-1])
    else:
        raise ValueError(f"Unsupported dataset: '{dataset}'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--base_path", type=str)
    #* WeLore args
    # parser.add_argument("--we_lore", type=int, default=0)
    # parser.add_argument("--we_lore_path_rank_k_checkpoint", type=str)
    # parser.add_argument("--we_lore_singular_value_path", type=str)
    # parser.add_argument("--we_lore_model_rank", type=int, default=50)
    # parser.add_argument("--we_lore_min_ratio", type=float, default=0.4999)
    
    args = parser.parse_args()

    if "gsm8k" in args.dataset or "mawps" in args.dataset or "svamp" in args.dataset:
        args.max_new_tokens = 256

    dist.init()
    devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
    torch.cuda.set_device(devices[0])
    max_memory = {device: torch.cuda.get_device_properties(device).total_memory for device in devices}

    model = AutoPeftModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory=max_memory,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model.peft_config["default"].base_model_name_or_path,
        model_max_length=512,
        padding_side="left",
        use_fast=False,
    )

    if len(tokenizer) > 32000: #* Llama3
        print("Using LLaMA 3 tokenizer")
        tokenizer.pad_token = "<|reserved_special_token_0|>"
        tokenizer.pad_token_id = 128002

    
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )

    metrics = {}
    for dataset in args.dataset.split("+") if "+" in args.dataset else [args.dataset]:
        with open(os.path.join("datasets", dataset, "test.json")) as fd:
            instances = json.load(fd)
        instances = instances[dist.rank() :: dist.size()]

        num_correct = 0
        num_samples = 0
        miss = 0.001
        for k in trange(0, len(instances), args.batch_size, disable=not dist.is_main(), desc=dataset):
            batch = instances[k : k + args.batch_size]
            targets = [instance["answer"] for instance in batch]

            inputs = [generate_prompt(instance["instruction"]) for instance in batch]
            input_ids = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True).input_ids.cuda()

            with torch.inference_mode():
                output_ids = model.generate(input_ids, generation_config=generation_config)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            outputs = [output.split("### Response:")[1].strip() for output in outputs]
            outputs = [extract_answer(output, dataset=dataset) for output in outputs]
            if dataset in ["gsm8k", "mawps", "svamp"]:
                num_correct += sum(abs(float(target) - output) <= miss  for output, target in zip(outputs, targets))
            else:
                num_correct += sum(output == target for output, target in zip(outputs, targets))
            num_samples += len(targets)

        metrics[dataset] = sum(dist.all_gather(num_correct)) / sum(dist.all_gather(num_samples))

    if dist.is_main():
        print(tabulate(metrics.items(), headers=["Dataset", "Accuracy"], tablefmt="simple_outline"))
        if os.path.exists(args.model_name_or_path):
            with open(os.path.join(args.model_name_or_path, "metrics.json"), "w") as fd:
                json.dump(metrics, fd, indent=4)


if __name__ == "__main__":
    main()
