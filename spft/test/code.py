import os
import sys
import re
from tqdm import tqdm
import math
from human_eval.data import write_jsonl, read_problems

import torch
import torch.distributed as dist

from datasets import Dataset


ALPACA_PREFIX_TEMPLATE_MD = """Below is an instruction that describes a task.\n Write a response that appropriately completes the request.

### Instruction:
Complete the following Python code: 
Notes: respond with the entire complete function definition
do not add any comments, be as concise in your code as possible
use only built-in libraries, assume no additional imports other than those provided (if any)
use `    ` (4 spaces) for each level of indentation

code:
```python
{PROMPT}
```

### Response:
```python
"""
# ALPACA_PREFIX_TEMPLATE_MD = """Below is an instruction that describes a task. Write a response that appropriately completes the request. 

# ### Instruction:
# {PROMPT}

# ### Response:
# """ 
    
def post_process(text):
    text = text.replace("```", "")
    text = text.replace("\t", "    ")
    text = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', text, flags=re.DOTALL)
    text = "\n".join([ll.rstrip() for ll in text.splitlines() if ll.strip()])
    lines = text.split("\n")
    spaces_for_each_line = []
    for line in lines:
        match = re.match(r'^( *)', line)
        if match:
            leading_spaces = len(match.group(1))
            spaces_for_each_line.append(leading_spaces)
    try:
        def_line = [i for i, line in enumerate(lines) if "def" in line][0]
        def_line_space = spaces_for_each_line[def_line]
    except:
        print("No def line found")
        print(text)
        def_line_space = 0
    rank_unique_spaces = sorted(list(set(spaces_for_each_line)))
    indentation_level = {}
    i = 0
    for space in rank_unique_spaces:
        if space <= def_line_space:
            indentation_level[space] = 0
        else:
            i += 1
            indentation_level[space] = i
    new_lines = []
    for line, space in zip(lines, spaces_for_each_line):
        new_lines.append("    " * indentation_level[space] + line.lstrip())
    return "\n".join(new_lines)

def split_dataset(dataset, rank, world_size):
    total_size = len(dataset)
    per_process_size = math.ceil(total_size / world_size)
    start_index = rank * per_process_size
    end_index = min(start_index + per_process_size, total_size)
    subset = torch.utils.data.Subset(dataset, list(range(start_index, end_index)))
    return subset

import argparse

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--base_model_name_or_path", type=str)
    
    args = parser.parse_args()
    
    seed = 42
    
    method = "base-lora"
    
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)

    
    # Step 1: load model
    from torch.utils.data import DataLoader
    from transformers import default_data_collator, AutoTokenizer, AutoModelForCausalLM
    from peft import AutoPeftModelForCausalLM
    
    model = AutoPeftModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16,
    device_map="auto",
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


    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s> "})
        model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    
    # Step 2: load dataset
    dataset = read_problems()
    dataset = [v for (k, v) in dataset.items()]
    
    dataset = split_dataset(dataset, local_rank, world_size)
    dataset = Dataset.from_list(dataset)
    
    def preprocess(examples):
        task_ids = [int(task_id.split("/")[-1]) for task_id in examples["task_id"]]
        input_texts = [(ALPACA_PREFIX_TEMPLATE_MD.format(PROMPT=prompt) + " ") for prompt in examples["prompt"]]
        
        encodings = tokenizer(input_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=768)
        
        results = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "task_ids": task_ids,
        }
        return results

    dataset = dataset.map(
        preprocess,
        batched=True,
        batch_size=1000,
        num_proc=1,
        desc="Running tokenizer on dataset",
        remove_columns=["entry_point", "canonical_solution", "test", "prompt"],
    )
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=default_data_collator)
    
    # Step 3: evaluation on test dataset
    all_predictions = []
    num_samples_per_task = 1
    
    model.eval()
    with torch.no_grad():
        for _ in range(num_samples_per_task):
            t = tqdm(dataloader, desc="Running Code Generation...") if dist.get_rank() == 0 else dataloader
            for batch in t:
                outputs = model.generate(
                    batch["input_ids"].to(model.device),
                    attention_mask=batch["attention_mask"].to(model.device),
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=False,
                    max_new_tokens=512,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=0.95,
                    temperature=0.8,
                )
                # wrong here
                predictions = tokenizer.batch_decode(outputs.sequences[:, 768:], skip_special_tokens=True)
                pred = []
                for pred_text in predictions:
                    pred.append(post_process(pred_text))
                batch_result = []
                for task_id, pred_text in zip(batch["task_ids"], pred):
                    batch_result.append(
                        dict(task_id=f"HumanEval/{task_id}", completion=pred_text)
                    )
                all_predictions.extend(batch_result)
            

    all_predictions = gather_from_all_processes(all_predictions)
    print("predictions", all_predictions[:5])
    if dist.get_rank() == 0:
        print(f"Size of predictions: {len(all_predictions)}")
        target_name = args.model_name_or_path + "generated_completions.jsonl" 
        write_jsonl(target_name, all_predictions)
        print(f"Generated samples saved to {target_name}")
        
    dist.destroy_process_group()
    
    
def gather_from_all_processes(data):
    """Gather data from all processes and concatenate."""
    gathered_data = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_data, data)
    # Flatten the list of lists
    return [item for sublist in gathered_data for item in sublist]

if __name__ == "__main__":
    main()

