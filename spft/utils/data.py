"""
Unified dataset loading and processing utilities for SparseLoRA training.
Supports math10k, csr170k, codefeedback, and wizardLM datasets.
"""

from functools import partial
from typing import Dict, List, Optional, Union, Any
import logging
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
from transformers import AutoTokenizer
from tqdm import tqdm

# Import from spft modules
from spft.data import generate_and_tokenize_prompt_math10k, generate_and_tokenize_prompt_csr170k, generate_and_tokenize_arc_agi, tokenize_arc_agi
import spft.utils.processor as processor

logger = logging.getLogger(__name__)


template_wo_input = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
'''


def load_math10k_dataset(
    data_args: Optional[Dict[str, Any]] = None,
    tokenizer: Optional[AutoTokenizer] = None,
) -> Dict[str, Any]:
    """
    Load and process the math10k dataset.
    
    Args:
        data_args: Data configuration arguments containing dataset path
        tokenizer: The tokenizer to use for processing
    
    Returns:
        Dictionary with 'train' and 'eval' splits
    """
    if data_args is None:
        raise ValueError("data_args is required and must contain 'dataset' key")
    
    # Load the dataset
    dataset = load_dataset("json", data_files=data_args.dataset)["train"]
    dataset = dataset.map(partial(generate_and_tokenize_prompt_math10k, tokenizer))
    
    return {
        "train": dataset,
        "eval": None
    }
    
def load_csr170k_dataset(
    data_args: Optional[Dict[str, Any]] = None,
    tokenizer: Optional[AutoTokenizer] = None,
) -> Dict[str, Any]:
    """
    Load and process the csr170k dataset.
    
    Args:
        data_args: Data configuration arguments containing dataset path
        tokenizer: The tokenizer to use for processing
    
    Returns:
        Dictionary with 'train' and 'eval' splits
    """
    if data_args is None:
        raise ValueError("data_args is required and must contain 'dataset' key")
    
    # Load the dataset
    dataset = load_dataset("json", data_files=data_args.dataset)["train"]
    dataset = dataset.map(partial(generate_and_tokenize_prompt_csr170k, tokenizer))
    
    return {
        "train": dataset,
        "eval": None
    }



def code_preprocess(data) -> Dict[str, str]:
    y = data['answer']
    y = "```".join(y.split("```")[:2]) + "```" # only keep the first code block
    return {
        "x": template_wo_input.format(
            instruction=data['query']
        ),
        "y": y,
    }

@processor.cache_to_disk("datasets")
def load_codefeedback_dataset(
    data_args: Optional[Dict[str, Any]] = None,
    tokenizer: Optional[AutoTokenizer] = None,
) -> Dict[str, Any]:
    """
    Load and process the CodeFeedback dataset.
    
    Args:
        data_args: Data configuration arguments containing dataset path and settings
        tokenizer: The tokenizer to use for processing
    
    Returns:
        Dictionary with 'train' and 'eval' splits
    """
    print("Loading CodeFeedback dataset...")
    max_tokens = data_args.max_seq_length
    
    dataset = load_dataset(data_args.dataset, split="train")
    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(seed=42)
    bar = tqdm(dataset, total=101000)
    total = 0
    ok = 0

    for sample in dataset:
        total += 1
        temp = code_preprocess(sample)
        if "```" not in sample['answer']:
            continue
        if len(tokenizer(temp['x']+' '+temp['y'])['input_ids']) >= max_tokens:
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = code_preprocess(sample)
        if count < 100000:  # First 100,000 samples for training
            train_samples.append(processed_sample)
        elif 100000 <= count < 101000:  # Next 10,000 samples for evaluation
            eval_samples.append(processed_sample)
        elif count >= 101000:  # Stop processing after collecting enough samples
            break
        count += 1
        
    # convert to hf dataset
    train_samples = Dataset.from_list(train_samples)
    eval_samples = Dataset.from_list(eval_samples)
    datasets = DatasetDict({
        "train": train_samples,
        "eval": eval_samples,
    })
    
    preprocessor = processor.CodeFeedback100k_Preprocessor(
        tokenizer=tokenizer,
        tokenizer_kwargs={
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "pt",
            "max_length": data_args.model_max_length
        },
    )
    
    datasets = datasets.map(
        preprocessor,
        batched=True,
        batch_size=5000,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    
    return {
        "train": datasets["train"],
        "eval": datasets["eval"]
    }


def chat_preprocess(data) -> Dict[str, str]:
    """
    Preprocess a single chat data point for WizardLM dataset.
    Args:
        data: Dictionary containing 'instruction' and 'output'
    Returns:
        Dictionary with formatted instruction and output
    """
    y = data['output']
    return {
        "instruction": template_wo_input.format(
            instruction=data['instruction']
        ),
        "output": y,
    }

@processor.cache_to_disk("datasets")
def load_wizardlm_dataset(
    data_args: Optional[Dict[str, Any]] = None,
    tokenizer: Optional[AutoTokenizer] = None,
) -> Dict[str, Any]:
    """
    Load and process the WizardLM dataset.
    
    Args:
        data_args: Data configuration arguments containing dataset path and settings
        tokenizer: The tokenizer to use for processing
    
    Returns:
        Dictionary with 'train' and 'eval' splits
    """
    print("Loading WizardLM dataset...")
    max_tokens = data_args.max_seq_length
    dataset = load_dataset(data_args.dataset, split="train")
    # Load the dataset from HuggingFace
    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(seed=42)
    bar = tqdm(dataset, total=55000)
    total = 0
    ok = 0
    for sample in dataset:
        total += 1
        temp = chat_preprocess(sample)
        if "sorry" in temp['output'].lower() or "as an ai" in temp['output'].lower():
            continue
        if len(tokenizer(temp['instruction']+' '+temp['output'])['input_ids']) >= max_tokens:
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = temp
        if count < 52000:
            train_samples.append(processed_sample)
        elif 52000 <= count < 55000:
            eval_samples.append(processed_sample)
        elif count >= 55000:  # Stop processing after collecting enough samples
            break
        count += 1
        
    # convert to hf dataset
    train_samples = Dataset.from_list(train_samples)
    eval_samples = Dataset.from_list(eval_samples)
    datasets = DatasetDict({
        "train": train_samples,
        "eval": eval_samples,
    })
    
    preprocessor = processor.WizardLM52k_Preprocessor(
            tokenizer=tokenizer,
            tokenizer_kwargs={
                "truncation": True,
                "return_tensors": "pt",
                "max_length": data_args.model_max_length,
            },
        )

    datasets = datasets.map(
        preprocessor,
        batched=True,
        batch_size=5000,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    
    return {
        "train": datasets["train"],
        "eval": datasets["eval"]
    }

@processor.cache_to_disk("datasets")
def load_arc_agi_dataset(
    data_args: Optional[Dict[str, Any]] = None,
    tokenizer: Optional[AutoTokenizer] = None,
) -> Dict[str, Any]:
    """
    Load and process the ARC AGI dataset.
    
    Args:
        data_args: Data configuration arguments containing dataset path
        tokenizer: The tokenizer to use for processing
    
    Returns:
        Dictionary with 'train' and 'eval' splits
    """
    if data_args is None:
        raise ValueError("data_args is required and must contain 'dataset' key")
    
    # Define the dataset names and sampling ratios
    dataset_mixer = {
        "barc0/transduction_formatted_test_time_finetune_for_evaluation": 1.0,
        "barc0/transduction_formatted_rearc_dataset_100k": 0.05,
        "barc0/transduction_heavy_100k_jsonl": 0.05
    }

    # Load and subsample each dataset
    datasets = []
    for dataset_name, sample_ratio in dataset_mixer.items():
        ds = load_dataset(dataset_name, split="train_sft")  # change 'train' to correct split if needed
        if sample_ratio < 1.0:
            ds = ds.shuffle(seed=42).select(range(int(len(ds) * sample_ratio)))
        datasets.append(ds)

    # Combine all datasets
    dataset = concatenate_datasets(datasets)
    
    print(f"Loaded {len(dataset)} samples from ARC AGI datasets, now processing...")
    
    #* First Filter Dataset:
    samples = []
    total_processed = 0
    samples_added = 0
    
    pbar = tqdm(dataset, desc="Filtering samples")
    for data_point in pbar:
        total_processed += 1
        messages = data_point["messages"]
        messages = [m for m in messages if m["content"].strip() != ""]
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("Last message must be from the assistant.")
        tokenized_full_prompt = tokenize_arc_agi(tokenizer, data_args.max_seq_length, messages, add_generation_prompt=False)
        
        if len(tokenized_full_prompt['input_ids']) > data_args.max_seq_length:
            # print(f"Skipping sample with length {len(tokenized_full_prompt['input_ids'])} > {data_args.max_seq_length}")
            continue
        else:
            samples.append(data_point)
            samples_added += 1
            
        # Update progress bar with current statistics
        pbar.set_postfix({
            'Added': samples_added,
            'Processed': total_processed,
            'Keep_Rate': f"{samples_added/total_processed:.2%}" if total_processed > 0 else "0%"
        })
            
    dataset = Dataset.from_list(samples)
    
    print(f"Loaded {len(dataset)} samples after filtering ARC AGI dataset")
    
    dataset = dataset.map(partial(generate_and_tokenize_arc_agi, tokenizer, data_args.max_seq_length),
                          batched=False, load_from_cache_file=False, desc="Processing ARC AGI dataset"
                          ).filter(lambda x: x is not None)
    
    print(f"Filtered {len(dataset)} samples from ARC AGI dataset")
    
    return {
        "train": dataset,
        "eval": None
    }
    

DATA_REGISTRY = {
    "CodeFeedback-Filtered-Instruction": load_codefeedback_dataset,
    "WizardLM_evol_instruct_70k": load_wizardlm_dataset,
    "math_10k": load_math10k_dataset,
    "commonsense_170k": load_csr170k_dataset,
    "commonsense_15k": load_csr170k_dataset,
    "arc_agi": load_arc_agi_dataset,
}

def load_dataset_by_name(
    data_args: Optional[Dict[str, Any]] = None,
    tokenizer: Optional[AutoTokenizer] = None,
) -> Dict[str, Any]:
    """
    Load a dataset by name with appropriate preprocessing.
    
    Args:
        data_args: Configuration arguments for the dataset
        tokenizer: The tokenizer to use
    
    Returns:
        Dictionary with 'train' and 'eval' splits
    """
    dataset_name = data_args.dataset.split("/")[-1].split(".")[0]
    
    fn = DATA_REGISTRY.get(dataset_name, None)
    
    if fn is not None:
        return fn(data_args, tokenizer)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported datasets: math10k, csr170k, codefeedback, wizardlm")
