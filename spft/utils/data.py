"""
Unified dataset loading and processing utilities for SparseLoRA training.
Supports math10k, csr170k, codefeedback, and wizardLM datasets.
"""

from functools import partial
from typing import Dict, List, Optional, Union, Any
import logging
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Import from spft modules
from spft.data import generate_and_tokenize_prompt_math10k, generate_and_tokenize_prompt_csr170k
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

@processor.cache_to_disk("data_cache")
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
                "padding": "max_length",
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
    

DATA_REGISTRY = {
    "CodeFeedback-Filtered-Instruction": load_codefeedback_dataset,
    "WizardLM_evol_instruct_70k": load_wizardlm_dataset,
    "math_10k": load_math10k_dataset,
    "commonsense_170k": load_csr170k_dataset,
    "commonsense_15k": load_csr170k_dataset,
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
