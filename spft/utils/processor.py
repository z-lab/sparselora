from datasets import load_dataset, Dataset, DatasetDict
import typing as tp
import functools
import os
import pickle

"""
This scripts preprocess any NLP dataset into a text-to-text format.
"""

from typing import Any, Callable, Dict
from transformers import AutoTokenizer
from filelock import FileLock
import torch.distributed as dist

def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

class DatasetPreprocessor:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        tokenizer_kwargs: Dict[str, Any] = None,
    ):
        """
        Initializes an instance of the datasets_preprocess class with a tokenizer object.

        Args:
            tokenizer: An instance of a tokenizer class used to preprocess text data.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs

def cache_to_disk(root_datadir):
    def decorator_cache(func):
        @functools.wraps(func)
        def wrapper_cache(*args, **kwargs):
            if not os.path.exists(root_datadir):
                os.makedirs(root_datadir, exist_ok=True)

            data_args, tokenizer = args[0], args[1]
            target_name = f"{data_args.max_seq_length}_{tokenizer.name_or_path.replace('/', '_')}_pad_{tokenizer.padding_side}"
            func_name = func.__name__.replace("/", "")
            cache_file = os.path.join(root_datadir, f"{func_name}_{target_name}.pkl")
            temp_file = cache_file + ".tmp"
            lock_file = cache_file + ".lock"

            if not os.path.exists(cache_file):
                if is_main_process():
                    print(f"[Rank 0] Cache not found. Generating and writing to {cache_file}...")
                    result = func(*args, **kwargs)
                    with FileLock(lock_file):
                        with open(temp_file, "wb") as f:
                            pickle.dump(result, f)
                        os.rename(temp_file, cache_file)
                else:
                    print(f"[Rank {dist.get_rank()}] Waiting for rank 0 to finish writing cache...")

            # All ranks wait for rank 0 to finish writing
            if dist.is_initialized():
                dist.barrier()

            # Now safe to read
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        return wrapper_cache
    return decorator_cache
class WizardLM52k_Preprocessor(DatasetPreprocessor):

    def __call__(self, example):
        """
        Preprocess the WizardLM dataset into a text-to-text format.
        """
        if isinstance(example["instruction"], str):
            raise NotImplementedError
    
        else:
            combined_text = [
                x + " " + y + self.tokenizer.eos_token 
                for (x, y) in zip(example["instruction"], example["output"])
            ]
            encodings = self.tokenizer(combined_text, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            input_text_length = [
                len(self.tokenizer(example["instruction"][i], return_tensors="pt")["input_ids"][0])
                for i in range(len(example["instruction"]))
            ]
            labels = encodings["input_ids"].clone()
            for i, l in enumerate(input_text_length):
                labels[i, :l] = -100
            labels[encodings["attention_mask"] == 0] = -100
            
            results = {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "labels": labels,
            }

            return results
        
class CodeFeedback100k_Preprocessor(DatasetPreprocessor):

    def __call__(self, example):
        """
        Preprocess the CoLA dataset into a text-to-text format.
        """
        if isinstance(example["x"], str):
            # not batched
            raise NotImplementedError
    
        else:
            combined_text = [(x + " " + y + self.tokenizer.eos_token) for (x, y) in zip(example["x"], example["y"])]
            encodings = self.tokenizer(combined_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)

            labels = encodings["input_ids"].clone()
            input_text_length = [
                len(self.tokenizer(example["x"][i], return_tensors="pt")["input_ids"][0])
                for i in range(len(example["x"]))
            ]
            for i, l in enumerate(input_text_length):
                labels[i, :l] = -100
            labels[encodings["attention_mask"] == 0] = -100
            
            results = {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "labels": labels,
            }

            return results

