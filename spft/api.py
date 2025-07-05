import types
from functools import partial
from typing import Dict, Optional, Tuple, Union, List

import torch
from torch import nn
from transformers import TrainerCallback
from spft.train.args import DataTrainingArguments, ModelArguments, TrainingArguments
from transformers.modeling_outputs import CausalLMOutputWithPast
import os
from .callbacks import SPFTCallback
from .modules import SPARSITY_MAPPING, SparseModule
from .utils import io, set_submodule
import peft
from tqdm import tqdm
from colorama import Fore, Style
desc_text = f"{Fore.GREEN}🚀 Patching Model --> Fast Fine-tuning!{Style.RESET_ALL}"
import torch.distributed as dist

__all__ = ["SPFTConfig", "get_spft_model", "get_spft_callback"]


class SPFTConfig:
    def __init__(
        self,
        sparsity: Dict[str, float],
        **kwargs,
    ) -> None:
        self.sparsity = sparsity
        self.current_step = 0

        for key, val in kwargs.items():
            setattr(self, key, val)
            io.rank0_print(f"[Init] Setting {key} to {val}")

    @classmethod
    def from_file(cls, path: str) -> "SPFTConfig":
        return cls(**io.load(path))

    def write_out(self, path: str) -> None:
        io.save(os.path.join(path, "args.json"), vars(self))
    
    def update(self, args: List[Union[ModelArguments, DataTrainingArguments, TrainingArguments]], prefix: Optional[str] = None, cli_keys: set = None) -> None:
        pre_set_args = set(vars(self).keys()) #- set(cli_keys)
        cli_set_args = set(cli_keys)  # these are the args passed via CLI
        
        for arg in args:
            for key, val in vars(arg).items():
                if key in cli_set_args: #* Explicity Overwrtten with CLI args
                    setattr(self, key.removeprefix(prefix), val)
                    io.rank0_print(f"[Update] Setting {key.removeprefix(prefix)} to {val}")
                    
                elif key.startswith(prefix) and key.removeprefix(prefix) not in pre_set_args: #* Not in the pre-set args
                    setattr(self, key.removeprefix(prefix), val)
                    io.rank0_print(f"[Update] Setting {key.removeprefix(prefix)} to {val}")
        
        # args[-1] is a TrainingArguments and args[-2] is DataTrainingArguments
        setattr(self, "per_device_train_batch_size", getattr(args[-1], "per_device_train_batch_size", None))
        setattr(self, "lora_target_modules", getattr(args[-1], "lora_target_modules", None))
        setattr(self, "model_max_length", getattr(args[-2], "model_max_length", None))
        setattr(self, "model_id", getattr(args[-3], "model_name_or_path", None))

def _patch_spft_forward(model: nn.Module, config: SPFTConfig) -> None:
    _unpatched_forward = model.forward

    def _patched_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        masks = None

        if labels is not None:
            masks = torch.zeros_like(input_ids, dtype=torch.bool)
            
                
            if config.skip_random_tokens:
                min_sparse_len = (labels == -100).sum(dim=-1).min().item()
                min_sparse_len = min(min_sparse_len, input_ids.shape[-1])
                
                num_random_tokens = labels.shape[-1] - min_sparse_len
                
                num_random_tokens = max(num_random_tokens, 1)
                
                T = labels.shape[-1]
                ids = []
                for i in range(labels.shape[0]):
                    idx = torch.randperm(T, device=input_ids.device)[:num_random_tokens]
                    masks[i, idx] = True
                    ids.append(idx)
                
            elif config.skip_sink_tokens:
                masks[..., : config.skip_sink_tokens] = True
            
            
            elif config.skip_output_tokens: 
                #* Left Bounds
                is_ctx = (labels == -100)  # shape (B, S)
                left_lengths = is_ctx.cumprod(dim=1).sum(dim=1)
                min_left = left_lengths.min().item()
                
                #* Right Bounds
                right_lengths = is_ctx.flip(dims=[1]).cumprod(dim=1).sum(dim=1).min().item()
                min_right = labels.shape[-1] - right_lengths if right_lengths > 0 else labels.shape[-1]
            
                # Tokens Orders: [...., min_left, output tokens, min_right, ...]
                masks[..., min_left :min_right] = True
                
                if config.sparse_output_tokens != 0: # Apply sparsity to ``some`` output tokens.
                    #* Percentage of output tokens to be sparse:
                    if config.sparse_output_tokens == 1.0:
                        masks = None
                    else:
                        num_out_tokens = min_right - min_left
                        num_sparse_tokens = int(num_out_tokens * config.sparse_output_tokens) if config.sparse_output_tokens > 0 else num_out_tokens
                        num_sparse_tokens = max(num_sparse_tokens, 1)
                        
                        #* Randomly select indices in the output tokens + min_left offset:
                        rand_indices = torch.randperm(num_out_tokens, device=masks.device)[:num_sparse_tokens] + min_left 
                        masks[..., rand_indices] = False  
                        
                        #* Print number of True vs False:
                        num_true = masks.sum().item() / masks.shape[0]
                        num_false = input_ids.shape[-1] - num_true
                        io.rank0_print(f"[SPFT] Dense to Sparse Ratio: {config.sparse_output_tokens}, True Tokens: {num_true}, False Tokens: {num_false} Total Out Tokens: {num_out_tokens}, Total Tokens: {input_ids.shape[-1]}")    
                    
                
                elif config.padding_side == "left":
                    #* No dense output tokens & left-padding:
                    masks = (masks, min_left) #* For easy slicing.
                
            if not (config.skip_sink_tokens or config.skip_output_tokens or config.skip_random_tokens):
                masks = None
            

        for module in self.model.modules():
            if isinstance(module, SparseModule):
                module.forward = partial(module.forward, masks=masks)

        return _unpatched_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    model.forward = types.MethodType(_patched_forward, model)
    
def _patch_spft_generate(model: nn.Module, config: SPFTConfig) -> None:
    _unpatched_generate = model.generate

    def _patched_generate(
        self,
        *args, **kwargs
    ):  
        masks = None
        

        for module in self.model.modules():
            if isinstance(module, SparseModule):
                module.forward = partial(module.forward, masks=masks)

        return _unpatched_generate(
            *args, **kwargs
        )

    model.generate = types.MethodType(_patched_generate, model)
    


def get_spft_model(model: nn.Module, config: SPFTConfig, **kwargs: Dict[str, str]) -> nn.Module:
    #* Patching the forward method of lora module
    from .modules import get_module_mapping, lora_forward, lora4bit_forward
    peft.tuners.lora.layer.Linear.forward = lora_forward
    peft.tuners.lora.Linear4bit.forward = lora4bit_forward
    
    _enable_unsloth = kwargs.get("enable_unsloth", False)
    io.rank0_print(f"Patching SparseLoRA onto {'Unsloth' if _enable_unsloth else 'HF'} model")
    
    MODEL_MAPPING = get_module_mapping(config, enable_unsloth=_enable_unsloth)
    
    if _enable_unsloth:
        assert not config.sparse_lora_branch, "Unsloth currently only supports sparsity on base branches. Please set `sparse_lora_branch` to False."
        
    svd_estimators_loaded = 0
    total_modules = sum(1 for _ in model.named_modules())
    with tqdm(
            total=total_modules,
            desc=desc_text,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ascii=" ░▒█",
            disable=(dist.is_initialized() and dist.get_rank() != 0)
        ) as pbar:
        for name, module in model.named_modules():
            l_name, sparsity = next(((suffix, val) for suffix, val in config.sparsity.items() if name.endswith(suffix)), (None, None))
            if sparsity is not None:
                kwargs = {"name": l_name, "idx":int(l_name.split(".")[1]), "sparsity": sparsity, "cfg": config}
                if type(module) in MODEL_MAPPING:
                    set_submodule(model, name, MODEL_MAPPING[type(module)](base=module, **kwargs))
                svd_estimators_loaded += 1
                for sub_name, sub_module in module.named_modules():
                    if isinstance(sub_module, nn.Linear):
                        mode = None if "lora_" in sub_name and not config.sparse_lora_branch else SPARSITY_MAPPING.get(sub_name, None)
                        set_submodule(model, f"{name}.{sub_name}", MODEL_MAPPING[type(sub_module)](base=sub_module, mode=mode, config=config))
                        svd_estimators_loaded += 1
            pbar.update(1)
        pbar.set_postfix({"SVD Estimators Loaded": svd_estimators_loaded})
    _patch_spft_forward(model, config)
    _patch_spft_generate(model, config)

    return model


def get_spft_callback(config: SPFTConfig) -> TrainerCallback:
    return SPFTCallback(start_step=config.start_step, end_step=config.end_step)
