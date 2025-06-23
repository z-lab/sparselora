from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    peft: Optional[str] = field(default=None)
    benchmark: bool = field(default=False)
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=64)
    lora_dropout: Optional[float] = field(default=0) #0.05)
    lora_target_modules: Optional[str] = field(default="q_proj,k_proj,v_proj,o_proj")

    spft: Optional[str] = field(default=None)
    spft_mode: Optional[str] = field(default="none")
    spft_predictor_basepath: Optional[str] = field(default="./spft/modules/low_rank_weights/")
    spft_skip_sink_tokens: int = field(default=0)
    spft_skip_output_tokens: bool = field(default=False)
    spft_skip_random_tokens: bool = field(default=False)
    spft_start_step: float = field(default=0)
    spft_end_step: float = field(default=1)
    spft_sparse_lora_branch: bool = field(default=False)
    spft_qk_per_head: bool = field(default=False)
    spft_qkvo_seq_avg: bool = field(default=False)
    spft_add_sparse_to_dense: bool = field(default=False)
    spft_dense_to_sparse_ratio: float = field(default=0)
    spft_mlp_seq_avg: bool = field(default=True)
    spft_min_sparse_length: int = field(default=0)
    spft_benchmark: bool = field(default=False)
    
    
    eval_only: bool = field(default=False)
    #* Store action true
    enable_unsloth: bool = field(
        default=False,
        metadata={"help": "Enable Unsloth mode."},
    )

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    
    #* UltraChat args:
    chat_template_format: Optional[str] = field(default="none")


@dataclass
class DataTrainingArguments:
    dataset: Optional[str] = field(default=None)
    eval_dataset: Optional[str] = field(default=None)
    model_max_length: Optional[int] = field(default=512)
    
    #* Instruction Following args:
    max_seq_length: Optional[int] = field(default=512)
    append_concat_token: bool = field(default=False)
    add_special_tokens: bool = field(default=False)
    splits: Optional[str] = field(default="train,test")
    packing: bool = field(default=False)
    dataset_text_field: Optional[str] = field(default="text")