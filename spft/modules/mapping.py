import torch.nn as nn
import bitsandbytes as bnb
from .mlp import SparseLlamaMLP
from .linear import SparseLinear
from .linear4bit import SparseLinear4bit
from transformers.models.llama.modeling_llama import LlamaMLP

__all__ = ["SPFT_MODULE_MAPPING", "UNSLOTH_MODULE_MAPPING", "SPARSITY_MAPPING"]


SPFT_MODULE_MAPPING = {
    LlamaMLP: SparseLlamaMLP,
    nn.Linear: SparseLinear,
    #* QLoRA mappings
    bnb.nn.modules.Linear4bit: SparseLinear4bit,
    
}

UNSLOTH_MODULE_MAPPING = {
    nn.Linear: SparseLinear,
}

try:
    from transformers.models.llama.modeling_llama import LlamaFlashAttention2
    from .attn import SparseLlamaFlashAttention
    SPFT_MODULE_MAPPING[LlamaFlashAttention2] = SparseLlamaFlashAttention
except ImportError or ModuleNotFoundError:
    # Fallback for newer versions of transformers to support Unsloth
    from transformers.models.llama.modeling_llama import LlamaAttention
    from .unsloth import UnslothSparseLlamaAttention, UnslothSparseLlamaMLP
    UNSLOTH_MODULE_MAPPING[LlamaAttention] = UnslothSparseLlamaAttention
    UNSLOTH_MODULE_MAPPING[LlamaMLP] = UnslothSparseLlamaMLP



SPARSITY_MAPPING = {
    
    "q_proj": "out",
    "q_proj.base_layer": "out_scatter",
    "q_proj.lora_B.default": "out_scatter",
    
    "k_proj": "out",
    "k_proj.base_layer": "out_scatter",
    "k_proj.lora_B.default": "out_scatter",
    
    "v_proj": "out",
    "v_proj.base_layer": "out_scatter",
    "v_proj.lora_B.default": "out_scatter",
    
    "o_proj": "in",
    "o_proj.base_layer": "in_gather",
    "o_proj.lora_A.default": "in_gather",
    
    "gate_proj": "out",
    "gate_proj.base_layer": "out_scatter",
    "gate_proj.lora_B.default": "out_scatter",
       
    "up_proj": "out",
    "up_proj.base_layer": "out_scatter",
    "up_proj.lora_B.default": "out_scatter",
    
    "down_proj": "in",
    "down_proj.base_layer": "in_gather",
    "down_proj.lora_A.default": "in_gather",
}