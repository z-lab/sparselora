"""Extensible registry mapping original module types to sparse replacements."""

from typing import Dict, Type

import torch.nn as nn

from .linear import SparseLinear

_MODULE_REGISTRY: Dict[Type[nn.Module], Type[nn.Module]] = {
    nn.Linear: SparseLinear,
}

_OUT_PROJS = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}
_IN_PROJS = {"o_proj", "down_proj"}


def register_sparse_module(original_cls: Type[nn.Module], sparse_cls: Type[nn.Module]) -> None:
    """Register a sparse replacement for a module class.

    Example::

        from sparselora.modules import register_sparse_module
        register_sparse_module(MistralMLP, SparseMistralMLP)
    """
    _MODULE_REGISTRY[original_cls] = sparse_cls


def get_module_mapping() -> Dict[Type[nn.Module], Type[nn.Module]]:
    """Return a copy of the current module mapping."""
    return dict(_MODULE_REGISTRY)


def get_sparsity_mode(sub_name: str):
    """Derive the sparsity direction for a linear sub-layer.

    Returns ``"out"``/``"out_scatter"`` for output-sparse projections,
    ``"in"``/``"in_gather"`` for input-sparse projections, or ``None``
    for layers that should stay dense (LoRA adapters).
    """
    if "lora_" in sub_name:
        return None
    proj = sub_name.split(".")[0]
    is_base_layer = ".base_layer" in sub_name
    if proj in _OUT_PROJS:
        return "out_scatter" if is_base_layer else "out"
    if proj in _IN_PROJS:
        return "in_gather" if is_base_layer else "in"
    return None


# Auto-register built-in Llama support
def _register_llama():
    from transformers.models.llama.modeling_llama import LlamaMLP, LlamaAttention
    from .llama import SparseLlamaMLP, SparseLlamaAttention
    register_sparse_module(LlamaMLP, SparseLlamaMLP)
    register_sparse_module(LlamaAttention, SparseLlamaAttention)


_register_llama()
