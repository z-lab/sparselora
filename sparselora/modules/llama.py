"""Sparse replacements for Llama MLP and Attention modules."""

from typing import Any, Optional

import torch
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward

from .base import SparseModule
from .svd import create_attn_predictor, create_mlp_predictor

_silu_mul = LigerSiLUMulFunction.apply


class SparseLlamaMLP(SparseModule):
    """Sparse replacement for ``LlamaMLP``."""

    inherited_attributes = ["gate_proj", "up_proj", "down_proj"]

    def __init__(self, base: nn.Module, *, name: str, idx: int, sparsity: float, cfg) -> None:
        super().__init__(base)
        self.sparsity = sparsity
        if self.sparsity > 0:
            self.pred = create_mlp_predictor(base, cfg.predictor_rank, name, cfg)

    def _block(self, x: torch.Tensor, **kw: Any) -> torch.Tensor:
        return self.down_proj(_silu_mul(self.gate_proj(x, **kw), self.up_proj(x, **kw)), **kw)

    def forward(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not (self.enabled and self.sparsity > 0):
            return self._block(x)

        indices = self.pred.predict(x, self.sparsity)
        if masks is None:
            return self._block(x, sparse_indices=indices)

        sparse_x, dense_x = self.token_splits(x, masks)
        return self.token_join(
            sparse=self._block(sparse_x, sparse_indices=indices),
            dense=self._block(dense_x),
            masks=masks,
        )


class SparseLlamaAttention(SparseModule):
    """Sparse replacement for ``LlamaAttention``."""

    inherited_attributes = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "config",
        "layer_idx",
        "head_dim",
        "num_key_value_groups",
        "scaling",
        "attention_dropout",
        "is_causal",
    ]

    def __init__(self, base: nn.Module, *, name: str, idx: int, sparsity: float = 0, cfg) -> None:
        super().__init__(base)
        self.sparsity = sparsity
        if self.sparsity > 0:
            self.pred = create_attn_predictor(base, cfg.predictor_rank, name, cfg)

    def _sparse_proj(self, proj, sparse_x, dense_x, indices, masks):
        return self.token_join(proj(sparse_x, indices), proj(dense_x), masks)

    def _proj_qkv(self, x, masks, q_i, k_i, v_i):
        if masks is None:
            return self.q_proj(x, q_i), self.k_proj(x, k_i), self.v_proj(x, v_i)
        sparse_x, dense_x = self.token_splits(x, masks)
        return (
            self._sparse_proj(self.q_proj, sparse_x, dense_x, q_i, masks),
            self._sparse_proj(self.k_proj, sparse_x, dense_x, k_i, masks),
            self._sparse_proj(self.v_proj, sparse_x, dense_x, v_i, masks),
        )

    def _proj_o(self, x, masks, v_i):
        if masks is None:
            return self.o_proj(x, v_i)
        sparse_x, dense_x = self.token_splits(x, masks)
        return self._sparse_proj(self.o_proj, sparse_x, dense_x, v_i, masks)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        sparse = self.enabled and self.sparsity > 0

        if sparse:
            q_i, k_i, v_i = self.pred.predict(hidden_states, self.sparsity)
            Q, K, V = self._proj_qkv(hidden_states, masks, q_i, k_i, v_i)
        else:
            Q, K, V = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)

        Q = Q.view(hidden_shape).transpose(1, 2)
        K = K.view(hidden_shape).transpose(1, 2)
        V = V.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            K, V = past_key_values.update(K, V, self.layer_idx, cache_kwargs)

        attention_fn = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)
        attn_output, attn_weights = attention_fn(
            self,
            Q,
            K,
            V,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self._proj_o(attn_output, masks, v_i) if sparse else self.o_proj(attn_output)
        return attn_output, attn_weights
