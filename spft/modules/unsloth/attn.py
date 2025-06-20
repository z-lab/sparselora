import torch
from typing import Optional, Tuple, List
from torch.nn.functional import scaled_dot_product_attention

from unsloth.kernels import (
    fast_rope_embedding,
)
# from ..tokenizer_utils import *
HAS_FLASH_ATTENTION = True
if HAS_FLASH_ATTENTION:
    from flash_attn import flash_attn_func

# Final patching code
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
)

# For Pytorch 2.1.1
try:
    from transformers.models.llama.modeling_llama import (
        LlamaSdpaAttention,
        LlamaFlashAttention2,
    )
except:
    LlamaSdpaAttention   = LlamaAttention
    LlamaFlashAttention2 = LlamaAttention
pass

from unsloth.models._utils import *

HAS_XFORMERS = xformers is not None
BlockDiagonalCausalMask = xformers.attn_bias.BlockDiagonalCausalMask if HAS_XFORMERS else None

from spft.modules import SparseModule
import torch.nn as nn


from unsloth.kernels.fast_lora import apply_lora_qkv
from unsloth.kernels.utils import get_lora_parameters, matmul_lora, fast_dequantize, torch_amp_custom_fwd, torch_amp_custom_bwd


__all__ = ["UnslothSparseLlamaAttention"]

from spft.modules.unsloth.lora import sparse_matmul_lora

class SparseLoRA_QKV(torch.autograd.Function):
    """
    ### LoRA weights
    Wq = Wq + Aq @ Bq
    Wk = Wk + Ak @ Bk
    Wv = Wv + Av @ Bv
    Q = X @ Wq = X @ Wq + X @ Aq @ Bq
    K = X @ Wk = X @ Wk + X @ Ak @ Bk
    V = X @ Wv = X @ Wv + X @ Av @ Bv

    ### Backpropagation chain rule
    See our blogpost for more details.

    dC/dWq = X.T @ D(Wq)
    dC/dWk = X.T @ D(Wk)
    dC/dWv = X.T @ D(Wv)
    We then sum them all find dC/dX

    ### Q projection LoRA weights
    dC/dAq =       X.T @ D(Wq) @ B.T
    dC/dBq = A.T @ X.T @ D(Wq)

    ### K projection LoRA weights
    dC/dAk =       X.T @ D(Wk) @ B.T
    dC/dBk = A.T @ X.T @ D(Wk)

    ### V projection LoRA weights
    dC/dAv =       X.T @ D(Wv) @ B.T
    dC/dBv = A.T @ X.T @ D(Wv)
    """
    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, X : torch.Tensor,
                QW, QW_quant, QA, QB, QS,
                KW, KW_quant, KA, KB, KS,
                VW, VW_quant, VA, VB, VS,
                inplace = True, sparse_indices: List[torch.Tensor] = None, sparse_lora_branch: bool = False) -> torch.Tensor:
        dtype = X.dtype

        ctx.sparse_indices = sparse_indices
        ctx.sparse_lora_branch = sparse_lora_branch
        if sparse_indices is not None:
            q_indices, k_indices, v_indices = sparse_indices
            Q = sparse_matmul_lora(X, QW, QW_quant, QA, QB, QS, sparse_indices = q_indices, sparse_lora_branch = sparse_lora_branch, mode = "out_scatter")
            K = sparse_matmul_lora(X, KW, KW_quant, KA, KB, KS, sparse_indices = k_indices, sparse_lora_branch = sparse_lora_branch, mode = "out_scatter")
            V = sparse_matmul_lora(X, VW, VW_quant, VA, VB, VS, sparse_indices = v_indices, sparse_lora_branch = sparse_lora_branch, mode = "out_scatter")

        else:
            Q = matmul_lora(X, QW, QW_quant, QA, QB, QS)
            K = matmul_lora(X, KW, KW_quant, KA, KB, KS)
            V = matmul_lora(X, VW, VW_quant, VA, VB, VS)

        ctx.custom_saved_tensors = (
            QW, QW_quant, QS,
            KW, KW_quant, KS,
            VW, VW_quant, VS,
        )
        ctx.save_for_backward(X, QA, QB, KA, KB, VA, VB,)
        ctx.inplace = inplace
        return Q, K, V
    pass

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dQ, dK, dV):
        QW, QW_quant, QS, KW, KW_quant, KS, VW, VW_quant, VS = \
            ctx.custom_saved_tensors
        X, QA, QB, KA, KB, VA, VB, = ctx.saved_tensors

        dQ = dQ.contiguous()
        dK = dK.contiguous()
        dV = dV.contiguous()
        
        batch, seq_len, hd = X.shape
        dQ = dQ.view(-1, dQ.shape[-1])
        dK = dK.reshape(-1, dK.shape[-1]) # view doesn't work on K.T
        dV = dV.view(-1, dV.shape[-1])
        X  = X .view(-1, X .shape[-1])
        dtype = X.dtype

        QA, QB, KA, KB, VA, VB = \
            QA.to(dtype), QB.to(dtype), KA.to(dtype), KB.to(dtype), VA.to(dtype), VB.to(dtype)

        QA, QB, KA, KB, VA, VB = \
            QA.t(), QB.t(), KA.t(), KB.t(), VA.t(), VB.t()
            
        ### Weight projection LoRA weights
        # See our blogpost for more details.
        d_QA = torch.empty_like(QA)
        d_QB = torch.empty_like(QB)
        d_KA = torch.empty_like(KA)
        d_KB = torch.empty_like(KB)
        d_VA = torch.empty_like(VA)
        d_VB = torch.empty_like(VB)

        # Q Projection
        # d_QA = X.t() @ (dQ @ QB.t())
        # d_QB = (QA.t() @ X.t()) @ dQ
        # d_QA *= QS
        # d_QB *= QS
        d_QA.addmm_(X.t(), dQ @ QB.t(), alpha = QS, beta = 0)
        d_QB.addmm_(QA.t() @ X.t(), dQ, alpha = QS, beta = 0)

        # K Projection
        # d_KA = X.t() @ (dK @ KB.t())
        # d_KB = (KA.t() @ X.t()) @ dK
        # d_KA *= KS
        # d_KB *= KS
        d_KA.addmm_(X.t(), dK @ KB.t(), alpha = KS, beta = 0)
        d_KB.addmm_(KA.t() @ X.t(), dK, alpha = KS, beta = 0)

        # V Projection
        # d_VA = X.t() @ (dV @ VB.t())
        # d_VB = (VA.t() @ X.t()) @ dV
        # d_VA *= VS
        # d_VB *= VS
        d_VA.addmm_(X.t(), dV @ VB.t(), alpha = VS, beta = 0)
        d_VB.addmm_(VA.t() @ X.t(), dV, alpha = VS, beta = 0)

        if ctx.sparse_indices is not None:
            q_indices, k_indices, v_indices = ctx.sparse_indices
            dQ_reshaped = dQ.view(batch, seq_len, dQ.shape[-1])
            dK_reshaped = dK.view(batch, seq_len, dK.shape[-1])
            dV_reshaped = dV.view(batch, seq_len, dV.shape[-1])
            
            dX_Q = sparse_matmul_lora(dQ_reshaped, QW, QW_quant, QB, QA, QS,
                                        sparse_indices=q_indices, bwd=True,
                                        sparse_lora_branch=ctx.sparse_lora_branch,
                                        mode="in_gather")
            
            dX_K = sparse_matmul_lora(dK_reshaped, KW, KW_quant, KB, KA, KS,
                                        sparse_indices=k_indices, bwd=True,
                                        sparse_lora_branch=ctx.sparse_lora_branch,
                                        mode="in_gather")
            
            dX_V = sparse_matmul_lora(dV_reshaped, VW, VW_quant, VB, VA, VS,
                                        sparse_indices=v_indices, bwd=True,
                                        sparse_lora_branch=ctx.sparse_lora_branch,
                                        mode="in_gather")
            
            # Combine derivatives to find dX
            dX = dX_Q + dX_K + dX_V
            del dX_Q, dX_K, dX_V
            dX = dX.view(-1, dX.shape[-1])  # Flatten to (batch * seq_len, hd)
            
        else:
            # Combine derivatives to find dX
            # dQ
            QW = fast_dequantize(QW.t(), QW_quant)
            dX = torch.matmul(dQ, QW.t(), out = X if ctx.inplace else None)
            del QW
            # dX += (dQ @ QB.to(dtype).t() @ (QS * QA.to(dtype).t()))
            dX.addmm_(dQ @ QB.t(), QA.t(), alpha = QS)

            # dK
            KW = fast_dequantize(KW.t(), KW_quant)
            # dX += dK @ KW.t()
            dX.addmm_(dK, KW.t())
            del KW
            # dX += dK @ KB.to(dtype).t() @ (KS * KA.to(dtype).t())
            dX.addmm_(dK @ KB.t(), KA.t(), alpha = KS)

            # dV
            VW = fast_dequantize(VW.t(), VW_quant)
            # dX += dV @ VW.t()
            dX.addmm_(dV, VW.t())
            del VW
            # dX += dV @ VB.to(dtype).t() @ (VS * VA.to(dtype).t())
            dX.addmm_(dV @ VB.t(), VA.t(), alpha = VS)

        # QW, QW_quant, QA, QB, QS,
        # KW, KW_quant, KA, KB, KS,
        # VW, VW_quant, VA, VB, VS,
        return dX.view(batch, seq_len, hd), \
            None, None, d_QA.t(), d_QB.t(), None, \
            None, None, d_KA.t(), d_KB.t(), None, \
            None, None, d_VA.t(), d_VB.t(), None, \
            None, \
                None, None
    pass
pass


def apply_sparselora_qkv(self, X, inplace = True, sparse_indices: Optional[torch.Tensor] = None, sparse_lora_branch: bool = False):
    QW, QW_quant, QA, QB, QS = get_lora_parameters(self.q_proj)
    KW, KW_quant, KA, KB, KS = get_lora_parameters(self.k_proj)
    VW, VW_quant, VA, VB, VS = get_lora_parameters(self.v_proj)
    Q, K, V = SparseLoRA_QKV.apply(X,
        QW, QW_quant, QA, QB, QS,
        KW, KW_quant, KA, KB, KS,
        VW, VW_quant, VA, VB, VS,
        inplace, sparse_indices, sparse_lora_branch
    )
    return Q, K, V
pass


class UnslothSparseLlamaAttention(SparseModule):
    """
    Unsloth Sparse Llama Attention module.
    This module is a sparse version of the LlamaAttention module, optimized for fast fine-tuning with unsloth.
    It uses fast RoPE embedding and supports grouped query attention.
    """
    
    inherited_attributes = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "rotary_emb",
        "head_dim",
        "num_key_value_groups",
        "attention_dropout",
        "scaling",
        "is_causal",
        "config",
        "num_key_value_groups",
        "apply_qkv", "apply_o", "rotary_emb"
    ]

    def __init__(self, base: nn.Module, *, name: str, idx: int, sparsity: float = 0, cfg) -> None:
        super().__init__(base)
        self.sparsity = sparsity
        self.layer_name = name
        self.layer_idx = idx #base.layer_idx
        if self.sparsity > 0:
            self.load_predictor(base,cfg)
                    
        self.per_channel = not cfg.qk_per_head
        self.sparse_lora_branch = cfg.sparse_lora_branch
        if cfg.add_sparse_to_dense:
            raise ValueError("Dense to Sparse Ratio is not supported in Llama Attention")
            self.dense_to_sparse_ratio = cfg.dense_to_sparse_ratio
            print("Addint output token sparsity at ratio: ", self.dense_to_sparse_ratio)
        else:
            self.dense_to_sparse_ratio = None


    def kernel_proj_o_forward(self, x, masks, vo_indices):
        
        if masks is None: # or self.layer_idx == 13: #* No Split
            return self.o_proj(x, vo_indices)
        
        else: #* Split
            if self.dense_to_sparse_ratio is not None:
                raise ValueError("Dense to Sparse Ratio is not supported in Llama Attention")
                #* Dense to Sparse Ratio:
                sparse_vo_indices, dense_vo_indices = vo_indices
            else:
                #* No Dense to Sparse Ratio:
                sparse_vo_indices = vo_indices
                dense_vo_indices = None
            # print("Sparse Train, ", dense_vo_indices.shape, sparse_vo_indices.shape)
            # sparse_vo_indices, dense_vo_indices = vo_indices
            sparse_x, dense_x = self.token_splits(x, masks)
            dense_o = self.o_proj(dense_x, dense_vo_indices)
            sparse_o = self.o_proj(sparse_x, sparse_vo_indices)
            
            # #* Token Order: [Sparse | Dense] --> [In | Out]
            out_o = self.token_join(sparse=sparse_o, dense=dense_o, masks=masks)
            return out_o
        
    def kernel_proj_forward(self, x, masks, indices):
        #* Unpack and clone indices
        sparse_q_indices, sparse_k_indices, sparse_v_indices = indices
        
        if masks is None: # or self.layer_idx == 13: #* No Split
            out_q, out_k, out_v = self.q_proj(x, sparse_q_indices), self.k_proj(x, sparse_k_indices), self.v_proj(x, sparse_v_indices)
            
        else: #* Split
            sparse_x, dense_x = self.token_splits(x, masks)
            #* Output Sparsity:
            if self.dense_to_sparse_ratio is not None:
                raise ValueError("Dense to Sparse Ratio is not supported in Llama Attention")
                #* Dense to Sparse Ratio:               
                sparse_q_indices, dense_q_indices = sparse_q_indices
                sparse_k_indices, dense_k_indices = sparse_k_indices
                sparse_v_indices, dense_v_indices = sparse_v_indices
            else:
                dense_q_indices, dense_k_indices, dense_v_indices = None, None, None
            # print("Sparse Train, ", dense_q_indices.shape, sparse_q_indices.shape)
            
            #* Dense Projections:
            dense_q, dense_k, dense_v = self.q_proj(dense_x, dense_q_indices), self.k_proj(dense_x, dense_k_indices), self.v_proj(dense_x, dense_v_indices)
            sparse_q, sparse_k, sparse_v = self.q_proj(sparse_x, sparse_q_indices), self.k_proj(sparse_x, sparse_k_indices), self.v_proj(sparse_x, sparse_v_indices) 

                #* Let's log the sparsity:
            self.stats["sparsity/q"] = 1-self.q_proj.sparsity
            self.stats["sparsity/k"] = 1-self.k_proj.sparsity
            self.stats["sparsity/v"] = 1-self.v_proj.sparsity
            
            # #* Token Order: [Sparse | Dense] --> [In | Out]
            out_q = self.token_join(sparse=sparse_q, dense=dense_q, masks=masks)
            out_k = self.token_join(sparse=sparse_k, dense=dense_k, masks=masks)
            out_v = self.token_join(sparse=sparse_v, dense=dense_v, masks=masks)
            
        return out_q, out_k, out_v
    
    def pred_attn(self, x, q=None, k=None, v=None, sparsity=None) -> torch.Tensor:
            
        with torch.no_grad():
            if self.mode == "svd":
                if self.pred.seq_avg:
                    # print("QKVO Seq Avg", x.shape)
                    x = x.mean(dim=1, keepdim=True)
                
                if self.pred.gqa:
                    x = x.view(-1, x.shape[-1])
                    x1 = x.expand(2, -1, -1)
                    if sparsity is not None:
                        return self.pred(x, x1, sparsity)
                    else:
                        return self.pred(x, x1, self.sparsity)
                else:
                    x_flat = x.view(1, -1, x.shape[-1]).expand(3, -1, -1)                
                    if sparsity is not None:
                        raise ValueError("Sparsity is not supported in SVD mode")
                        return self.pred(x_flat, sparsity)
                    else:
                        return self.pred(x_flat, self.sparsity)

            else: raise ValueError("Not implemented")

    def forward(
        self,
        hidden_states:       torch.Tensor,
        causal_mask:         Optional[BlockDiagonalCausalMask] = None,
        attention_mask:      Optional[torch.Tensor] = None,
        position_ids:        Optional[torch.LongTensor] = None,
        past_key_value:      Optional[Tuple[torch.Tensor]] = None,
        output_attentions:   bool = False,
        use_cache:           bool = False,
        padding_mask:        Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        masks:              Optional[torch.Tensor] = None,
        *args, **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        # Clear inference
        if hasattr(self, "paged_attention"):
            del self.paged_attention_K
            del self.paged_attention_V
            del self.paged_attention
            del self.temp_QA
            del self.temp_KV
            del self.RH_Q
            del self.attention
        pass

        SPARSE = True if self.enabled and self.sparsity > 0 and self.mode == "svd" else False
        
        bsz, q_len, _ = hidden_states.size()

        n_heads    = self.config.num_attention_heads
        n_groups   = self.num_key_value_groups
        n_kv_heads = self.config.num_key_value_heads
        head_dim   = self.head_dim
        assert(n_kv_heads * n_groups == n_heads)
        
        dense_x = hidden_states
        if SPARSE:
            indices  = self.pred_attn(hidden_states)
            sparse_x, dense_x = self.token_splits(hidden_states, masks)
            Q_sparse, K_sparse, V_sparse = apply_sparselora_qkv(self, sparse_x, inplace = True, sparse_indices = indices, sparse_lora_branch = self.sparse_lora_branch)
            vo_indices = indices[-1]
        # else:
            # raise ValueError("Sparse Attention is not supported in Llama Attention")
        # Q, K, V = self.apply_qkv(self, dense_x)
        Q, K, V = apply_sparselora_qkv(self, dense_x)
        if SPARSE:
            # #* Token Order: [Sparse | Dense] --> [In | Out]
            Q = self.token_join(sparse=Q_sparse, dense=Q, masks=masks)
            K = self.token_join(sparse=K_sparse, dense=K, masks=masks)
            V = self.token_join(sparse=V_sparse, dense=V, masks=masks)
        
        
        Q = Q.view(bsz, q_len, n_heads,    head_dim).transpose(1, 2)
        K = K.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
        V = V.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

        kv_seq_len = K.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if position_embeddings:
            cos, sin = position_embeddings
        else:
            # Extend RoPE dynamically to fit in VRA
            rotary_emb = self.rotary_emb
            rotary_emb.extend_rope_embedding(V, seq_len = kv_seq_len)

            if position_ids is None:
                # Useful for LongRoPE
                cos, sin = rotary_emb.get_cached(kv_seq_len)
            else:
                cos, sin = rotary_emb(V, seq_len = kv_seq_len)

        # Q, K = (
        #     fast_rope_embedding(Q, K, cos, sin)
        #     if position_ids is None
        #     else inplace_rope_embedding(Q, K, cos, sin, position_ids)
        # )
        Q, K = fast_rope_embedding(Q, K, cos, sin)

        if past_key_value is not None:
            K = torch.cat([past_key_value[0], K], dim = 2)
            V = torch.cat([past_key_value[1], V], dim = 2)
        pass
        past_key_value = (K, V) if use_cache else None

        # Attention module
        if (not HAS_FLASH_ATTENTION and HAS_XFORMERS and attention_mask is None):
            # Xformers memory efficient attention
            # Also has Flash Attention v2 dispatching
            Q = Q.transpose(1, 2)
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)

            # Group query attention
            if n_groups != 1:
                K = K  .view(bsz, kv_seq_len, n_kv_heads,        1, head_dim)
                V = V  .view(bsz, kv_seq_len, n_kv_heads,        1, head_dim)
                K = K.expand(bsz, kv_seq_len, n_kv_heads, n_groups, head_dim)
                V = V.expand(bsz, kv_seq_len, n_kv_heads, n_groups, head_dim)
                if hidden_states.requires_grad:
                    K = K.reshape(bsz, kv_seq_len, n_heads, head_dim)
                    V = V.reshape(bsz, kv_seq_len, n_heads, head_dim)
                else:
                    Q = Q.view(bsz, q_len, n_kv_heads, n_groups, head_dim)
            pass
            A = xformers_attention(Q, K, V, attn_bias = causal_mask)
            A = A.view(bsz, q_len, n_heads, head_dim)

        elif HAS_FLASH_ATTENTION and attention_mask is None:
            Q = Q.transpose(1, 2)
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)
            A = flash_attn_func(Q, K, V, causal = True)
        else:
            # Grouped query attention
            if SDPA_HAS_GQA:
                # Needs (batch_size, n_heads, seq_len, head_dim)
                # is_casual and attention_mask must not be both set!
                A = scaled_dot_product_attention(Q, K, V, attn_mask = attention_mask, is_causal = False, enable_gqa = n_groups != 1)
                # Go back to (batch_size, seq_len, n_heads, head_dim)
                A = A.transpose(1, 2)#.contiguous()
            else:
                if n_groups != 1:
                    K = K[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, head_dim)
                    V = V[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, head_dim)
                    K = K.reshape(bsz, n_heads, kv_seq_len, head_dim)
                    V = V.reshape(bsz, n_heads, kv_seq_len, head_dim)
                pass
                # Must be contiguous or else results are False!
                # https://github.com/pytorch/pytorch/issues/112577
                Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
                # Needs (batch_size, n_heads, seq_len, head_dim)
                # is_casual and attention_mask must not be both set!
                A = scaled_dot_product_attention(Q, K, V, attn_mask = attention_mask, is_causal = False)
                # Go back to (batch_size, seq_len, n_heads, head_dim)
                A = A.transpose(1, 2).contiguous()
            pass
        pass
        attn_output = A.reshape(bsz, q_len, n_heads*head_dim)
        
        
        dense_x = attn_output
        if SPARSE:
            #* Apply the SVD predictor
            sparse_x, dense_x = self.token_splits(attn_output, masks)
            sparse_attn_output = self.o_proj(sparse_x, vo_indices)
        attn_output = self.apply_o(self, dense_x)
        attn_output = self.token_join(sparse=sparse_attn_output, dense=attn_output, masks=masks) if SPARSE else attn_output
        
        attn_weights = None
        return attn_output, attn_weights, past_key_value