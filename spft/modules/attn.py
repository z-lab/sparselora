from typing import Optional, Tuple

import torch
from torch import nn
from transformers.cache_utils import Cache, StaticCache

from transformers.modeling_flash_attention_utils import _flash_attention_forward

from .base import SparseModule
from liger_kernel.transformers import liger_rotary_pos_emb 

__all__ = ["SparseLlamaFlashAttention"]


class SparseLlamaFlashAttention(SparseModule):
    inherited_attributes = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "rotary_emb",
        "hidden_size",
        "head_dim",
        "num_heads",
        "num_key_value_heads",
        "attention_dropout",
        "is_causal",
        "_flash_attn_uses_top_left_mask",
    ]

    def __init__(self, base: nn.Module, *, name: str, idx: int, sparsity: float = 0, cfg) -> None:
        super().__init__(base)
        self.sparsity = sparsity
        self.layer_name = name
        self.layer_idx = idx #base.layer_idx
        if self.sparsity > 0:
            self.load_predictor(base,cfg)
        
        self.per_channel = not cfg.qk_per_head
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
            # print(f"sparse_q: {sparse_q.shape}, dense_q: {dense_q.shape}, sparse_k: {sparse_k.shape}, dense_k: {dense_k.shape}, sparse_v: {sparse_v.shape}, dense_v: {dense_v.shape}")
            
            #* Let's log the sparsity:
            self.stats["sparsity/q"] = 1-self.q_proj.sparsity
            self.stats["sparsity/k"] = 1-self.k_proj.sparsity
            self.stats["sparsity/v"] = 1-self.v_proj.sparsity
            
            # #* Token Order: [Sparse | Dense] --> [In | Out]
            out_q = self.token_join(sparse=sparse_q, dense=dense_q, masks=masks)
            out_k = self.token_join(sparse=sparse_k, dense=dense_k, masks=masks)
            out_v = self.token_join(sparse=sparse_v, dense=dense_v, masks=masks)
            
            
        
           
            
        return out_q, out_k, out_v
        
    def mask_proj_o_forward(self, attn_output, masks, vo_indices):
        if vo_indices is None:
            return self.o_proj(attn_output)

        if masks is None: #* No Split
            binary_mask = torch.zeros_like(attn_output)
            binary_mask[:, :, vo_indices] = 1
            attn_output = attn_output * binary_mask
            
        else:
            id_split = self.split_idx(masks)
            binary_mask = torch.zeros_like(attn_output[:, :id_split, :])
            binary_mask[:, :, vo_indices] = 1
            attn_output[:, :id_split, :] *= binary_mask
            
        return self.o_proj(attn_output)
    
    def mask_proj_forward(self, query_states, key_states, value_states, masks, indices):
        q_indices, k_indices, v_indices = indices
        
        if masks is None: #* No Split
            binary_mask = torch.zeros_like(query_states)
            binary_mask[:, :, q_indices] = 1
            query_states = query_states * binary_mask
            
    
            binary_mask = torch.zeros_like(key_states)
            binary_mask[:, :, k_indices] = 1
            key_states = key_states * binary_mask
            
            
            binary_mask = torch.zeros_like(value_states)
            binary_mask[:, :, v_indices] = 1
            value_states = value_states * binary_mask
            
        else:
            id_split = self.split_idx(masks)
            
            if q_indices is not None:
                binary_mask = torch.zeros_like(query_states[:, :id_split, :])
                binary_mask[:, :, q_indices] = 1
                query_states[:, :id_split, :] *= binary_mask
                
                
                binary_mask = torch.zeros_like(key_states[:, :id_split, :])
                binary_mask[:, :, k_indices] = 1
                key_states[:, :id_split, :] *= binary_mask
            
                self.stats["sparsity/q"] = torch.count_nonzero(query_states[:, :id_split, :]) / query_states[:, :id_split, :].numel()
                self.stats["sparsity/k"] = torch.count_nonzero(key_states[:, :id_split, :]) / key_states[:, :id_split, :].numel()

                
            if v_indices is not None:            
                binary_mask = torch.zeros_like(value_states[:, :id_split, :])
                binary_mask[:, :, v_indices] = 1
                value_states[:, :id_split, :] *= binary_mask
            
                self.stats["sparsity/v"] = torch.count_nonzero(value_states[:, :id_split, :]) / value_states[:, :id_split, :].numel()
                
        return query_states, key_states, value_states
    
    def pred_attn(self, x, q=None, k=None, v=None, sparsity=None) -> torch.Tensor:
            
        with torch.no_grad():
            if self.mode == "svd":
                if self.pred.seq_avg:
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
            
            elif "oracle" in self.mode:
                
                if "svd" in self.mode:
                    if self.pred.seq_avg:
                        x = x.mean(dim=1, keepdim=True)
                    
                    if self.pred.gqa:
                        x = x.view(-1, x.shape[-1])
                        x1 = x.expand(2, -1, -1)
                        return self.pred(x, x1, self.sparsity)
                    else:
                        x_flat = x.view(1, -1, x.shape[-1]).expand(3, -1, -1)                
                        return self.pred(x_flat, self.sparsity)
                
                if "attention_norm" in self.mode:
                    q = q.flatten(0, 1).norm(dim=0)
                    k = k.flatten(0, 1).norm(dim=0)
                    
                    groups = q.shape[-1] // k.shape[-1]
                    k_exp, q = k, q.view(groups, k.shape[-1]).mean(dim=0)
                    
                    qk = q * k_exp
                    tk = int(qk.shape[-1] * (1-self.sparsity))

                    v = v.flatten(0, 1).norm(dim=0)
                    
                    assert qk.shape[-1] == v.shape[-1], "QK and V should have the same shape [even in gqa]"

                    if self.per_channel:
                        k, v = qk.topk(tk, dim=-1).indices, v.topk(tk, dim=-1).indices
            
                        #* Obtain original co-responding q-indices
                        #* qk is shape [k], q is shape [groups, k]
                        q = k.unsqueeze(1) * groups + torch.arange(groups, device=k.device).unsqueeze(0)
                        q_index, k_index, v_index = q.flatten(), k.flatten(), v.flatten()
                    else:
                        #* Implement topk per head and then map indieces back to full-dim selection:
                        qk_head = qk.view(self.num_key_value_heads, -1)
                        v_head = v.view(self.num_key_value_heads, -1)
                        tk = int(qk_head.shape[-1] * (1-self.sparsity))
                        
                        #* Head_based Indices:
                        qk_head_indices = qk_head.topk(tk, dim=-1).indices
                        v_head_indices = v_head.topk(tk, dim=-1).indices
                        
                        #* Map to full-dim indices:
                        offset = torch.arange(self.num_key_value_heads, device=qk.device).unsqueeze(1) * qk_head.shape[-1]  # [num_heads, 1]
                        k_index = (qk_head_indices + offset).flatten()  # [num_heads * tk]
                        v_index = (v_head_indices + offset).flatten()   # [num_heads * tk]
                        
                        # q is still [groups, k.shape[-1]] like in the per-channel version (for GQA)
                        # For consistency with per-channel indexing, generate q_index accordingly:
                        # Each selected index in k corresponds to multiple queries (1 per group)
                        q_index = k_index.unsqueeze(1) * groups + torch.arange(groups, device=k_index.device).unsqueeze(0)  # [num_heads * tk, groups]
                        q_index = q_index.flatten()
                        
                elif "random" in self.mode:
                    tk = int(q.shape[-1] * (1-self.sparsity))
                    q = torch.randn(q.shape[-1], device=x.device)
                    q_index = torch.topk(q, tk, dim=-1).indices.flatten()
                    
                    tk = int(k.shape[-1] * (1-self.sparsity))
                    k = torch.randn(k.shape[-1], device=x.device)
                    k_index = torch.topk(k, tk, dim=-1).indices.flatten()
                    
                    v = torch.randn(v.shape[-1], device=x.device)
                    v_index = torch.topk(v, tk, dim=-1).indices.flatten()
                    
                    
                elif "l2norm" in self.mode:     
                    tk = int(q.shape[-1] * (1-self.sparsity))
                    q = q.flatten(0, 1).norm(dim=0)
                    q_index = torch.topk(q, tk, dim=-1).indices.flatten()
                    
                    tk = int(k.shape[-1] * (1-self.sparsity))
                    k = k.flatten(0, 1).norm(dim=0)
                    k_index = torch.topk(k, tk, dim=-1).indices.flatten()
                    
                    v = v.flatten(0, 1).norm(dim=0)
                    v_index = torch.topk(v, tk, dim=-1).indices.flatten()
                else:
                    raise ValueError("Invalid oracle mode {}".format(self.mode))
                
                
                if "qk" in self.mode:
                    return q_index, k_index, None
                elif "vo" in self.mode:
                    return None, None, v_index
                else:
                    return q_index, k_index, v_index

            else: raise ValueError("Not implemented")

        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )
        
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()
        

        if self.enabled and self.sparsity > 0:
            
            if self.mode == "svd":
                
                indices  = self.pred_attn(hidden_states)
                if self.dense_to_sparse_ratio is not None:
                    raise ValueError("Dense to Sparse Ratio is not supported in Llama Attention")
                    dense_indices  = self.pred_attn(hidden_states, sparsity=self.dense_to_sparse_ratio*self.sparsity)
                    
                    indices = (
                        (indices[0], dense_indices[0]),
                        (indices[1], dense_indices[1]),
                        (indices[2], dense_indices[2]),
                    )
                query_states, key_states, value_states = self.kernel_proj_forward(hidden_states, masks, indices)
                
                
            elif "oracle" in self.mode: 
                query_states, key_states, value_states = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
                with torch.no_grad():
                    indices  = self.pred_attn(hidden_states, q=query_states, k=key_states, v=value_states) 
                
                query_states, key_states, value_states = self.mask_proj_forward(query_states, key_states, value_states, masks, indices)
        
            #* Assign for latter:
            vo_indices = indices[2]
            
        else:
            
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            
            
        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, -1).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, -1).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, -1).transpose(1, 2)

        cos, sin = position_embeddings
    
        query_states, key_states = liger_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)
        
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=self.attention_dropout if self.training else 0.0,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
            
            
        if self.enabled and self.sparsity > 0 and self.mode == "svd":
            
            if self.mode == "svd":
                attn_output = self.kernel_proj_o_forward(attn_output, masks, vo_indices)
                
            elif "oracle" in self.mode: 
                attn_output = self.mask_proj_o_forward(attn_output, masks, vo_indices)
            
        else:
            attn_output = self.o_proj(attn_output)
        
            
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
