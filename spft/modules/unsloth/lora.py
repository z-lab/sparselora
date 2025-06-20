import torch
from typing import Optional, Any
from unsloth.kernels.utils import fast_dequantize
from torch import nn
import torch.nn.functional as F

torch_matmul = torch.matmul

def sparse_torch_matmul(x: torch.Tensor, w: torch.Tensor, sparse_indices: Optional[torch.Tensor] = None, out: Optional[torch.Tensor] = None, sparse_lora_branch: bool = False, mode: Optional[str] = None) -> torch.Tensor:
    x = x.contiguous()
    
    if mode == "in_gather":
        sparse_indices_exp = sparse_indices.unsqueeze(0).unsqueeze(0).expand(x.shape[0],x.shape[1], -1)
        with torch.no_grad(): #* Might throw errors if LoRA branch is sparse!
            x = torch.gather(x, 2, sparse_indices_exp).contiguous()

    
    w_sparse = w[:, sparse_indices].contiguous() if mode.startswith("in") else w[sparse_indices].contiguous()
    x = F.linear(x, w_sparse)
        
    if mode == "out_scatter" and not sparse_lora_branch and x.shape[-1] != w.shape[0]:
        #* SparseLoRA QKV
        with torch.no_grad():
            out_shape = (x.shape[0], x.shape[1], w.shape[0])
            sparse_indices = sparse_indices.unsqueeze(0).unsqueeze(0).expand(x.shape[0],x.shape[1], -1)
            x = torch.zeros(out_shape, dtype=x.dtype, device=x.device).scatter_add(2, sparse_indices, x).contiguous()
    
    return x.contiguous()

def sparse_torch_matmul_bwd(x: torch.Tensor, w: torch.Tensor, sparse_indices: Optional[torch.Tensor] = None, out: Optional[torch.Tensor] = None, sparse_lora_branch: bool = False, mode: Optional[str] = None) -> torch.Tensor:
    x = x.contiguous()
    
    if mode == "in_gather":
        sparse_indices_exp = sparse_indices.unsqueeze(0).unsqueeze(0).expand(x.shape[0],x.shape[1], -1)
        with torch.no_grad(): #* Might throw errors if LoRA branch is sparse!
            x = torch.gather(x, 2, sparse_indices_exp).contiguous()

    #! Swap Weight Handle
    w_sparse = w[sparse_indices].contiguous() if mode.startswith("in") else w[:, sparse_indices].contiguous()
    w_sparse = w_sparse.t().contiguous()
    x = F.linear(x, w_sparse)
        
    if mode == "out_scatter" and not sparse_lora_branch and x.shape[-1] != w.shape[0]:
        #* SparseLoRA QKV
        with torch.no_grad():
            out_shape = (x.shape[0], x.shape[1], w.shape[0])
            sparse_indices = sparse_indices.unsqueeze(0).unsqueeze(0).expand(x.shape[0],x.shape[1], -1)
            x = torch.zeros(out_shape, dtype=x.dtype, device=x.device).scatter_add(2, sparse_indices, x).contiguous()
    
    return x.contiguous()



def sparse_matmul_lora(X, W, W_quant, A, B, s, out = None, \
                       sparse_indices: Optional[torch.Tensor] = None, bwd: bool = False,
                       sparse_lora_branch: bool = False, mode: Optional[str] = None) -> torch.Tensor:
    dtype = X.dtype
    W = fast_dequantize(W, W_quant, use_global_buffer = True)
        
    batch, seq_len, d = X.shape
    if not bwd:
        out = sparse_torch_matmul(X, W, sparse_indices = sparse_indices, \
                                out = out, sparse_lora_branch = sparse_lora_branch, \
                                mode = mode).flatten(0, 1)
    else:
        out = sparse_torch_matmul_bwd(X, W, sparse_indices = sparse_indices, \
                                out = out, sparse_lora_branch = sparse_lora_branch, \
                                mode = mode).flatten(0, 1)
    X = X.view(-1, X.shape[-1])
        
    if W_quant is not None: del W

    if A is not None:
        # LoRA is enabled
        A, B = A.t(), B.t()
        
        if not bwd and sparse_lora_branch and sparse_indices is not None and mode.startswith("in"): #! Reversed Shapes for LoRA weight addmm_
            with torch.no_grad():
                A = A[sparse_indices.clone()].contiguous()
            
        XA = torch_matmul(X, A.to(dtype))
        
        if not bwd and sparse_lora_branch and sparse_indices is not None and not mode.startswith("in"): #! Reversed Shapes for LoRA weight addmm_
            with torch.no_grad():
                B = B[:, sparse_indices.clone()].contiguous()
            
        out.addmm_(XA, B.to(dtype), alpha = s)
    
    return out.view(batch, seq_len, -1).contiguous()
pass
