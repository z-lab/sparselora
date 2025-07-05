from typing import Any, Optional

import torch
from torch import nn

from unsloth.kernels.utils import get_lora_parameters, matmul_lora, fast_dequantize, torch_amp_custom_fwd, torch_amp_custom_bwd
from unsloth.kernels.fast_lora import LoRA_MLP, swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel

from spft.modules import SparseModule
from spft.modules.unsloth.lora import sparse_matmul_lora

__all__ = ["UnslothSparseLlamaMLP"]


class SparseLoRA_MLP(torch.autograd.Function):
    """
    ### LoRA weights
    G = G + Ag @ Bg
    U = U + Au @ Bu
    W = W + Aw @ Bw

    ### SwiGLU(X)
    e = X @ G
    f = e * sigmoid(e)
    g = X @ U
    h = f * g
    i = h @ W

    ### Backpropagation chain rule
    See our blog post for more details

    df = sigmoid(e) * (1 - f) + f
    dC/dW = h.T @ dY
    dC/dU = X.T @ (D @ W.T * f)
    dC/dG = X.T @ (D @ W.T * df * g)

    ### Down projection LoRA weights
    dC/dAw = dC/dW @ B.T
    dC/dBw = A.T @ dC/dW
    dC/dAw =       h.T @ dY @ B.T
    dC/dBw = A.T @ h.T @ dY

    ### Up projection LoRA weights
    dC/dAu =       X.T @ (D @ W.T * f) @ B.T
    dC/dBu = A.T @ X.T @ (D @ W.T * f)

    ### Gate projection LoRA weights
    dC/dAg =       X.T @ (D @ W.T * df * g) @ B.T
    dC/dBg = A.T @ X.T @ (D @ W.T * df * g)

    Don't forget to see our blog post for more details!
    """
    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, X : torch.Tensor,
                gateW, gateW_quant, gateA, gateB, gateS,
                  upW,   upW_quant, upA,   upB,   upS,
                downW, downW_quant, downA, downB, downS,
                _forward_function, _backward_function,
                inplace = True, sparse_indices: Optional[torch.Tensor] = None, sparse_lora_branch: bool = False) -> torch.Tensor:
        dtype = X.dtype

        ctx.sparse_indices = sparse_indices
        ctx.sparse_lora_branch = sparse_lora_branch
        if sparse_indices is not None:
            e = sparse_matmul_lora(X, gateW, gateW_quant, gateA, gateB, gateS, sparse_indices = sparse_indices, sparse_lora_branch = sparse_lora_branch, mode = "out_scatter")
            g = sparse_matmul_lora(X,   upW,   upW_quant,   upA,   upB,   upS, sparse_indices = sparse_indices, sparse_lora_branch = sparse_lora_branch, mode = "out_scatter")
            h = _forward_function(e, g)
            i = sparse_matmul_lora(h, downW, downW_quant, downA, downB, downS, sparse_indices = sparse_indices, sparse_lora_branch = sparse_lora_branch, mode = "in_gather")
            
        else:
            e = matmul_lora(X, gateW, gateW_quant, gateA, gateB, gateS)
            g = matmul_lora(X,   upW,   upW_quant,   upA,   upB,   upS)
            h = _forward_function(e, g)
            i = matmul_lora(h, downW, downW_quant, downA, downB, downS)

        ctx.custom_saved_tensors = (
            gateW, gateW_quant, gateS,
            upW, upW_quant, upS,
            downW, downW_quant, downS,
            _backward_function,
        )
        ctx.save_for_backward(gateA, gateB, upA, upB, downA, downB,
                              X, e, g)
        ctx.inplace = inplace
        return i
    pass


    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dY : torch.Tensor):
        gateW, gateW_quant, gateS, upW, upW_quant, upS, downW, downW_quant, downS, \
            _backward_function = ctx.custom_saved_tensors
        gateA, gateB, upA, upB, downA, downB, \
            X, e, g = ctx.saved_tensors
            
        dY = dY.contiguous()
        
        # if ctx.sparse_indices is not None:
        #     gateW = gateW[ctx.sparse_indices].contiguous()
        #     # gateW_quant = gateW_quant[:, ctx.sparse_indices]
        #     upW = upW[ctx.sparse_indices].contiguous()
        #     # upW_quant = upW_quant[:, ctx.sparse_indices]
        #     downW = downW[:, ctx.sparse_indices].contiguous()
            # downW_quant = downW_quant[:, ctx.sparse_indices]

        batch, seq_len, hd = X.shape
        dY = dY.view(-1, dY.shape[-1])
        X  = X .view(-1, X .shape[-1])
        e  = e .view(-1, e .shape[-1])
        g  = g .view(-1, g .shape[-1])
        dtype = X.dtype

        gateA, gateB, upA, upB, downA, downB = \
            gateA.to(dtype), gateB.to(dtype), upA.to(dtype), upB.to(dtype), downA.to(dtype), downB.to(dtype)

        gateA, gateB, upA, upB, downA, downB = \
            gateA.t(), gateB.t(), upA.t(), upB.t(), downA.t(), downB.t()

        DW = matmul_lora(dY, downW.t(), downW_quant, downB, downA, downS)
            
        DW, e, g = _backward_function(DW, e, g)
        h, df, de = DW, e, g

        d_downA = torch.empty_like(downA)
        d_downB = torch.empty_like(downB)
        d_gateA = torch.empty_like(gateA)
        d_gateB = torch.empty_like(gateB)
        d_upA   = torch.empty_like(upA)
        d_upB   = torch.empty_like(upB)

        # Down projection LoRA weights
        # d_downA = h.t() @ (dY @ downB.t())
        # d_downB = (downA.t() @ h.t()) @ dY
        # d_downA *= downS
        # d_downB *= downS
        d_downA.addmm_(h.t(), dY @ downB.t(), alpha = downS, beta = 0)
        d_downB.addmm_(downA.t() @ h.t(), dY, alpha = downS, beta = 0)

        # Up projection LoRA weights
        # d_upA   = X.t() @ (df @ upB.t())
        # d_upB   = (upA.t() @ X.t()) @ df
        # d_upA  *= upS
        # d_upB  *= upS
        d_upA.addmm_(X.t(), df @ upB.t(), alpha = upS, beta = 0)
        d_upB.addmm_(upA.t() @ X.t(), df, alpha = upS, beta = 0)

        # Gate projection LoRA weights
        # d_gateA = X.t() @ (de @ gateB.t())
        # d_gateB = (gateA.t() @ X.t()) @ de
        # d_gateA *= gateS
        # d_gateB *= gateS
        d_gateA.addmm_(X.t(), de @ gateB.t(), alpha = gateS, beta = 0)
        d_gateB.addmm_(gateA.t() @ X.t(), de, alpha = gateS, beta = 0)

        # dX  = matmul_lora(df, upW.t(), upW_quant, upB, upA, upS)
        # dX += matmul_lora(de, gateW.t(), gateW_quant, gateB, gateA, gateS)
        
        if ctx.sparse_indices is not None:
            df_reshaped = df.view(batch, seq_len, df.shape[-1])
            de_reshaped = de.view(batch, seq_len, de.shape[-1])
            
            dX_up = sparse_matmul_lora(df_reshaped, upW, upW_quant, upB, upA, upS,
                                        sparse_indices=ctx.sparse_indices, bwd=True,
                                        sparse_lora_branch=ctx.sparse_lora_branch,
                                        mode="in_gather")

            dX_gate = sparse_matmul_lora(de_reshaped, gateW, gateW_quant, gateB, gateA, gateS,
                                        sparse_indices=ctx.sparse_indices, bwd=True,
                                        sparse_lora_branch=ctx.sparse_lora_branch,
                                        mode="in_gather")

            dX = dX_up + dX_gate
            dX = dX.view(-1, dX.shape[-1])

        else:

            upW = fast_dequantize(upW.t(), upW_quant)
            dX = torch.matmul(df, upW.t(), out = X if ctx.inplace else None)
            del upW
            # dX += df @ upB.to(dtype).t() @ (upS * upA.to(dtype).t())
            dX.addmm_(df @ upB.t(), upA.t(), alpha = upS)

            gateW = fast_dequantize(gateW.t(), gateW_quant)
            # dX += de @ gateW.t()
            dX.addmm_(de, gateW.t())
            del gateW
            # dX += de @ gateB.to(dtype).t() @ (gateS * gateA.to(dtype).t())
            dX.addmm_(de @ gateB.t(), gateA.t(), alpha = gateS)

        # gateW, gateW_quant, gateA, gateB, gateS,
        #  upW,    upW_quant,   upA,   upB,   upS,
        # downW, downW_quant, downA, downB, downS,
        return dX.view(batch, seq_len, hd), \
            None, None, d_gateA.t(), d_gateB.t(), None, \
            None, None,   d_upA.t(),   d_upB.t(), None, \
            None, None, d_downA.t(), d_downB.t(), None, \
            None, None, None, None, None, # _backward and _forward and inplace and sparse_indices and sparse_lora_branch
    pass
pass





class UnslothSparseLlamaMLP(SparseModule):
    inherited_attributes = [
        "gate_proj",
        "up_proj",
        "down_proj",
        "act_fn",
    ]

    def __init__(self, base: nn.Module, *, name: str, idx: int, sparsity: float, cfg) -> None:
        super().__init__(base)
        self.sparsity = sparsity
        self.layer_name = name
        self.layer_idx = idx
        if self.sparsity > 0:
            self.load_predictor(base,cfg)
            
        self.sparse_lora_branch = cfg.sparse_lora_branch
    
    def pred_mlp(self, x: torch.Tensor, sparsity= None) -> torch.Tensor:
        with torch.no_grad():
            if self.mode == "svd":
                if sparsity is not None:
                    return self.pred(x, sparsity)
                else:
                    return self.pred(x, self.sparsity)
            else: raise ValueError("Not implemented")

    
    def forward(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:      
        
        inplace = kwargs.get("inplace", True)
        SPARSE = True if self.enabled and self.sparsity > 0 and self.mode == "svd" else False
        
        gateW, gateW_quant, gateA, gateB, gateS = get_lora_parameters(self.gate_proj)
        upW,     upW_quant,   upA,   upB,   upS = get_lora_parameters(self.  up_proj)
        downW, downW_quant, downA, downB, downS = get_lora_parameters(self.down_proj)
        
        
        dense_x = x
        
        if SPARSE:
            indices = self.pred_mlp(x)
            sparse_x, dense_x = self.token_splits(x, masks)
            
            sparse_x = SparseLoRA_MLP.apply(sparse_x,
                        gateW, gateW_quant, gateA, gateB, gateS,
                        upW,     upW_quant, upA,   upB,   upS,
                        downW, downW_quant, downA, downB, downS,
                        swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel,
                        inplace, indices, self.sparse_lora_branch)         
            
        dense_x = SparseLoRA_MLP.apply(dense_x,
                        gateW, gateW_quant, gateA, gateB, gateS,
                        upW,     upW_quant, upA,   upB,   upS,
                        downW, downW_quant, downA, downB, downS,
                        swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel,
                        inplace)   
        
           
        out = self.token_join(sparse=sparse_x, dense=dense_x, masks=masks) if SPARSE else dense_x

        return out