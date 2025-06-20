from typing import Optional, Any

import torch
import torch.nn as nn
import bitsandbytes.functional as F
import bitsandbytes as bnb
from math import prod

class MatMul4Bit(torch.autograd.Function):
    # forward is the same, but we added the fallback for pre-turing GPUs
    # backward is mostly the same, but adds one extra clause (see "elif state.CxB is not None")

    @staticmethod
    def forward(ctx, A, B, out=None, bias=None, quant_state: Optional[F.QuantState] = None, sparse_indices: Optional[torch.Tensor] = None, mode: Optional[str] = None):
        # default of pytorch behavior if inputs are empty
        ctx.is_empty = False
        if prod(A.shape) == 0:
            ctx.is_empty = True
            ctx.A = A
            ctx.B = B
            ctx.bias = bias
            B_shape = quant_state.shape
            if A.shape[-1] == B_shape[0]:
                return torch.empty(A.shape[:-1] + B_shape[1:], dtype=A.dtype, device=A.device)
            else:
                return torch.empty(A.shape[:-1] + B_shape[:1], dtype=A.dtype, device=A.device)

        # 1. Dequantize
        # 2. MatmulnN
        w = F.dequantize_4bit(B, quant_state).to(A.dtype).t()
        ctx.sparse_indices = sparse_indices
        ctx.mode = mode
        if sparse_indices is not None and mode is not None:
            w = w[:, sparse_indices].contiguous() if mode.startswith("in") else w[sparse_indices].contiguous()
        output = torch.nn.functional.linear(A, w, bias)

        # 3. Save state
        ctx.state = quant_state
        ctx.dtype_A, ctx.dtype_B, ctx.dtype_bias = A.dtype, B.dtype, None if bias is None else bias.dtype

        if any(ctx.needs_input_grad[:2]):
            ctx.tensors = (None, B)
        else:
            ctx.tensors = (None, None)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_empty:
            bias_grad = None if ctx.bias is None else torch.zeros_like(ctx.bias)
            return torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, bias_grad, None

        req_gradA, _, _, req_gradBias, _, _, _ = ctx.needs_input_grad
        _, B = ctx.tensors

        grad_A, grad_B, grad_bias = None, None, None

        if req_gradBias:
            # compute grad_bias first before changing grad_output dtype
            grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)

        # not supported by PyTorch. TODO: create work-around
        # if req_gradB: grad_B = torch.matmul(grad_output.t(), A)
        if req_gradA:
            w = F.dequantize_4bit(B, ctx.state).to(grad_output.dtype).t()
            if ctx.sparse_indices is not None and ctx.mode is not None:
                w = w[:, ctx.sparse_indices].contiguous() if ctx.mode.startswith("in") else w[ctx.sparse_indices].contiguous()
            grad_A = torch.matmul(grad_output, w)

        return grad_A, grad_B, None, grad_bias, None, None, None
    

def matmul_4bit(
    A: torch.Tensor,
    B: torch.Tensor,
    quant_state: F.QuantState,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    sparse_indices: Optional[torch.Tensor] = None,
    mode: Optional[str] = None,
):
    assert quant_state is not None

    if A.numel() == A.shape[-1] and A.requires_grad == False:
        if A.shape[-1] % quant_state.blocksize != 0:
            # warn(
            #     f"Some matrices hidden dimension is not a multiple of {quant_state.blocksize} and efficient inference kernels are not supported for these (slow). Matrix input size found: {A.shape}",
            # )
            return MatMul4Bit.apply(A, B, out, bias, quant_state)
        else:
            out = F.gemv_4bit(A, B.t(), out, state=quant_state)
            if bias is not None:
                out += bias
            return out
    else:
        return MatMul4Bit.apply(A, B, out, bias, quant_state, sparse_indices,mode)
    
    
    
class SparseLinear4bit(bnb.nn.modules.Linear4bit):
    def __init__(self, base: nn.Module, mode: str, config: Any) -> None:
        super(SparseLinear4bit, self).__init__(
            base.in_features, 
            base.out_features, 
            bias=True if base.bias is not None else False)  
        self.weight = base.weight
        self.mode = mode
        self.sparse_lora_branch = config.sparse_lora_branch
       
    
    def forward(self, x: torch.Tensor, sparse_indices: Optional[torch.Tensor] = None, *args: Any, **kwargs: Any) -> torch.Tensor:
        bnb.nn.modules.fix_4bit_weight_quant_state_from_module(self)
        x = x.contiguous()
        
        if not self.compute_type_is_set:
            self.set_compute_type(x)
            self.compute_type_is_set = True
            
        inp_dtype = x.dtype
        bias = None
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)
            
        if sparse_indices is None or self.mode is None: #dense
            return matmul_4bit(x, self.weight.data.t(), bias=bias, quant_state=self.weight.quant_state).to(inp_dtype)
            # return F.linear(x, self.weight)
        
        
        
        if self.mode == "in_gather":
            sparse_indices_exp = sparse_indices.unsqueeze(0).unsqueeze(0).expand(x.shape[0],x.shape[1], -1)
            with torch.no_grad(): #* Might throw errors if LoRA branch is sparse!
                x = torch.gather(x, 2, sparse_indices_exp).contiguous()
                          
        x = matmul_4bit(x, self.weight.data.t(), bias=bias, quant_state=self.weight.quant_state, sparse_indices=sparse_indices, mode = self.mode).to(inp_dtype)
        
        if self.mode == "out_scatter" and not self.sparse_lora_branch and x.shape[-1] != self.weight.quant_state.shape[0]:
            #* SparseLoRA QKV
            with torch.no_grad():
                out_shape = (x.shape[0], x.shape[1], self.weight.quant_state.shape[0])
                sparse_indices = sparse_indices.unsqueeze(0).unsqueeze(0).expand(x.shape[0],x.shape[1], -1)
                x = torch.zeros(out_shape, dtype=x.dtype, device=x.device).scatter_add(2, sparse_indices, x).contiguous()
        
        return x


#* Just patch-ing the forward to accept/pass indices

def lora4bit_forward(self, x: torch.Tensor, indices: Optional[torch.Tensor] = None, output_mask: Optional[torch.Tensor] = None, *args: Any, **kwargs: Any) -> torch.Tensor:
    self._check_forward_args(x, *args, **kwargs)

    result = self.base_layer(x, *args, **kwargs)
    # As per Tim Dettmers, for 4bit, we need to defensively clone here.
    # The reason is that in some cases, an error can occur that backprop
    # does not work on a manipulated view. This issue may be solved with
    # newer PyTorch versions but this would need extensive testing to be
    # sure.
    result = result.clone()
    
    if indices is not None and not self.sparse_lora_branch:
        #* Scattered back!
        self.sparsity =  torch.count_nonzero(result) / result.numel()
        

    for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A.keys():
            continue
        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]

        requires_conversion = not torch.is_autocast_enabled()
        if requires_conversion:
            expected_dtype = result.dtype
            x = self._cast_input_dtype(x, lora_A.weight.dtype)

        output = lora_B(lora_A(dropout(x), indices, *args, **kwargs), indices, *args, **kwargs)
        if requires_conversion:
            output = output.to(expected_dtype)
        result = torch.add(result, output, alpha=scaling)
        
        if indices is not None and self.base_layer.sparse_lora_branch:
            #* Scattered back!
            self.sparsity =  torch.count_nonzero(result) / result.numel()
            
            if lora_B.mode == "out_scatter" : #* Support for Sparse LoRA Branch
                out_shape = (x.shape[0], x.shape[1], self.base_layer.weight.shape[0])
                indices = indices.unsqueeze(0).unsqueeze(0).expand(x.shape[0],x.shape[1], -1)
                result = torch.zeros(out_shape, dtype=x.dtype, device=x.device).scatter_add(2, indices, result).contiguous()
                

    return result