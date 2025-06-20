from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseLinear(nn.Linear):
    def __init__(self, base: nn.Module, mode: str, config: Any) -> None:
        super(SparseLinear, self).__init__(
            base.in_features, 
            base.out_features, 
            bias=True if base.bias is not None else False)   
        self.weight = base.weight
        self.mode = mode
        self.sparse_lora_branch = config.sparse_lora_branch
    
    def forward(self, x: torch.Tensor, sparse_indices: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
        x = x.contiguous()
        if sparse_indices is None or self.mode is None: #dense
            return F.linear(x, self.weight)
        
        
        
        if self.mode == "in_gather":
            sparse_indices_exp = sparse_indices.unsqueeze(0).unsqueeze(0).expand(x.shape[0],x.shape[1], -1)
            with torch.no_grad(): #* Might throw errors if LoRA branch is sparse!
                x = torch.gather(x, 2, sparse_indices_exp).contiguous()

        w = self.weight[:, sparse_indices].contiguous() if self.mode.startswith("in") else self.weight[sparse_indices].contiguous()
            
        x = F.linear(x, w)
        
        if self.mode == "out_scatter" and not self.sparse_lora_branch and x.shape[-1] != self.weight.shape[0]:
            #* SparseLoRA QKV
            with torch.no_grad():
                out_shape = (x.shape[0], x.shape[1], self.weight.shape[0])
                sparse_indices = sparse_indices.unsqueeze(0).unsqueeze(0).expand(x.shape[0],x.shape[1], -1)
                x = torch.zeros(out_shape, dtype=x.dtype, device=x.device).scatter_add(2, sparse_indices, x).contiguous()
        
        return x


#* Just patch-ing the forward to accept/pass indices
def lora_forward(self, x: torch.Tensor, indices: Optional[torch.Tensor] = None, *args: Any, **kwargs: Any) -> torch.Tensor:
    self._check_forward_args(x, *args, **kwargs)
    
    result = self.base_layer(x, indices)
    
    if indices is not None and not self.base_layer.sparse_lora_branch:
        #* Scattered back!
        self.sparsity =  torch.count_nonzero(result) / result.numel()
        
    
    torch_result_dtype = result.dtype
    for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A.keys():
            continue
        
        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        scaling = self.scaling[active_adapter]
        x = x.to(lora_A.weight.dtype)

        if not self.use_dora[active_adapter]: #* Implement Changes Here (pass *args/**kwargs)
            x = lora_A(x, indices, *args, **kwargs)
            x = lora_B(x, indices, *args, **kwargs)
            result = torch.add(result, x, alpha=scaling)
            
            if indices is not None and self.base_layer.sparse_lora_branch:
                #* Scattered back!
                self.sparsity =  torch.count_nonzero(result) / result.numel()
                
                if lora_B.mode == "out_scatter" : #* Support for Sparse LoRA Branch
                    out_shape = (x.shape[0], x.shape[1], self.base_layer.weight.shape[0])
                    indices = indices.unsqueeze(0).unsqueeze(0).expand(x.shape[0],x.shape[1], -1)
                    result = torch.zeros(out_shape, dtype=x.dtype, device=x.device).scatter_add(2, indices, result).contiguous()
                    
        else:
            result = result + self.lora_magnitude_vector[active_adapter](
                x,
                lora_A=lora_A,
                lora_B=lora_B,
                scaling=scaling,
                base_layer=self.get_base_layer(),
            )

        result = result.to(torch_result_dtype)

    return result


