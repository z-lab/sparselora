from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseLinear(nn.Linear):
    def __init__(self, base: nn.Module, mode: str) -> None:
        super().__init__(base.in_features, base.out_features, bias=base.bias is not None)
        self.weight = base.weight
        self.mode = mode

    def forward(self, x: torch.Tensor, sparse_indices: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
        x = x.contiguous()
        if sparse_indices is None or self.mode is None:
            return F.linear(x, self.weight)

        if self.mode == "in_gather":
            idx = sparse_indices.unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], -1)
            with torch.no_grad():
                x = torch.gather(x, 2, idx).contiguous()

        if self.mode.startswith("in"):
            w = self.weight[:, sparse_indices].contiguous()
        else:
            w = self.weight[sparse_indices].contiguous()
        x = F.linear(x, w)

        if self.mode == "out_scatter" and x.shape[-1] != self.weight.shape[0]:
            with torch.no_grad():
                out = torch.zeros(
                    x.shape[0],
                    x.shape[1],
                    self.weight.shape[0],
                    dtype=x.dtype,
                    device=x.device,
                )
                idx = sparse_indices.unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], -1)
                x = out.scatter_add(2, idx, x).contiguous()

        return x


def lora_forward(
    self, x: torch.Tensor, indices: Optional[torch.Tensor] = None, *args: Any, **kwargs: Any
) -> torch.Tensor:
    """Patched LoRA Linear forward that passes sparse indices through."""
    self._check_forward_args(x, *args, **kwargs)

    result = self.base_layer(x, indices, *args, **kwargs)
    result_dtype = result.dtype

    for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A.keys():
            continue

        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        scaling = self.scaling[active_adapter]

        lora_dropout = getattr(self, "lora_dropout", None)
        if lora_dropout is not None and active_adapter in lora_dropout:
            x_in = lora_dropout[active_adapter](x)
        else:
            x_in = x

        x_in = x_in.to(lora_A.weight.dtype)
        lora_out = lora_B(lora_A(x_in, indices, *args, **kwargs), indices, *args, **kwargs)
        lora_out = lora_out.to(result_dtype)
        result = torch.add(result, lora_out, alpha=scaling)

    return result
