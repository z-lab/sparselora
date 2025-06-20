import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

import torch._dynamo                                                                                                                                                                                 
torch._dynamo.config.suppress_errors = True  

silu_mul = LigerSiLUMulFunction.apply


class QKPredictor(nn.Module):
    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.register_buffer("w1", torch.randn(2, hidden_size, rank))
        self.register_buffer("w2", torch.randn(2, rank, hidden_size))

    @torch.inference_mode
    def forward(self, x: torch.Tensor, sparsity: float) -> torch.Tensor:
        x = x.view(1, -1, x.shape[-1]).expand(2, -1, -1)

        x = torch.bmm(x, self.w1)
        x = torch.bmm(x, self.w2)

        x = x.norm(dim=1)
        k = int(qk.shape[-1] * (1-sparsity))
        
        qk = x[0] * x[1]
        qk = qk.topk(k, dim=-1).indices.flatten()
        return qk


class VOPredictor(nn.Module):
    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.register_buffer("w1", torch.randn(rank, hidden_size))
        self.register_buffer("w2", torch.randn(hidden_size, rank))
    
    @torch.inference_mode
    def forward(self, x: torch.Tensor, sparsity: float) -> torch.Tensor:
        x = F.linear(x, self.w1)
        x = F.linear(x, self.w2)

        x = x.flatten(0, 1).norm(dim=0)

        k = int(x.shape[-1] * (1-sparsity))
        x = x.topk(k, dim=-1).indices.flatten()
        return x

class QKVOFusedPredictor(nn.Module):
    """QKVOFusedPredictor is a module that combines QK and VO prediction into a single forward pass.
    Not Compatible with GQA -- use GQA Version.
    """
    def __init__(self, hidden_size: int, kv_size: int, rank: int, seq_avg: bool = False) -> None:
        super().__init__()
        assert hidden_size == kv_size, "Hidden size and KV size must be the same"
        self.register_buffer("w1", torch.randn(3, hidden_size, rank))
        self.register_buffer("w2", torch.randn(3, rank, hidden_size))
        self.seq_avg = seq_avg
        self.gqa = False
       
    @torch.inference_mode 
    def forward(self, x: torch.Tensor, sparsity: float) -> torch.Tensor:

        x = torch.bmm(x, self.w1)
        x = torch.bmm(x, self.w2)
        
        x = x.norm(dim=1)

        qk = x[0] * x[1]
        k = int(qk.shape[-1] * (1-sparsity))
        
        qk = qk.topk(k, dim=-1).indices.flatten()
        
        v = x[2]
        v = v.topk(k, dim=-1).indices.flatten()
        return qk, qk, v
    
    
class QKVOFusedGQAPredictor(nn.Module):
    """QKVOFusedGQAPredictor is a module that combines QK and VO prediction into a single forward pass.
    Compatible with GQA.
    """
    def __init__(self, hidden_size: int, kv_size: int, rank: int, seq_avg: bool = False, per_head: bool = False) -> None:
        super().__init__()
        self.register_buffer("w1", torch.randn(2, hidden_size, rank))
        self.register_buffer("w2", torch.randn(2, rank, kv_size))
        
        self.register_buffer("q1", torch.randn(rank, hidden_size))
        self.register_buffer("q2", torch.randn(hidden_size, rank))
        
        self.seq_avg = seq_avg
        self.gqa = True
        self.per_head = per_head
        self.per_channel = not per_head
    
    @torch.inference_mode
    def forward(self, x: torch.Tensor, x1: torch.Tensor, sparsity: float) -> torch.Tensor:
        
        #* Compute Q
        q = F.linear(x, self.q1)
        q = F.linear(q, self.q2)
        q = q.norm(dim=0)
        
        #* Compute KV
        kv = torch.bmm(x1, self.w1)
        kv = torch.bmm(kv, self.w2)
        kv = kv.norm(dim=1)
        k,v = kv[0], kv[1]
        
        groups = q.shape[-1] // k.shape[-1]
        
        #* We either repeat k or mean q
        # k_exp, q = k.repeat_interleave(groups), q
        k_exp, q = k, q.view(groups, k.shape[-1]).mean(dim=0)
        
        #* Compute QK
        qk = q * k_exp
        tk = int(qk.shape[-1] * (1-sparsity))
        
        if self.per_channel:
            k, v = qk.topk(tk, dim=-1).indices, v.topk(tk, dim=-1).indices
            
            #* Obtain original co-responding q-indices
            #* qk is shape [k], q is shape [groups, k]
            q = k.unsqueeze(1) * groups + torch.arange(groups, device=k.device).unsqueeze(0)
            q, k, v = q.flatten(), k.flatten(), v.flatten()
        else:
            raise NotImplementedError("Per Head not implemented")  
        
        return q, k, v
    
    

class FFNPredictor(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, rank: int, seq_avg: bool = False) -> None:
        super().__init__()
        self.register_buffer("w1", torch.randn(2, hidden_size, rank))
        self.register_buffer("w2", torch.randn(2, rank, intermediate_size))
        self.seq_avg = seq_avg
    
    @torch.inference_mode
    def forward(self, x: torch.Tensor, sparsity: float) -> torch.Tensor:
        if self.seq_avg:
            x = x.mean(dim=1, keepdim=True)
            # print("FFN Seq Avg", x.shape)
        
        x = x.view(1, -1, x.shape[-1]).expand(2, -1, -1)

        x = torch.bmm(x, self.w1)
        x = torch.bmm(x, self.w2)
        
        x = silu_mul(x[0], x[1])
        x = x.norm(dim=0)

        k = int(x.shape[-1] * (1-sparsity))
        x = x.topk(k, dim=-1).indices.flatten()
        
        return x


def cuda_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def benchmark(model, x, num_repeats: int = 10000, num_warmup: int = 10000):
    device = torch.device("cuda")
    dtype = torch.float16

    model = torch.compile(model)

    model = model.to(device=device, dtype=dtype)
    x = x.to(device=device, dtype=dtype)

    for _ in range(num_warmup):
        _ = model(x)
    
    start = cuda_time()
    for _ in range(num_repeats):
        _ = model(x)
    end = cuda_time()
    return (end - start) / num_repeats * 1000


@torch.inference_mode
def main() -> None:
    hidden_size = 4096
    intermediate_size = 11008
    rank = 8

    x = torch.randn(8, 512, hidden_size)
    total = 0

    model = QKPredictor(hidden_size=hidden_size, rank=rank)
    time = benchmark(model, x)
    print("qk:", time)
    total += time

    model = VOPredictor(hidden_size=hidden_size, rank=rank)
    time = benchmark(model, x)
    print("vo:", time)
    total += time

    model = FFNPredictor(hidden_size=hidden_size, intermediate_size=intermediate_size, rank=rank)
    time = benchmark(model, x)
    print("ffn:", time)
    total += time

    print("total:", total)


if __name__ == "__main__":
    main()