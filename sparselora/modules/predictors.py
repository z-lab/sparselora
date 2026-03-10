"""SVD-based sparsity predictors.

Each predictor has a uniform ``predict(hidden_states, sparsity)`` interface:

- :class:`FFNPredictor` returns a 1-D index tensor of intermediate dims to keep.
- :class:`AttentionPredictor` / :class:`GQAAttentionPredictor` return
  ``(q_idx, k_idx, v_idx)`` tuples.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

_silu_mul = LigerSiLUMulFunction.apply


class FFNPredictor(nn.Module):
    """Predicts which intermediate dimensions to keep in an MLP layer."""

    def __init__(self, hidden_size: int, intermediate_size: int, rank: int) -> None:
        super().__init__()
        self.register_buffer("w1", torch.empty(2, hidden_size, rank))
        self.register_buffer("w2", torch.empty(2, rank, intermediate_size))

    @torch.inference_mode()
    def predict(self, x: torch.Tensor, sparsity: float) -> torch.Tensor:
        x = x.mean(dim=1, keepdim=True).view(1, -1, x.shape[-1]).expand(2, -1, -1)
        scores = _silu_mul(*torch.bmm(torch.bmm(x, self.w1), self.w2).unbind(0)).norm(dim=0)
        k = int(scores.shape[-1] * (1 - sparsity))
        return scores.topk(k).indices.flatten()


class AttentionPredictor(nn.Module):
    """Predicts Q/K/V sparsity indices for non-GQA attention layers."""

    def __init__(self, hidden_size: int, kv_size: int, rank: int) -> None:
        super().__init__()
        assert hidden_size == kv_size, "Use GQAAttentionPredictor when Q and KV sizes differ"
        self.register_buffer("w1", torch.empty(3, hidden_size, rank))
        self.register_buffer("w2", torch.empty(3, rank, hidden_size))

    @torch.inference_mode()
    def predict(self, x: torch.Tensor, sparsity: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.view(1, -1, x.shape[-1]).expand(3, -1, -1)
        scores = torch.bmm(torch.bmm(x, self.w1), self.w2).norm(dim=1)
        qk = scores[0] * scores[1]
        k = int(qk.shape[-1] * (1 - sparsity))
        qk_idx = qk.topk(k).indices.flatten()
        v_idx = scores[2].topk(k).indices.flatten()
        return qk_idx, qk_idx, v_idx


class GQAAttentionPredictor(nn.Module):
    """Predicts Q/K/V sparsity indices for grouped-query attention layers."""

    def __init__(self, hidden_size: int, kv_size: int, rank: int) -> None:
        super().__init__()
        self.register_buffer("w1", torch.empty(2, hidden_size, rank))
        self.register_buffer("w2", torch.empty(2, rank, kv_size))
        self.register_buffer("q1", torch.empty(rank, hidden_size))
        self.register_buffer("q2", torch.empty(hidden_size, rank))

    @torch.inference_mode()
    def predict(self, x: torch.Tensor, sparsity: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_scores = F.linear(F.linear(x, self.q1), self.q2).flatten(0, -2).norm(dim=0)
        kv = torch.bmm(torch.bmm(x.reshape(1, -1, x.shape[-1]).expand(2, -1, -1), self.w1), self.w2).norm(dim=1)
        k_scores, v_scores = kv[0], kv[1]

        groups = q_scores.shape[-1] // k_scores.shape[-1]
        qk = q_scores.view(groups, k_scores.shape[-1]).mean(0) * k_scores

        tk = int(qk.shape[-1] * (1 - sparsity))
        k_idx = qk.topk(tk).indices
        v_idx = v_scores.topk(tk).indices
        q_idx = (k_idx.unsqueeze(1) * groups + torch.arange(groups, device=k_idx.device)).flatten()
        return q_idx, k_idx.flatten(), v_idx.flatten()
