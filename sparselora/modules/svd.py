"""Load SVD-based sparsity predictors from safetensors."""

import os
from typing import Any

import torch
from safetensors.torch import load_file

from sparselora.modules.predictors import FFNPredictor, AttentionPredictor, GQAAttentionPredictor


def _load_tensors(cfg: Any, device, dtype) -> dict:
    path = os.path.join(cfg.path, "model.safetensors")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SVD predictors not found at {path}. "
            f"Set config.path to a directory containing model.safetensors."
        )
    return {k: v.to(device=device, dtype=dtype) for k, v in load_file(path).items()}


def create_mlp_predictor(base, rank: int, layer_name: str, cfg: Any) -> FFNPredictor:
    """Load an SVD predictor for an MLP layer from safetensors."""
    device, dtype = base.gate_proj.weight.device, base.gate_proj.weight.dtype
    tensors = _load_tensors(cfg, device, dtype)

    prefix = layer_name
    w1 = torch.stack([tensors[f"{prefix}.gate_proj.w1"], tensors[f"{prefix}.up_proj.w1"]])
    w2 = torch.stack([tensors[f"{prefix}.gate_proj.w2"], tensors[f"{prefix}.up_proj.w2"]])

    pred = FFNPredictor(w1.shape[1], w2.shape[2], rank)
    pred.load_state_dict({"w1": w1, "w2": w2})
    pred.eval()
    return pred.to(device=device, dtype=torch.bfloat16)


def create_attn_predictor(base, rank: int, layer_name: str, cfg: Any):
    """Load an SVD predictor for an attention layer from safetensors."""
    device, dtype = base.q_proj.base_layer.weight.device, base.q_proj.base_layer.weight.dtype
    tensors = _load_tensors(cfg, device, dtype)

    prefix = layer_name
    q = [tensors[f"{prefix}.q_proj.w1"], tensors[f"{prefix}.q_proj.w2"]]
    k = [tensors[f"{prefix}.k_proj.w1"], tensors[f"{prefix}.k_proj.w2"]]
    v = [tensors[f"{prefix}.v_proj.w1"], tensors[f"{prefix}.v_proj.w2"]]

    is_gqa = q[1].shape != k[1].shape
    if is_gqa:
        state_dict = {
            "w1": torch.stack([k[0], v[0]]),
            "w2": torch.stack([k[1], v[1]]),
            "q1": q[0].transpose(0, 1),
            "q2": q[1].transpose(0, 1),
        }
        pred = GQAAttentionPredictor(q[0].shape[0], k[1].shape[-1], rank)
    else:
        state_dict = {
            "w1": torch.stack([q[0], k[0], v[0]]),
            "w2": torch.stack([q[1], k[1], v[1]]),
        }
        pred = AttentionPredictor(q[0].shape[0], k[1].shape[-1], rank)

    pred.load_state_dict(state_dict)
    pred.eval()
    pred = torch.compile(pred)
    return pred.to(device=device, dtype=torch.bfloat16)
