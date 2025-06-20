import torch
import os
from torch import nn
from spft.modules.pred import (
    FFNPredictor,
    QKPredictor,
    VOPredictor,
    QKVOFusedPredictor,
    QKVOFusedGQAPredictor,
)
from typing import Any

def create_mlp_svd_pred(base, rank: int, layer_name: str, cfg: Any):
    model_name = cfg.model_id
    basepath = cfg.predictor_basepath
    
    base_dir = f"{basepath}/{model_name}/r_{rank}/mlp"
    os.makedirs(base_dir, exist_ok=True)

    base_path = f"{base_dir}/{layer_name}"
    device, dtype = base.gate_proj.weight.device, base.gate_proj.weight.dtype

    try:
        w1 = torch.load(base_path + "_low_rank_a.pt", map_location=device, weights_only=True).to(dtype)
        w2 = torch.load(base_path + "_low_rank_b.pt", map_location=device, weights_only=True).to(dtype)
    except FileNotFoundError:
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            raise RuntimeError("SVD Predictor should be created before run-time.")
        #* Weights are store Output x Input --> Let's Transpose
        gate_w = base.gate_proj.weight.to(torch.float32).transpose(0, 1)
        up_w = base.up_proj.weight.to(torch.float32).transpose(0, 1)

        def svd_decompose(W):
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            A = U[:, :rank] @ torch.diag(S[:rank]).sqrt()
            B = torch.diag(S[:rank]).sqrt() @ Vh[:rank, :]
            return A.to(device).to(dtype), B.to(device).to(dtype)

        gate = svd_decompose(gate_w)
        up = svd_decompose(up_w)

        #* Stack Weights:
        w1 = torch.stack([gate[0], up[0]], dim=0)
        w2 = torch.stack([gate[1], up[1]], dim=0)
        
        #* Save Weights:
        torch.save(w1, base_path + "_low_rank_a.pt")
        torch.save(w2, base_path + "_low_rank_b.pt")
        
    state_dict = {
        "w1": w1,
        "w2": w2,
    }
    
    #* Create Predictor & Load Weights:
    pred = FFNPredictor(w1.shape[1], w2.shape[2], rank, seq_avg=cfg.mlp_seq_avg)
    pred.load_state_dict(state_dict)
    pred.eval()
    # pred = torch.compile(pred)
    pred = pred.to(device=device, dtype=torch.bfloat16)
        
    return pred

def create_attn_svd_pred(base, rank: int, layer_name: str, cfg: Any):
    model_name = cfg.model_id
    basepath = cfg.predictor_basepath
    
    base_dir = f"{basepath}/{model_name}/r_{rank}/attn"
    os.makedirs(base_dir, exist_ok=True)
    base_path = f"{base_dir}/{layer_name}"
    device, dtype = base.q_proj.base_layer.weight.device, base.q_proj.base_layer.weight.dtype

    try:
        q = [torch.load(base_path + f"_q_low_rank_{p}.pt", map_location=device, weights_only=True).to(dtype) for p in ("a", "b")]
        k = [torch.load(base_path + f"_k_low_rank_{p}.pt", map_location=device, weights_only=True).to(dtype) for p in ("a", "b")]
        v = [torch.load(base_path + f"_v_low_rank_{p}.pt", map_location=device, weights_only=True).to(dtype) for p in ("a", "b")]
    except FileNotFoundError:
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            raise RuntimeError("SVD Predictor should be created before run-time.")
        def svd_decompose(W):
            U, S, Vh = torch.linalg.svd(W.to(torch.float32), full_matrices=False)
            A = U[:, :rank] @ torch.diag(S[:rank]).sqrt()
            B = torch.diag(S[:rank]).sqrt() @ Vh[:rank, :]
            return [A.to(device).to(dtype), B.to(device).to(dtype)]

        # Weights are store Output x Input --> Let's Transpose
        q = svd_decompose(base.q_proj.base_layer.weight.transpose(0, 1))
        k = svd_decompose(base.k_proj.base_layer.weight.transpose(0, 1))
        v = svd_decompose(base.v_proj.base_layer.weight.transpose(0, 1))

        for name, pair in zip(('q', 'k', 'v'), (q, k, v)):
            torch.save(pair[0], base_path + f"_{name}_low_rank_a.pt")
            torch.save(pair[1], base_path + f"_{name}_low_rank_b.pt")
            

    #* Either GQA or Non-GQA Models:
    if q[-1].shape[0] != k[-1].shape[0] or q[-1].shape[1] != k[-1].shape[1]:
    # if cfg.gqa_ratio > 0: #* We have a GQA Model
        state_dict = {
            "w1": torch.stack([k[0], v[0]], dim=0),
            "w2": torch.stack([k[1], v[1]], dim=0),
            
            "q1": q[0].transpose(0, 1), #* Back into Output x Input Weight Format
            "q2": q[1].transpose(0, 1), #* Back into Output x Input Weight Format
        }
        pred_class = QKVOFusedGQAPredictor
    else:
        state_dict = {
            "w1": torch.stack([q[0], k[0], v[0]], dim=0),
            "w2": torch.stack([q[1], k[1], v[1]], dim=0),
        }
        pred_class = QKVOFusedPredictor
        
    pred = pred_class(q[0].shape[0], k[-1].shape[-1], rank, seq_avg=cfg.qkvo_seq_avg)
    pred.load_state_dict(state_dict)
    pred.eval()
    #?@ Skhaki: Fix torch.compile issues on FusedGQAPredictor || For others it works well.
    pred = torch.compile(pred)
    pred = pred.to(device=device, dtype=torch.bfloat16)
    
    return pred 