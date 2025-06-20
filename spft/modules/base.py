from typing import List
import torch.nn as nn
import torch
from typing import Optional
from .svd import create_mlp_svd_pred, create_attn_svd_pred
__all__ = ["SparseModule"]


class SparseModule(nn.Module):
    enabled: bool = True
    inherited_attributes: List[str] = []

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.stats = {}
        for name in self.inherited_attributes:
            setattr(self, name, getattr(base, name))

    def load_predictor(self, base, cfg):
        
        if "svd" in cfg.mode:
            self.mode = "svd"
            rank = int(cfg.mode.split("_")[-1])
            if "mlp" in self.layer_name:
                self.pred = create_mlp_svd_pred(base, rank, self.layer_name, cfg)
            elif "attn" in self.layer_name:
                self.pred = create_attn_svd_pred(base, rank, self.layer_name, cfg)
            else:
                raise ValueError("Not implemented")
        
        #* Can be SVD + Oracle == Masked SVD    
        if "oracle" in cfg.mode:
            self.mode = cfg.mode
        
        return None
    
    def split_idx(self, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        if isinstance(masks, tuple):
            return masks[1]
        else:
            raise ValueError("Should use slice not mask")
        
    def token_splits(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        #?@ Fix here!
        if isinstance(masks, tuple):
            masks, id_split = masks
            sparse_x = x[:, :id_split, :]
            dense_x = x[:, id_split:, :]
        else:
            dense_x = x[masks].view(x.shape[0], -1, x.shape[-1])
            sparse_x = x[~masks].view(x.shape[0], -1, x.shape[-1])
        
        self.stats["token_split/sparse"] = sparse_x.shape[1]
        self.stats["token_split/dense"] = dense_x.shape[1]
        
        return sparse_x.contiguous(), dense_x.contiguous()
    
    def token_join(self, sparse, dense, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        if masks is not None and not isinstance(masks, tuple):
            res = torch.zeros((sparse.shape[0], sparse.shape[1] + dense.shape[1], sparse.shape[-1]), device=sparse.device, dtype=sparse.dtype)
            res[masks] = dense.view(-1, dense.shape[-1])
            res[~masks] = sparse.view(-1, sparse.shape[-1])
        else:
            #* Token Order: [Sparse | Dense] --> [In | Out]
            res = torch.cat([sparse, dense], dim=1).contiguous()
        return res.contiguous()
        
