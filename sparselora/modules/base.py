"""Base class for sparse transformer modules."""

from typing import List, Tuple

import torch
import torch.nn as nn


class SparseModule(nn.Module):
    """Base for sparse module replacements.

    Provides token split/join utilities shared by all architecture-specific
    sparse modules (MLP, attention).
    """

    enabled: bool = True
    inherited_attributes: List[str] = []

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        for attr in self.inherited_attributes:
            setattr(self, attr, getattr(base, attr))

    def token_splits(self, x: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split *x* into ``(sparse_tokens, dense_tokens)`` using a boolean mask."""
        return (
            x[~masks].view(x.shape[0], -1, x.shape[-1]).contiguous(),
            x[masks].view(x.shape[0], -1, x.shape[-1]).contiguous(),
        )

    @staticmethod
    def token_join(sparse: torch.Tensor, dense: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Reassemble sparse and dense tokens back to original token order."""
        out = torch.empty(
            sparse.shape[0], sparse.shape[1] + dense.shape[1], sparse.shape[-1],
            device=sparse.device, dtype=sparse.dtype,
        )
        out[masks] = dense.reshape(-1, dense.shape[-1])
        out[~masks] = sparse.reshape(-1, sparse.shape[-1])
        return out
