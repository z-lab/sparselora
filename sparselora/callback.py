"""Trainer callback for SparseLoRA sparsity scheduling."""

import torch.distributed as dist
import transformers
from torch import nn

from sparselora.modules import SparseModule

__all__ = ["SparseLoRACallback"]


class SparseLoRACallback(transformers.TrainerCallback):
    """Enables sparsity after ``start_step`` and disables it after ``end_step``.

    Both values are fractions of ``max_steps`` when <= 1, or absolute step
    counts when > 1.
    """

    def __init__(self, start_step: float = 0, end_step: float = 1) -> None:
        self.start_step = start_step
        self.end_step = end_step
        self._logged = False

    def on_step_begin(self, args, state, control, model: nn.Module = None, **kwargs) -> None:
        start = self.start_step * state.max_steps if self.start_step <= 1 else self.start_step
        end = self.end_step * state.max_steps if self.end_step <= 1 else self.end_step

        enabled = start <= state.global_step < end
        for m in model.modules():
            if isinstance(m, SparseModule):
                m.enabled = enabled

        if enabled and not self._logged:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"[SparseLoRA] Enabled at step {state.global_step}/{state.max_steps}")
            self._logged = True
