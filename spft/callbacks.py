import transformers
from torch import nn
from transformers import TrainerControl, TrainerState, TrainingArguments

import wandb
from spft.modules import SparseModule
from spft.utils.io import rank0_print

__all__ = ["SPFTCallback"]


class SPFTCallback(transformers.TrainerCallback):
    def __init__(self, start_step: float = 0, end_step: float = 1) -> None:
        self.start_step = start_step
        self.end_step = end_step
        self.prev_state = False

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: nn.Module,
        **kwargs,
    ) -> None:
        start_step = self.start_step * state.max_steps if self.start_step <= 1 else self.start_step
        end_step = self.end_step * state.max_steps if self.end_step <= 1 else self.end_step

        
        for module in model.modules():
            if isinstance(module, SparseModule):
                module.enabled = start_step <= state.global_step < end_step
                current_state = module.enabled
                
        if current_state and not self.prev_state:
            rank0_print(f"SPFT: Enabled at step {state.global_step} / {state.max_steps}")
            self.prev_state = True
            
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: nn.Module,
        **kwargs,
    ) -> None:
        if wandb.run is None:
            return
        for name, module in model.named_modules():
            if isinstance(module, SparseModule):
                for key, val in module.stats.items():
                    wandb.log({f"stats/{name}/{key}": val}, step=state.global_step)
                module.stats.clear()

class EvaluateFirstStepCallback(transformers.integrations.integration_utils.TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True