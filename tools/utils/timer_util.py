import time
import torch
import transformers
from typing import Dict, Union, Any
from torch import nn
class TimedTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_times = []
        self.backward_times = []


    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        start_fwd = torch.cuda.Event(enable_timing=True)
        end_fwd = torch.cuda.Event(enable_timing=True)
        start_bwd = torch.cuda.Event(enable_timing=True)
        end_bwd = torch.cuda.Event(enable_timing=True)

        start_fwd.record()
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        end_fwd.record()
        del inputs
        kwargs = {}
        start_bwd.record()
        
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss, **kwargs)
        # Finally we need to normalize the loss for reporting
        if num_items_in_batch is None:
            return loss.detach() / self.args.gradient_accumulation_steps
        end_bwd.record()
        torch.cuda.synchronize()
        
        fwd_time = start_fwd.elapsed_time(end_fwd)
        bwd_time = start_bwd.elapsed_time(end_bwd)

        self.forward_times.append(fwd_time)
        self.backward_times.append(bwd_time)
        
        return loss.detach()
    
class _Timer:
    """An internal timer."""

    def __init__(self, name: str):
        self.name = name
        self.started = False
        self.start_time = None

        # start-stop timestamp pairs
        self.start_times = []
        self.stop_times = []
        self.costs = []

        self.start_event = None
        self.stop_event = None

    def start(self, sync_func=torch.cuda.synchronize):
        """Start the timer."""
        assert not self.started, f"timer {self.name} has already been started."
        if sync_func:
            sync_func()

        # self.start_time = time.perf_counter()
        # self.start_times.append(self.start_time)
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.stop_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        self.started = True

    def stop(self, sync_func=torch.cuda.synchronize):
        """Stop the timer."""
        assert self.started, f"timer {self.name} is not started."
        self.stop_event.record()
        if sync_func:
            sync_func()

        # stop_time = time.perf_counter()
        # self.costs.append(stop_time - self.start_time)
        # self.stop_times.append(stop_time)
        self.costs.append(self.start_event.elapsed_time(self.stop_event))
        self.started = False
        self.start_event = None
        self.stop_event = None

    def reset(self):
        """Reset timer."""
        self.started = False
        self.start_time = None
        self.start_event = None
        self.stop_event = None
        self.start_times = []
        self.stop_times = []
        self.costs = []

    def elapsed(self, mode: str = "average"):
        """Calculate the elapsed time."""
        if not self.costs:
            return 0.0
        if mode == "average":
            return sum(self.costs) / len(self.costs)
        elif mode == "sum":
            return sum(self.costs)
        else:
            raise RuntimeError("Supported mode is: average | sum")


class Timers:
    """A group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name: str):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def __contains__(self, name: str):
        return name in self.timers