"""Core API for applying SparseLoRA to PEFT models."""

import types
from functools import partial

import torch
import torch.distributed as dist
import transformers
from torch import nn
from tqdm import tqdm

import peft

from .callback import SparseLoRACallback
from .config import SparseLoRAConfig
from .modules import SparseModule, get_module_mapping, get_sparsity_mode, lora_forward

__all__ = ["apply_sparselora"]

_trainer_init_patched = False


def _set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    for part in parts[:-1]:
        model = getattr(model, part)
    setattr(model, parts[-1], module)


def _compute_output_token_mask(labels: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Build a boolean mask that is True for output (non-context) tokens."""
    is_ctx = (labels == -100)
    left = is_ctx.cumprod(dim=1).sum(dim=1).min().item()
    right_pad = is_ctx.flip(dims=[1]).cumprod(dim=1).sum(dim=1).min().item()
    right = labels.shape[-1] - right_pad if right_pad > 0 else labels.shape[-1]
    masks = torch.zeros_like(input_ids, dtype=torch.bool)
    masks[..., left:right] = True
    return masks


def _patch_forward(model: nn.Module, config: SparseLoRAConfig) -> None:
    _orig = model.forward

    def _forward(self, *args, **kwargs):
        labels = kwargs.get("labels")
        masks = _compute_output_token_mask(labels, kwargs.get("input_ids")) if labels is not None else None
        for m in self.model.modules():
            if isinstance(m, SparseModule):
                m.forward = partial(m.forward, masks=masks)
        return _orig(*args, **kwargs)

    model.forward = types.MethodType(_forward, model)


def _patch_generate(model: nn.Module) -> None:
    _orig = model.generate

    def _generate(self, *args, **kwargs):
        for m in self.model.modules():
            if isinstance(m, SparseModule):
                m.forward = partial(m.forward, masks=None)
        return _orig(*args, **kwargs)

    model.generate = types.MethodType(_generate, model)


def _patch_trainer(config: SparseLoRAConfig) -> None:
    """Monkey-patch ``Trainer.__init__`` to auto-inject :class:`SparseLoRACallback`."""
    global _trainer_init_patched
    if _trainer_init_patched:
        return
    _trainer_init_patched = True

    _orig_init = transformers.Trainer.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        has_sparselora = any(isinstance(m, SparseModule) for m in self.model.modules())
        already_registered = any(isinstance(cb, SparseLoRACallback) for cb in self.callback_handler.callbacks)
        if has_sparselora and not already_registered:
            self.add_callback(SparseLoRACallback(config.start_step, config.end_step))

    transformers.Trainer.__init__ = _patched_init


def apply_sparselora(model: nn.Module, config: SparseLoRAConfig) -> nn.Module:
    """Apply SparseLoRA contextual sparsity to a PEFT model.

    Call **after** ``peft.get_peft_model()``.  Replaces eligible modules with
    sparse counterparts and patches ``forward`` / ``generate``.  The
    :class:`SparseLoRACallback` is automatically registered with any
    ``transformers.Trainer`` created afterwards.

    Args:
        model: A PEFT-wrapped model.
        config: A :class:`SparseLoRAConfig`.

    Returns:
        The same model, modified in-place.
    """
    peft.tuners.lora.layer.Linear.forward = lora_forward

    module_map = get_module_mapping()
    show = not (dist.is_initialized() and dist.get_rank() != 0)

    with tqdm(total=sum(1 for _ in model.named_modules()), desc="[SparseLoRA] Patching", disable=not show) as pbar:
        for name, module in model.named_modules():
            layer_name, sparsity = next(
                ((s, v) for s, v in config.layer_sparsity.items() if name.endswith(s)),
                (None, None),
            )
            if sparsity is not None:
                parts = layer_name.split(".")
                idx = next(int(p) for p in parts if p.isdigit())
                kw = {"name": layer_name, "idx": idx, "sparsity": sparsity, "cfg": config}
                if type(module) in module_map:
                    _set_submodule(model, name, module_map[type(module)](base=module, **kw))
                for sub_name, sub_mod in module.named_modules():
                    if isinstance(sub_mod, nn.Linear):
                        _set_submodule(
                            model, f"{name}.{sub_name}",
                            module_map[type(sub_mod)](base=sub_mod, mode=get_sparsity_mode(sub_name)),
                        )
            pbar.update(1)

    _patch_forward(model, config)
    _patch_generate(model)
    _patch_trainer(config)
    return model
