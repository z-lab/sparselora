from torch import nn

__all__ = ["set_submodule"]


def set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    for part in parts[:-1]:
        model = getattr(model, part)
    setattr(model, parts[-1], module)
