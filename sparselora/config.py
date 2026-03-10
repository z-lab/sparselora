import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional


@dataclass
class SparseLoRAConfig:
    """Configuration for SparseLoRA contextual sparsity.

    Args:
        layer_sparsity: Per-layer sparsity fractions
            (e.g. ``{"model.layers.3.mlp": 0.5}``).
        predictor_rank: Rank of the SVD sparsity predictor.
        path: Local directory or HuggingFace repo ID (e.g. ``z-lab/Meta-Llama-3-8B-Instruct-SparseLoRA``)
            containing ``model.safetensors`` and ``config.json``.
        start_step: Fraction of training at which to enable sparsity (0-1).
        end_step: Fraction of training at which to disable sparsity (0-1).

    Example::

        config = SparseLoRAConfig.from_pretrained(
            "z-lab/Meta-Llama-3-8B-Instruct-SparseLoRA", mode="o2",
        )
        model = apply_sparselora(model, config)
    """

    layer_sparsity: Dict[str, float] = field(default_factory=dict)
    predictor_rank: int = 8
    path: Optional[str] = None
    start_step: float = 0.0
    end_step: float = 1.0

    @classmethod
    def from_pretrained(cls, path: str, mode: str = "o1", **kwargs) -> "SparseLoRAConfig":
        """Load from a SparseLoRA model directory.

        Args:
            path: Local directory or HuggingFace repo ID.
            mode: Sparsity mode (``"o1"`` = conservative, ``"o2"`` = aggressive).
            **kwargs: Override any config field (e.g. ``start_step=0.05``).
        """
        if os.path.isdir(path):
            config_path = os.path.join(path, "config.json")
        else:
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(path, "config.json")

        with open(config_path) as f:
            data = json.load(f)

        modes = data.get("modes", {})
        if mode not in modes:
            raise ValueError(f"Mode '{mode}' not found. Available: {list(modes.keys())}")

        config = cls(
            layer_sparsity=modes[mode]["layer_sparsity"],
            predictor_rank=data.get("predictor_rank", 8),
            path=path,
        )
        for k, v in kwargs.items():
            setattr(config, k, v)
        return config

    @classmethod
    def from_yaml(cls, path: str) -> "SparseLoRAConfig":
        """Load from a YAML file."""
        import yaml

        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def from_dict(cls, data: dict) -> "SparseLoRAConfig":
        """Create from a dictionary, ignoring unknown keys."""
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def to_dict(self) -> dict:
        return asdict(self)

    def save_pretrained(self, out: str) -> None:
        """Save as ``sparselora_config.json`` in the given directory."""
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "sparselora_config.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=2)
