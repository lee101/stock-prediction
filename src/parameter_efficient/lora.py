from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "LoRALinear",
    "freeze_module_parameters",
    "inject_lora_adapters",
    "iter_lora_parameters",
    "save_lora_adapter",
]


class LoRALinear(nn.Module):
    """
    Lightweight wrapper around ``nn.Linear`` that injects a trainable
    low-rank offset (LoRA) while freezing the base weights.
    """

    def __init__(self, base_layer: nn.Linear, *, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout: nn.Module
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze the base layer weights/bias to ensure only the adapters train.
        for param in self.base_layer.parameters():
            param.requires_grad_(False)

        in_features = self.base_layer.in_features
        out_features = self.base_layer.out_features

        # Create LoRA parameters on the same device as the base layer
        device = self.base_layer.weight.device
        dtype = self.base_layer.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(self.rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank, device=device, dtype=dtype))

        # Flag these parameters so they can be easily filtered later.
        self.lora_A._is_lora_param = True  # type: ignore[attr-defined]
        self.lora_B._is_lora_param = True  # type: ignore[attr-defined]

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Follow the standard LoRA initialisation: A ~ kaiming_uniform, B zeros.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self) -> nn.Parameter:
        return self.base_layer.weight

    @property
    def bias(self) -> Optional[nn.Parameter]:
        return self.base_layer.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - exercised indirectly
        base_out = self.base_layer(inputs)
        if self.rank == 0:
            return base_out

        dropped = self.lora_dropout(inputs)
        lora_intermediate = F.linear(dropped, self.lora_A)
        lora_out = F.linear(lora_intermediate, self.lora_B)
        return base_out + self.scaling * lora_out


def freeze_module_parameters(module: nn.Module) -> None:
    """Set ``requires_grad=False`` for every parameter inside ``module``."""
    for param in module.parameters():
        param.requires_grad_(False)


def _should_match(name: str, patterns: Sequence[str]) -> bool:
    if not patterns:
        return True
    return any(pattern in name for pattern in patterns)


def inject_lora_adapters(
    module: nn.Module,
    *,
    target_patterns: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
    module_filter: Optional[Callable[[str, nn.Module], bool]] = None,
) -> List[str]:
    """
    Replace matching ``nn.Linear`` layers with :class:`LoRALinear`.

    Args:
        module: Root module to traverse.
        target_patterns: Collection of substrings; a module path is wrapped when
            any pattern is contained within it. An empty sequence matches all linear layers.
        rank: LoRA rank ``r``.
        alpha: Scaling factor (``alpha / r`` applied to the LoRA branch).
        dropout: Dropout probability applied before the rank reduction.
        module_filter: Optional callback receiving ``(full_name, child_module)``;
            only when it returns ``True`` does the replacement occur.

    Returns:
        List of dotted module names that were wrapped.

    Raises:
        ValueError: If no modules were matched.
    """
    replaced: List[str] = []

    for name, parent in list(module.named_modules()):
        for child_name, child in list(parent.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            if not isinstance(child, nn.Linear):
                continue
            if not _should_match(full_name, target_patterns):
                continue
            if module_filter and not module_filter(full_name, child):
                continue

            lora_layer = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent, child_name, lora_layer)
            replaced.append(full_name)

    if not replaced:
        raise ValueError(
            "No modules matched for LoRA injection. "
            "Adjust `target_patterns` or ensure the model contains Linear layers."
        )
    return replaced


def iter_lora_parameters(module: nn.Module) -> Iterator[Tuple[str, nn.Parameter]]:
    """Yield ``(name, parameter)`` pairs for LoRA-specific parameters."""
    for name, param in module.named_parameters():
        if getattr(param, "_is_lora_param", False):
            yield name, param


@dataclass
class LoraMetadata:
    adapter_type: str
    rank: int
    alpha: float
    dropout: float
    targets: Sequence[str]
    base_model: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "adapter_type": self.adapter_type,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "targets": list(self.targets),
            "base_model": self.base_model,
        }


def save_lora_adapter(
    module: nn.Module,
    path: Path,
    *,
    metadata: Optional[LoraMetadata] = None,
) -> None:
    """
    Persist only the LoRA trainable weights alongside optional metadata.
    """
    state: Dict[str, torch.Tensor] = {}
    for name, tensor in module.state_dict().items():
        if "lora_" in name:
            state[name] = tensor.cpu()

    if not state:
        raise ValueError("Module does not contain LoRA parameters to save.")

    payload: Dict[str, object] = {"state_dict": state}
    if metadata is not None:
        payload["metadata"] = metadata.to_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)

    if metadata is not None:
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
