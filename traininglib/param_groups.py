"""
Helper for splitting model parameters into decay / no-decay groups.

Keeping the logic in one place avoids re-implementing LayerNorm/bias filtering
everywhere we construct optimizers.  The heuristics follow the pattern used in
nanochat (and Hugging Face) so the default behaviour is predictable.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

import torch

_NO_DECAY_PATTERN = re.compile(
    r"(?:bias|bn\d*\.weight|batchnorm\d*\.weight|layernorm\d*\.weight|"
    r"ln\d*\.weight|norm\d*\.weight|embedding\.weight)$",
    flags=re.IGNORECASE,
)


def parameter_groups(
    model: torch.nn.Module,
    *,
    weight_decay: float,
    extra_no_decay: Iterable[str] | None = None,
) -> List[Dict]:
    """Return parameter groups with transparent weight decay policies."""
    extra = set(extra_no_decay or ())
    decay, no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if _NO_DECAY_PATTERN.search(name) or any(token in name for token in extra) or param.ndim <= 1:
            no_decay.append(param)
        else:
            decay.append(param)

    groups: List[Dict] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups

