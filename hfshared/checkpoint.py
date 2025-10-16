#!/usr/bin/env python3
"""Checkpoint helpers shared across training/inference."""
from __future__ import annotations

from typing import Optional, Dict, Any


def infer_input_dim_from_state(state_dict: Dict[str, Any]) -> Optional[int]:
    """Infer input dimension from a model state dict by inspecting the input projection weight."""
    if not isinstance(state_dict, dict):
        return None
    try:
        if 'input_projection.weight' in state_dict:
            w = state_dict['input_projection.weight']
            if hasattr(w, 'shape') and len(w.shape) == 2:
                return int(w.shape[1])
        for k, v in state_dict.items():
            if k.endswith('input_projection.weight') and hasattr(v, 'shape') and len(v.shape) == 2:
                return int(v.shape[1])
    except Exception:
        return None
    return None

