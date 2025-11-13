"""Lightweight utilities shared by pctdiff strategy components."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_CLIP_LOGGED = False


def reset_pctdiff_clip_flag() -> None:
    global _CLIP_LOGGED
    _CLIP_LOGGED = False


def clip_pctdiff_returns(values: np.ndarray, *, max_abs_return: float) -> np.ndarray:
    if values.size == 0 or max_abs_return <= 0:
        return values

    clipped = np.clip(values, -max_abs_return, max_abs_return)
    if np.any(clipped != values):
        global _CLIP_LOGGED
        if not _CLIP_LOGGED:
            max_obs = float(np.max(np.abs(values)))
            logger.warning(
                "Clipped pctdiff returns to ±%.2f (observed %.4f exceeded limit)",
                max_abs_return,
                max_obs,
            )
            _CLIP_LOGGED = True
    return clipped


def pctdiff_midpoint_stub_returns(enabled: bool = False, reason: str = "not_implemented") -> Tuple[np.ndarray, Dict[str, object]]:
    metadata = {
        "pctdiff_midpoint_enabled": enabled,
        "pctdiff_midpoint_reason": reason,
        "pctdiff_midpoint_sharpe": 0.0,
        "pctdiff_midpoint_avg_daily": 0.0,
    }
    return np.zeros(0, dtype=float), metadata
