"""Utility helpers for Kronos training metrics."""

from __future__ import annotations

import numpy as np


def compute_mae_percent(mae: float, actual_values: np.ndarray, eps: float = 1e-8) -> float:
    """Return MAE expressed as a percentage of the mean absolute actual value.

    Args:
        mae: Mean absolute error expressed in the same units as ``actual_values``.
        actual_values: Sequence of observed target values.
        eps: Small constant to avoid division by extremely small scales.

    Returns:
        The MAE as a percentage of the mean absolute actual magnitude. When the
        average absolute actual value is extremely small, returns ``inf`` if the
        MAE is non-zero (indicating an undefined scale) or ``0.0`` otherwise.
    """

    if mae < 0:
        raise ValueError("mae must be non-negative")

    if actual_values.size == 0:
        raise ValueError("actual_values must be non-empty")

    mean_abs_actual = float(np.mean(np.abs(actual_values)))
    if mean_abs_actual < eps:
        return float("inf") if mae > 0 else 0.0
    return float((mae / mean_abs_actual) * 100.0)


__all__ = ["compute_mae_percent"]
