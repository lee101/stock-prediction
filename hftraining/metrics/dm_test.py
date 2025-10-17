from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def dm_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    horizon: int = 1,
) -> Tuple[float, float]:
    """
    Diebold-Mariano test statistic comparing two loss sequences.

    Args:
        loss_a: Array of losses for model A.
        loss_b: Array of losses for model B.
        horizon: Forecast horizon (controls Newey-West lag truncation).

    Returns:
        (dm_stat, p_value)
    """
    if loss_a.shape != loss_b.shape:
        raise ValueError("Loss arrays must share identical shapes.")

    diff = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    T = diff.size
    d_bar = diff.mean()

    # Newey-West HAC variance estimate
    gamma0 = np.mean((diff - d_bar) ** 2)
    var = gamma0
    for lag in range(1, horizon):
        cov = np.mean((diff[:-lag] - d_bar) * (diff[lag:] - d_bar))
        var += 2.0 * (1.0 - lag / horizon) * cov

    denominator = math.sqrt(max(var / max(T, 1), 1e-12))
    dm_stat = d_bar / denominator if denominator > 0 else 0.0

    # two-sided p-value under asymptotic normality
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(dm_stat) / math.sqrt(2.0))))
    return dm_stat, p_value
