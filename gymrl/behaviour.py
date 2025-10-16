"""
Baseline behaviour policies for offline dataset generation.

These heuristics convert forecast features into portfolio weights that can seed
offline RL algorithms such as CQL/IQL.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np


def topk_equal_weight(
    forecast_scores: np.ndarray,
    k: int = 2,
    *,
    long_only: bool = True,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Construct equal-weight behaviour allocations by picking the top-k assets.

    Args:
        forecast_scores: Array with shape (T, N) capturing signal strength (e.g., forecast_mu).
        k: Number of assets to hold per step.
        long_only: If False, negative scores will generate short allocations mirroring longs.
        threshold: Minimum score absolute value required for inclusion.

    Returns:
        weights: Array with shape (T, N) of behaviour weights per time-step.
    """

    scores = np.asarray(forecast_scores, dtype=np.float32)
    T, N = scores.shape
    weights = np.zeros_like(scores)

    for t in range(T):
        row = scores[t]
        if long_only:
            candidates = np.where(row > threshold)[0]
            if candidates.size == 0:
                continue
            selected = candidates[np.argsort(row[candidates])[::-1][:k]]
            weight = 1.0 / max(len(selected), 1)
            weights[t, selected] = weight
        else:
            ordered = np.argsort(np.abs(row))[::-1][: 2 * k]
            longs = [idx for idx in ordered if row[idx] > threshold][:k]
            shorts = [idx for idx in ordered if row[idx] < -threshold][:k]
            if longs:
                weights[t, longs] = 0.5 / len(longs)
            if shorts:
                weights[t, shorts] = -0.5 / len(shorts)
            balance = weights[t].sum()
            if abs(balance) > 1e-6:
                weights[t, :] -= balance / N

    return weights


def kelly_fractional(
    expected_returns: np.ndarray,
    variances: np.ndarray,
    *,
    cap: float = 0.3,
    risk_aversion: float = 1.0,
) -> np.ndarray:
    """
    Kelly-inspired sizing per asset with variance regularisation.

    Args:
        expected_returns: Shape (T, N) array of forecast mean simple returns.
        variances: Shape (T, N) array of forecast variances (sigma^2).
        cap: Maximum absolute weight per asset.
        risk_aversion: Shrinkage factor applied to Kelly sizing.

    Returns:
        weights: Shape (T, N) array of allocations (long-only, renormalised).
    """

    mu = np.asarray(expected_returns, dtype=np.float32)
    var = np.asarray(variances, dtype=np.float32)
    kelly = np.divide(mu, var + 1e-8) / max(risk_aversion, 1e-6)
    kelly = np.clip(kelly, -cap, cap)

    weights = np.maximum(kelly, 0.0)
    normaliser = weights.sum(axis=1, keepdims=True)
    normaliser[normaliser < 1e-6] = 1.0
    weights /= normaliser
    return weights


def blend_policies(
    weights_long_only: np.ndarray,
    weights_kelly: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Blend two behaviour policies with convex combination.

    Args:
        weights_long_only: Baseline long-only allocations.
        weights_kelly: Risk-adjusted allocations.
        alpha: Blend coefficient toward the first policy.

    Returns:
        Convex combination of weight matrices.
    """

    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * weights_long_only + (1.0 - alpha) * weights_kelly


__all__ = ["topk_equal_weight", "kelly_fractional", "blend_policies"]

