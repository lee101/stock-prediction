from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_active_mask(predictions: np.ndarray) -> np.ndarray:
    if predictions.ndim != 2:
        raise ValueError("predictions must be a 2D array")
    return np.any(np.abs(predictions) > 1e-12, axis=1)


def apply_soft_rank_sizing(
    predictions: np.ndarray,
    action_rows: pd.DataFrame,
    rank_signal: np.ndarray,
    *,
    min_score: float,
    reference_quantile: float,
    regime_floor_scale: float,
    name_floor_scale: float,
    regime_power: float,
    rank_power: float,
) -> np.ndarray:
    if len(predictions) == 0:
        return predictions.astype(np.float32, copy=False)
    if len(predictions) != len(action_rows) or len(predictions) != len(rank_signal):
        raise ValueError("predictions, action_rows, and rank_signal must have matching lengths")

    scaled = predictions.astype(np.float32, copy=True)
    scores = np.asarray(rank_signal, dtype=np.float32).reshape(-1)
    timestamps = pd.to_datetime(action_rows["timestamp"], utc=True)
    active_mask = _compute_active_mask(scaled)
    quantile = float(np.clip(reference_quantile, 0.0, 1.0))
    regime_floor = float(np.clip(regime_floor_scale, 0.0, 1.0))
    name_floor = float(np.clip(name_floor_scale, 0.0, 1.0))
    regime_exponent = float(max(regime_power, 0.0))
    rank_exponent = float(max(rank_power, 0.0))
    score_denom = max(1.0 - float(min_score), 1e-6)

    for _, group_index in timestamps.groupby(timestamps, sort=False).groups.items():
        group_indices = np.asarray(group_index, dtype=np.int64)
        candidate_indices = group_indices[active_mask[group_indices]]
        if candidate_indices.size == 0:
            continue

        candidate_scores = scores[candidate_indices]
        top_score = float(np.max(candidate_scores))
        strength = float(np.clip((top_score - float(min_score)) / score_denom, 0.0, 1.0))
        regime_scale = regime_floor + (1.0 - regime_floor) * (strength**regime_exponent)

        anchor_score = float(np.quantile(candidate_scores.astype(np.float64, copy=False), quantile))
        relative_denom = max(top_score - anchor_score, 1e-6)
        relative_scores = np.clip((candidate_scores - anchor_score) / relative_denom, 0.0, 1.0)
        rank_scale = name_floor + (1.0 - name_floor) * np.power(relative_scores, rank_exponent)
        scaled[candidate_indices] *= (regime_scale * rank_scale).astype(np.float32, copy=False)[:, None]

    return scaled.astype(np.float32, copy=False)
