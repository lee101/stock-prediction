from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_active_mask(predictions: np.ndarray) -> np.ndarray:
    if predictions.ndim != 2:
        raise ValueError("predictions must be a 2D array")
    return np.any(np.abs(predictions) > 1e-12, axis=1)


def apply_dynamic_score_floor_gate(
    predictions: np.ndarray,
    action_rows: pd.DataFrame,
    rank_signal: np.ndarray,
    *,
    min_score: float,
    min_strength: float,
    floor_quantile: float,
    floor_gap_scale: float,
) -> np.ndarray:
    if len(predictions) == 0:
        return predictions.astype(np.float32, copy=False)
    if len(predictions) != len(action_rows) or len(predictions) != len(rank_signal):
        raise ValueError("predictions, action_rows, and rank_signal must have matching lengths")

    gated = predictions.astype(np.float32, copy=True)
    scores = np.asarray(rank_signal, dtype=np.float32).reshape(-1)
    timestamps = pd.to_datetime(action_rows["timestamp"], utc=True)
    active_mask = _compute_active_mask(gated)
    keep_mask = np.zeros((len(scores),), dtype=bool)
    quantile = float(np.clip(floor_quantile, 0.0, 1.0))
    scale = float(max(floor_gap_scale, 0.0))
    base_floor = float(min_score)
    required_top_score = base_floor + float(max(min_strength, 0.0))
    denom = max(1.0 - required_top_score, 1e-6)

    for _, group_index in timestamps.groupby(timestamps, sort=False).groups.items():
        group_indices = np.asarray(group_index, dtype=np.int64)
        candidate_indices = group_indices[active_mask[group_indices]]
        if candidate_indices.size == 0:
            continue

        candidate_scores = scores[candidate_indices]
        top_score = float(np.max(candidate_scores))
        if top_score < required_top_score:
            continue

        anchor_score = float(np.quantile(candidate_scores.astype(np.float64, copy=False), quantile))
        dispersion = max(top_score - anchor_score, 0.0)
        strength = float(np.clip((top_score - required_top_score) / denom, 0.0, 1.0))
        dynamic_floor = max(base_floor, top_score - strength * scale * dispersion)
        keep_mask[candidate_indices[candidate_scores >= dynamic_floor]] = True

    gated[~keep_mask] = 0.0
    return gated.astype(np.float32, copy=False)
