from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..ambiguity_plan_gate import (
    PLAN_LONG_STRONG,
    PLAN_LONG_WEAK,
    PLAN_SHORT_STRONG,
    PLAN_SHORT_WEAK,
)


def compute_cross_sectional_rank_signal(
    plan_logits: torch.Tensor,
    margin_logits: torch.Tensor,
    *,
    weak_action_scale: float,
) -> torch.Tensor:
    plan_probs = F.softmax(plan_logits, dim=-1)
    plan_scale = (
        plan_probs[:, PLAN_SHORT_STRONG]
        + plan_probs[:, PLAN_LONG_STRONG]
        + float(weak_action_scale) * (plan_probs[:, PLAN_SHORT_WEAK] + plan_probs[:, PLAN_LONG_WEAK])
    )
    margin_scale = torch.sigmoid(margin_logits.squeeze(-1))
    return plan_scale * margin_scale


def _compute_active_mask(predictions: np.ndarray) -> np.ndarray:
    if predictions.ndim != 2:
        raise ValueError("predictions must be a 2D array")
    return np.any(np.abs(predictions) > 1e-12, axis=1)


def apply_topk_cross_sectional_gate(
    predictions: np.ndarray,
    action_rows: pd.DataFrame,
    rank_signal: np.ndarray,
    *,
    top_k: int,
    min_score: float,
) -> np.ndarray:
    if len(predictions) == 0:
        return predictions.astype(np.float32, copy=False)
    if len(predictions) != len(action_rows) or len(predictions) != len(rank_signal):
        raise ValueError("predictions, action_rows, and rank_signal must have matching lengths")
    if int(top_k) < 1:
        raise ValueError("top_k must be at least 1")

    gated = predictions.astype(np.float32, copy=True)
    scores = np.asarray(rank_signal, dtype=np.float32).reshape(-1)
    timestamps = pd.to_datetime(action_rows["timestamp"], utc=True)
    active_mask = _compute_active_mask(gated)
    keep_mask = np.zeros((len(scores),), dtype=bool)

    for _, group_index in timestamps.groupby(timestamps, sort=False).groups.items():
        group_indices = np.asarray(group_index, dtype=np.int64)
        group_scores = scores[group_indices]
        eligible = group_indices[np.logical_and(active_mask[group_indices], group_scores >= float(min_score))]
        if eligible.size == 0:
            continue
        order = np.argsort(scores[eligible])[::-1]
        keep_mask[eligible[order[: int(top_k)]]] = True

    gated[~keep_mask] = 0.0
    return gated.astype(np.float32, copy=False)


def apply_dynamic_cross_sectional_gate(
    predictions: np.ndarray,
    action_rows: pd.DataFrame,
    rank_signal: np.ndarray,
    *,
    min_score: float,
    min_keep: int,
    max_keep: int,
    reference_quantile: float,
    gap_scale: float,
) -> np.ndarray:
    if len(predictions) == 0:
        return predictions.astype(np.float32, copy=False)
    if len(predictions) != len(action_rows) or len(predictions) != len(rank_signal):
        raise ValueError("predictions, action_rows, and rank_signal must have matching lengths")
    if int(min_keep) < 1:
        raise ValueError("min_keep must be at least 1")
    if int(max_keep) < int(min_keep):
        raise ValueError("max_keep must be at least min_keep")

    gated = predictions.astype(np.float32, copy=True)
    scores = np.asarray(rank_signal, dtype=np.float32).reshape(-1)
    timestamps = pd.to_datetime(action_rows["timestamp"], utc=True)
    active_mask = _compute_active_mask(gated)
    keep_mask = np.zeros((len(scores),), dtype=bool)
    quantile = float(np.clip(reference_quantile, 0.0, 1.0))
    scale = float(np.clip(gap_scale, 0.0, 1.0))

    for _, group_index in timestamps.groupby(timestamps, sort=False).groups.items():
        group_indices = np.asarray(group_index, dtype=np.int64)
        group_scores = scores[group_indices]
        eligible = group_indices[np.logical_and(active_mask[group_indices], group_scores >= float(min_score))]
        if eligible.size == 0:
            continue

        eligible_scores = scores[eligible]
        order = np.argsort(eligible_scores)[::-1]
        sorted_indices = eligible[order]
        sorted_scores = eligible_scores[order]

        if sorted_scores.size <= int(min_keep):
            keep_mask[sorted_indices] = True
            continue

        anchor_score = float(np.quantile(sorted_scores.astype(np.float64, copy=False), quantile))
        top_score = float(sorted_scores[0])
        dynamic_threshold = max(float(min_score), top_score - scale * (top_score - anchor_score))
        selected_mask = sorted_scores >= dynamic_threshold
        selected_indices = sorted_indices[selected_mask]

        if selected_indices.size < int(min_keep):
            selected_indices = sorted_indices[: int(min_keep)]
        if selected_indices.size > int(max_keep):
            selected_indices = sorted_indices[: int(max_keep)]
        keep_mask[selected_indices] = True

    gated[~keep_mask] = 0.0
    return gated.astype(np.float32, copy=False)
