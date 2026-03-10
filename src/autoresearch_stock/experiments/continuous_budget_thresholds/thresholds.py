from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..cross_sectional_rank_gate.rank_gate import _compute_active_mask


def _interpolate_count(
    probs: np.ndarray,
    *,
    skip_value: int,
    selective_value: int,
    broad_value: int,
) -> float:
    values = np.asarray(
        [float(skip_value), float(selective_value), float(broad_value)],
        dtype=np.float32,
    )
    return float(np.dot(probs.astype(np.float32, copy=False), values))


def apply_continuous_budget_gate(
    predictions: np.ndarray,
    action_rows: pd.DataFrame,
    rank_signal: np.ndarray,
    budget_logits: np.ndarray,
    *,
    min_score: float,
    base_min_keep: int,
    selective_top_k: int,
    selective_max_keep: int,
    broad_top_k: int,
    broad_max_keep: int,
    reference_quantile: float,
    selective_gap_scale: float,
    broad_gap_scale: float,
    skip_gap_scale: float,
    skip_min_score_scale: float,
    selective_min_score_scale: float,
    broad_min_score_scale: float,
) -> np.ndarray:
    if len(predictions) == 0:
        return predictions.astype(np.float32, copy=False)
    if len(predictions) != len(action_rows) or len(predictions) != len(rank_signal) or len(predictions) != len(budget_logits):
        raise ValueError("predictions, action_rows, rank_signal, and budget_logits must have matching lengths")
    if int(base_min_keep) < 1:
        raise ValueError("base_min_keep must be at least 1")
    if int(selective_top_k) < int(base_min_keep):
        raise ValueError("selective_top_k must be at least base_min_keep")
    if int(selective_max_keep) < int(base_min_keep):
        raise ValueError("selective_max_keep must be at least base_min_keep")
    if int(broad_top_k) < int(selective_top_k):
        raise ValueError("broad_top_k must be at least selective_top_k")
    if int(broad_max_keep) < int(selective_max_keep):
        raise ValueError("broad_max_keep must be at least selective_max_keep")

    gated = predictions.astype(np.float32, copy=True)
    scores = np.asarray(rank_signal, dtype=np.float32).reshape(-1)
    budget_values = np.asarray(budget_logits, dtype=np.float32)
    timestamps = pd.to_datetime(action_rows["timestamp"], utc=True)
    active_mask = _compute_active_mask(gated)
    keep_mask = np.zeros((len(scores),), dtype=bool)
    fractional_scales = np.zeros((len(scores),), dtype=np.float32)
    quantile = float(np.clip(reference_quantile, 0.0, 1.0))
    min_score_scales = np.asarray(
        [
            float(max(skip_min_score_scale, 0.0)),
            float(max(selective_min_score_scale, 0.0)),
            float(max(broad_min_score_scale, 0.0)),
        ],
        dtype=np.float32,
    )
    gap_scales = np.asarray(
        [
            float(np.clip(skip_gap_scale, 0.0, 1.0)),
            float(np.clip(selective_gap_scale, 0.0, 1.0)),
            float(np.clip(broad_gap_scale, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )

    for _, group_index in timestamps.groupby(timestamps, sort=False).groups.items():
        group_indices = np.asarray(group_index, dtype=np.int64)
        group_probs = F.softmax(torch.from_numpy(budget_values[group_indices]), dim=-1).mean(dim=0).cpu().numpy()
        group_min_score = float(
            np.clip(float(min_score) * float(np.dot(group_probs, min_score_scales)), 0.0, 1.0)
        )
        eligible = group_indices[np.logical_and(active_mask[group_indices], scores[group_indices] >= group_min_score)]
        if eligible.size == 0:
            continue

        top_k_float = _interpolate_count(
            group_probs,
            skip_value=int(base_min_keep),
            selective_value=int(selective_top_k),
            broad_value=int(broad_top_k),
        )
        max_keep_float = _interpolate_count(
            group_probs,
            skip_value=int(base_min_keep),
            selective_value=int(selective_max_keep),
            broad_value=int(broad_max_keep),
        )
        group_top_k = int(np.clip(math.ceil(top_k_float), int(base_min_keep), int(broad_top_k)))
        group_max_keep = float(np.clip(max_keep_float, float(base_min_keep), float(broad_max_keep)))
        gap_scale = float(np.dot(group_probs, gap_scales))

        eligible_scores = scores[eligible]
        order = np.argsort(eligible_scores)[::-1]
        prefiltered = eligible[order[:group_top_k]]
        prefiltered_scores = scores[prefiltered]

        if prefiltered_scores.size <= int(base_min_keep):
            keep_mask[prefiltered] = True
            continue

        anchor_score = float(np.quantile(prefiltered_scores.astype(np.float64, copy=False), quantile))
        top_score = float(prefiltered_scores[0])
        dynamic_threshold = max(group_min_score, top_score - gap_scale * (top_score - anchor_score))
        selected_mask = prefiltered_scores >= dynamic_threshold
        selected_indices = prefiltered[selected_mask]

        if selected_indices.size < int(base_min_keep):
            selected_indices = prefiltered[: int(base_min_keep)]

        full_keep = int(math.floor(group_max_keep))
        partial_keep = float(group_max_keep - float(full_keep))
        full_keep = int(np.clip(full_keep, int(base_min_keep), int(broad_max_keep)))

        primary_indices = selected_indices[:full_keep]
        keep_mask[primary_indices] = True

        if selected_indices.size > full_keep and partial_keep > 1e-6:
            fractional_scales[selected_indices[full_keep]] = max(
                fractional_scales[selected_indices[full_keep]],
                float(np.clip(partial_keep, 0.0, 1.0)),
            )
        if selected_indices.size > int(math.ceil(group_max_keep)):
            overflow = selected_indices[int(math.ceil(group_max_keep)) :]
            gated[overflow] = 0.0

    partial_mask = fractional_scales > 0.0
    inactive_mask = ~(keep_mask | partial_mask)
    gated[inactive_mask] = 0.0
    gated[partial_mask] *= fractional_scales[partial_mask, None]
    return gated.astype(np.float32, copy=False)
