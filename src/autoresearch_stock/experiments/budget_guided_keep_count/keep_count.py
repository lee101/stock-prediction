from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..cross_sectional_rank_gate.rank_gate import _compute_active_mask
from ..timestamp_budget_head import BUDGET_BROAD, BUDGET_SELECTIVE


def apply_budget_guided_rank_gate(
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
    quantile = float(np.clip(reference_quantile, 0.0, 1.0))

    for _, group_index in timestamps.groupby(timestamps, sort=False).groups.items():
        group_indices = np.asarray(group_index, dtype=np.int64)
        eligible = group_indices[np.logical_and(active_mask[group_indices], scores[group_indices] >= float(min_score))]
        if eligible.size == 0:
            continue

        group_probs = F.softmax(torch.from_numpy(budget_values[group_indices]), dim=-1).mean(dim=0).cpu().numpy()
        dominant_class = int(np.argmax(group_probs))
        if dominant_class == int(BUDGET_BROAD):
            group_top_k = int(broad_top_k)
            group_max_keep = int(broad_max_keep)
            group_gap_scale = float(np.clip(broad_gap_scale, 0.0, 1.0))
        elif dominant_class == int(BUDGET_SELECTIVE):
            group_top_k = int(selective_top_k)
            group_max_keep = int(selective_max_keep)
            group_gap_scale = float(np.clip(selective_gap_scale, 0.0, 1.0))
        else:
            group_top_k = int(base_min_keep)
            group_max_keep = int(base_min_keep)
            group_gap_scale = float(np.clip(skip_gap_scale, 0.0, 1.0))

        eligible_scores = scores[eligible]
        order = np.argsort(eligible_scores)[::-1]
        prefiltered = eligible[order[:group_top_k]]
        prefiltered_scores = scores[prefiltered]

        if prefiltered_scores.size <= int(base_min_keep):
            keep_mask[prefiltered] = True
            continue

        anchor_score = float(np.quantile(prefiltered_scores.astype(np.float64, copy=False), quantile))
        top_score = float(prefiltered_scores[0])
        dynamic_threshold = max(float(min_score), top_score - group_gap_scale * (top_score - anchor_score))
        selected_mask = prefiltered_scores >= dynamic_threshold
        selected_indices = prefiltered[selected_mask]

        if selected_indices.size < int(base_min_keep):
            selected_indices = prefiltered[: int(base_min_keep)]
        if selected_indices.size > int(group_max_keep):
            selected_indices = prefiltered[: int(group_max_keep)]
        keep_mask[selected_indices] = True

    gated[~keep_mask] = 0.0
    return gated.astype(np.float32, copy=False)
