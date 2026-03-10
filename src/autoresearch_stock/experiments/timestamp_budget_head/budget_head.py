from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..ambiguity_plan_gate import PLAN_FLAT
from ..soft_rank_sizing import apply_soft_rank_sizing
from ...prepare import (
    TaskConfig,
    _build_feature_frame,
    _build_sequence_block,
    _feature_names_for_frequency,
    _load_symbol_bars,
)

BUDGET_SKIP = 0
BUDGET_SELECTIVE = 1
BUDGET_BROAD = 2
BUDGET_LABEL_COUNT = 3


def rebuild_split_sample_rows(
    config: TaskConfig,
    *,
    spread_profile_bps: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames_by_symbol: dict[str, pd.DataFrame] = {}
    feature_names = _feature_names_for_frequency(config.frequency)

    for symbol in config.symbols:
        raw = _load_symbol_bars(symbol, config)
        enriched = _build_feature_frame(raw, config, float(spread_profile_bps[str(symbol).upper()]))
        frames_by_symbol[str(symbol).upper()] = enriched

    all_timestamps = sorted(
        {
            pd.Timestamp(ts)
            for frame in frames_by_symbol.values()
            for ts in frame["timestamp"].tolist()
        }
    )
    reserve = max(config.eval_windows) + config.hold_bars + config.decision_lag_bars
    train_end_ts = all_timestamps[-reserve - 1]
    train_candidate_ts = [ts for ts in all_timestamps if ts <= train_end_ts]
    val_start_idx = max(config.sequence_length + 1, int(len(train_candidate_ts) * 0.85))
    val_start_ts = train_candidate_ts[val_start_idx]

    train_rows_blocks: list[pd.DataFrame] = []
    val_rows_blocks: list[pd.DataFrame] = []
    for symbol in config.symbols:
        frame = frames_by_symbol[str(symbol).upper()]
        ts = pd.to_datetime(frame["timestamp"], utc=True)
        train_mask = (ts <= train_end_ts).to_numpy(dtype=bool) & (ts < val_start_ts).to_numpy(dtype=bool)
        val_mask = (ts <= train_end_ts).to_numpy(dtype=bool) & (ts >= val_start_ts).to_numpy(dtype=bool)

        _, _, train_rows = _build_sequence_block(
            frame,
            feature_names=feature_names,
            sequence_length=config.sequence_length,
            row_mask=train_mask,
            include_targets=True,
        )
        _, _, val_rows = _build_sequence_block(
            frame,
            feature_names=feature_names,
            sequence_length=config.sequence_length,
            row_mask=val_mask,
            include_targets=True,
        )
        if not train_rows.empty:
            train_rows_blocks.append(train_rows.reset_index(drop=True))
        if not val_rows.empty:
            val_rows_blocks.append(val_rows.reset_index(drop=True))

    train_rows = pd.concat(train_rows_blocks, ignore_index=True) if train_rows_blocks else pd.DataFrame()
    val_rows = pd.concat(val_rows_blocks, ignore_index=True) if val_rows_blocks else pd.DataFrame()
    return train_rows, val_rows


def derive_budget_labels(
    sample_rows: pd.DataFrame,
    plan_labels: np.ndarray,
    *,
    broad_count_threshold: int = 2,
) -> np.ndarray:
    if len(sample_rows) != len(plan_labels):
        raise ValueError("sample_rows and plan_labels must have matching lengths")
    if len(sample_rows) == 0:
        return np.zeros((0,), dtype=np.int64)
    if int(broad_count_threshold) < 2:
        raise ValueError("broad_count_threshold must be at least 2")

    labels = np.full((len(sample_rows),), BUDGET_SKIP, dtype=np.int64)
    timestamps = pd.to_datetime(sample_rows["timestamp"], utc=True)
    active_mask = np.asarray(plan_labels, dtype=np.int64) != int(PLAN_FLAT)

    for _, group_index in timestamps.groupby(timestamps, sort=False).groups.items():
        group_indices = np.asarray(group_index, dtype=np.int64)
        active_count = int(np.count_nonzero(active_mask[group_indices]))
        if active_count >= int(broad_count_threshold):
            labels[group_indices] = BUDGET_BROAD
        elif active_count >= 1:
            labels[group_indices] = BUDGET_SELECTIVE

    return labels.astype(np.int64, copy=False)


def derive_budget_class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels.astype(np.int64, copy=False), minlength=BUDGET_LABEL_COUNT).astype(
        np.float32,
        copy=False,
    )
    valid = counts > 0.0
    weights = np.ones((BUDGET_LABEL_COUNT,), dtype=np.float32)
    if not np.any(valid):
        return weights
    weights[valid] = np.sqrt(float(counts[valid].sum()) / (counts[valid] * float(np.count_nonzero(valid))))
    weights /= float(weights[valid].mean())
    return weights.astype(np.float32, copy=False)


def _compute_budget_scales(
    budget_logits: np.ndarray,
    action_rows: pd.DataFrame,
    *,
    skip_scale: float,
    selective_scale: float,
) -> np.ndarray:
    if len(budget_logits) != len(action_rows):
        raise ValueError("budget_logits and action_rows must have matching lengths")
    if len(budget_logits) == 0:
        return np.zeros((0,), dtype=np.float32)

    probs = F.softmax(torch.from_numpy(budget_logits.astype(np.float32, copy=False)), dim=-1).cpu().numpy()
    scales = np.ones((len(action_rows),), dtype=np.float32)
    timestamps = pd.to_datetime(action_rows["timestamp"], utc=True)
    skip_weight = float(np.clip(skip_scale, 0.0, 1.0))
    selective_weight = float(np.clip(selective_scale, skip_weight, 1.0))

    for _, group_index in timestamps.groupby(timestamps, sort=False).groups.items():
        group_indices = np.asarray(group_index, dtype=np.int64)
        group_probs = probs[group_indices].mean(axis=0)
        group_scale = (
            skip_weight * float(group_probs[BUDGET_SKIP])
            + selective_weight * float(group_probs[BUDGET_SELECTIVE])
            + float(group_probs[BUDGET_BROAD])
        )
        scales[group_indices] = float(np.clip(group_scale, skip_weight, 1.0))
    return scales.astype(np.float32, copy=False)


def apply_budget_aware_soft_rank_sizing(
    predictions: np.ndarray,
    action_rows: pd.DataFrame,
    rank_signal: np.ndarray,
    budget_logits: np.ndarray,
    *,
    min_score: float,
    reference_quantile: float,
    regime_floor_scale: float,
    name_floor_scale: float,
    regime_power: float,
    rank_power: float,
    budget_skip_scale: float,
    budget_selective_scale: float,
) -> np.ndarray:
    scaled = apply_soft_rank_sizing(
        predictions,
        action_rows,
        rank_signal,
        min_score=min_score,
        reference_quantile=reference_quantile,
        regime_floor_scale=regime_floor_scale,
        name_floor_scale=name_floor_scale,
        regime_power=regime_power,
        rank_power=rank_power,
    )
    budget_scales = _compute_budget_scales(
        budget_logits,
        action_rows,
        skip_scale=budget_skip_scale,
        selective_scale=budget_selective_scale,
    )
    return (scaled * budget_scales[:, None]).astype(np.float32, copy=False)
