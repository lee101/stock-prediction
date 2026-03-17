from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from binanceneural.inference import generate_actions_from_frame
from binanceneural.model import BinanceHourlyPolicy

from .config import ExperimentConfig
from .data import FeatureNormalizer


@dataclass
class AggregationResult:
    aggregated: pd.DataFrame
    per_context: List[pd.DataFrame]


def _trimmed_mean(values: np.ndarray, trim_ratio: float) -> float:
    if values.size == 0:
        return float("nan")
    if values.size < 3 or trim_ratio <= 0:
        return float(np.mean(values))
    k = int(values.size * trim_ratio)
    if k == 0 or k * 2 >= values.size:
        return float(np.mean(values))
    sorted_vals = np.sort(values)
    trimmed = sorted_vals[k:-k]
    return float(np.mean(trimmed))


def aggregate_actions(
    actions_list: Sequence[pd.DataFrame],
    *,
    trim_ratio: float = 0.2,
) -> pd.DataFrame:
    if not actions_list:
        return pd.DataFrame()
    merged = None
    action_fields = (
        "buy_price",
        "sell_price",
        "buy_amount",
        "sell_amount",
        "trade_amount",
        "allocation_fraction",
        "hold_hours",
    )
    for idx, frame in enumerate(actions_list):
        if frame.empty:
            continue
        suffix = f"_c{idx}"
        rename_cols = {field: f"{field}{suffix}" for field in action_fields if field in frame.columns}
        if not rename_cols:
            continue
        subset = frame[["timestamp", "symbol", *rename_cols.keys()]].rename(columns=rename_cols)
        if merged is None:
            merged = subset
        else:
            merged = merged.merge(subset, on=["timestamp", "symbol"], how="inner")
    if merged is None or merged.empty:
        return pd.DataFrame()

    out_rows = []
    value_cols = {field: [] for field in action_fields}
    for col in merged.columns:
        for key in value_cols:
            if col.startswith(key):
                value_cols[key].append(col)

    for row in merged.itertuples(index=False):
        row_dict = {"timestamp": getattr(row, "timestamp"), "symbol": getattr(row, "symbol")}
        for key, cols in value_cols.items():
            if not cols:
                continue
            values = np.array([float(getattr(row, col)) for col in cols], dtype=np.float64)
            row_dict[key] = _trimmed_mean(values, trim_ratio)
        out_rows.append(row_dict)
    return pd.DataFrame(out_rows)


def blend_actions(
    actions_list: Sequence[pd.DataFrame],
    *,
    weights: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    if not actions_list:
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    weight_values: List[float] = []
    if weights is None:
        weights = [1.0] * len(actions_list)
    if len(weights) != len(actions_list):
        raise ValueError("Weights length must match actions list length.")
    for frame, weight in zip(actions_list, weights):
        if frame is None or frame.empty:
            continue
        frames.append(frame)
        weight_values.append(float(weight))
    if not frames:
        return pd.DataFrame()
    weight_array = np.asarray(weight_values, dtype=np.float64)
    if np.allclose(weight_array, 0.0):
        raise ValueError("At least one blend weight must be non-zero.")

    merged = None
    suffixes: List[str] = []
    action_fields = (
        "buy_price",
        "sell_price",
        "buy_amount",
        "sell_amount",
        "trade_amount",
        "allocation_fraction",
        "hold_hours",
    )
    for idx, frame in enumerate(frames):
        suffix = f"_b{idx}"
        rename_cols = {field: f"{field}{suffix}" for field in action_fields if field in frame.columns}
        if not rename_cols:
            continue
        suffixes.append(suffix)
        subset = frame[["timestamp", "symbol", *rename_cols.keys()]].rename(columns=rename_cols)
        if merged is None:
            merged = subset
        else:
            merged = merged.merge(subset, on=["timestamp", "symbol"], how="inner")
    if merged is None or merged.empty:
        return pd.DataFrame()

    value_cols = {
        field: [f"{field}{suffix}" for suffix in suffixes if f"{field}{suffix}" in merged.columns]
        for field in action_fields
    }

    out_rows = []
    for row in merged.itertuples(index=False):
        row_dict = {"timestamp": getattr(row, "timestamp"), "symbol": getattr(row, "symbol")}
        for key, cols in value_cols.items():
            if not cols:
                continue
            values = np.array([float(getattr(row, col)) for col in cols], dtype=np.float64)
            row_dict[key] = _weighted_mean(values, weight_array)
        out_rows.append(row_dict)
    return pd.DataFrame(out_rows)


def generate_actions_multi_context(
    *,
    model: BinanceHourlyPolicy,
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    normalizer: FeatureNormalizer,
    base_sequence_length: int,
    horizon: int,
    experiment: Optional[ExperimentConfig] = None,
    device: Optional[torch.device] = None,
) -> AggregationResult:
    cfg = experiment or ExperimentConfig()
    per_context: List[pd.DataFrame] = []
    for ctx in cfg.context_lengths:
        seq_len = min(int(ctx), int(base_sequence_length))
        if seq_len < 2:
            continue
        actions = generate_actions_from_frame(
            model=model,
            frame=frame,
            feature_columns=feature_columns,
            normalizer=normalizer,
            sequence_length=seq_len,
            horizon=horizon,
            device=device,
        )
        per_context.append(actions)
    aggregated = aggregate_actions(per_context, trim_ratio=cfg.trim_ratio)
    return AggregationResult(aggregated=aggregated, per_context=per_context)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    mask = np.isfinite(values)
    if not mask.any():
        return float("nan")
    w = weights[mask]
    if np.allclose(w.sum(), 0.0):
        return float("nan")
    return float(np.sum(values[mask] * w) / w.sum())


__all__ = ["AggregationResult", "aggregate_actions", "blend_actions", "generate_actions_multi_context"]
