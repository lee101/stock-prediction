from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..ambiguity_plan_gate import (
    PLAN_FLAT,
    PLAN_LONG_STRONG,
    PLAN_LONG_WEAK,
    PLAN_SHORT_STRONG,
    PLAN_SHORT_WEAK,
)


def _compute_best_edge(
    features: np.ndarray,
    targets: np.ndarray,
    symbol_ids: np.ndarray,
    *,
    spread_feature_index: int,
    symbol_fee_rates: np.ndarray,
    entry_slippage_bps: float,
    exit_slippage_bps: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(targets) == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty

    spread_bps = np.maximum(features[:, -1, spread_feature_index], 0.0).astype(np.float32, copy=False) * 100.0
    spread_frac = spread_bps / 1e4
    fee_rates = symbol_fee_rates[symbol_ids.astype(np.int64, copy=False)]
    round_trip_cost = (
        spread_frac
        + (float(entry_slippage_bps) + float(exit_slippage_bps)) / 1e4
        + 2.0 * fee_rates
    )

    zeros = np.zeros((len(targets),), dtype=np.float32)
    upside = np.maximum.reduce([targets[:, 0], targets[:, 2], zeros]).astype(np.float32, copy=False)
    downside = np.maximum.reduce([-targets[:, 1], -targets[:, 2], zeros]).astype(np.float32, copy=False)
    best_edge = np.maximum(upside - round_trip_cost, downside - round_trip_cost)
    return best_edge.astype(np.float32, copy=False), spread_frac.astype(np.float32, copy=False)


def compute_edge_quantiles(
    features: np.ndarray,
    targets: np.ndarray,
    symbol_ids: np.ndarray,
    *,
    spread_feature_index: int,
    symbol_fee_rates: np.ndarray,
    min_edge_bps: float,
    entry_slippage_bps: float,
    exit_slippage_bps: float,
    floor_quantile: float = 0.25,
    ceiling_quantile: float = 0.75,
) -> tuple[float, float]:
    best_edge, _ = _compute_best_edge(
        features,
        targets,
        symbol_ids,
        spread_feature_index=spread_feature_index,
        symbol_fee_rates=symbol_fee_rates,
        entry_slippage_bps=entry_slippage_bps,
        exit_slippage_bps=exit_slippage_bps,
    )
    if len(best_edge) == 0:
        floor = float(min_edge_bps) / 1e4
        return floor, floor + 1e-4
    floor = max(
        float(min_edge_bps) / 1e4,
        float(np.quantile(best_edge.astype(np.float64, copy=False), np.clip(float(floor_quantile), 0.0, 1.0))),
    )
    ceiling = float(
        np.quantile(
            best_edge.astype(np.float64, copy=False),
            max(np.clip(float(ceiling_quantile), 0.0, 1.0), np.clip(float(floor_quantile), 0.0, 1.0)),
        )
    )
    return floor, max(ceiling, floor + 1e-4)


def derive_margin_targets(
    features: np.ndarray,
    targets: np.ndarray,
    symbol_ids: np.ndarray,
    plan_labels: np.ndarray,
    *,
    spread_feature_index: int,
    volatility_feature_index: int,
    symbol_fee_rates: np.ndarray,
    entry_slippage_bps: float,
    exit_slippage_bps: float,
    edge_floor: float,
    edge_ceiling: float,
    spread_margin_scale: float = 4.0,
    volatility_margin_scale: float = 0.5,
) -> np.ndarray:
    if len(targets) == 0:
        return np.zeros((0,), dtype=np.float32)

    best_edge, spread_frac = _compute_best_edge(
        features,
        targets,
        symbol_ids,
        spread_feature_index=spread_feature_index,
        symbol_fee_rates=symbol_fee_rates,
        entry_slippage_bps=entry_slippage_bps,
        exit_slippage_bps=exit_slippage_bps,
    )
    volatility = np.clip(
        np.nan_to_num(features[:, -1, volatility_feature_index], nan=0.0, posinf=0.2, neginf=0.0),
        0.0,
        0.2,
    ).astype(np.float32, copy=False)
    margin_threshold = (
        float(edge_floor)
        + spread_frac * float(spread_margin_scale)
        + volatility * float(volatility_margin_scale)
    )
    span = max(float(edge_ceiling) - float(edge_floor), 1e-4)
    margin_targets = np.clip((best_edge - margin_threshold) / span, 0.0, 1.0).astype(np.float32, copy=False)
    active_mask = (plan_labels.astype(np.int64, copy=False) != PLAN_FLAT).astype(np.float32, copy=False)
    return (margin_targets * active_mask).astype(np.float32, copy=False)


def apply_cost_margin_gate(
    raw_predictions: torch.Tensor,
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
    return raw_predictions * (plan_scale * margin_scale).unsqueeze(-1)
