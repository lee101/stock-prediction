from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

PLAN_SHORT_WEAK = 0
PLAN_SHORT_STRONG = 1
PLAN_FLAT = 2
PLAN_LONG_WEAK = 3
PLAN_LONG_STRONG = 4
PLAN_LABEL_COUNT = 5


def derive_plan_labels(
    features: np.ndarray,
    targets: np.ndarray,
    symbol_ids: np.ndarray,
    *,
    spread_feature_index: int,
    symbol_fee_rates: np.ndarray,
    min_edge_bps: float,
    entry_slippage_bps: float,
    exit_slippage_bps: float,
    ambiguity_floor: float,
    strong_gap_multiplier: float = 2.0,
) -> np.ndarray:
    if len(targets) == 0:
        return np.zeros((0,), dtype=np.int64)

    spread_bps = np.maximum(features[:, -1, spread_feature_index], 0.0).astype(np.float32, copy=False) * 100.0
    fee_rates = symbol_fee_rates[symbol_ids.astype(np.int64, copy=False)]
    round_trip_cost = (
        spread_bps / 1e4
        + (float(entry_slippage_bps) + float(exit_slippage_bps)) / 1e4
        + 2.0 * fee_rates
    )

    zeros = np.zeros((len(targets),), dtype=np.float32)
    upside = np.maximum.reduce([targets[:, 0], targets[:, 2], zeros]).astype(np.float32, copy=False)
    downside = np.maximum.reduce([-targets[:, 1], -targets[:, 2], zeros]).astype(np.float32, copy=False)
    direction_gap = np.abs(upside - downside)
    long_edge = upside - round_trip_cost
    short_edge = downside - round_trip_cost
    best_edge = np.maximum(long_edge, short_edge)
    tradable = best_edge > (float(min_edge_bps) / 1e4)

    weak_gap = max(float(ambiguity_floor), 1e-4)
    strong_gap = weak_gap * max(float(strong_gap_multiplier), 1.0)
    labels = np.full((len(targets),), PLAN_FLAT, dtype=np.int64)
    long_side = long_edge >= short_edge
    strong_mask = tradable & (direction_gap > strong_gap)
    weak_mask = tradable & ~strong_mask & (direction_gap > weak_gap)

    labels[weak_mask & long_side] = PLAN_LONG_WEAK
    labels[strong_mask & long_side] = PLAN_LONG_STRONG
    labels[weak_mask & ~long_side] = PLAN_SHORT_WEAK
    labels[strong_mask & ~long_side] = PLAN_SHORT_STRONG
    return labels


def derive_plan_class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels.astype(np.int64, copy=False), minlength=PLAN_LABEL_COUNT).astype(np.float32, copy=False)
    valid = counts > 0.0
    weights = np.ones((PLAN_LABEL_COUNT,), dtype=np.float32)
    if not np.any(valid):
        return weights
    weights[valid] = np.sqrt(float(counts[valid].sum()) / (counts[valid] * float(np.count_nonzero(valid))))
    weights /= float(weights[valid].mean())
    return weights.astype(np.float32, copy=False)


def apply_plan_gate(
    raw_predictions: torch.Tensor,
    plan_logits: torch.Tensor,
    *,
    weak_action_scale: float,
) -> torch.Tensor:
    plan_probs = F.softmax(plan_logits, dim=-1)
    trade_scale = (
        plan_probs[:, PLAN_SHORT_STRONG]
        + plan_probs[:, PLAN_LONG_STRONG]
        + float(weak_action_scale) * (plan_probs[:, PLAN_SHORT_WEAK] + plan_probs[:, PLAN_LONG_WEAK])
    )
    return raw_predictions * trade_scale.unsqueeze(-1)
