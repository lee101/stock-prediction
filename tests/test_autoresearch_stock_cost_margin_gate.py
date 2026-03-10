from __future__ import annotations

import numpy as np
import torch

from src.autoresearch_stock.experiments.ambiguity_plan_gate import PLAN_FLAT, PLAN_LONG_STRONG, PLAN_LONG_WEAK
from src.autoresearch_stock.experiments.cost_margin_gate import (
    apply_cost_margin_gate,
    compute_edge_quantiles,
    derive_margin_targets,
)


def test_derive_margin_targets_tracks_excess_edge() -> None:
    features = np.zeros((3, 2, 2), dtype=np.float32)
    features[:, -1, 0] = 0.02
    features[:, -1, 1] = np.asarray([0.004, 0.012, 0.006], dtype=np.float32)
    targets = np.asarray(
        [
            [0.080, -0.002, 0.055],
            [0.050, -0.002, 0.035],
            [0.015, -0.003, 0.010],
        ],
        dtype=np.float32,
    )
    symbol_ids = np.zeros((3,), dtype=np.int64)
    plan_labels = np.asarray([PLAN_LONG_STRONG, PLAN_LONG_WEAK, PLAN_FLAT], dtype=np.int64)

    edge_floor, edge_ceiling = compute_edge_quantiles(
        features,
        targets,
        symbol_ids,
        spread_feature_index=0,
        symbol_fee_rates=np.zeros((1,), dtype=np.float32),
        min_edge_bps=4.0,
        entry_slippage_bps=1.0,
        exit_slippage_bps=1.0,
        floor_quantile=0.25,
        ceiling_quantile=0.75,
    )
    margin_targets = derive_margin_targets(
        features,
        targets,
        symbol_ids,
        plan_labels,
        spread_feature_index=0,
        volatility_feature_index=1,
        symbol_fee_rates=np.zeros((1,), dtype=np.float32),
        entry_slippage_bps=1.0,
        exit_slippage_bps=1.0,
        edge_floor=edge_floor,
        edge_ceiling=edge_ceiling,
        spread_margin_scale=4.0,
        volatility_margin_scale=0.5,
    )

    assert margin_targets[0] > margin_targets[1] > 0.0
    assert margin_targets[2] == 0.0


def test_apply_cost_margin_gate_combines_plan_and_margin() -> None:
    raw_predictions = torch.tensor(
        [
            [0.02, -0.01, 0.01],
            [0.02, -0.01, 0.01],
            [0.02, -0.01, 0.01],
        ],
        dtype=torch.float32,
    )
    plan_logits = torch.tensor(
        [
            [-8.0, -8.0, -8.0, -8.0, 8.0],
            [-8.0, -8.0, -8.0, 8.0, -8.0],
            [-8.0, -8.0, 8.0, -8.0, -8.0],
        ],
        dtype=torch.float32,
    )
    margin_logits = torch.tensor([[8.0], [-8.0], [8.0]], dtype=torch.float32)

    gated = apply_cost_margin_gate(
        raw_predictions,
        plan_logits,
        margin_logits,
        weak_action_scale=0.4,
    )

    assert torch.allclose(gated[0], raw_predictions[0], atol=5e-4)
    assert torch.allclose(gated[1], raw_predictions[1] * (0.4 * torch.sigmoid(torch.tensor(-8.0))), atol=5e-4)
    assert torch.allclose(gated[2], torch.zeros_like(raw_predictions[2]), atol=1e-4)
