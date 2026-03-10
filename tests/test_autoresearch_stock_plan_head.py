from __future__ import annotations

import numpy as np
import torch

from src.autoresearch_stock.experiments.ambiguity_plan_gate.plan_head import (
    PLAN_FLAT,
    PLAN_LONG_STRONG,
    PLAN_LONG_WEAK,
    PLAN_SHORT_STRONG,
    apply_plan_gate,
    derive_plan_labels,
)


def test_derive_plan_labels_respects_cost_and_ambiguity() -> None:
    features = np.zeros((4, 2, 1), dtype=np.float32)
    features[:, -1, 0] = 0.02
    targets = np.asarray(
        [
            [0.020, -0.002, 0.015],
            [0.010, -0.002, 0.006],
            [0.004, -0.003, 0.001],
            [0.003, -0.020, -0.015],
        ],
        dtype=np.float32,
    )
    labels = derive_plan_labels(
        features,
        targets,
        np.zeros((4,), dtype=np.int64),
        spread_feature_index=0,
        symbol_fee_rates=np.zeros((1,), dtype=np.float32),
        min_edge_bps=4.0,
        entry_slippage_bps=1.0,
        exit_slippage_bps=1.0,
        ambiguity_floor=0.005,
        strong_gap_multiplier=2.0,
    )

    assert labels.tolist() == [
        PLAN_LONG_STRONG,
        PLAN_LONG_WEAK,
        PLAN_FLAT,
        PLAN_SHORT_STRONG,
    ]


def test_apply_plan_gate_scales_weak_and_flat_actions() -> None:
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
            [8.0, -8.0, -8.0, -8.0, -8.0],
            [-8.0, -8.0, 8.0, -8.0, -8.0],
        ],
        dtype=torch.float32,
    )

    gated = apply_plan_gate(raw_predictions, plan_logits, weak_action_scale=0.4)

    assert torch.allclose(gated[0], raw_predictions[0], atol=1e-4)
    assert torch.allclose(gated[1], raw_predictions[1] * 0.4, atol=1e-4)
    assert torch.allclose(gated[2], torch.zeros_like(raw_predictions[2]), atol=1e-4)
