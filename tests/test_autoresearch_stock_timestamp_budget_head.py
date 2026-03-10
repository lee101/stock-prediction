from __future__ import annotations

import numpy as np
import pandas as pd

from src.autoresearch_stock.experiments.timestamp_budget_head import (
    BUDGET_BROAD,
    BUDGET_SELECTIVE,
    BUDGET_SKIP,
    apply_budget_aware_soft_rank_sizing,
    derive_budget_labels,
)


def test_derive_budget_labels_groups_active_names_per_timestamp() -> None:
    sample_rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T10:00:00Z",
                    "2024-01-01T10:00:00Z",
                    "2024-01-01T11:00:00Z",
                    "2024-01-01T11:00:00Z",
                    "2024-01-01T12:00:00Z",
                    "2024-01-01T12:00:00Z",
                ],
                utc=True,
            ),
            "symbol": ["A", "B", "A", "B", "A", "B"],
        }
    )
    plan_labels = np.asarray([2, 2, 4, 2, 4, 1], dtype=np.int64)

    labels = derive_budget_labels(sample_rows, plan_labels, broad_count_threshold=2)

    assert labels.tolist() == [
        BUDGET_SKIP,
        BUDGET_SKIP,
        BUDGET_SELECTIVE,
        BUDGET_SELECTIVE,
        BUDGET_BROAD,
        BUDGET_BROAD,
    ]


def test_apply_budget_aware_soft_rank_sizing_uses_budget_regime_signal() -> None:
    predictions = np.asarray(
        [
            [0.05, -0.01, 0.03],
            [0.04, -0.01, 0.025],
            [0.05, -0.01, 0.03],
            [0.04, -0.01, 0.025],
        ],
        dtype=np.float32,
    )
    action_rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T10:00:00Z",
                    "2024-01-01T10:00:00Z",
                    "2024-01-01T11:00:00Z",
                    "2024-01-01T11:00:00Z",
                ],
                utc=True,
            ),
            "symbol": ["A", "B", "A", "B"],
        }
    )
    rank_signal = np.asarray([0.92, 0.75, 0.92, 0.75], dtype=np.float32)
    budget_logits = np.asarray(
        [
            [8.0, -8.0, -8.0],
            [8.0, -8.0, -8.0],
            [-8.0, -8.0, 8.0],
            [-8.0, -8.0, 8.0],
        ],
        dtype=np.float32,
    )

    scaled = apply_budget_aware_soft_rank_sizing(
        predictions,
        action_rows,
        rank_signal,
        budget_logits,
        min_score=0.20,
        reference_quantile=0.25,
        regime_floor_scale=0.75,
        name_floor_scale=0.55,
        regime_power=0.75,
        rank_power=1.5,
        budget_skip_scale=0.55,
        budget_selective_scale=0.82,
    )

    weak_hour_scale = float(scaled[0, 0] / predictions[0, 0])
    strong_hour_scale = float(scaled[2, 0] / predictions[2, 0])
    assert strong_hour_scale > weak_hour_scale
