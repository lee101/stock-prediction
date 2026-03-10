from __future__ import annotations

import numpy as np
import pandas as pd

from src.autoresearch_stock.experiments.continuous_budget_thresholds import apply_continuous_budget_gate


def test_apply_continuous_budget_gate_expands_broad_hours_and_constrains_skip_hours() -> None:
    predictions = np.asarray(
        [
            [0.05, -0.01, 0.03],
            [0.045, -0.01, 0.03],
            [0.04, -0.01, 0.025],
            [0.035, -0.01, 0.02],
            [0.03, -0.01, 0.02],
            [0.05, -0.01, 0.03],
            [0.045, -0.01, 0.03],
            [0.04, -0.01, 0.025],
            [0.035, -0.01, 0.02],
            [0.03, -0.01, 0.02],
        ],
        dtype=np.float32,
    )
    action_rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T10:00:00Z",
                    "2024-01-01T10:00:00Z",
                    "2024-01-01T10:00:00Z",
                    "2024-01-01T10:00:00Z",
                    "2024-01-01T10:00:00Z",
                    "2024-01-01T11:00:00Z",
                    "2024-01-01T11:00:00Z",
                    "2024-01-01T11:00:00Z",
                    "2024-01-01T11:00:00Z",
                    "2024-01-01T11:00:00Z",
                ],
                utc=True,
            ),
            "symbol": ["A", "B", "C", "D", "E", "A", "B", "C", "D", "E"],
        }
    )
    rank_signal = np.asarray([0.95, 0.90, 0.86, 0.80, 0.74, 0.95, 0.90, 0.86, 0.80, 0.74], dtype=np.float32)
    budget_logits = np.asarray(
        [
            [-8.0, -8.0, 8.0],
            [-8.0, -8.0, 8.0],
            [-8.0, -8.0, 8.0],
            [-8.0, -8.0, 8.0],
            [-8.0, -8.0, 8.0],
            [8.0, -8.0, -8.0],
            [8.0, -8.0, -8.0],
            [8.0, -8.0, -8.0],
            [8.0, -8.0, -8.0],
            [8.0, -8.0, -8.0],
        ],
        dtype=np.float32,
    )

    gated = apply_continuous_budget_gate(
        predictions,
        action_rows,
        rank_signal,
        budget_logits,
        min_score=0.20,
        base_min_keep=1,
        selective_top_k=3,
        selective_max_keep=2,
        broad_top_k=5,
        broad_max_keep=4,
        reference_quantile=0.25,
        selective_gap_scale=0.50,
        broad_gap_scale=0.75,
        skip_gap_scale=0.35,
        skip_min_score_scale=1.10,
        selective_min_score_scale=1.00,
        broad_min_score_scale=0.80,
    )

    broad_count = int(np.count_nonzero(np.any(np.abs(gated[:5]) > 1e-9, axis=1)))
    skip_count = int(np.count_nonzero(np.any(np.abs(gated[5:]) > 1e-9, axis=1)))

    assert broad_count > skip_count
    assert broad_count >= 3
    assert skip_count == 1


def test_apply_continuous_budget_gate_partially_scales_marginal_candidates() -> None:
    predictions = np.asarray(
        [
            [0.05, -0.01, 0.03],
            [0.04, -0.01, 0.025],
            [0.03, -0.01, 0.02],
        ],
        dtype=np.float32,
    )
    action_rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T10:00:00Z",
                    "2024-01-01T10:00:00Z",
                    "2024-01-01T10:00:00Z",
                ],
                utc=True,
            ),
            "symbol": ["A", "B", "C"],
        }
    )
    rank_signal = np.asarray([0.95, 0.94, 0.80], dtype=np.float32)
    budget_logits = np.asarray(
        [
            [0.0, 0.0, -20.0],
            [0.0, 0.0, -20.0],
            [0.0, 0.0, -20.0],
        ],
        dtype=np.float32,
    )

    gated = apply_continuous_budget_gate(
        predictions,
        action_rows,
        rank_signal,
        budget_logits,
        min_score=0.20,
        base_min_keep=1,
        selective_top_k=3,
        selective_max_keep=2,
        broad_top_k=5,
        broad_max_keep=4,
        reference_quantile=0.0,
        selective_gap_scale=1.0,
        broad_gap_scale=1.0,
        skip_gap_scale=1.0,
        skip_min_score_scale=1.0,
        selective_min_score_scale=1.0,
        broad_min_score_scale=1.0,
    )

    assert np.allclose(gated[0], predictions[0])
    assert np.all(np.abs(gated[1]) > 0.0)
    assert np.all(np.abs(gated[1]) < np.abs(predictions[1]))
    assert np.allclose(gated[2], 0.0)
