from __future__ import annotations

import numpy as np
import pandas as pd

from src.autoresearch_stock.experiments.budget_consensus_dispersion import apply_budget_consensus_dispersion_gate


def test_budget_consensus_dispersion_gate_keeps_consensus_broad_hours_wider_than_disagreed_broad_hours() -> None:
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
    rank_signal = np.asarray([0.95, 0.92, 0.89, 0.84, 0.80, 0.95, 0.92, 0.89, 0.84, 0.80], dtype=np.float32)
    budget_logits = np.asarray(
        [
            [-8.0, -8.0, 8.0],
            [-8.0, -8.0, 8.0],
            [-8.0, -8.0, 8.0],
            [-8.0, -8.0, 8.0],
            [-8.0, -8.0, 8.0],
            [-8.0, -8.0, 8.0],
            [-8.0, 8.0, -8.0],
            [-8.0, -8.0, 8.0],
            [8.0, -8.0, -8.0],
            [-8.0, 8.0, -8.0],
        ],
        dtype=np.float32,
    )

    gated = apply_budget_consensus_dispersion_gate(
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
        broad_consensus_power=1.6,
        selective_reallocation=0.85,
        consensus_gap_floor=0.80,
        consensus_fractional_floor=0.75,
    )

    consensus_count = int(np.count_nonzero(np.any(np.abs(gated[:5]) > 1e-9, axis=1)))
    disagreed_count = int(np.count_nonzero(np.any(np.abs(gated[5:]) > 1e-9, axis=1)))

    assert consensus_count > disagreed_count
    assert consensus_count >= 3
    assert disagreed_count >= 1


def test_budget_consensus_dispersion_gate_preserves_confident_skip_concentration() -> None:
    predictions = np.asarray(
        [
            [0.05, -0.01, 0.03],
            [0.045, -0.01, 0.03],
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
                    "2024-01-01T10:00:00Z",
                ],
                utc=True,
            ),
            "symbol": ["A", "B", "C"],
        }
    )
    rank_signal = np.asarray([0.95, 0.90, 0.85], dtype=np.float32)
    budget_logits = np.asarray(
        [
            [8.0, -8.0, -8.0],
            [8.0, -8.0, -8.0],
            [8.0, -8.0, -8.0],
        ],
        dtype=np.float32,
    )

    gated = apply_budget_consensus_dispersion_gate(
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
        broad_consensus_power=1.6,
        selective_reallocation=0.85,
        consensus_gap_floor=0.80,
        consensus_fractional_floor=0.75,
    )

    kept_count = int(np.count_nonzero(np.any(np.abs(gated) > 1e-9, axis=1)))
    assert kept_count == 1
