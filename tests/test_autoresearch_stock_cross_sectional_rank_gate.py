from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.autoresearch_stock.experiments.cross_sectional_rank_gate import (
    apply_dynamic_cross_sectional_gate,
    apply_topk_cross_sectional_gate,
    compute_cross_sectional_rank_signal,
)
from src.autoresearch_stock.experiments.dynamic_score_floor import apply_dynamic_score_floor_gate
from src.autoresearch_stock.experiments.soft_rank_sizing import apply_soft_rank_sizing


def test_compute_cross_sectional_rank_signal_tracks_plan_and_margin() -> None:
    plan_logits = torch.tensor(
        [
            [-8.0, -8.0, -8.0, -8.0, 8.0],
            [-8.0, -8.0, -8.0, 8.0, -8.0],
            [-8.0, -8.0, 8.0, -8.0, -8.0],
        ],
        dtype=torch.float32,
    )
    margin_logits = torch.tensor([[8.0], [0.0], [8.0]], dtype=torch.float32)

    signal = compute_cross_sectional_rank_signal(
        plan_logits,
        margin_logits,
        weak_action_scale=0.4,
    )

    assert float(signal[0]) > float(signal[1]) > float(signal[2])


def test_apply_topk_cross_sectional_gate_keeps_top_scores_per_timestamp() -> None:
    predictions = np.asarray(
        [
            [0.04, -0.01, 0.02],
            [0.03, -0.01, 0.02],
            [0.02, -0.01, 0.01],
            [0.05, -0.01, 0.03],
            [0.03, -0.01, 0.02],
            [0.01, -0.01, 0.00],
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
                    "2024-01-01T11:00:00Z",
                    "2024-01-01T11:00:00Z",
                    "2024-01-01T11:00:00Z",
                ],
                utc=True,
            ),
            "symbol": ["A", "B", "C", "A", "B", "C"],
        }
    )
    rank_signal = np.asarray([0.95, 0.70, 0.10, 0.90, 0.65, 0.30], dtype=np.float32)

    gated = apply_topk_cross_sectional_gate(
        predictions,
        action_rows,
        rank_signal,
        top_k=2,
        min_score=0.20,
    )

    assert np.allclose(gated[0], predictions[0])
    assert np.allclose(gated[1], predictions[1])
    assert np.allclose(gated[2], 0.0)
    assert np.allclose(gated[3], predictions[3])
    assert np.allclose(gated[4], predictions[4])
    assert np.allclose(gated[5], 0.0)


def test_apply_dynamic_cross_sectional_gate_uses_score_dispersion() -> None:
    predictions = np.asarray(
        [
            [0.04, -0.01, 0.02],
            [0.03, -0.01, 0.02],
            [0.02, -0.01, 0.01],
            [0.05, -0.01, 0.03],
            [0.045, -0.01, 0.03],
            [0.01, -0.01, 0.00],
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
                    "2024-01-01T11:00:00Z",
                    "2024-01-01T11:00:00Z",
                    "2024-01-01T11:00:00Z",
                ],
                utc=True,
            ),
            "symbol": ["A", "B", "C", "A", "B", "C"],
        }
    )
    rank_signal = np.asarray([0.92, 0.50, 0.30, 0.90, 0.88, 0.40], dtype=np.float32)

    gated = apply_dynamic_cross_sectional_gate(
        predictions,
        action_rows,
        rank_signal,
        min_score=0.20,
        min_keep=1,
        max_keep=4,
        reference_quantile=0.25,
        gap_scale=0.5,
    )

    assert np.allclose(gated[0], predictions[0])
    assert np.allclose(gated[1], 0.0)
    assert np.allclose(gated[2], 0.0)
    assert np.allclose(gated[3], predictions[3])
    assert np.allclose(gated[4], predictions[4])
    assert np.allclose(gated[5], 0.0)


def test_apply_dynamic_score_floor_gate_skips_weak_hours_and_expands_strong_hours() -> None:
    predictions = np.asarray(
        [
            [0.04, -0.01, 0.02],
            [0.03, -0.01, 0.02],
            [0.025, -0.01, 0.01],
            [0.05, -0.01, 0.03],
            [0.045, -0.01, 0.03],
            [0.035, -0.01, 0.02],
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
                    "2024-01-01T11:00:00Z",
                    "2024-01-01T11:00:00Z",
                    "2024-01-01T11:00:00Z",
                ],
                utc=True,
            ),
            "symbol": ["A", "B", "C", "A", "B", "C"],
        }
    )
    rank_signal = np.asarray([0.25, 0.23, 0.21, 0.90, 0.74, 0.60], dtype=np.float32)

    gated = apply_dynamic_score_floor_gate(
        predictions,
        action_rows,
        rank_signal,
        min_score=0.20,
        min_strength=0.06,
        floor_quantile=0.25,
        floor_gap_scale=1.60,
    )

    assert np.allclose(gated[0], 0.0)
    assert np.allclose(gated[1], 0.0)
    assert np.allclose(gated[2], 0.0)
    assert np.allclose(gated[3], predictions[3])
    assert np.allclose(gated[4], predictions[4])
    assert np.allclose(gated[5], predictions[5])


def test_apply_dynamic_cross_sectional_gate_respects_pre_zeroed_candidates() -> None:
    predictions = np.asarray(
        [
            [0.05, -0.01, 0.03],
            [0.0, 0.0, 0.0],
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
    rank_signal = np.asarray([0.95, 0.90, 0.50], dtype=np.float32)

    gated = apply_dynamic_cross_sectional_gate(
        predictions,
        action_rows,
        rank_signal,
        min_score=0.20,
        min_keep=1,
        max_keep=3,
        reference_quantile=0.25,
        gap_scale=0.75,
    )

    assert np.allclose(gated[0], predictions[0])
    assert np.allclose(gated[1], 0.0)
    assert np.allclose(gated[2], 0.0)


def test_apply_soft_rank_sizing_scales_by_hour_strength_and_relative_rank() -> None:
    predictions = np.asarray(
        [
            [0.05, -0.01, 0.03],
            [0.04, -0.01, 0.025],
            [0.03, -0.01, 0.02],
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
                    "2024-01-01T11:00:00Z",
                    "2024-01-01T11:00:00Z",
                    "2024-01-01T11:00:00Z",
                ],
                utc=True,
            ),
            "symbol": ["A", "B", "C", "A", "B", "C"],
        }
    )
    rank_signal = np.asarray([0.90, 0.78, 0.62, 0.32, 0.28, 0.22], dtype=np.float32)

    scaled = apply_soft_rank_sizing(
        predictions,
        action_rows,
        rank_signal,
        min_score=0.20,
        reference_quantile=0.25,
        regime_floor_scale=0.75,
        name_floor_scale=0.55,
        regime_power=0.75,
        rank_power=1.5,
    )

    strong_hour_scales = scaled[:3, 0] / predictions[:3, 0]
    weak_hour_scales = scaled[3:, 0] / predictions[3:, 0]

    assert float(strong_hour_scales[0]) > float(strong_hour_scales[1]) > float(strong_hour_scales[2])
    assert float(weak_hour_scales[0]) > float(weak_hour_scales[1]) > float(weak_hour_scales[2])
    assert float(strong_hour_scales[0]) > float(weak_hour_scales[0])
