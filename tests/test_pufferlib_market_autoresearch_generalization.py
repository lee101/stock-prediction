from __future__ import annotations

import pytest

from pufferlib_market.autoresearch_rl import compute_generalization_metrics, select_rank_score


def test_compute_generalization_metrics_penalizes_false_positive_replay_winner() -> None:
    metrics = compute_generalization_metrics(
        {
            "replay_combo_score": 39.1369,
            "holdout_robust_score": -170.7026,
            "holdout_negative_return_rate": 0.8333,
            "train_return": -0.1507,
            "val_return": 0.0370,
            "replay_hourly_return_pct": 25.5629,
        }
    )

    assert metrics["generalization_score"] < 0.0
    assert metrics["overfit_gap_score"] > 100.0


def test_compute_generalization_metrics_rewards_consistent_run() -> None:
    metrics = compute_generalization_metrics(
        {
            "replay_combo_score": 18.0,
            "holdout_robust_score": 12.0,
            "holdout_negative_return_rate": 0.1,
            "train_return": 0.08,
            "val_return": 0.06,
            "replay_hourly_return_pct": 5.5,
        }
    )

    assert metrics["generalization_score"] > 0.0
    assert metrics["train_val_gap_pct"] == pytest.approx(2.0)
    assert metrics["val_replay_gap_pct"] == pytest.approx(0.5)


def test_select_rank_score_prefers_generalization_score_when_present() -> None:
    metric_name, score = select_rank_score(
        {
            "generalization_score": 7.5,
            "replay_combo_score": 20.0,
            "holdout_robust_score": -40.0,
        },
        rank_metric="auto",
    )

    assert metric_name == "generalization_score"
    assert score == pytest.approx(7.5)
