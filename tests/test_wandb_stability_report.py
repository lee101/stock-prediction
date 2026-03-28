from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = str(REPO / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import wandb_stability_report as module


def test_compute_metric_stability_rewards_smooth_improvement_for_min_goal() -> None:
    smooth = module.compute_metric_stability([5.0, 4.0, 3.0, 2.0], goal="min")
    noisy = module.compute_metric_stability([5.0, 3.0, 4.5, 2.0], goal="min")

    assert smooth.direction_match_rate == 1.0
    assert smooth.stability_score > noisy.stability_score
    assert smooth.finish_gap_ratio == 0.0


def test_compute_metric_stability_rewards_smooth_improvement_for_max_goal() -> None:
    smooth = module.compute_metric_stability([1.0, 2.0, 3.0, 4.0], goal="max")
    stalled = module.compute_metric_stability([1.0, 2.0, 2.0, 2.0], goal="max")

    assert smooth.direction_match_rate == 1.0
    assert smooth.relative_improvement > stalled.relative_improvement
    assert smooth.stability_score > stalled.stability_score


def test_compute_metric_stability_handles_empty_values() -> None:
    stability = module.compute_metric_stability([], goal="min")

    assert stability.count == 0
    assert stability.start is None
    assert stability.stability_score == 0.0


def test_format_markdown_includes_top_run() -> None:
    rows = [
        {
            "name": "stable_run",
            "stability": {
                "stability_score": 1.2,
                "direction_match_rate": 0.9,
                "smoothness_ratio": 0.1,
                "finish_gap_ratio": 0.0,
                "relative_improvement": 0.4,
                "count": 20,
            },
        },
        {
            "name": "noisy_run",
            "stability": {
                "stability_score": 0.2,
                "direction_match_rate": 0.5,
                "smoothness_ratio": 0.6,
                "finish_gap_ratio": 0.3,
                "relative_improvement": 0.1,
                "count": 20,
            },
        },
    ]

    text = module.format_markdown(rows, project="stock", entity=None, metric_key="val/loss", goal="min")

    assert "stable_run" in text
    assert "Most Stable Run" in text
    assert "val/loss" in text


def test_compute_metric_stability_rejects_bad_goal() -> None:
    with pytest.raises(ValueError):
        module.compute_metric_stability([1.0, 2.0], goal="sideways")


def test_fetch_run_stability_rows_skips_missing_run_ids() -> None:
    class _Api:
        def run(self, _path: str):
            raise RuntimeError("missing")

    class _Wandb:
        def Api(self):
            return _Api()

    rows = module.fetch_run_stability_rows(
        wandb=_Wandb(),
        project="stock",
        entity=None,
        metric_key="train/policy_loss",
        goal="min",
        run_ids=["missing"],
        group=None,
        last_n_runs=1,
        history_samples=10,
    )

    assert rows == []
