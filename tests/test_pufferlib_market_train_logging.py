from __future__ import annotations

import sys

import numpy as np
import pytest

import pufferlib_market.train as train_module
from pufferlib_market.train import (
    _summarize_validation_results,
    _wandb_summary_update,
)


class _DummyRun:
    def __init__(self) -> None:
        self.summary: dict[str, object] = {}


def test_summarize_validation_results_aggregates_core_metrics() -> None:
    summary = _summarize_validation_results(
        val_returns=[0.05, 0.02, -0.01],
        val_sortinos=[2.0, 1.0, -0.5],
        val_max_drawdowns=[0.03, 0.02, 0.08],
        val_trades=[4.0, 5.0, 0.0],
    )

    assert summary["val/return"] == 0.02
    assert summary["val/sortino"] == 1.0
    assert summary["val/max_drawdown"] == 0.08
    assert summary["val/negative_windows"] == 1.0
    assert summary["val/avg_trades"] == (4.0 + 5.0 + 0.0) / 3.0
    assert summary["val/score"] == 0.0
    assert 0.0 <= summary["smooth_score"] <= 1.0


def test_summarize_validation_results_penalizes_near_zero_trading() -> None:
    summary = _summarize_validation_results(
        val_returns=[0.04, 0.03],
        val_sortinos=[1.2, 1.1],
        val_max_drawdowns=[0.02, 0.03],
        val_trades=[0.0, 0.5],
    )

    assert summary["val/avg_trades"] < 1.0
    assert summary["val/score"] == -100.0


def test_wandb_summary_update_skips_nonfinite_values() -> None:
    run = _DummyRun()

    _wandb_summary_update(
        run,
        {
            "val/return": np.float32(0.12),
            "train/final_return": 0.18,
            "bad": float("inf"),
            "none_value": None,
        },
    )

    assert run.summary["val/return"] == pytest.approx(0.12)
    assert run.summary["train/final_return"] == 0.18
    assert "bad" not in run.summary
    assert "none_value" not in run.summary


def test_train_main_parses_wandb_resume_arguments(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_train(args) -> None:
        captured["args"] = args

    monkeypatch.setattr(train_module, "train", _fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--data-path",
            "train.bin",
            "--wandb-project",
            "stock",
            "--wandb-run-id",
            "shared-run-123",
            "--wandb-resume",
            "allow",
        ],
    )

    train_module.main()

    args = captured["args"]
    assert args.wandb_run_id == "shared-run-123"
    assert args.wandb_resume == "allow"
