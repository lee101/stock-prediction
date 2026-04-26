from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

import pufferlib_market.train as train_module
from pufferlib_market.train import (
    ResidualTradingPolicy,
    _mask_short_logits,
    _checkpoint_payload,
    _summarize_validation_results,
    _validation_simulation_kwargs,
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


def test_mask_short_logits_handles_action_grids() -> None:
    logits = torch.zeros((1, 13), dtype=torch.float32)

    masked = _mask_short_logits(logits, num_actions=13)

    min_val = torch.finfo(masked.dtype).min
    assert torch.isfinite(masked[0, :7]).all()
    assert torch.equal(masked[0, 7:], torch.full((6,), min_val))


def test_validation_simulation_kwargs_match_realistic_training_args() -> None:
    args = type(
        "Args",
        (),
        {
            "fee_rate": 0.001,
            "fill_slippage_bps": 5.0,
            "max_leverage": 1.0,
            "periods_per_year": 365.0,
            "short_borrow_apr": 0.0625,
        },
    )()
    action_meta = {
        "action_allocation_bins": 2,
        "action_level_bins": 3,
        "action_max_offset_bps": 25.0,
    }

    kwargs = _validation_simulation_kwargs(args, action_meta, window_steps=90)

    assert kwargs["max_steps"] == 90
    assert kwargs["fee_rate"] == pytest.approx(0.001)
    assert kwargs["slippage_bps"] == pytest.approx(5.0)
    assert kwargs["fill_buffer_bps"] == pytest.approx(5.0)
    assert kwargs["max_leverage"] == pytest.approx(1.0)
    assert kwargs["periods_per_year"] == pytest.approx(365.0)
    assert kwargs["short_borrow_apr"] == pytest.approx(0.0625)
    assert kwargs["action_allocation_bins"] == 2
    assert kwargs["action_level_bins"] == 3
    assert kwargs["action_max_offset_bps"] == pytest.approx(25.0)
    assert kwargs["enable_drawdown_profit_early_exit"] is False


def test_checkpoint_payload_requires_explicit_arch_metadata() -> None:
    policy = ResidualTradingPolicy(obs_size=22, num_actions=3, hidden=16)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    payload = _checkpoint_payload(
        policy,
        optimizer,
        update=1,
        global_step=128,
        best_return=0.25,
        disable_shorts=True,
        action_meta={
            "action_allocation_bins": 1,
            "action_level_bins": 1,
            "action_max_offset_bps": 0.0,
        },
        arch="resmlp",
    )

    assert payload["arch"] == "resmlp"
    assert "input_proj.weight" in payload["model"]


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
