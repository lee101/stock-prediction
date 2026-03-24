from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from unified_hourly_experiment.marketsimulator.portfolio_simulator import PortfolioConfig
from unified_hourly_experiment.sweep_meta_portfolio import (
    StrategyModel,
    build_strategy_action_cache_key,
    evaluate_strategy_baselines,
    load_or_generate_strategy_frames,
    rank_key,
    resolve_execution_scenarios,
)


def test_rank_key_prefers_goodness_when_available() -> None:
    stronger = {
        "min_goodness_score": 1.2,
        "mean_goodness_score": 1.8,
        "min_sortino": 0.1,
        "mean_sortino": 0.2,
        "min_return_pct": 0.3,
        "mean_return_pct": 0.4,
        "mean_max_drawdown_pct": 10.0,
    }
    weaker = {
        "min_goodness_score": 1.1,
        "mean_goodness_score": 5.0,
        "min_sortino": 9.0,
        "mean_sortino": 9.0,
        "min_return_pct": 9.0,
        "mean_return_pct": 9.0,
        "mean_max_drawdown_pct": 1.0,
    }
    assert rank_key(stronger) > rank_key(weaker)


def test_evaluate_strategy_baselines_emits_goodness_fields(monkeypatch) -> None:
    def fake_run_portfolio_simulation(
        bars_eval: pd.DataFrame,
        actions_eval: pd.DataFrame,
        base_cfg: PortfolioConfig,
        horizon: int = 1,
    ) -> SimpleNamespace:
        assert not bars_eval.empty
        assert not actions_eval.empty
        assert isinstance(base_cfg, PortfolioConfig)
        assert horizon == 1
        return SimpleNamespace(
            metrics={
                "total_return": 0.12,
                "annualized_return": 0.34,
                "sortino": 1.7,
                "max_drawdown": 0.08,
                "pnl_smoothness": 0.004,
                "ulcer_index": 0.01,
                "trade_rate": 0.05,
                "goodness_score": 2.4,
            }
        )

    monkeypatch.setattr(
        "unified_hourly_experiment.sweep_meta_portfolio.run_portfolio_simulation",
        fake_run_portfolio_simulation,
    )

    bars = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2026-03-01T15:00:00Z"), "symbol": "NVDA"},
            {"timestamp": pd.Timestamp("2026-03-02T15:00:00Z"), "symbol": "NVDA"},
        ]
    )
    actions = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2026-03-01T15:00:00Z"), "symbol": "NVDA"},
            {"timestamp": pd.Timestamp("2026-03-02T15:00:00Z"), "symbol": "NVDA"},
        ]
    )

    rows, summaries = evaluate_strategy_baselines(
        bars=bars,
        actions_by_strategy={"baseline": actions},
        holdout_days=[0],
        base_cfg=PortfolioConfig(),
        execution_scenarios=resolve_execution_scenarios(base_cfg=PortfolioConfig()),
    )

    assert rows[0]["strategy"] == "baseline"
    assert rows[0]["period"] == "all"
    assert rows[0]["holdout_days"] == 0
    assert rows[0]["return_pct"] == 12.0
    assert rows[0]["annualized_return_pct"] == 34.0
    assert rows[0]["sortino"] == 1.7
    assert rows[0]["max_drawdown_pct"] == 8.0
    assert rows[0]["pnl_smoothness"] == 0.004
    assert rows[0]["ulcer_index"] == 0.01
    assert rows[0]["trade_rate"] == 0.05
    assert rows[0]["goodness_score"] == 2.4
    assert rows[0]["execution_scenario_count"] == 1
    assert rows[0]["scenario_mean_goodness_score"] == 2.4
    assert rows[0]["execution_scenarios"] == [
        {
            "label": "bm0p0005_ttl0",
            "bar_margin": 0.0005,
            "entry_order_ttl_hours": 0,
            "return_pct": 12.0,
            "annualized_return_pct": 34.0,
            "sortino": 1.7,
            "max_drawdown_pct": 8.0,
            "pnl_smoothness": 0.004,
            "ulcer_index": 0.01,
            "trade_rate": 0.05,
            "goodness_score": 2.4,
            "num_buys": 0,
            "num_sells": 0,
        }
    ]

    assert summaries == [
        {
            "strategy": "baseline",
            "min_sortino": 1.7,
            "mean_sortino": 1.7,
            "min_return_pct": 12.0,
            "mean_return_pct": 12.0,
            "min_annualized_return_pct": 34.0,
            "mean_annualized_return_pct": 34.0,
            "mean_max_drawdown_pct": 8.0,
            "mean_pnl_smoothness": 0.004,
            "mean_ulcer_index": 0.01,
            "mean_trade_rate": 0.05,
            "min_goodness_score": 2.4,
            "mean_goodness_score": 2.4,
            "execution_scenario_count": 1,
            "mean_scenario_mean_return_pct": 12.0,
            "mean_scenario_mean_annualized_return_pct": 34.0,
            "mean_scenario_mean_sortino": 1.7,
            "mean_scenario_mean_max_drawdown_pct": 8.0,
            "mean_scenario_mean_pnl_smoothness": 0.004,
            "mean_scenario_mean_ulcer_index": 0.01,
            "mean_scenario_mean_trade_rate": 0.05,
            "mean_scenario_mean_goodness_score": 2.4,
        }
    ]


def test_evaluate_strategy_baselines_uses_worst_case_execution_scenario(monkeypatch) -> None:
    def fake_run_portfolio_simulation(
        bars_eval: pd.DataFrame,
        actions_eval: pd.DataFrame,
        base_cfg: PortfolioConfig,
        horizon: int = 1,
    ) -> SimpleNamespace:
        assert not bars_eval.empty
        assert not actions_eval.empty
        assert horizon == 1
        if base_cfg.bar_margin < 0.001:
            metrics = {
                "total_return": 0.12,
                "annualized_return": 0.34,
                "sortino": 1.7,
                "max_drawdown": 0.08,
                "pnl_smoothness": 0.004,
                "ulcer_index": 0.01,
                "trade_rate": 0.05,
                "goodness_score": 2.4,
                "num_buys": 5,
                "num_sells": 5,
            }
        else:
            metrics = {
                "total_return": 0.03,
                "annualized_return": 0.08,
                "sortino": 0.4,
                "max_drawdown": 0.14,
                "pnl_smoothness": 0.010,
                "ulcer_index": 0.03,
                "trade_rate": 0.02,
                "goodness_score": 0.8,
                "num_buys": 2,
                "num_sells": 2,
            }
        return SimpleNamespace(metrics=metrics)

    monkeypatch.setattr(
        "unified_hourly_experiment.sweep_meta_portfolio.run_portfolio_simulation",
        fake_run_portfolio_simulation,
    )

    bars = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2026-03-01T15:00:00Z"), "symbol": "NVDA"},
            {"timestamp": pd.Timestamp("2026-03-02T15:00:00Z"), "symbol": "NVDA"},
        ]
    )
    actions = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2026-03-01T15:00:00Z"), "symbol": "NVDA"},
            {"timestamp": pd.Timestamp("2026-03-02T15:00:00Z"), "symbol": "NVDA"},
        ]
    )

    scenarios = resolve_execution_scenarios(
        base_cfg=PortfolioConfig(bar_margin=0.0005, entry_order_ttl_hours=0),
        bar_margins=[0.0005, 0.0013],
        entry_order_ttls=[0],
    )
    rows, summaries = evaluate_strategy_baselines(
        bars=bars,
        actions_by_strategy={"baseline": actions},
        holdout_days=[0],
        base_cfg=PortfolioConfig(bar_margin=0.0005, entry_order_ttl_hours=0),
        execution_scenarios=scenarios,
    )

    assert rows[0]["execution_scenario_count"] == 2
    assert rows[0]["return_pct"] == 3.0
    assert rows[0]["annualized_return_pct"] == 8.0
    assert rows[0]["sortino"] == 0.4
    assert rows[0]["max_drawdown_pct"] == pytest.approx(14.0)
    assert rows[0]["goodness_score"] == 0.8
    assert rows[0]["num_buys"] == 2
    assert rows[0]["scenario_mean_goodness_score"] == 1.6
    assert summaries[0]["min_goodness_score"] == 0.8
    assert summaries[0]["mean_goodness_score"] == 0.8
    assert summaries[0]["min_annualized_return_pct"] == 8.0
    assert summaries[0]["mean_scenario_mean_goodness_score"] == 1.6


def test_load_or_generate_strategy_frames_reuses_action_cache(monkeypatch, tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "epoch_001.pt").write_bytes(b"weights")
    (checkpoint_dir / "config.json").write_text("{}")
    (checkpoint_dir / "training_meta.json").write_text("{}")

    strategy = StrategyModel(
        name="s1",
        checkpoint_dir=checkpoint_dir,
        checkpoint_name="epoch_001.pt",
        model=object(),
        feature_columns=["return_1h"],
        sequence_length=2,
        normalizer=None,
        horizons=[1],
    )
    frame = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2026-03-01T15:00:00Z"), "symbol": "NVDA", "close": 100.0},
            {"timestamp": pd.Timestamp("2026-03-02T15:00:00Z"), "symbol": "NVDA", "close": 101.0},
        ]
    )
    data_cache = {("NVDA", (1,), 2): SimpleNamespace(frame=frame, normalizer=None)}
    action_cache_dir = tmp_path / "action_cache"

    calls = {"count": 0}

    def fake_generate_actions_from_frame(**_: object) -> pd.DataFrame:
        calls["count"] += 1
        return pd.DataFrame(
            [
                {"timestamp": pd.Timestamp("2026-03-01T15:00:00Z"), "symbol": "NVDA", "action": 1},
                {"timestamp": pd.Timestamp("2026-03-02T15:00:00Z"), "symbol": "NVDA", "action": -1},
            ]
        )

    monkeypatch.setattr(
        "unified_hourly_experiment.sweep_meta_portfolio.generate_actions_from_frame",
        fake_generate_actions_from_frame,
    )

    expected_key = build_strategy_action_cache_key(strategy=strategy, symbols=["NVDA"], data_cache=data_cache)
    expected_cache_path = action_cache_dir / f"strategy_actions_{expected_key}.pkl"

    actions_a, bars_a = load_or_generate_strategy_frames(
        strategy=strategy,
        symbols=["NVDA"],
        data_cache=data_cache,
        device=torch.device("cpu"),
        action_cache_dir=action_cache_dir,
    )
    assert calls["count"] == 1
    assert expected_cache_path.exists()

    def fail_generate_actions_from_frame(**_: object) -> pd.DataFrame:
        raise AssertionError("cache should have been reused instead of regenerating actions")

    monkeypatch.setattr(
        "unified_hourly_experiment.sweep_meta_portfolio.generate_actions_from_frame",
        fail_generate_actions_from_frame,
    )

    actions_b, bars_b = load_or_generate_strategy_frames(
        strategy=strategy,
        symbols=["NVDA"],
        data_cache=data_cache,
        device=torch.device("cpu"),
        action_cache_dir=action_cache_dir,
    )

    pd.testing.assert_frame_equal(actions_a, actions_b)
    pd.testing.assert_frame_equal(bars_a, bars_b)
