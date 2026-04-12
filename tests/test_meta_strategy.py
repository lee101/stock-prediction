"""Tests for meta_strategy_backtest.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pufferlib_market.hourly_replay import MktdData, INITIAL_CASH
from scripts.meta_strategy_backtest import (
    ModelTrace,
    build_mktd_from_csvs,
    select_top_k_momentum,
    run_meta_portfolio,
)


def _make_trace(name: str, returns: list[float], symbols: list[str]) -> ModelTrace:
    eq = [INITIAL_CASH]
    for r in returns:
        eq.append(eq[-1] * (1 + r))
    T = len(returns)
    actions = np.ones(T, dtype=np.int32)
    return ModelTrace(
        name=name,
        actions=actions,
        equity_curve=np.array(eq),
        held_symbols=symbols[:T] if len(symbols) >= T else symbols + [None] * (T - len(symbols)),
        daily_returns=np.array(returns),
    )


def _make_data(n_days: int = 50, n_sym: int = 3) -> MktdData:
    symbols = [f"SYM{i}" for i in range(n_sym)]
    features = np.random.randn(n_days, n_sym, 16).astype(np.float32) * 0.1
    prices = np.ones((n_days, n_sym, 5), dtype=np.float32) * 100
    for t in range(1, n_days):
        prices[t] = prices[t - 1] * (1 + np.random.randn(n_sym, 5).astype(np.float32) * 0.01)
    prices = np.clip(prices, 1.0, None)
    tradable = np.ones((n_days, n_sym), dtype=np.uint8)
    return MktdData(version=2, symbols=symbols, features=features, prices=prices, tradable=tradable)


class TestSelectTopKMomentum:
    def test_basic_selection(self):
        t1 = _make_trace("good", [0.01] * 30, ["SYM0"] * 30)
        t2 = _make_trace("bad", [-0.01] * 30, ["SYM1"] * 30)
        t3 = _make_trace("mid", [0.005] * 30, ["SYM2"] * 30)
        selected = select_top_k_momentum([t1, t2, t3], day_idx=25, lookback=20, top_k=2)
        assert len(selected) == 2
        assert 0 in selected  # good model
        assert 2 in selected  # mid model
        assert 1 not in selected  # bad model

    def test_top_k_1(self):
        t1 = _make_trace("best", [0.02] * 20, ["SYM0"] * 20)
        t2 = _make_trace("ok", [0.01] * 20, ["SYM1"] * 20)
        selected = select_top_k_momentum([t1, t2], day_idx=15, lookback=10, top_k=1)
        assert selected == [0]

    def test_early_days(self):
        t1 = _make_trace("a", [0.01] * 5, ["SYM0"] * 5)
        t2 = _make_trace("b", [-0.01] * 5, ["SYM1"] * 5)
        selected = select_top_k_momentum([t1, t2], day_idx=2, lookback=20, top_k=2)
        assert len(selected) == 2


class TestRunMetaPortfolio:
    def test_basic_meta(self):
        data = _make_data(n_days=60, n_sym=3)
        t1 = _make_trace("m1", [0.005] * 59, ["SYM0"] * 59)
        t2 = _make_trace("m2", [-0.002] * 59, ["SYM1"] * 59)
        t3 = _make_trace("m3", [0.003] * 59, ["SYM2"] * 59)
        result = run_meta_portfolio(
            data, [t1, t2, t3],
            top_k=2, lookback=10, warmup=10,
            fee_rate=0.001, slippage_bps=0.0,
        )
        assert result.equity_curve is not None
        assert len(result.equity_curve) > 0
        assert result.num_trades > 0

    def test_meta_stays_flat_during_warmup(self):
        data = _make_data(n_days=40, n_sym=2)
        t1 = _make_trace("m1", [0.01] * 39, ["SYM0"] * 39)
        t2 = _make_trace("m2", [0.01] * 39, ["SYM1"] * 39)
        result = run_meta_portfolio(
            data, [t1, t2],
            top_k=1, lookback=5, warmup=20,
            fee_rate=0.001, slippage_bps=0.0,
        )
        # During first 20 days, returns should be ~0
        for i in range(20):
            assert abs(result.daily_returns[i]) < 1e-6

    def test_meta_selects_winning_model(self):
        data = _make_data(n_days=80, n_sym=2)
        t1 = _make_trace("winner", [0.02] * 79, ["SYM0"] * 79)
        t2 = _make_trace("loser", [-0.02] * 79, ["SYM1"] * 79)
        result = run_meta_portfolio(
            data, [t1, t2],
            top_k=1, lookback=10, warmup=15,
            fee_rate=0.0, slippage_bps=0.0,
        )
        # After warmup, meta should mostly select the winner
        for models in result.selected_models[20:]:
            if models:
                assert "winner" in models


class TestBuildMktdFromCsvs:
    def test_loads_csvs(self, tmp_path):
        for sym in ["AAA", "BBB"]:
            dates = pd.date_range("2024-01-01", periods=200, freq="B", tz="UTC")
            df = pd.DataFrame({
                "timestamp": dates,
                "open": 100 + np.random.randn(200).cumsum(),
                "high": 105 + np.random.randn(200).cumsum(),
                "low": 95 + np.random.randn(200).cumsum(),
                "close": 100 + np.random.randn(200).cumsum(),
                "volume": np.random.randint(1000, 10000, 200).astype(float),
                "symbol": sym,
            })
            df.to_csv(tmp_path / f"{sym}.csv", index=False)

        data, dates = build_mktd_from_csvs(tmp_path, ["AAA", "BBB"])
        assert data.num_symbols == 2
        assert data.num_timesteps > 100
        assert data.features.shape[2] == 16
        assert data.prices.shape[2] == 5
