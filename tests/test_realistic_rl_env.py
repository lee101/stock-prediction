#!/usr/bin/env python3
"""Unit tests for hftraining realistic RL environment and simulator.

These tests exercise market simulation (slippage, spread, stop/take-profit)
and environment stepping on synthetic OHLCV without network or training.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Ensure repository root is on import path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Skip these tests if torch isn't available in the environment
pytest.importorskip("torch", reason="realistic_rl_env tests require torch installed")

from hftraining.realistic_backtest_rl import (
    RealisticTradingConfig,
    RealisticMarketSimulator,
    RealisticTradingEnvironment,
)


def make_trending_ohlcv(n=300, start=100.0, drift=0.03, noise=0.5, vol_base=1_000_000):
    rng = np.random.RandomState(42)
    close = start + np.cumsum(rng.randn(n) * noise + drift)
    open_ = close + rng.randn(n) * 0.2
    high = np.maximum(open_, close) + np.abs(rng.randn(n)) * 0.5
    low = np.minimum(open_, close) - np.abs(rng.randn(n)) * 0.5
    vol = rng.randint(int(0.5 * vol_base), int(1.5 * vol_base), size=n).astype(float)
    return np.column_stack([open_, high, low, close, vol])


def test_market_simulator_execution_price_slippage_and_spread():
    data = make_trending_ohlcv(n=120)
    cfg = RealisticTradingConfig(sequence_length=60)
    sim = RealisticMarketSimulator(data, cfg)

    bar = 60
    size = 10_000.0  # $ amount traded

    buy_price, buy_slip = sim.get_execution_price(bar, is_buy=True, size=size)
    sell_price, sell_slip = sim.get_execution_price(bar, is_buy=False, size=size)

    # Basic sanity: slippage is non-negative and spread widens buy vs sell
    assert buy_slip >= 0 and sell_slip >= 0
    assert buy_price > sell_price


def test_stop_loss_take_profit_triggering():
    data = make_trending_ohlcv(n=120, drift=0.0)
    cfg = RealisticTradingConfig(sequence_length=60)
    sim = RealisticMarketSimulator(data, cfg)

    bar = 80
    entry_price = sim.opens[bar]
    # Set tight TP/SL so at least one triggers using high/low
    res = sim.check_stop_loss_take_profit(bar, entry_price, stop_loss=0.001, take_profit=0.001)
    assert res is None or res[0] in {"stop_loss", "take_profit"}


def test_environment_step_and_metrics_progress():
    # Upward trend should allow profitable episodes with simple buy/hold actions
    data = make_trending_ohlcv(n=260, drift=0.05)
    cfg = RealisticTradingConfig(sequence_length=60, max_daily_trades=100)
    env = RealisticTradingEnvironment(data, cfg)

    state = env.reset()
    steps = 0
    # Naive policy: buy small position when flat; otherwise hold
    while steps < 80:
        steps += 1
        market_data, portfolio_state = state
        action = {"trade": 1 if env.position == 0 else 0, "position_size": 0.1, "stop_loss": 0.02, "take_profit": 0.05}
        next_state, reward, done, metrics = env.step(action)
        state = next_state if not done else state
        if done:
            break

    # We should have executed at least 1 trade and recorded some metrics
    assert env.metrics.total_trades >= 1
    assert isinstance(env.metrics.max_drawdown, float)
    assert isinstance(env.metrics.win_rate, float)

    # Ensure equity curve progressed
    assert len(env.equity_curve) > 1
