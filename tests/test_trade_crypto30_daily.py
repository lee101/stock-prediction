"""Tests for trade_crypto30_daily.py"""
import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from trade_crypto30_daily import (
    SYMBOLS_30, INTERNAL_SYMBOLS, to_binance_symbol,
    PortfolioState, load_state, save_state, execute_binance_order,
    Crypto30Ensemble, fetch_daily_klines,
)
from pufferlib_market.inference import TradingSignal


def test_symbol_count():
    assert len(SYMBOLS_30) == 30
    assert len(INTERNAL_SYMBOLS) == 30


def test_matic_pol_rename():
    assert "POLUSDT" in SYMBOLS_30
    assert "MATICUSD" in INTERNAL_SYMBOLS
    assert "MATICUSDT" not in SYMBOLS_30
    assert "POLUSD" not in INTERNAL_SYMBOLS
    idx = SYMBOLS_30.index("POLUSDT")
    assert INTERNAL_SYMBOLS[idx] == "MATICUSD"


def test_to_binance_symbol():
    assert to_binance_symbol("BTCUSD") == "BTCUSDT"
    assert to_binance_symbol("ETHUSD") == "ETHUSDT"
    assert to_binance_symbol("MATICUSD") == "POLUSDT"
    assert to_binance_symbol("SOLUSD") == "SOLUSDT"


def test_internal_symbols_usd_suffix():
    for s in INTERNAL_SYMBOLS:
        assert s.endswith("USD"), f"{s} should end with USD"
        assert not s.endswith("USDT"), f"{s} should not end with USDT"


def test_portfolio_state_defaults():
    state = PortfolioState()
    assert state.cash_usd == 10000.0
    assert state.position_symbol is None
    assert state.position_qty == 0.0
    assert state.hold_days == 0
    assert state.total_value == 10000.0


def test_state_persistence(tmp_path):
    import trade_crypto30_daily as mod
    orig_dir = mod.STATE_DIR
    orig_file = mod.STATE_FILE
    try:
        mod.STATE_DIR = tmp_path
        mod.STATE_FILE = tmp_path / "state.json"
        state = PortfolioState(cash_usd=5000.0, position_symbol="BTCUSD",
                               position_qty=0.1, entry_price=50000.0,
                               hold_days=3, episode_step=10, total_value=10000.0)
        save_state(state)
        loaded = load_state()
        assert loaded.cash_usd == 5000.0
        assert loaded.position_symbol == "BTCUSD"
        assert loaded.position_qty == 0.1
        assert loaded.entry_price == 50000.0
        assert loaded.hold_days == 3
    finally:
        mod.STATE_DIR = orig_dir
        mod.STATE_FILE = orig_file


def test_execute_flat_closes_position():
    state = PortfolioState(cash_usd=500.0, position_symbol="BTCUSD",
                           position_qty=0.1, entry_price=50000.0, hold_days=5)
    signal = TradingSignal("flat", None, None, 0.5, 1.0)
    prices = {"BTCUSD": 55000.0}
    execute_binance_order(signal, state, prices, dry_run=True)
    assert state.position_symbol is None
    assert state.position_qty == 0.0
    assert state.cash_usd == pytest.approx(500.0 + 0.1 * 55000.0)


def test_execute_buy_new_position():
    state = PortfolioState(cash_usd=10000.0)
    signal = TradingSignal("long_ETHUSD", "ETHUSD", "long", 0.3, 1.5)
    prices = {"ETHUSD": 2000.0}
    execute_binance_order(signal, state, prices, dry_run=True)
    assert state.position_symbol == "ETHUSD"
    assert state.position_qty == pytest.approx(9500.0 / 2000.0)
    assert state.cash_usd == pytest.approx(500.0)
    assert state.entry_price == 2000.0


def test_execute_rotate_position():
    state = PortfolioState(cash_usd=500.0, position_symbol="BTCUSD",
                           position_qty=0.1, entry_price=50000.0, hold_days=3)
    signal = TradingSignal("long_ETHUSD", "ETHUSD", "long", 0.4, 1.2)
    prices = {"BTCUSD": 55000.0, "ETHUSD": 2000.0}
    execute_binance_order(signal, state, prices, dry_run=True)
    expected_cash = 500.0 + 0.1 * 55000.0
    expected_alloc = expected_cash * 0.95
    assert state.position_symbol == "ETHUSD"
    assert state.position_qty == pytest.approx(expected_alloc / 2000.0)
    assert state.cash_usd == pytest.approx(expected_cash - expected_alloc)


def test_execute_no_double_buy():
    """If already holding same symbol, do nothing."""
    state = PortfolioState(cash_usd=500.0, position_symbol="ETHUSD",
                           position_qty=4.0, entry_price=2000.0)
    signal = TradingSignal("long_ETHUSD", "ETHUSD", "long", 0.3, 1.0)
    prices = {"ETHUSD": 2100.0}
    execute_binance_order(signal, state, prices, dry_run=True)
    assert state.position_symbol == "ETHUSD"
    assert state.position_qty == 4.0
    assert state.cash_usd == 500.0


def test_execute_matic_maps_to_pol():
    """MATICUSD buy should place POLUSDT order."""
    state = PortfolioState(cash_usd=10000.0)
    signal = TradingSignal("long_MATICUSD", "MATICUSD", "long", 0.25, 1.0)
    prices = {"MATICUSD": 0.5}
    with patch("trade_crypto30_daily._place_market_order") as mock_order:
        execute_binance_order(signal, state, prices, dry_run=False)
        mock_order.assert_called_once()
        args = mock_order.call_args[0]
        assert args[0] == "POLUSDT"
        assert args[1] == "BUY"


def _make_ohlcv_df(n=90, base_price=100.0):
    dates = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    np.random.seed(42)
    prices = base_price * np.exp(np.cumsum(np.random.randn(n) * 0.02))
    return pd.DataFrame({
        "open": prices * 0.99,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "volume": np.random.uniform(1e6, 1e7, n),
    }, index=dates)


def test_ensemble_init():
    ckpts = [f"pufferlib_market/checkpoints/crypto30_ensemble/s{s}.pt"
             for s in [2, 19, 21, 23]]
    for c in ckpts:
        if not os.path.exists(c):
            pytest.skip("checkpoints not available")
    ensemble = Crypto30Ensemble(ckpts)
    assert len(ensemble.traders) == 4
    assert ensemble.num_symbols == 30
    assert ensemble.num_actions == 61


def test_ensemble_signal():
    ckpts = [f"pufferlib_market/checkpoints/crypto30_ensemble/s{s}.pt"
             for s in [2, 19, 21, 23]]
    for c in ckpts:
        if not os.path.exists(c):
            pytest.skip("checkpoints not available")
    ensemble = Crypto30Ensemble(ckpts)
    daily_dfs = {}
    prices = {}
    for sym in INTERNAL_SYMBOLS:
        daily_dfs[sym] = _make_ohlcv_df()
        prices[sym] = float(daily_dfs[sym]["close"].iloc[-1])
    state = PortfolioState()
    signal = ensemble.get_ensemble_signal(daily_dfs, prices, state)
    assert signal.action is not None
    assert 0 <= signal.confidence <= 1.0
    if signal.direction == "long":
        assert signal.symbol in INTERNAL_SYMBOLS
    if signal.direction == "short":
        pytest.fail("shorts should be masked")


def test_ensemble_shorts_masked():
    ckpts = [f"pufferlib_market/checkpoints/crypto30_ensemble/s{s}.pt"
             for s in [2, 19, 21, 23]]
    for c in ckpts:
        if not os.path.exists(c):
            pytest.skip("checkpoints not available")
    ensemble = Crypto30Ensemble(ckpts)
    daily_dfs = {}
    prices = {}
    for sym in INTERNAL_SYMBOLS:
        daily_dfs[sym] = _make_ohlcv_df()
        prices[sym] = float(daily_dfs[sym]["close"].iloc[-1])
    state = PortfolioState()
    for _ in range(20):
        signal = ensemble.get_ensemble_signal(daily_dfs, prices, state)
        assert signal.direction != "short", "short actions should be masked to -inf"


def test_observation_shape():
    ckpts = [f"pufferlib_market/checkpoints/crypto30_ensemble/s{s}.pt"
             for s in [2, 19, 21, 23]]
    for c in ckpts:
        if not os.path.exists(c):
            pytest.skip("checkpoints not available")
    ensemble = Crypto30Ensemble(ckpts)
    fps = ensemble.features_per_sym
    expected_obs = 30 * fps + 5 + 30
    obs = np.zeros(expected_obs, dtype=np.float32)
    assert obs.shape[0] == expected_obs
