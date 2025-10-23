import math
import numpy as np
import pandas as pd

from pufferlibtraining.envs.stock_env import StockTradingEnv
from src.fees import get_fee_for_symbol


def make_frame(days=40, open_start=100.0, close_delta=0.0):
    dates = pd.date_range("2020-01-01", periods=days, freq="D")
    opens = np.full(days, open_start, dtype=np.float32)
    closes = opens + float(close_delta)
    highs = np.maximum(opens, closes)
    lows = np.minimum(opens, closes)
    return pd.DataFrame({
        "date": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.full(days, 1_000_000, dtype=np.float32),
    })


def test_base_fee_detection_crypto_vs_equity():
    frames = {"AAPL": make_frame(), "BTCUSD": make_frame()}
    env = StockTradingEnv(frames, window_size=5)
    # Ensure base fee rates match fee utility behaviour
    aapl_fee = get_fee_for_symbol("AAPL")
    btc_fee = get_fee_for_symbol("BTCUSD")
    assert math.isclose(float(env.base_fee_rates[0].item()), aapl_fee, rel_tol=1e-6)
    assert math.isclose(float(env.base_fee_rates[1].item()), btc_fee, rel_tol=1e-6)


def test_open_timing_deleverage_to_overnight_cap():
    # Construct action that produces intraday gross > 2× but <= 4×, triggering auto-deleverage.
    frames = {"AAPL": make_frame(close_delta=1.0), "AMZN": make_frame(close_delta=0.5)}
    env = StockTradingEnv(frames, window_size=5, trade_timing="open", risk_scale=1.0)
    obs, _ = env.reset()
    # Target ~1.5× per asset intraday => tanh(x)*4 ≈ 1.5 ==> x ≈ atanh(0.375)
    raw = float(np.arctanh(0.375))
    action = np.array([raw, raw], dtype=np.float32)
    _, _, term, trunc, info = env.step(action)
    assert not (term or trunc)
    # After step, weights are auto-reduced so overnight gross equals 2×
    weights_after = np.array(env.trades[-1]["weights_after"], dtype=np.float32)
    assert math.isclose(float(np.abs(weights_after).sum()), 2.0, rel_tol=1e-5)
    # Intraday gross exposure reported in info should be > overnight cap
    assert info["max_intraday_leverage"] >= 4.0 - 1e-6
    assert info["max_overnight_leverage"] <= info["max_intraday_leverage"]


def test_close_timing_holds_then_trades():
    # With close timing, first step should realise zero PnL from zero holdings, then trade.
    frames = {"AAPL": make_frame(close_delta=10.0), "NVDA": make_frame(close_delta=-5.0)}
    env = StockTradingEnv(frames, window_size=5, trade_timing="close", risk_scale=1.0)
    env.reset()
    action = np.array([0.5, 0.5], dtype=np.float32)
    _, _, _, _, _ = env.step(action)
    last_trade = env.trades[-1]
    # From zero starting weights, raw_profit should be ~0 on first day
    assert abs(last_trade["raw_profit"]) < 1e-6
    # Weights after should be non-zero (we did trade at close)
    assert np.abs(np.array(last_trade["weights_after"]).sum()) > 0.0

