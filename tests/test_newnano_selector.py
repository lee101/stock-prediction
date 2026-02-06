from __future__ import annotations

import math

import pandas as pd

from newnanoalpacahourlyexp.marketsimulator.selector import SelectionConfig, run_best_trade_simulation


def _make_two_step_frames(
    *,
    symbol: str,
    t0: str,
    t1: str,
    buy_price: float,
    sell_price: float,
    low0: float,
    high0: float,
    low1: float,
    high1: float,
    close0: float,
    close1: float,
    buy_amount0: float = 1.0,
    sell_amount0: float = 0.0,
    buy_amount1: float = 0.0,
    sell_amount1: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.to_datetime([t0, t1], utc=True)
    bars = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": [symbol, symbol],
            "open": [close0, close1],
            "high": [high0, high1],
            "low": [low0, low1],
            "close": [close0, close1],
            "predicted_high_p50_h1": [high0 + 10.0, high1 + 10.0],
            "predicted_low_p50_h1": [low0 - 10.0, low1 - 10.0],
            "predicted_close_p50_h1": [close0, close1],
        }
    )
    actions = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": [symbol, symbol],
            "buy_price": [buy_price, buy_price],
            "sell_price": [sell_price, sell_price],
            "buy_amount": [buy_amount0, buy_amount1],
            "sell_amount": [sell_amount0, sell_amount1],
        }
    )
    return bars, actions


def test_selector_executes_buy_then_sell_within_bars_no_fee():
    bars, actions = _make_two_step_frames(
        symbol="BTCUSD",
        t0="2026-01-01T00:00:00Z",
        t1="2026-01-01T01:00:00Z",
        buy_price=100.0,
        sell_price=112.0,
        low0=99.0,
        high0=101.0,
        low1=100.0,
        high1=115.0,
        close0=100.0,
        close1=114.0,
    )
    cfg = SelectionConfig(
        initial_cash=10_000.0,
        min_edge=0.0,
        risk_weight=0.0,
        edge_mode="high_low",
        enforce_market_hours=True,
        close_at_eod=False,
        fee_by_symbol={"BTCUSD": 0.0},
    )
    result = run_best_trade_simulation(bars, actions, cfg, horizon=1)
    assert [t.side for t in result.trades] == ["buy", "sell"]
    assert math.isclose(result.final_cash, 11_200.0, rel_tol=0, abs_tol=1e-6)
    assert result.open_symbol is None


def test_selector_skips_buy_when_price_never_trades_through():
    bars, actions = _make_two_step_frames(
        symbol="BTCUSD",
        t0="2026-01-01T00:00:00Z",
        t1="2026-01-01T01:00:00Z",
        buy_price=100.0,
        sell_price=112.0,
        low0=101.0,  # buy_price below the bar range -> no fill
        high0=102.0,
        low1=100.0,
        high1=115.0,
        close0=101.5,
        close1=114.0,
        sell_amount1=1.0,
    )
    cfg = SelectionConfig(
        initial_cash=10_000.0,
        min_edge=0.0,
        risk_weight=0.0,
        edge_mode="high_low",
        enforce_market_hours=True,
        close_at_eod=False,
        fee_by_symbol={"BTCUSD": 0.0},
    )
    result = run_best_trade_simulation(bars, actions, cfg, horizon=1)
    assert result.trades == []
    assert math.isclose(result.final_cash, 10_000.0, rel_tol=0, abs_tol=1e-9)


def test_selector_applies_fees_on_round_trip():
    fee = 0.001
    bars, actions = _make_two_step_frames(
        symbol="BTCUSD",
        t0="2026-01-01T00:00:00Z",
        t1="2026-01-01T01:00:00Z",
        buy_price=100.0,
        sell_price=112.0,
        low0=99.0,
        high0=101.0,
        low1=100.0,
        high1=115.0,
        close0=100.0,
        close1=114.0,
    )
    cfg = SelectionConfig(
        initial_cash=10_000.0,
        min_edge=0.0,
        risk_weight=0.0,
        edge_mode="high_low",
        enforce_market_hours=True,
        close_at_eod=False,
        fee_by_symbol={"BTCUSD": fee},
    )
    result = run_best_trade_simulation(bars, actions, cfg, horizon=1)
    assert [t.side for t in result.trades] == ["buy", "sell"]

    qty = 10_000.0 / (100.0 * (1.0 + fee))
    expected_final = qty * 112.0 * (1.0 - fee)
    assert math.isclose(result.final_cash, expected_final, rel_tol=0, abs_tol=1e-6)


def test_stock_market_hours_blocks_trades_outside_session():
    bars, actions = _make_two_step_frames(
        symbol="NVDA",
        t0="2026-01-05T03:00:00Z",  # 22:00 (previous day) NY time, outside regular session
        t1="2026-01-05T04:00:00Z",
        buy_price=100.0,
        sell_price=112.0,
        low0=99.0,
        high0=101.0,
        low1=100.0,
        high1=115.0,
        close0=100.0,
        close1=114.0,
    )
    cfg = SelectionConfig(
        initial_cash=10_000.0,
        min_edge=0.0,
        risk_weight=0.0,
        edge_mode="high_low",
        enforce_market_hours=True,
        close_at_eod=False,
        fee_by_symbol={"NVDA": 0.0},
    )
    result = run_best_trade_simulation(bars, actions, cfg, horizon=1)
    assert result.trades == []
    assert math.isclose(result.final_cash, 10_000.0, rel_tol=0, abs_tol=1e-9)


def test_selector_force_closes_after_max_hold_hours():
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T01:00:00Z",
                ],
                utc=True,
            ),
            "symbol": ["BTCUSD", "BTCUSD"],
            "open": [100.0, 110.0],
            "high": [101.0, 111.0],
            "low": [99.0, 109.0],
            "close": [100.0, 110.0],
            "predicted_high_p50_h1": [120.0, 120.0],
            "predicted_low_p50_h1": [80.0, 80.0],
            "predicted_close_p50_h1": [100.0, 110.0],
        }
    )
    actions = pd.DataFrame(
        {
            "timestamp": bars["timestamp"],
            "symbol": ["BTCUSD", "BTCUSD"],
            "buy_price": [100.0, 100.0],
            "sell_price": [200.0, 200.0],  # never hit
            "buy_amount": [1.0, 0.0],
            "sell_amount": [0.0, 0.0],
        }
    )
    cfg = SelectionConfig(
        initial_cash=10_000.0,
        min_edge=0.0,
        risk_weight=0.0,
        edge_mode="high_low",
        max_hold_hours=1,
        enforce_market_hours=True,
        close_at_eod=False,
        fee_by_symbol={"BTCUSD": 0.0},
    )
    result = run_best_trade_simulation(bars, actions, cfg, horizon=1)
    assert [t.side for t in result.trades] == ["buy", "sell"]
    assert result.trades[1].reason == "max_hold"
    assert math.isclose(result.final_cash, 11_000.0, rel_tol=0, abs_tol=1e-6)
    assert result.open_symbol is None


def test_selector_can_short_when_enabled_for_short_only_stock() -> None:
    bars, actions = _make_two_step_frames(
        symbol="EBAY",
        t0="2026-01-02T19:30:00Z",
        t1="2026-01-02T20:30:00Z",
        buy_price=92.0,
        sell_price=104.0,
        low0=99.0,
        high0=106.0,
        low1=90.0,
        high1=97.0,
        close0=100.0,
        close1=95.0,
        buy_amount0=0.0,
        sell_amount0=1.0,
        buy_amount1=1.0,
        sell_amount1=0.0,
    )
    cfg = SelectionConfig(
        initial_cash=1000.0,
        min_edge=0.0,
        risk_weight=0.0,
        edge_mode="high_low",
        enforce_market_hours=False,
        close_at_eod=False,
        fee_by_symbol={"EBAY": 0.0},
        allow_short=True,
    )
    result = run_best_trade_simulation(bars, actions, cfg, horizon=1)
    assert [t.side for t in result.trades] == ["sell", "buy"]
    assert result.open_symbol is None
    assert result.final_cash > 1000.0
