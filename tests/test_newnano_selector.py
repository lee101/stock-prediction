from __future__ import annotations

import math

import pandas as pd
import pytest

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
    volume0: float = 1e9,
    volume1: float = 1e9,
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
            "volume": [volume0, volume1],
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


def test_selector_decision_lag_bars_shifts_actions_forward_one_bar() -> None:
    ts = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T01:00:00Z",
            "2026-01-01T02:00:00Z",
        ],
        utc=True,
    )
    bars = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["BTCUSD", "BTCUSD", "BTCUSD"],
            "open": [100.0, 100.0, 114.0],
            "high": [101.0, 101.0, 115.0],
            "low": [99.0, 99.0, 110.0],
            "close": [100.0, 100.0, 114.0],
            "volume": [1e9, 1e9, 1e9],
            "predicted_high_p50_h1": [110.0, 110.0, 110.0],
            "predicted_low_p50_h1": [90.0, 90.0, 90.0],
            "predicted_close_p50_h1": [100.0, 100.0, 114.0],
        }
    )
    actions = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["BTCUSD", "BTCUSD", "BTCUSD"],
            "buy_price": [100.0, 100.0, 100.0],
            "sell_price": [112.0, 112.0, 112.0],
            "buy_amount": [1.0, 0.0, 0.0],
            "sell_amount": [0.0, 1.0, 0.0],
        }
    )
    cfg = SelectionConfig(
        initial_cash=10_000.0,
        min_edge=0.0,
        risk_weight=0.0,
        edge_mode="high_low",
        enforce_market_hours=True,
        close_at_eod=False,
        fee_by_symbol={"BTCUSD": 0.0},
        decision_lag_bars=1,
    )
    result = run_best_trade_simulation(bars, actions, cfg, horizon=1)
    assert [t.side for t in result.trades] == ["buy", "sell"]
    assert result.trades[0].timestamp == ts[1]
    assert result.trades[1].timestamp == ts[2]
    assert math.isclose(result.final_cash, 11_200.0, rel_tol=0, abs_tol=1e-6)
    assert result.open_symbol is None


def test_selector_caps_fills_by_volume_fraction() -> None:
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
        volume0=1.0,
        volume1=1.0,
        buy_amount0=1.0,
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
        max_volume_fraction=0.1,
    )
    result = run_best_trade_simulation(bars, actions, cfg, horizon=1)
    assert [t.side for t in result.trades] == ["buy", "sell"]
    assert math.isclose(result.trades[0].quantity, 0.1, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(result.final_cash, 10_001.2, rel_tol=0, abs_tol=1e-6)


def test_selector_volume_cap_requires_volume_column() -> None:
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
    bars = bars.drop(columns=["volume"])
    cfg = SelectionConfig(
        initial_cash=10_000.0,
        min_edge=0.0,
        risk_weight=0.0,
        edge_mode="high_low",
        enforce_market_hours=True,
        close_at_eod=False,
        fee_by_symbol={"BTCUSD": 0.0},
        max_volume_fraction=0.1,
    )
    with pytest.raises(ValueError, match="volume"):
        run_best_trade_simulation(bars, actions, cfg, horizon=1)


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


def test_selector_bar_margin_requires_trade_through_limit_buffer():
    bars, actions = _make_two_step_frames(
        symbol="BTCUSD",
        t0="2026-01-01T00:00:00Z",
        t1="2026-01-01T01:00:00Z",
        buy_price=100.0,
        sell_price=112.0,
        low0=99.95,  # Touches limit, but not 10 bps through.
        high0=101.0,
        low1=100.0,
        high1=115.0,
        close0=100.2,
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
        bar_margin=0.001,  # 10 bps buffer
    )
    result = run_best_trade_simulation(bars, actions, cfg, horizon=1)
    assert result.trades == []
    assert math.isclose(result.final_cash, 10_000.0, rel_tol=0, abs_tol=1e-9)


def test_selector_penetration_fill_model_scales_entry_size() -> None:
    bars, actions = _make_two_step_frames(
        symbol="BTCUSD",
        t0="2026-01-01T00:00:00Z",
        t1="2026-01-01T01:00:00Z",
        buy_price=100.0,
        sell_price=140.0,
        low0=99.8,
        high0=100.2,
        low1=105.0,
        high1=106.0,
        close0=100.0,
        close1=105.5,
        sell_amount1=0.0,
    )
    cfg = SelectionConfig(
        initial_cash=10_000.0,
        min_edge=0.0,
        risk_weight=0.0,
        edge_mode="high_low",
        enforce_market_hours=True,
        close_at_eod=False,
        fee_by_symbol={"BTCUSD": 0.0},
        limit_fill_model="penetration",
    )
    result = run_best_trade_simulation(bars, actions, cfg, horizon=1)

    assert [t.side for t in result.trades] == ["buy"]
    assert result.trades[0].quantity == pytest.approx(50.0, abs=1e-9)
    assert result.final_cash == pytest.approx(5_000.0, abs=1e-6)
    assert result.final_inventory == pytest.approx(50.0, abs=1e-9)
    assert result.open_symbol == "BTCUSD"


def test_selector_penetration_fill_model_scales_exit_size_for_seeded_position() -> None:
    ts = pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True)
    bars = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["BTCUSD"],
            "open": [109.8],
            "high": [110.5],
            "low": [109.5],
            "close": [110.0],
            "volume": [1e9],
            "predicted_high_p50_h1": [111.0],
            "predicted_low_p50_h1": [109.0],
            "predicted_close_p50_h1": [110.0],
        }
    )
    actions = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["BTCUSD"],
            "buy_price": [100.0],
            "sell_price": [110.0],
            "buy_amount": [0.0],
            "sell_amount": [1.0],
        }
    )
    cfg = SelectionConfig(
        initial_cash=0.0,
        initial_inventory=10.0,
        initial_symbol="BTCUSD",
        initial_open_price=100.0,
        initial_open_ts=ts[0],
        min_edge=0.0,
        risk_weight=0.0,
        edge_mode="high_low",
        enforce_market_hours=False,
        close_at_eod=False,
        fee_by_symbol={"BTCUSD": 0.0},
        periods_per_year_by_symbol={"BTCUSD": 24 * 365},
        symbols=["BTCUSD"],
        limit_fill_model="penetration",
    )
    result = run_best_trade_simulation(bars, actions, cfg, horizon=1)

    assert [t.side for t in result.trades] == ["sell"]
    assert result.trades[0].quantity == pytest.approx(5.0, abs=1e-9)
    assert result.final_cash == pytest.approx(550.0, abs=1e-6)
    assert result.final_inventory == pytest.approx(5.0, abs=1e-9)
    assert result.open_symbol == "BTCUSD"


def test_selector_penetration_fill_model_can_require_more_than_exact_touch() -> None:
    bars, actions = _make_two_step_frames(
        symbol="BTCUSD",
        t0="2026-01-01T00:00:00Z",
        t1="2026-01-01T01:00:00Z",
        buy_price=100.0,
        sell_price=140.0,
        low0=100.0,
        high0=100.5,
        low1=105.0,
        high1=106.0,
        close0=100.2,
        close1=105.5,
        sell_amount1=0.0,
    )
    base_kwargs = dict(
        initial_cash=10_000.0,
        min_edge=0.0,
        risk_weight=0.0,
        edge_mode="high_low",
        enforce_market_hours=True,
        close_at_eod=False,
        fee_by_symbol={"BTCUSD": 0.0},
        limit_fill_model="penetration",
    )

    no_touch_fill = run_best_trade_simulation(bars, actions, SelectionConfig(**base_kwargs), horizon=1)
    with_touch_fill = run_best_trade_simulation(
        bars,
        actions,
        SelectionConfig(**{**base_kwargs, "touch_fill_fraction": 0.25}),
        horizon=1,
    )

    assert no_touch_fill.trades == []
    assert with_touch_fill.trades[0].quantity == pytest.approx(25.0, abs=1e-9)


def test_selector_realistic_mode_does_not_use_fillability_for_symbol_selection() -> None:
    ts = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T01:00:00Z",
        ],
        utc=True,
    )
    bars = pd.DataFrame(
        [
            # t0: AAA has higher edge but does not fill (low above buy_price).
            {
                "timestamp": ts[0],
                "symbol": "AAAUSD",
                "open": 130.0,
                "high": 140.0,
                "low": 120.0,
                "close": 130.0,
                "volume": 1e9,
                "predicted_high_p50_h1": 200.0,
                "predicted_low_p50_h1": 90.0,
                "predicted_close_p50_h1": 130.0,
            },
            # t0: BBB has lower edge but would fill.
            {
                "timestamp": ts[0],
                "symbol": "BBBUSD",
                "open": 130.0,
                "high": 140.0,
                "low": 99.0,
                "close": 130.0,
                "volume": 1e9,
                "predicted_high_p50_h1": 105.0,
                "predicted_low_p50_h1": 95.0,
                "predicted_close_p50_h1": 130.0,
            },
            # t1: allow BBB to exit if it entered; disable entries.
            {
                "timestamp": ts[1],
                "symbol": "AAAUSD",
                "open": 130.0,
                "high": 140.0,
                "low": 120.0,
                "close": 130.0,
                "volume": 1e9,
                "predicted_high_p50_h1": 200.0,
                "predicted_low_p50_h1": 90.0,
                "predicted_close_p50_h1": 130.0,
            },
            {
                "timestamp": ts[1],
                "symbol": "BBBUSD",
                "open": 130.0,
                "high": 115.0,
                "low": 100.0,
                "close": 114.0,
                "volume": 1e9,
                "predicted_high_p50_h1": 105.0,
                "predicted_low_p50_h1": 95.0,
                "predicted_close_p50_h1": 114.0,
            },
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "timestamp": ts[0],
                "symbol": "AAAUSD",
                "buy_price": 100.0,
                "sell_price": 150.0,
                "buy_amount": 1.0,
                "sell_amount": 0.0,
            },
            {
                "timestamp": ts[0],
                "symbol": "BBBUSD",
                "buy_price": 100.0,
                "sell_price": 110.0,
                "buy_amount": 1.0,
                "sell_amount": 0.0,
            },
            {
                "timestamp": ts[1],
                "symbol": "AAAUSD",
                "buy_price": 100.0,
                "sell_price": 150.0,
                "buy_amount": 0.0,
                "sell_amount": 0.0,
            },
            {
                "timestamp": ts[1],
                "symbol": "BBBUSD",
                "buy_price": 100.0,
                "sell_price": 110.0,
                "buy_amount": 0.0,
                "sell_amount": 1.0,
            },
        ]
    )

    base_cfg = SelectionConfig(
        initial_cash=10_000.0,
        min_edge=0.0,
        risk_weight=0.0,
        edge_mode="high_low",
        enforce_market_hours=False,
        close_at_eod=False,
        fee_by_symbol={"AAAUSD": 0.0, "BBBUSD": 0.0},
        periods_per_year_by_symbol={"AAAUSD": 24 * 365, "BBBUSD": 24 * 365},
        symbols=["AAAUSD", "BBBUSD"],
    )

    legacy = run_best_trade_simulation(bars, actions, base_cfg, horizon=1)
    assert [t.side for t in legacy.trades] == ["buy", "sell"]
    assert legacy.open_symbol is None
    assert legacy.final_cash > 10_000.0

    realistic_cfg = SelectionConfig(**{**base_cfg.__dict__, "select_fillable_only": False})
    realistic = run_best_trade_simulation(bars, actions, realistic_cfg, horizon=1)
    assert realistic.trades == []
    assert realistic.open_symbol is None
    assert math.isclose(realistic.final_cash, 10_000.0, rel_tol=0, abs_tol=1e-9)


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


def test_stock_market_hours_blocks_covering_short_outside_session() -> None:
    bars, actions = _make_two_step_frames(
        symbol="EBAY",
        t0="2026-01-02T19:30:00Z",  # 14:30 NY time, within regular session
        t1="2026-01-03T03:00:00Z",  # 22:00 NY time (previous day), outside regular session
        buy_price=92.0,
        sell_price=104.0,
        low0=99.0,
        high0=106.0,
        low1=90.0,
        high1=97.0,
        close0=100.0,
        close1=95.0,
        buy_amount0=0.0,
        sell_amount0=1.0,  # open short
        buy_amount1=1.0,  # attempt to cover out-of-session
        sell_amount1=0.0,
    )
    cfg = SelectionConfig(
        initial_cash=1000.0,
        min_edge=0.0,
        risk_weight=0.0,
        edge_mode="high_low",
        enforce_market_hours=True,
        close_at_eod=False,
        fee_by_symbol={"EBAY": 0.0},
        allow_short=True,
    )
    result = run_best_trade_simulation(bars, actions, cfg, horizon=1)
    assert [t.side for t in result.trades] == ["sell"]
    assert result.open_symbol == "EBAY"
    assert result.final_inventory < 0.0


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
        max_leverage_stock=2.0,
    )
    result = run_best_trade_simulation(bars, actions, cfg, horizon=1)
    assert [t.side for t in result.trades] == ["sell", "buy"]
    assert result.trades[0].quantity == pytest.approx(1000.0 * 2.0 / 104.0, abs=1e-9)
    assert result.open_symbol is None
    assert result.final_cash > 1000.0


def test_selector_long_leverage_override_takes_precedence_over_base_cap() -> None:
    bars, actions = _make_two_step_frames(
        symbol="NVDA",
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
        enforce_market_hours=False,
        close_at_eod=False,
        fee_by_symbol={"NVDA": 0.0},
        max_leverage_stock=1.0,
        long_max_leverage_stock=3.0,
    )
    result = run_best_trade_simulation(bars, actions, cfg, horizon=1)
    assert [t.side for t in result.trades] == ["buy", "sell"]
    assert result.trades[0].quantity == pytest.approx(300.0, abs=1e-9)
    assert result.final_cash == pytest.approx(13_600.0, abs=1e-6)


def test_selector_allows_stock_leverage_when_configured() -> None:
    bars, actions = _make_two_step_frames(
        symbol="NVDA",
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
        enforce_market_hours=False,
        close_at_eod=False,
        fee_by_symbol={"NVDA": 0.0},
        max_leverage_stock=2.0,
    )
    result = run_best_trade_simulation(bars, actions, cfg, horizon=1)
    assert [t.side for t in result.trades] == ["buy", "sell"]
    assert result.trades[0].quantity == pytest.approx(200.0, abs=1e-9)
    assert result.final_cash == pytest.approx(12_400.0, abs=1e-6)


def test_selector_short_leverage_override_takes_precedence_over_base_cap() -> None:
    bars, actions = _make_two_step_frames(
        symbol="EBAY",
        t0="2026-01-02T19:30:00Z",
        t1="2026-01-02T20:30:00Z",
        buy_price=90.0,
        sell_price=100.0,
        low0=99.0,
        high0=101.0,
        low1=89.0,
        high1=95.0,
        close0=100.0,
        close1=90.0,
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
        max_leverage_stock=2.0,
        short_max_leverage_stock=0.5,
    )
    result = run_best_trade_simulation(bars, actions, cfg, horizon=1)
    assert [t.side for t in result.trades] == ["sell", "buy"]
    assert result.trades[0].quantity == pytest.approx(5.0, abs=1e-9)
    assert result.final_cash == pytest.approx(1050.0, abs=1e-6)


def test_selector_charges_margin_interest_on_debit_cash() -> None:
    bars, actions = _make_two_step_frames(
        symbol="NVDA",
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
    annual_rate = 0.0675
    cfg = SelectionConfig(
        initial_cash=10_000.0,
        min_edge=0.0,
        risk_weight=0.0,
        edge_mode="high_low",
        enforce_market_hours=False,
        close_at_eod=False,
        fee_by_symbol={"NVDA": 0.0},
        max_leverage_stock=2.0,
        margin_interest_annual=annual_rate,
    )
    result = run_best_trade_simulation(bars, actions, cfg, horizon=1)
    expected_cost = 10_000.0 * annual_rate / (365.0 * 24.0)
    assert result.metrics.get("financing_cost_paid") == pytest.approx(expected_cost, rel=0, abs=1e-6)
    assert result.final_cash == pytest.approx(12_400.0 - expected_cost, rel=0, abs=1e-5)
