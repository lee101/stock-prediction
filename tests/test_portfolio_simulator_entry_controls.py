from __future__ import annotations

import pandas as pd
import pytest

from unified_hourly_experiment.marketsimulator import PortfolioConfig, run_portfolio_simulation
from unified_hourly_experiment.marketsimulator.portfolio_simulator import _drawdown_entry_scale


def _two_symbol_bars(periods: int = 8) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=periods, freq="h")
    rows = []
    for ts_idx, ts in enumerate(timestamps):
        close = 100.0 + ts_idx
        for symbol in ("AAAUSDT", "BBBUSDT"):
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    "open": close,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "close": close,
                    "predicted_high_p50_h1": close * 1.05,
                    "predicted_low_p50_h1": close * 0.95,
                    "predicted_close_p50_h1": close * 1.01,
                }
            )
    return pd.DataFrame(rows)


def test_concentrated_allocator_can_enforce_second_position_floor():
    bars = _two_symbol_bars(periods=2)
    ts = bars["timestamp"].min()
    actions = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "AAAUSDT",
                "buy_price": 99.0,
                "sell_price": 110.0,
                "buy_amount": 100.0,
                "trade_amount": 100.0,
            },
            {
                "timestamp": ts,
                "symbol": "BBBUSDT",
                "buy_price": 99.0,
                "sell_price": 104.0,
                "buy_amount": 100.0,
                "trade_amount": 100.0,
            },
        ]
    )

    result = run_portfolio_simulation(
        bars,
        actions,
        PortfolioConfig(
            max_positions=2,
            max_leverage=1.0,
            decision_lag_bars=0,
            enforce_market_hours=False,
            close_at_eod=False,
            int_qty=False,
            fee_by_symbol={"AAAUSDT": 0.0, "BBBUSDT": 0.0},
            entry_allocator_mode="concentrated",
            entry_allocator_edge_power=4.0,
            entry_allocator_max_single_position_fraction=0.95,
            entry_allocator_min_second_position_fraction=0.25,
            apply_leverage_to_crypto=True,
        ),
        horizon=1,
    )

    entries = [trade for trade in result.trades if trade.side == "buy" and trade.reason == "entry"]
    notionals = {trade.symbol: trade.quantity * trade.price for trade in entries}
    assert set(notionals) == {"AAAUSDT", "BBBUSDT"}
    total = sum(notionals.values())
    assert notionals["BBBUSDT"] / total >= 0.25


def test_entry_corr_gate_skips_redundant_same_direction_candidate():
    bars = _two_symbol_bars(periods=8)
    ts = bars["timestamp"].max()
    actions = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": symbol,
                "buy_price": 105.0,
                "sell_price": 112.0,
                "buy_amount": 100.0,
                "trade_amount": 100.0,
            }
            for symbol in ("AAAUSDT", "BBBUSDT")
        ]
    )

    result = run_portfolio_simulation(
        bars,
        actions,
        PortfolioConfig(
            max_positions=2,
            decision_lag_bars=0,
            enforce_market_hours=False,
            close_at_eod=False,
            int_qty=False,
            fee_by_symbol={"AAAUSDT": 0.0, "BBBUSDT": 0.0},
            entry_corr_window_bars=6,
            entry_corr_min_periods=3,
            entry_corr_max_signed=0.5,
            apply_leverage_to_crypto=True,
        ),
        horizon=1,
    )

    entries = [trade for trade in result.trades if trade.side == "buy" and trade.reason == "entry"]
    assert len(entries) == 1


def test_drawdown_entry_scale_linearly_reduces_to_floor():
    assert _drawdown_entry_scale(equity=100.0, peak_equity=100.0, start=0.05, full=0.20, floor=0.4) == 1.0
    assert _drawdown_entry_scale(equity=80.0, peak_equity=100.0, start=0.05, full=0.20, floor=0.4) == pytest.approx(0.4)
    assert _drawdown_entry_scale(equity=87.5, peak_equity=100.0, start=0.05, full=0.20, floor=0.4) == pytest.approx(0.7)
