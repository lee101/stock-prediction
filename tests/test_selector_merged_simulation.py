from __future__ import annotations

import pandas as pd
import pytest

from newnanoalpacahourlyexp.marketsimulator.selector import (
    SelectionConfig,
    run_best_trade_simulation,
    run_best_trade_simulation_merged,
)


@pytest.mark.unit
def test_run_best_trade_simulation_merged_matches_unmerged() -> None:
    ts0 = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
    ts1 = pd.Timestamp("2020-01-01 01:00:00", tz="UTC")

    bars = pd.DataFrame(
        [
            {
                "timestamp": ts0,
                "symbol": "BTCUSD",
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "predicted_high_p50_h1": 110.0,
                "predicted_low_p50_h1": 98.0,
                "predicted_close_p50_h1": 105.0,
            },
            {
                "timestamp": ts1,
                "symbol": "BTCUSD",
                "high": 111.0,
                "low": 100.0,
                "close": 110.0,
                "predicted_high_p50_h1": 112.0,
                "predicted_low_p50_h1": 100.0,
                "predicted_close_p50_h1": 110.0,
            },
            {
                "timestamp": ts0,
                "symbol": "ETHUSD",
                "high": 51.0,
                "low": 49.0,
                "close": 50.0,
                "predicted_high_p50_h1": 52.0,
                "predicted_low_p50_h1": 48.0,
                "predicted_close_p50_h1": 50.5,
            },
            {
                "timestamp": ts1,
                "symbol": "ETHUSD",
                "high": 51.0,
                "low": 49.0,
                "close": 50.0,
                "predicted_high_p50_h1": 52.0,
                "predicted_low_p50_h1": 48.0,
                "predicted_close_p50_h1": 50.0,
            },
        ]
    )

    actions = pd.DataFrame(
        [
            {
                "timestamp": ts0,
                "symbol": "BTCUSD",
                "buy_price": 99.5,
                "sell_price": 109.0,
                "buy_amount": 1.0,
                "sell_amount": 0.0,
            },
            {
                "timestamp": ts1,
                "symbol": "BTCUSD",
                "buy_price": 0.0,
                "sell_price": 109.0,
                "buy_amount": 0.0,
                "sell_amount": 1.0,
            },
            {
                "timestamp": ts0,
                "symbol": "ETHUSD",
                "buy_price": 50.5,
                "sell_price": 55.0,
                "buy_amount": 1.0,
                "sell_amount": 0.0,
            },
            {
                "timestamp": ts1,
                "symbol": "ETHUSD",
                "buy_price": 0.0,
                "sell_price": 55.0,
                "buy_amount": 0.0,
                "sell_amount": 0.0,
            },
        ]
    )

    cfg = SelectionConfig(
        initial_cash=1_000.0,
        min_edge=0.0,
        risk_weight=0.5,
        edge_mode="high_low",
        enforce_market_hours=False,
        close_at_eod=False,
        fee_by_symbol={"BTCUSD": 0.0, "ETHUSD": 0.0},
        periods_per_year_by_symbol={"BTCUSD": 24 * 365, "ETHUSD": 24 * 365},
        symbols=["BTCUSD", "ETHUSD"],
    )

    unmerged = run_best_trade_simulation(bars, actions, cfg, horizon=1)
    merged_df = bars.merge(actions, on=["timestamp", "symbol"], how="inner")
    merged = run_best_trade_simulation_merged(merged_df, cfg, horizon=1)

    assert merged.open_symbol == unmerged.open_symbol
    assert merged.final_cash == pytest.approx(unmerged.final_cash, rel=0.0, abs=1e-12)
    assert merged.final_inventory == pytest.approx(unmerged.final_inventory, rel=0.0, abs=1e-12)
    assert merged.metrics["total_return"] == pytest.approx(unmerged.metrics["total_return"], rel=0.0, abs=1e-12)
    assert merged.metrics["sortino"] == pytest.approx(unmerged.metrics["sortino"], rel=0.0, abs=1e-12)
    assert merged.metrics["mean_hourly_return"] == pytest.approx(
        unmerged.metrics["mean_hourly_return"], rel=0.0, abs=1e-12
    )


@pytest.mark.unit
def test_run_best_trade_simulation_supports_seeded_initial_position() -> None:
    ts0 = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
    ts1 = pd.Timestamp("2020-01-01 01:00:00", tz="UTC")

    merged = pd.DataFrame(
        [
            {
                "timestamp": ts0,
                "symbol": "BTCUSD",
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "buy_price": 99.5,
                "sell_price": 101.0,
                "buy_amount": 0.0,
                "sell_amount": 0.0,
                "predicted_high_p50_h1": 101.0,
                "predicted_low_p50_h1": 99.0,
                "predicted_close_p50_h1": 100.0,
            },
            {
                "timestamp": ts1,
                "symbol": "BTCUSD",
                "high": 111.0,
                "low": 109.0,
                "close": 110.0,
                "buy_price": 109.5,
                "sell_price": 111.0,
                "buy_amount": 0.0,
                "sell_amount": 0.0,
                "predicted_high_p50_h1": 111.0,
                "predicted_low_p50_h1": 109.0,
                "predicted_close_p50_h1": 110.0,
            },
        ]
    )

    cfg = SelectionConfig(
        initial_cash=500.0,
        initial_inventory=1.0,
        initial_symbol="BTCUSD",
        max_hold_hours=1,
        min_edge=0.0,
        risk_weight=0.0,
        edge_mode="high_low",
        enforce_market_hours=False,
        close_at_eod=False,
        fee_by_symbol={"BTCUSD": 0.0},
        periods_per_year_by_symbol={"BTCUSD": 24 * 365},
        symbols=["BTCUSD"],
    )

    result = run_best_trade_simulation_merged(merged, cfg, horizon=1)

    assert result.trades[-1].reason == "max_hold"
    assert result.trades[-1].side == "sell"
    assert result.trades[-1].price == pytest.approx(110.0)
    assert result.final_cash == pytest.approx(610.0)
    assert result.final_inventory == pytest.approx(0.0)
    assert result.open_symbol is None
    assert result.metrics["total_return"] == pytest.approx((610.0 - 600.0) / 600.0)


@pytest.mark.unit
def test_run_best_trade_simulation_requires_symbol_for_seeded_position() -> None:
    ts0 = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
    merged = pd.DataFrame(
        [
            {
                "timestamp": ts0,
                "symbol": "BTCUSD",
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "buy_price": 99.5,
                "sell_price": 101.0,
                "buy_amount": 0.0,
                "sell_amount": 0.0,
                "predicted_high_p50_h1": 101.0,
                "predicted_low_p50_h1": 99.0,
                "predicted_close_p50_h1": 100.0,
            }
        ]
    )

    cfg = SelectionConfig(
        initial_cash=500.0,
        initial_inventory=1.0,
        min_edge=0.0,
        risk_weight=0.0,
        edge_mode="high_low",
        enforce_market_hours=False,
        close_at_eod=False,
        fee_by_symbol={"BTCUSD": 0.0},
        periods_per_year_by_symbol={"BTCUSD": 24 * 365},
        symbols=["BTCUSD"],
    )

    with pytest.raises(ValueError, match="initial_symbol is required"):
        run_best_trade_simulation_merged(merged, cfg, horizon=1)
