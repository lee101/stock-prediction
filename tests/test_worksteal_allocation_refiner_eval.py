import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from binance_worksteal.allocation_refiner_eval import (
    ForecastLookup,
    build_feature_row,
    heuristic_chronos_scale,
    heuristic_score_scale,
)
from binance_worksteal.strategy import EntrySizingContext, WorkStealConfig, run_worksteal_backtest


def make_daily_rows(rows, start="2026-01-01", symbol="BTCUSD"):
    dates = pd.date_range(start, periods=len(rows), freq="D", tz="UTC")
    return pd.DataFrame(
        [
            {
                "timestamp": d,
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": 1000.0,
                "symbol": symbol,
            }
            for d, (o, h, l, c) in zip(dates, rows)
        ]
    )


def make_hourly_rows(rows, start="2026-01-01 01:00:00+00:00", symbol="BTCUSD"):
    dates = pd.date_range(start, periods=len(rows), freq="h", tz="UTC")
    return pd.DataFrame(
        [
            {
                "timestamp": d,
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": 250.0,
                "symbol": symbol,
            }
            for d, (o, h, l, c) in zip(dates, rows)
        ]
    )


def test_allocation_scale_one_matches_intraday_baseline():
    daily = {
        "BTCUSD": make_daily_rows(
            [
                (100, 100, 100, 100),
                (110, 110, 110, 110),
                (120, 120, 120, 120),
                (108, 108, 108, 108),
                (112, 112, 112, 112),
            ],
            symbol="BTCUSD",
        )
    }
    intraday = {
        "BTCUSD": make_hourly_rows(
            [
                (108.5, 109.0, 107.8, 108.2),
                (113.0, 114.5, 112.5, 114.0),
            ],
            start="2026-01-04 06:00:00+00:00",
            symbol="BTCUSD",
        )
    }
    config = WorkStealConfig(
        dip_pct=0.10,
        proximity_pct=0.02,
        entry_proximity_bps=float("inf"),
        lookback_days=3,
        profit_target_pct=0.05,
        stop_loss_pct=0.08,
        max_hold_days=5,
    )

    eq_base, trades_base, metrics_base = run_worksteal_backtest(daily, config, intraday_bars=intraday)
    eq_hook, trades_hook, metrics_hook = run_worksteal_backtest(
        daily,
        config,
        intraday_bars=intraday,
        allocation_scale_fn=lambda _context: 1.0,
    )

    assert len(trades_base) == len(trades_hook)
    assert [trade.side for trade in trades_base] == [trade.side for trade in trades_hook]
    assert [trade.timestamp for trade in trades_base] == [trade.timestamp for trade in trades_hook]
    assert [trade.symbol for trade in trades_base] == [trade.symbol for trade in trades_hook]
    assert [trade.quantity for trade in trades_base] == pytest.approx([trade.quantity for trade in trades_hook])
    assert eq_base["equity"].tolist() == pytest.approx(eq_hook["equity"].tolist())
    assert metrics_base["total_return_pct"] == pytest.approx(metrics_hook["total_return_pct"])


def test_allocation_scale_half_reduces_quantity():
    daily = {
        "BTCUSD": make_daily_rows(
            [
                (100, 100, 100, 100),
                (110, 110, 110, 110),
                (120, 120, 120, 120),
                (108, 108, 108, 108),
                (112, 112, 112, 112),
            ],
            symbol="BTCUSD",
        )
    }
    intraday = {
        "BTCUSD": make_hourly_rows(
            [
                (108.5, 109.0, 107.8, 108.2),
                (113.0, 114.5, 112.5, 114.0),
            ],
            start="2026-01-04 06:00:00+00:00",
            symbol="BTCUSD",
        )
    }
    config = WorkStealConfig(
        dip_pct=0.10,
        proximity_pct=0.02,
        entry_proximity_bps=float("inf"),
        lookback_days=3,
        profit_target_pct=0.05,
        stop_loss_pct=0.08,
        max_hold_days=5,
    )

    _, trades_base, _ = run_worksteal_backtest(daily, config, intraday_bars=intraday)
    _, trades_half, _ = run_worksteal_backtest(
        daily,
        config,
        intraday_bars=intraday,
        allocation_scale_fn=lambda _context: 0.5,
    )

    base_buy = next(trade for trade in trades_base if trade.side == "buy")
    half_buy = next(trade for trade in trades_half if trade.side == "buy")
    assert half_buy.quantity == pytest.approx(base_buy.quantity * 0.5)


def test_build_feature_row_uses_forecast_cache(tmp_path):
    cache_dir = tmp_path / "forecast_cache"
    cache_dir.mkdir()
    forecast_ts = pd.Timestamp("2026-01-04 06:00:00+00:00")
    pd.DataFrame(
        [
            {
                "timestamp": forecast_ts,
                "symbol": "BTCUSD",
                "issued_at": forecast_ts - pd.Timedelta(hours=1),
                "target_timestamp": forecast_ts + pd.Timedelta(hours=24),
                "horizon_hours": 24,
                "predicted_close_p50": 114.0,
                "predicted_close_p10": 109.0,
                "predicted_close_p90": 118.0,
                "predicted_high_p50": 116.0,
                "predicted_low_p50": 110.0,
            }
        ]
    ).to_parquet(cache_dir / "BTCUSD.parquet", index=False)

    history = make_daily_rows(
        [
            (100, 101, 99, 100),
            (102, 103, 101, 102),
            (104, 105, 103, 104),
            (106, 107, 105, 106),
            (108, 109, 107, 108),
        ],
        symbol="BTCUSD",
    )
    signal_bar = history.iloc[-1]
    execution_bar = pd.Series({"open": 108.5, "high": 109.0, "low": 107.8, "close": 108.2})
    context = EntrySizingContext(
        timestamp=forecast_ts,
        signal_timestamp=pd.Timestamp("2026-01-04 00:00:00+00:00"),
        symbol="BTCUSD",
        direction="long",
        score=0.01,
        fill_price=108.0,
        candidate_rank=1,
        candidate_count=3,
        slots_remaining=2,
        current_position_count=1,
        cash=7000.0,
        base_equity=10000.0,
        current_equity=10000.0,
        market_breadth=0.2,
        hold_base_asset=False,
        signal_bar=signal_bar,
        history=history,
        execution_bar=execution_bar,
    )

    lookup = ForecastLookup(cache_dir)
    features = build_feature_row(context, lookup)

    assert features["has_forecast"] == pytest.approx(1.0)
    assert features["forecast_close_delta"] > 0.0
    assert 0.0 <= heuristic_score_scale(features) <= 1.35
    assert 0.0 <= heuristic_chronos_scale(features) <= 1.35
