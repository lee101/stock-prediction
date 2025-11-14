"""Tests for global PnL gating logic used in strategytraining sizing tests."""

from __future__ import annotations

import pandas as pd
import pytest

from marketsimulator.sizing_strategies import FixedFractionStrategy
from strategytraining.test_sizing_on_precomputed_pnl import (
    GlobalGateConfig,
    PrecomputedPnLSizingTester,
    build_daily_metrics_df,
)


def _build_trades(pnls: list[float], initial_capital: float = 10_000.0) -> pd.DataFrame:
    """Create a minimal trades dataframe covering sequential days."""

    entry_price = 100.0
    baseline_fraction = 0.5
    position_size = (baseline_fraction * initial_capital) / entry_price

    rows = []
    start_ts = pd.Timestamp("2025-01-01T10:00:00Z")

    for offset, pnl in enumerate(pnls):
        timestamp = start_ts + pd.Timedelta(days=offset)
        pnl_pct = pnl / (position_size * entry_price) if position_size else 0.0
        rows.append(
            {
                "entry_timestamp": timestamp.isoformat(),
                "symbol": "AAPL",
                "is_crypto": False,
                "pnl_pct": pnl_pct,
                "position_size": position_size,
                "entry_price": entry_price,
                "pnl": pnl,
            }
        )

    return pd.DataFrame(rows)


def test_day_positive_probe_gates_next_day_after_loss():
    """Day+1 probe gate should scale positions after a losing day."""

    trades = _build_trades([1000.0, -2000.0, 1500.0])
    tester = PrecomputedPnLSizingTester(trades, initial_capital=10_000.0)
    gate = GlobalGateConfig(
        name="day_probe",
        window_days=1,
        fail_mode="probe",
        probe_fraction=0.1,
        min_positive=1e-9,
    )

    result = tester.run_strategy(
        FixedFractionStrategy(0.5),
        "Fixed_50pct_DayProbeTest",
        gate_config=gate,
    )

    # Previous day's loss should force exactly one probe day
    assert result.gate_probe_days == 1
    assert result.gate_blocked_days == 0
    # Day3 profits shrink to 10% of baseline -> 1000 - 2000 + 150 = -850
    assert result.total_pnl == pytest.approx(-850.0)


def test_two_day_block_halts_trading_until_window_positive():
    """Two-day block should skip trades until trailing window > 0."""

    trades = _build_trades([-500.0, -100.0, 1200.0, 800.0])
    tester = PrecomputedPnLSizingTester(trades, initial_capital=10_000.0)
    gate = GlobalGateConfig(
        name="two_day_block",
        window_days=2,
        fail_mode="block",
        min_positive=1e-9,
    )

    result = tester.run_strategy(
        FixedFractionStrategy(0.5),
        "Fixed_50pct_BlockTest",
        gate_config=gate,
    )

    assert result.gate_blocked_days == 1
    assert result.gate_probe_days == 0
    # Day3 is fully blocked (0 PnL), trading resumes Day4 => -500 -100 + 0 + 800
    assert result.total_pnl == pytest.approx(200.0)


def test_unprofit_shutdown_blocks_based_on_strategy_pnl():
    """Dynamic gate should use realized strategy PnL, not dataset baseline."""

    trades = _build_trades([-100.0, -200.0, 500.0, 600.0], initial_capital=5_000.0)
    tester = PrecomputedPnLSizingTester(trades, initial_capital=5_000.0)
    gate = GlobalGateConfig(
        name="unprofit",
        window_days=2,
        fail_mode="block",
        min_positive=1e-9,
        use_strategy_pnl=True,
    )

    result = tester.run_strategy(
        FixedFractionStrategy(0.5),
        "Fixed_50pct_UnprofitTest",
        gate_config=gate,
    )

    # After two losing days, third day should be blocked entirely
    assert result.gate_blocked_days >= 1
    # total PnL should exclude day3 but include day4 profits
    assert result.total_pnl == pytest.approx(300.0)


def test_stock_dir_shutdown_blocks_symbol_trades_only():
    """Symbol-level gate should block trades without affecting other symbols."""

    trades = _build_trades([-50.0, -60.0, 400.0, 500.0], initial_capital=4_000.0)
    tester = PrecomputedPnLSizingTester(trades, initial_capital=4_000.0)
    gate = GlobalGateConfig(
        name="stockdir",
        window_days=2,
        window_trades=2,
        fail_mode="block",
        scope="symbol_side",
        use_strategy_pnl=True,
    )

    result = tester.run_strategy(
        FixedFractionStrategy(0.5),
        "Fixed_50pct_StockDirTest",
        gate_config=gate,
    )

    assert result.symbol_gate_blocks >= 1
    assert result.gate_blocked_days == 0


def test_daily_curve_and_metrics_dataframe_alignment():
    """Daily curve should record rolling Sharpe and export cleanly."""

    trades = _build_trades([100.0, -50.0, 75.0])
    tester = PrecomputedPnLSizingTester(trades, initial_capital=5_000.0)

    result = tester.run_strategy(
        FixedFractionStrategy(0.5),
        "Fixed_50pct_CurveTest",
    )

    assert len(result.daily_curve) == 3
    assert result.daily_curve[0]['rolling_sharpe'] == 0.0
    assert all('rolling_sharpe' in point for point in result.daily_curve)
    assert all('rolling_sortino' in point for point in result.daily_curve)
    assert all('rolling_ann_return' in point for point in result.daily_curve)
    assert result.sortino_ratio is not None
    assert result.annualized_return_pct is not None

    df = build_daily_metrics_df([result])
    assert len(df) == 3
    assert df['strategy'].unique().tolist() == ["Fixed_50pct_CurveTest"]
    assert set(df['mode']) == {"normal"}
    assert {'rolling_sortino', 'rolling_ann_return', 'annualization_days', 'day_class'} <= set(df.columns)
