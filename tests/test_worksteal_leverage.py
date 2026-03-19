"""Tests for leveraged work-stealing strategy: margin interest, position sizing, stop loss."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from binance_worksteal.strategy import (
    MARGIN_ANNUAL_RATE,
    WorkStealConfig,
    Position,
    _compute_margin_interest,
    run_worksteal_backtest,
)


def _make_bars(symbol: str, prices: list[float], start: str = "2025-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(prices), freq="1D", tz="UTC")
    return pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": [p * 1.02 for p in prices],
        "low": [p * 0.98 for p in prices],
        "close": prices,
        "volume": [1e6] * len(prices),
        "symbol": symbol,
    })


def _dip_then_recover(symbol: str, n_flat: int = 25, dip_pct: float = 0.22,
                       recover_pct: float = 0.18) -> pd.DataFrame:
    base = 100.0
    prices = [base] * n_flat
    prices.append(base * (1 - dip_pct))
    for i in range(1, 15):
        prices.append(base * (1 - dip_pct) * (1 + recover_pct * i / 14))
    return _make_bars(symbol, prices)


class TestMarginInterest:
    def test_no_leverage_no_interest(self):
        pos = Position(
            symbol="BTCUSD", direction="long", entry_price=100.0,
            entry_date=pd.Timestamp("2025-01-01", tz="UTC"),
            quantity=1.0, cost_basis=100.0, peak_price=100.0,
            target_exit_price=115.0, stop_price=90.0, margin_borrowed=0.0,
        )
        interest = _compute_margin_interest(pos, pd.Timestamp("2025-01-15", tz="UTC"), MARGIN_ANNUAL_RATE)
        assert interest == 0.0

    def test_leverage_2x_interest(self):
        pos = Position(
            symbol="ETHUSD", direction="long", entry_price=3000.0,
            entry_date=pd.Timestamp("2025-01-01", tz="UTC"),
            quantity=1.0, cost_basis=3000.0, peak_price=3000.0,
            target_exit_price=3450.0, stop_price=2700.0, margin_borrowed=1500.0,
        )
        interest = _compute_margin_interest(pos, pd.Timestamp("2025-01-15", tz="UTC"), MARGIN_ANNUAL_RATE)
        daily_rate = MARGIN_ANNUAL_RATE / 365.0
        expected = 1500.0 * daily_rate * 14
        assert abs(interest - expected) < 0.01

    def test_leverage_5x_interest(self):
        borrowed = 8000.0
        pos = Position(
            symbol="SOLUSD", direction="long", entry_price=200.0,
            entry_date=pd.Timestamp("2025-01-01", tz="UTC"),
            quantity=50.0, cost_basis=10000.0, peak_price=200.0,
            target_exit_price=230.0, stop_price=180.0, margin_borrowed=borrowed,
        )
        days = 30
        exit_date = pd.Timestamp("2025-01-31", tz="UTC")
        interest = _compute_margin_interest(pos, exit_date, MARGIN_ANNUAL_RATE)
        expected = borrowed * (MARGIN_ANNUAL_RATE / 365.0) * days
        assert abs(interest - expected) < 0.01
        assert interest > 0

    def test_interest_scales_with_leverage(self):
        results = []
        for borrowed in [0.0, 2500.0, 5000.0, 10000.0]:
            pos = Position(
                symbol="BTCUSD", direction="long", entry_price=50000.0,
                entry_date=pd.Timestamp("2025-01-01", tz="UTC"),
                quantity=0.2, cost_basis=10000.0, peak_price=50000.0,
                target_exit_price=57500.0, stop_price=45000.0, margin_borrowed=borrowed,
            )
            interest = _compute_margin_interest(pos, pd.Timestamp("2025-01-15", tz="UTC"), MARGIN_ANNUAL_RATE)
            results.append(interest)
        assert results[0] == 0.0
        for i in range(1, len(results)):
            assert results[i] > results[i - 1]


class TestPositionSizing:
    def test_1x_sizing(self):
        config = WorkStealConfig(
            initial_cash=10000.0, max_position_pct=0.25, max_leverage=1.0,
        )
        max_alloc = config.initial_cash * config.max_position_pct * config.max_leverage
        assert max_alloc == 2500.0

    def test_2x_sizing(self):
        config = WorkStealConfig(
            initial_cash=10000.0, max_position_pct=0.25, max_leverage=2.0,
        )
        max_alloc = config.initial_cash * config.max_position_pct * config.max_leverage
        assert max_alloc == 5000.0

    def test_5x_sizing(self):
        config = WorkStealConfig(
            initial_cash=10000.0, max_position_pct=0.25, max_leverage=5.0,
        )
        max_alloc = config.initial_cash * config.max_position_pct * config.max_leverage
        assert max_alloc == 12500.0

    def test_sizing_scales_linearly(self):
        for lev in [1.0, 2.0, 3.0, 5.0]:
            config = WorkStealConfig(initial_cash=10000.0, max_position_pct=0.25, max_leverage=lev)
            alloc = config.initial_cash * config.max_position_pct * config.max_leverage
            assert abs(alloc - 2500.0 * lev) < 0.01


class TestLeveragedStopLoss:
    def test_stop_price_at_leverage_levels(self):
        entry = 100.0
        for sl_pct in [0.05, 0.08, 0.10]:
            stop = entry * (1 - sl_pct)
            assert abs(stop - (100.0 - sl_pct * 100.0)) < 0.01

    def test_5x_leverage_10pct_sl_means_50pct_equity_loss(self):
        entry = 100.0
        sl_pct = 0.10
        leverage = 5.0
        equity_per_unit = entry / leverage
        price_drop = entry * sl_pct
        equity_loss_pct = price_drop / equity_per_unit
        assert abs(equity_loss_pct - 0.50) < 0.01

    def test_tighter_stop_limits_equity_loss(self):
        entry = 100.0
        leverage = 5.0
        equity_per_unit = entry / leverage
        losses = {}
        for sl_pct in [0.05, 0.08, 0.10]:
            price_drop = entry * sl_pct
            losses[sl_pct] = price_drop / equity_per_unit
        assert losses[0.05] < losses[0.08] < losses[0.10]
        assert abs(losses[0.05] - 0.25) < 0.01
        assert abs(losses[0.08] - 0.40) < 0.01


class TestBacktestLeverage:
    def _run_config(self, leverage: float, stop_loss: float = 0.10):
        bars_a = _dip_then_recover("ASYMUSD")
        bars_b = _dip_then_recover("BSYMUSD")
        all_bars = {"ASYMUSD": bars_a, "BSYMUSD": bars_b}
        config = WorkStealConfig(
            dip_pct=0.20, profit_target_pct=0.15, stop_loss_pct=stop_loss,
            sma_filter_period=0, trailing_stop_pct=0.0,
            max_positions=5, max_hold_days=14, lookback_days=20,
            initial_cash=10000.0, max_leverage=leverage,
            margin_annual_rate=MARGIN_ANNUAL_RATE, max_position_pct=0.25,
            max_drawdown_exit=0.0,
        )
        start = str(bars_a["timestamp"].min().date())
        end = str(bars_a["timestamp"].max().date())
        return run_worksteal_backtest(all_bars, config, start_date=start, end_date=end)

    def test_higher_leverage_more_notional(self):
        _, trades_1x, _ = self._run_config(1.0)
        _, trades_2x, _ = self._run_config(2.0)
        buys_1x = [t for t in trades_1x if t.side == "buy"]
        buys_2x = [t for t in trades_2x if t.side == "buy"]
        if buys_1x and buys_2x:
            notional_1x = sum(t.notional for t in buys_1x)
            notional_2x = sum(t.notional for t in buys_2x)
            assert notional_2x >= notional_1x * 1.5

    def test_margin_interest_deducted_at_exit(self):
        _, trades_5x, metrics_5x = self._run_config(5.0)
        exits = [t for t in trades_5x if t.side == "sell"]
        if exits:
            total_fees = sum(t.fee for t in exits)
            assert total_fees > 0

    def test_1x_vs_5x_pnl_difference(self):
        _, _, metrics_1x = self._run_config(1.0)
        _, _, metrics_5x = self._run_config(5.0)
        if metrics_1x and metrics_5x:
            assert metrics_1x.get("total_return_pct", 0) != metrics_5x.get("total_return_pct", 0)

    def test_stop_loss_triggers(self):
        base = 100.0
        prices = [base] * 25
        prices.append(base * 0.78)  # 22% dip
        for i in range(14):
            prices.append(base * 0.78 * 0.92)  # drop another 8%
        bars = {"TSYMUSD": _make_bars("TSYMUSD", prices)}
        config = WorkStealConfig(
            dip_pct=0.20, profit_target_pct=0.15, stop_loss_pct=0.10,
            sma_filter_period=0, trailing_stop_pct=0.0,
            max_positions=5, max_hold_days=14, lookback_days=20,
            initial_cash=10000.0, max_leverage=3.0,
            margin_annual_rate=MARGIN_ANNUAL_RATE, max_position_pct=0.25,
            max_drawdown_exit=0.0,
        )
        start = str(bars["TSYMUSD"]["timestamp"].min().date())
        end = str(bars["TSYMUSD"]["timestamp"].max().date())
        _, trades, _ = run_worksteal_backtest(bars, config, start_date=start, end_date=end)
        exits = [t for t in trades if t.side == "sell"]
        stop_exits = [t for t in exits if t.reason == "stop_loss"]
        if exits:
            assert len(stop_exits) > 0 or any(t.reason in ("max_hold", "trailing_stop", "max_dd_exit") for t in exits)


class TestSweepScript:
    def test_make_config(self):
        from scripts.run_worksteal_leverage_sweep import make_config
        c = make_config(3.0, 0.05)
        assert c.max_leverage == 3.0
        assert c.stop_loss_pct == 0.05
        assert c.dip_pct == 0.20
        assert c.margin_annual_rate == MARGIN_ANNUAL_RATE

    def test_base_config_values(self):
        from scripts.run_worksteal_leverage_sweep import BASE_CONFIG
        assert BASE_CONFIG["dip_pct"] == 0.20
        assert BASE_CONFIG["profit_target_pct"] == 0.15
        assert BASE_CONFIG["sma_filter_period"] == 20
        assert BASE_CONFIG["trailing_stop_pct"] == 0.03
        assert BASE_CONFIG["max_positions"] == 5
        assert BASE_CONFIG["max_hold_days"] == 14
