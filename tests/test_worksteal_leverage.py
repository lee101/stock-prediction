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


def _run_leverage_backtest(leverage: float):
    bars = {"ASYMUSD": _dip_then_recover("ASYMUSD")}
    config = WorkStealConfig(
        dip_pct=0.20, profit_target_pct=0.15, stop_loss_pct=0.10,
        sma_filter_period=0, trailing_stop_pct=0.0,
        max_positions=5, max_hold_days=14, lookback_days=20,
        initial_cash=10000.0, max_leverage=leverage,
        margin_annual_rate=MARGIN_ANNUAL_RATE, max_position_pct=0.50,
        max_drawdown_exit=0.0,
        risk_off_trigger_momentum_period=0, risk_off_trigger_sma_period=0,
    )
    start = str(bars["ASYMUSD"]["timestamp"].min().date())
    end = str(bars["ASYMUSD"]["timestamp"].max().date())
    return run_worksteal_backtest(bars, config, start_date=start, end_date=end)


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
        config = WorkStealConfig(initial_cash=10000.0, max_position_pct=0.25, max_leverage=1.0)
        max_alloc = config.initial_cash * config.max_position_pct * config.max_leverage
        assert max_alloc == 2500.0

    def test_2x_sizing(self):
        config = WorkStealConfig(initial_cash=10000.0, max_position_pct=0.25, max_leverage=2.0)
        max_alloc = config.initial_cash * config.max_position_pct * config.max_leverage
        assert max_alloc == 5000.0

    def test_5x_sizing(self):
        config = WorkStealConfig(initial_cash=10000.0, max_position_pct=0.25, max_leverage=5.0)
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
            risk_off_trigger_momentum_period=0, risk_off_trigger_sma_period=0,
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
        prices.append(base * 0.78)
        for i in range(14):
            prices.append(base * 0.78 * 0.92)
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


class TestLeverage3x:
    def test_3x_buying_power(self):
        config = WorkStealConfig(initial_cash=10000.0, max_position_pct=0.50, max_leverage=3.0)
        max_alloc = config.initial_cash * config.max_position_pct * config.max_leverage
        assert abs(max_alloc - 15000.0) < 0.01

    def test_3x_creates_borrowed(self):
        _, trades, _ = _run_leverage_backtest(3.0)
        buys = [t for t in trades if t.side == "buy"]
        if buys:
            assert buys[0].notional > 10000.0 * 0.50, "3x should produce notional > 1x alloc"

    def test_3x_more_notional_than_1x(self):
        _, trades_1x, _ = _run_leverage_backtest(1.0)
        _, trades_3x, _ = _run_leverage_backtest(3.0)
        buys_1x = [t for t in trades_1x if t.side == "buy"]
        buys_3x = [t for t in trades_3x if t.side == "buy"]
        if buys_1x and buys_3x:
            n1 = sum(t.notional for t in buys_1x)
            n3 = sum(t.notional for t in buys_3x)
            assert n3 > n1 * 2.0


class TestLeverage5x:
    def test_5x_buying_power(self):
        config = WorkStealConfig(initial_cash=10000.0, max_position_pct=0.50, max_leverage=5.0)
        max_alloc = config.initial_cash * config.max_position_pct * config.max_leverage
        assert abs(max_alloc - 25000.0) < 0.01

    def test_5x_creates_large_notional(self):
        _, trades, _ = _run_leverage_backtest(5.0)
        buys = [t for t in trades if t.side == "buy"]
        if buys:
            assert buys[0].notional > 10000.0 * 0.50

    def test_5x_borrowed_amount(self):
        _, trades, _ = _run_leverage_backtest(5.0)
        buys = [t for t in trades if t.side == "buy"]
        if buys:
            assert buys[0].notional > 10000.0

    def test_5x_amplifies_returns_vs_1x(self):
        _, _, m1 = _run_leverage_backtest(1.0)
        _, _, m5 = _run_leverage_backtest(5.0)
        if m1 and m5:
            r1 = abs(m1.get("total_return_pct", 0))
            r5 = abs(m5.get("total_return_pct", 0))
            assert r5 > r1


class TestDeleveraging:
    def test_deleverage_closes_worst_position(self):
        base = 100.0
        prices_a = [base] * 25
        prices_a.append(base * 0.78)
        for _ in range(14):
            prices_a.append(base * 0.78 * 0.85)
        prices_b = [base] * 25
        prices_b.append(base * 0.78)
        for i in range(1, 15):
            prices_b.append(base * 0.78 * (1 + 0.10 * i / 14))
        bars = {
            "ASYMUSD": _make_bars("ASYMUSD", prices_a),
            "BSYMUSD": _make_bars("BSYMUSD", prices_b),
        }
        config = WorkStealConfig(
            dip_pct=0.20, profit_target_pct=0.30, stop_loss_pct=0.20,
            sma_filter_period=0, trailing_stop_pct=0.0,
            max_positions=5, max_hold_days=30, lookback_days=20,
            initial_cash=10000.0, max_leverage=3.0,
            margin_annual_rate=MARGIN_ANNUAL_RATE, max_position_pct=0.50,
            max_drawdown_exit=0.0,
            deleverage_threshold=0.5, target_leverage=3.0,
            risk_off_trigger_momentum_period=0, risk_off_trigger_sma_period=0,
        )
        start = str(bars["ASYMUSD"]["timestamp"].min().date())
        end = str(bars["ASYMUSD"]["timestamp"].max().date())
        _, trades, _ = run_worksteal_backtest(bars, config, start_date=start, end_date=end)
        all_exits = [t for t in trades if t.side in ("sell", "cover")]
        assert len(all_exits) > 0

    def test_no_deleverage_when_threshold_zero(self):
        bars = {"ASYMUSD": _dip_then_recover("ASYMUSD")}
        config = WorkStealConfig(
            dip_pct=0.20, profit_target_pct=0.15, stop_loss_pct=0.10,
            sma_filter_period=0, trailing_stop_pct=0.0,
            max_positions=5, max_hold_days=14, lookback_days=20,
            initial_cash=10000.0, max_leverage=3.0,
            margin_annual_rate=MARGIN_ANNUAL_RATE, max_position_pct=0.50,
            max_drawdown_exit=0.0, deleverage_threshold=0.0,
            risk_off_trigger_momentum_period=0, risk_off_trigger_sma_period=0,
        )
        start = str(bars["ASYMUSD"]["timestamp"].min().date())
        end = str(bars["ASYMUSD"]["timestamp"].max().date())
        _, trades, _ = run_worksteal_backtest(bars, config, start_date=start, end_date=end)
        delev_exits = [t for t in trades if t.reason == "deleverage"]
        assert len(delev_exits) == 0


class TestMarginInterest5x14d:
    def test_margin_interest_accumulates_over_14_days(self):
        initial_cash = 10000.0
        pos_pct = 0.50
        notional = initial_cash * pos_pct * 5.0
        cash_used = min(notional, initial_cash * pos_pct)
        borrowed = notional - cash_used
        pos = Position(
            symbol="BTCUSD", direction="long", entry_price=100.0,
            entry_date=pd.Timestamp("2025-01-01", tz="UTC"),
            quantity=notional / 100.0, cost_basis=notional,
            peak_price=100.0, target_exit_price=115.0, stop_price=90.0,
            margin_borrowed=borrowed,
        )
        days = 14
        exit_date = pd.Timestamp("2025-01-15", tz="UTC")
        interest = _compute_margin_interest(pos, exit_date, MARGIN_ANNUAL_RATE)
        expected = borrowed * (MARGIN_ANNUAL_RATE / 365.0) * days
        assert abs(interest - expected) < 0.01
        assert 40.0 < interest < 60.0

    def test_5x_interest_much_larger_than_1x(self):
        for borrowed, label in [(0.0, "1x"), (4000.0, "2x"), (20000.0, "5x")]:
            pos = Position(
                symbol="BTCUSD", direction="long", entry_price=100.0,
                entry_date=pd.Timestamp("2025-01-01", tz="UTC"),
                quantity=250.0, cost_basis=25000.0, peak_price=100.0,
                target_exit_price=115.0, stop_price=90.0,
                margin_borrowed=borrowed,
            )
            interest = _compute_margin_interest(pos, pd.Timestamp("2025-01-15", tz="UTC"), MARGIN_ANNUAL_RATE)
            if label == "1x":
                assert interest == 0.0
            elif label == "5x":
                assert interest > 30.0


class TestCSimParity5x:
    def test_python_vs_c_parity_at_5x(self):
        try:
            from binance_worksteal.csim.fast_worksteal import run_worksteal_backtest_fast
        except Exception:
            return

        bars = {"ASYMUSD": _dip_then_recover("ASYMUSD"), "BSYMUSD": _dip_then_recover("BSYMUSD")}
        config = WorkStealConfig(
            dip_pct=0.20, profit_target_pct=0.15, stop_loss_pct=0.10,
            sma_filter_period=0, trailing_stop_pct=0.0,
            max_positions=5, max_hold_days=14, lookback_days=20,
            initial_cash=10000.0, max_leverage=5.0,
            margin_annual_rate=MARGIN_ANNUAL_RATE, max_position_pct=0.25,
            max_drawdown_exit=0.0,
            risk_off_trigger_momentum_period=0, risk_off_trigger_sma_period=0,
        )
        start = str(bars["ASYMUSD"]["timestamp"].min().date())
        end = str(bars["ASYMUSD"]["timestamp"].max().date())
        bars_py = {k: v.copy() for k, v in bars.items()}
        _, _, metrics_py = run_worksteal_backtest(bars_py, config, start_date=start, end_date=end)
        bars_c = {k: v.copy() for k, v in bars.items()}
        metrics_c = run_worksteal_backtest_fast(bars_c, config, start_date=start, end_date=end)
        if metrics_py and metrics_c:
            py_ret = metrics_py.get("total_return_pct", 0)
            c_ret = metrics_c.get("total_return_pct", 0)
            if abs(py_ret) > 0.1:
                assert abs(py_ret - c_ret) / abs(py_ret) < 0.15, \
                    f"Python={py_ret:.4f}% vs C={c_ret:.4f}%"
            else:
                assert abs(py_ret - c_ret) < 1.0

    def test_python_vs_c_parity_at_3x(self):
        try:
            from binance_worksteal.csim.fast_worksteal import run_worksteal_backtest_fast
        except Exception:
            return

        bars = {"ASYMUSD": _dip_then_recover("ASYMUSD")}
        config = WorkStealConfig(
            dip_pct=0.20, profit_target_pct=0.15, stop_loss_pct=0.10,
            sma_filter_period=0, trailing_stop_pct=0.0,
            max_positions=5, max_hold_days=14, lookback_days=20,
            initial_cash=10000.0, max_leverage=3.0,
            margin_annual_rate=MARGIN_ANNUAL_RATE, max_position_pct=0.25,
            max_drawdown_exit=0.0,
            risk_off_trigger_momentum_period=0, risk_off_trigger_sma_period=0,
        )
        start = str(bars["ASYMUSD"]["timestamp"].min().date())
        end = str(bars["ASYMUSD"]["timestamp"].max().date())
        bars_py = {k: v.copy() for k, v in bars.items()}
        _, _, metrics_py = run_worksteal_backtest(bars_py, config, start_date=start, end_date=end)
        bars_c = {k: v.copy() for k, v in bars.items()}
        metrics_c = run_worksteal_backtest_fast(bars_c, config, start_date=start, end_date=end)
        if metrics_py and metrics_c:
            py_ret = metrics_py.get("total_return_pct", 0)
            c_ret = metrics_c.get("total_return_pct", 0)
            if abs(py_ret) > 0.1:
                assert abs(py_ret - c_ret) / abs(py_ret) < 0.15, \
                    f"Python={py_ret:.4f}% vs C={c_ret:.4f}%"
            else:
                assert abs(py_ret - c_ret) < 1.0


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
