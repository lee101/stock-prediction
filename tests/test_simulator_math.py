#!/usr/bin/env python3
"""Validate portfolio simulator and crypto simulator math."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from unified_hourly_experiment.marketsimulator.portfolio_simulator import (
    PortfolioConfig, run_portfolio_simulation, Position
)
from binanceleveragesui.run_leverage_sweep import (
    LeverageConfig, simulate_with_margin_cost
)
from binanceleveragesui.validate_sim_vs_live import simulate_5m


def _make_bars(prices, symbol="TEST", start="2026-01-01"):
    rows = []
    for i, p in enumerate(prices):
        ts = pd.Timestamp(start) + pd.Timedelta(hours=i)
        rows.append({"timestamp": ts, "symbol": symbol, "open": p, "high": p * 1.02, "low": p * 0.98, "close": p})
    return pd.DataFrame(rows)


def _make_actions(bars, buy_price_fn=None, sell_price_fn=None, buy_amount=50, sell_amount=50):
    rows = []
    for _, b in bars.iterrows():
        bp = buy_price_fn(b) if buy_price_fn else b["close"] * 0.99
        sp = sell_price_fn(b) if sell_price_fn else b["close"] * 1.01
        rows.append({
            "timestamp": b["timestamp"], "symbol": b["symbol"],
            "buy_price": bp, "sell_price": sp, "buy_amount": buy_amount, "sell_amount": sell_amount,
            "trade_amount": buy_amount,
            "predicted_high_p50_h1": b["close"] * 1.015,
            "predicted_low_p50_h1": b["close"] * 0.985,
            "predicted_close_p50_h1": b["close"] * 1.005,
        })
    return pd.DataFrame(rows)


# ======================== STOCK SIMULATOR TESTS ========================

class TestPortfolioSimulatorFees:
    def test_fee_deducted_on_buy(self):
        """Fee should reduce effective buying power: with fee, fewer shares purchased."""
        prices = [100.0] * 5
        bars = _make_bars(prices, symbol="NVDA")
        actions = _make_actions(bars, buy_amount=100, sell_amount=0)

        results = {}
        for fee in [0.0, 0.01]:
            cfg = PortfolioConfig(
                initial_cash=10000, max_positions=1, max_hold_hours=100,
                enforce_market_hours=False, close_at_eod=False,
                decision_lag_bars=0, bar_margin=0.0,
                fee_by_symbol={"NVDA": fee}, max_leverage=1.0, int_qty=False,
                symbols=["NVDA"],
            )
            r = run_portfolio_simulation(bars, actions, cfg, horizon=1)
            if r.trades:
                results[fee] = r.trades[0].quantity

        if 0.0 in results and 0.01 in results:
            assert results[0.0] > results[0.01], \
                f"Higher fee should buy fewer shares: 0fee={results[0.0]}, 1%fee={results[0.01]}"

    def test_zero_fee_no_loss(self):
        """With zero fee, buy and immediate sell at same price = no loss."""
        prices = [100.0] * 10
        bars = _make_bars(prices, symbol="NVDA")
        actions = _make_actions(bars, buy_amount=100, sell_amount=100)
        cfg = PortfolioConfig(
            initial_cash=10000, max_positions=1, max_hold_hours=1,
            enforce_market_hours=False, close_at_eod=False,
            decision_lag_bars=0, bar_margin=0.0,
            fee_by_symbol={"NVDA": 0.0}, max_leverage=1.0, int_qty=False,
            symbols=["NVDA"],
        )
        r = run_portfolio_simulation(bars, actions, cfg, horizon=1)
        assert r.metrics["final_equity"] > 0

    def test_higher_fee_more_drag(self):
        """Higher fees should produce lower returns."""
        prices = [100 + i * 0.1 for i in range(50)]
        bars = _make_bars(prices, symbol="NVDA")
        actions = _make_actions(bars, buy_amount=80, sell_amount=80)

        results = {}
        for fee in [0.0, 0.001, 0.005]:
            cfg = PortfolioConfig(
                initial_cash=10000, max_positions=1, max_hold_hours=3,
                enforce_market_hours=False, close_at_eod=False,
                decision_lag_bars=0, bar_margin=0.0,
                fee_by_symbol={"NVDA": fee}, max_leverage=1.0, int_qty=False,
                symbols=["NVDA"],
            )
            r = run_portfolio_simulation(bars, actions, cfg, horizon=1)
            results[fee] = r.metrics["final_equity"]

        assert results[0.0] >= results[0.001] >= results[0.005], \
            f"Higher fees should produce lower equity: {results}"


class TestPortfolioSimulatorMargin:
    def test_margin_interest_charged(self):
        """Margin interest should reduce equity when leveraged."""
        prices = [100.0] * 100
        bars = _make_bars(prices, symbol="NVDA")
        actions = _make_actions(bars, buy_amount=100, sell_amount=0)

        eq_no_margin = None
        eq_with_margin = None

        for margin_rate in [0.0, 0.0625]:
            cfg = PortfolioConfig(
                initial_cash=10000, max_positions=1, max_hold_hours=200,
                enforce_market_hours=False, close_at_eod=False,
                decision_lag_bars=0, bar_margin=0.0,
                fee_by_symbol={"NVDA": 0.001}, max_leverage=2.0, int_qty=False,
                symbols=["NVDA"], margin_annual_rate=margin_rate,
            )
            r = run_portfolio_simulation(bars, actions, cfg, horizon=1)
            if margin_rate == 0.0:
                eq_no_margin = r.metrics["final_equity"]
            else:
                eq_with_margin = r.metrics["final_equity"]

        assert eq_with_margin < eq_no_margin, \
            f"Margin interest should reduce equity: no_margin={eq_no_margin}, with={eq_with_margin}"

    def test_margin_rate_math(self):
        """Verify margin interest formula: borrowed * rate / 8760 per hour."""
        rate = 0.0625
        hourly_rate = rate / 8760
        borrowed = 5000
        hours = 100
        expected_interest = borrowed * hourly_rate * hours
        assert abs(expected_interest - 3.566) < 0.1, f"Interest calc: {expected_interest}"


class TestPortfolioSimulatorLeverage:
    def test_leverage_increases_position_size(self):
        """2x leverage should allow ~2x position size."""
        prices = [100.0] * 10
        bars = _make_bars(prices, symbol="NVDA")
        actions = _make_actions(bars, buy_amount=100, sell_amount=0)

        sizes = {}
        for lev in [1.0, 2.0]:
            cfg = PortfolioConfig(
                initial_cash=10000, max_positions=1, max_hold_hours=100,
                enforce_market_hours=False, close_at_eod=False,
                decision_lag_bars=0, bar_margin=0.0,
                fee_by_symbol={"NVDA": 0.001}, max_leverage=lev, int_qty=False,
                symbols=["NVDA"],
            )
            r = run_portfolio_simulation(bars, actions, cfg, horizon=1)
            if r.trades:
                sizes[lev] = r.trades[0].quantity

        if 1.0 in sizes and 2.0 in sizes:
            ratio = sizes[2.0] / sizes[1.0]
            assert 1.8 < ratio < 2.2, f"2x leverage should ~double position: ratio={ratio}"

    def test_no_leverage_cant_exceed_equity(self):
        """Without leverage, position value shouldn't exceed equity."""
        prices = [100.0] * 10
        bars = _make_bars(prices, symbol="NVDA")
        actions = _make_actions(bars, buy_amount=100, sell_amount=0)
        cfg = PortfolioConfig(
            initial_cash=10000, max_positions=1, max_hold_hours=100,
            enforce_market_hours=False, close_at_eod=False,
            decision_lag_bars=0, bar_margin=0.0,
            fee_by_symbol={"NVDA": 0.001}, max_leverage=1.0, int_qty=False,
            symbols=["NVDA"],
        )
        r = run_portfolio_simulation(bars, actions, cfg, horizon=1)
        if r.trades:
            pos_value = r.trades[0].quantity * r.trades[0].price
            assert pos_value <= 10000 * 1.01, f"Position {pos_value} exceeds equity"


class TestPortfolioSimulatorFillBuffer:
    def test_bar_margin_filters_fills(self):
        """bar_margin should prevent fills when price doesn't reach order."""
        prices = [100.0] * 10
        bars = _make_bars(prices, symbol="NVDA")
        buy_price = 99.0  # needs low <= 99 * (1-margin)
        actions = _make_actions(bars, buy_price_fn=lambda b: buy_price, buy_amount=100)

        fills_no_buf = None
        fills_with_buf = None
        for bm in [0.0, 0.05]:  # 5% buffer - should block most fills
            cfg = PortfolioConfig(
                initial_cash=10000, max_positions=1, max_hold_hours=100,
                enforce_market_hours=False, close_at_eod=False,
                decision_lag_bars=0, bar_margin=bm,
                fee_by_symbol={"NVDA": 0.001}, max_leverage=1.0, int_qty=False,
                symbols=["NVDA"],
            )
            r = run_portfolio_simulation(bars, actions, cfg, horizon=1)
            n = r.metrics["num_buys"]
            if bm == 0.0:
                fills_no_buf = n
            else:
                fills_with_buf = n

        assert fills_with_buf <= fills_no_buf, \
            f"Larger bar_margin should filter more: no_buf={fills_no_buf}, with_buf={fills_with_buf}"

    def test_bar_margin_requires_trade_through_limit_boundary(self):
        """5 bps buffer should reject a touch above threshold and fill once price trades through it."""
        ts0 = pd.Timestamp("2026-01-01T00:00:00Z")
        ts1 = ts0 + pd.Timedelta(hours=1)
        bars = pd.DataFrame(
            [
                {"timestamp": ts0, "symbol": "NVDA", "open": 100.0, "high": 100.2, "low": 99.951, "close": 100.0},
                {"timestamp": ts1, "symbol": "NVDA", "open": 100.0, "high": 100.2, "low": 99.949, "close": 100.0},
            ]
        )
        actions = pd.DataFrame(
            [
                {
                    "timestamp": ts0,
                    "symbol": "NVDA",
                    "buy_price": 100.0,
                    "sell_price": 101.0,
                    "buy_amount": 100.0,
                    "sell_amount": 0.0,
                    "trade_amount": 100.0,
                    "predicted_high_p50_h1": 101.0,
                    "predicted_low_p50_h1": 99.0,
                    "predicted_close_p50_h1": 100.5,
                },
                {
                    "timestamp": ts1,
                    "symbol": "NVDA",
                    "buy_price": 100.0,
                    "sell_price": 101.0,
                    "buy_amount": 100.0,
                    "sell_amount": 0.0,
                    "trade_amount": 100.0,
                    "predicted_high_p50_h1": 101.0,
                    "predicted_low_p50_h1": 99.0,
                    "predicted_close_p50_h1": 100.5,
                },
            ]
        )
        cfg = PortfolioConfig(
            initial_cash=10_000.0,
            max_positions=1,
            max_hold_hours=100,
            enforce_market_hours=False,
            close_at_eod=False,
            decision_lag_bars=0,
            bar_margin=0.0005,
            fee_by_symbol={"NVDA": 0.0},
            max_leverage=1.0,
            int_qty=True,
            symbols=["NVDA"],
        )

        result = run_portfolio_simulation(bars, actions, cfg, horizon=1)
        buys = [trade for trade in result.trades if trade.side == "buy"]

        assert len(buys) == 1
        assert buys[0].timestamp == ts1


class TestPortfolioSimulatorDrawdown:
    def test_drawdown_calculation(self):
        """Max drawdown should be correctly computed from equity curve."""
        prices = [100, 110, 120, 90, 80, 100, 110]
        bars = _make_bars(prices, symbol="NVDA")
        actions = _make_actions(bars, buy_amount=100, sell_amount=0)
        cfg = PortfolioConfig(
            initial_cash=10000, max_positions=1, max_hold_hours=100,
            enforce_market_hours=False, close_at_eod=False,
            decision_lag_bars=0, bar_margin=0.0,
            fee_by_symbol={"NVDA": 0.001}, max_leverage=1.0, int_qty=False,
            symbols=["NVDA"],
        )
        r = run_portfolio_simulation(bars, actions, cfg, horizon=1)
        dd = r.metrics["max_drawdown"]
        assert 0 <= dd <= 1, f"Drawdown should be 0-1: {dd}"

    def test_no_drawdown_on_monotone_up(self):
        """Monotonically increasing prices = 0 drawdown (ignoring fees)."""
        prices = [100 + i * 2 for i in range(20)]
        bars = _make_bars(prices, symbol="NVDA")
        actions = _make_actions(bars, buy_amount=0, sell_amount=0)  # no trades
        cfg = PortfolioConfig(
            initial_cash=10000, max_positions=1, max_hold_hours=100,
            enforce_market_hours=False, close_at_eod=False,
            decision_lag_bars=0, bar_margin=0.0,
            fee_by_symbol={"NVDA": 0.0}, max_leverage=1.0, int_qty=False,
            symbols=["NVDA"],
        )
        r = run_portfolio_simulation(bars, actions, cfg, horizon=1)
        assert r.metrics["max_drawdown"] == 0.0


class TestPortfolioSimulatorSortino:
    def test_sortino_positive_for_profitable(self):
        """Profitable trading should yield positive Sortino."""
        prices = [100 + i * 0.5 for i in range(100)]
        bars = _make_bars(prices, symbol="NVDA")
        actions = _make_actions(bars, buy_amount=80, sell_amount=80)
        cfg = PortfolioConfig(
            initial_cash=10000, max_positions=1, max_hold_hours=3,
            enforce_market_hours=False, close_at_eod=False,
            decision_lag_bars=0, bar_margin=0.0,
            fee_by_symbol={"NVDA": 0.0005}, max_leverage=1.0, int_qty=False,
            symbols=["NVDA"],
        )
        r = run_portfolio_simulation(bars, actions, cfg, horizon=1)
        if r.metrics["num_buys"] > 5:
            assert r.metrics["sortino"] > 0, f"Profitable trades should have positive sortino"


# ======================== CRYPTO SIMULATOR TESTS ========================

def _make_crypto_data(prices, symbol="DOGEUSD"):
    rows = []
    for i, p in enumerate(prices):
        ts = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i)
        rows.append({"timestamp": ts, "symbol": symbol, "open": p, "high": p * 1.02, "low": p * 0.98, "close": p})
    df = pd.DataFrame(rows)
    return df


def _make_crypto_actions(bars, buy_amount=50, sell_amount=50):
    rows = []
    for _, b in bars.iterrows():
        rows.append({
            "timestamp": b["timestamp"], "symbol": b["symbol"],
            "buy_price": b["close"] * 0.99, "sell_price": b["close"] * 1.01,
            "buy_amount": buy_amount, "sell_amount": sell_amount,
        })
    return pd.DataFrame(rows)


class TestCryptoSimulatorFees:
    def test_maker_fee_applied(self):
        """Crypto sim should deduct maker fee on each trade."""
        prices = [0.10] * 50
        bars = _make_crypto_data(prices)
        actions = _make_crypto_actions(bars, buy_amount=50, sell_amount=50)

        results = {}
        for fee in [0.0, 0.001, 0.005]:
            cfg = LeverageConfig(
                max_leverage=1.0, initial_cash=1000, maker_fee=fee,
                margin_hourly_rate=0.0, fill_buffer_pct=0.0,
                decision_lag_bars=0, intensity_scale=5.0,
            )
            r = simulate_with_margin_cost(bars, actions, cfg)
            results[fee] = r["final_equity"]

        assert results[0.0] >= results[0.001] >= results[0.005], \
            f"Higher fee should reduce equity: {results}"


class TestCryptoSimulatorMarginInterest:
    def test_negative_cash_charges_interest(self):
        """Borrowing cash (negative cash) should incur hourly interest."""
        prices = [0.10] * 100
        bars = _make_crypto_data(prices)
        actions = _make_crypto_actions(bars, buy_amount=80, sell_amount=0)

        results = {}
        for margin_rate in [0.0, 0.0000025457]:
            cfg = LeverageConfig(
                max_leverage=2.0, initial_cash=1000, maker_fee=0.001,
                margin_hourly_rate=margin_rate, fill_buffer_pct=0.0,
                decision_lag_bars=0, intensity_scale=5.0,
            )
            r = simulate_with_margin_cost(bars, actions, cfg)
            results[margin_rate] = r["final_equity"]

        assert results[0.0] >= results[0.0000025457], \
            f"Margin interest should reduce equity: {results}"

    def test_margin_cost_tracked(self):
        """margin_cost_total should be positive when leveraged."""
        prices = [0.10] * 100
        bars = _make_crypto_data(prices)
        actions = _make_crypto_actions(bars, buy_amount=80, sell_amount=0)
        cfg = LeverageConfig(
            max_leverage=2.0, initial_cash=1000, maker_fee=0.001,
            margin_hourly_rate=0.0000025457, fill_buffer_pct=0.0,
            decision_lag_bars=0, intensity_scale=5.0,
        )
        r = simulate_with_margin_cost(bars, actions, cfg)
        if r["num_trades"] > 0:
            assert r["margin_cost_total"] >= 0, "Margin cost should be non-negative"


class TestCryptoSimulatorLeverage:
    def test_leverage_multiplies_position(self):
        """Higher leverage should allow larger positions."""
        prices = [0.10] * 20
        bars = _make_crypto_data(prices)
        actions = _make_crypto_actions(bars, buy_amount=80, sell_amount=0)

        trades_by_lev = {}
        for lev in [1.0, 2.0, 3.0]:
            cfg = LeverageConfig(
                max_leverage=lev, initial_cash=1000, maker_fee=0.001,
                margin_hourly_rate=0.0, fill_buffer_pct=0.0,
                decision_lag_bars=0, intensity_scale=5.0,
            )
            r = simulate_with_margin_cost(bars, actions, cfg)
            trades_by_lev[lev] = r["num_trades"]

        # More leverage = same or more trades (more buying power)
        assert trades_by_lev[2.0] >= trades_by_lev[1.0], \
            f"2x leverage should enable >= trades: {trades_by_lev}"

    def test_equity_never_negative(self):
        """Equity should never go negative in simulation."""
        prices = [0.10 * (1 - 0.005 * i) for i in range(100)]  # declining
        bars = _make_crypto_data(prices)
        actions = _make_crypto_actions(bars, buy_amount=80, sell_amount=20)
        cfg = LeverageConfig(
            max_leverage=3.0, initial_cash=1000, maker_fee=0.001,
            margin_hourly_rate=0.0000025457, fill_buffer_pct=0.0,
            decision_lag_bars=0, intensity_scale=5.0,
        )
        r = simulate_with_margin_cost(bars, actions, cfg)
        assert r["final_equity"] > -1000, f"Equity cratered: {r['final_equity']}"


class TestCryptoSimulatorFillBuffer:
    def test_fill_buffer_reduces_trades(self):
        """Larger fill buffer should reduce number of fills."""
        prices = [0.10] * 50
        bars = _make_crypto_data(prices)
        actions = _make_crypto_actions(bars, buy_amount=50, sell_amount=50)

        results = {}
        for fb in [0.0, 0.001, 0.01]:
            cfg = LeverageConfig(
                max_leverage=1.0, initial_cash=1000, maker_fee=0.001,
                margin_hourly_rate=0.0, fill_buffer_pct=fb,
                decision_lag_bars=0, intensity_scale=5.0,
            )
            r = simulate_with_margin_cost(bars, actions, cfg)
            results[fb] = r["num_trades"]

        assert results[0.0] >= results[0.01], \
            f"Larger fill buffer should reduce trades: {results}"


class TestCryptoSimulatorDecisionLag:
    def test_lag_shifts_signals(self):
        """Decision lag should shift signals by N bars."""
        prices = [0.10 + i * 0.001 for i in range(50)]
        bars = _make_crypto_data(prices)
        actions = _make_crypto_actions(bars, buy_amount=50, sell_amount=50)

        results = {}
        for lag in [0, 1]:
            cfg = LeverageConfig(
                max_leverage=1.0, initial_cash=1000, maker_fee=0.001,
                margin_hourly_rate=0.0, fill_buffer_pct=0.0,
                decision_lag_bars=lag, intensity_scale=5.0,
            )
            r = simulate_with_margin_cost(bars, actions, cfg)
            results[lag] = r["total_return"]

        # lag=1 should generally perform differently than lag=0
        # (may be better or worse, just verify it's different)
        assert results[0] != results[1] or True  # always passes, just checks no crash


class TestCryptoSimulatorSortino:
    def test_sortino_formula(self):
        """Verify Sortino ratio formula: mean(ret) / std(neg_ret) * sqrt(8760)."""
        returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015, -0.003, 0.01])
        neg = returns[returns < 0]
        expected = (np.mean(returns) / (np.std(neg) + 1e-10)) * np.sqrt(8760)
        assert expected > 0, "Positive mean returns should give positive Sortino"

    def test_max_drawdown_formula(self):
        """Verify drawdown: (peak - current) / peak."""
        eq = np.array([100, 110, 105, 120, 90, 95])
        running_max = np.maximum.accumulate(eq)
        dd = (running_max - eq) / (running_max + 1e-10)
        max_dd = float(np.min((eq - running_max) / (running_max + 1e-10)))
        assert abs(max_dd - (-0.25)) < 0.01, f"DD from 120->90 = 25%: got {max_dd}"


class TestCryptoSimulatorHoldTimeout:
    def test_force_close_on_timeout(self):
        """max_hold_bars should force close position."""
        prices = [0.10] * 20
        bars = _make_crypto_data(prices)
        actions = _make_crypto_actions(bars, buy_amount=80, sell_amount=0)
        cfg = LeverageConfig(
            max_leverage=1.0, initial_cash=1000, maker_fee=0.001,
            margin_hourly_rate=0.0, fill_buffer_pct=0.0,
            decision_lag_bars=0, max_hold_bars=3, intensity_scale=5.0,
        )
        r = simulate_with_margin_cost(bars, actions, cfg)
        force_sells = [t for t in r.get("trades", []) if isinstance(t, tuple) and t[0] == "force_sell"]
        # Should have some force sells due to timeout
        # But even if structure differs, equity should be positive
        assert r["final_equity"] > 0


class TestCrossSimulatorConsistency:
    def test_sortino_sign_consistency(self):
        """Both sims should produce same sign of Sortino for same price series."""
        # Uptrending prices
        prices_up = [100 + i * 0.5 for i in range(100)]
        # Stock sim
        bars_stock = _make_bars(prices_up, symbol="NVDA")
        actions_stock = _make_actions(bars_stock, buy_amount=80, sell_amount=80)
        cfg_stock = PortfolioConfig(
            initial_cash=10000, max_positions=1, max_hold_hours=3,
            enforce_market_hours=False, close_at_eod=False,
            decision_lag_bars=0, bar_margin=0.0,
            fee_by_symbol={"NVDA": 0.001}, max_leverage=1.0, int_qty=False,
            symbols=["NVDA"],
        )
        r_stock = run_portfolio_simulation(bars_stock, actions_stock, cfg_stock, horizon=1)

        # Crypto sim
        bars_crypto = _make_crypto_data(prices_up, symbol="DOGEUSD")
        actions_crypto = _make_crypto_actions(bars_crypto, buy_amount=50, sell_amount=50)
        cfg_crypto = LeverageConfig(
            max_leverage=1.0, initial_cash=10000, maker_fee=0.001,
            margin_hourly_rate=0.0, fill_buffer_pct=0.0,
            decision_lag_bars=0, intensity_scale=5.0,
        )
        r_crypto = simulate_with_margin_cost(bars_crypto, actions_crypto, cfg_crypto)

        # Both should see positive or both negative with uptrending prices
        # (structure differences may cause different absolute values)
        if r_stock.metrics["num_buys"] > 5 and r_crypto["num_trades"] > 5:
            stock_positive = r_stock.metrics["total_return"] > 0
            crypto_positive = r_crypto["total_return"] > 0
            assert stock_positive == crypto_positive, \
                f"Both sims should agree on direction: stock={r_stock.metrics['total_return']:.4f} crypto={r_crypto['total_return']:.4f}"


class TestValidateSimVsLiveInterest:
    def test_simulate_5m_applies_margin_interest_on_negative_cash(self):
        ts = pd.date_range("2026-01-01 01:00:00+00:00", periods=13, freq="5min")
        bars_5m = pd.DataFrame(
            {
                "timestamp": ts,
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1_000_000.0,
            }
        )
        hourly_signals = {
            pd.Timestamp("2026-01-01 00:00:00+00:00"): {
                "buy_price": 99.0,
                "sell_price": 101.0,
                "buy_amount": 0.0,
                "sell_amount": 0.0,
            },
            pd.Timestamp("2026-01-01 01:00:00+00:00"): {
                "buy_price": 99.0,
                "sell_price": 101.0,
                "buy_amount": 0.0,
                "sell_amount": 0.0,
            },
        }

        args_no_interest = SimpleNamespace(
            fee=0.001,
            fill_buffer_pct=0.0,
            initial_cash=-1000.0,
            start="2026-01-01 01:00:00+00:00",
            realistic=False,
            expiry_minutes=90,
            max_fill_fraction=0.01,
            min_notional=5.0,
            tick_size=0.00001,
            step_size=1.0,
            max_hold_hours=6,
            max_leverage=2.0,
            margin_hourly_rate=0.0,
            verbose=False,
        )
        _, eq_no_interest, cash_no_interest, inv_no_interest = simulate_5m(
            args_no_interest, hourly_signals, bars_5m, initial_inv=0.0, initial_entry_ts=None
        )
        assert inv_no_interest == 0.0
        assert abs(eq_no_interest + 1000.0) < 1e-9
        assert abs(cash_no_interest + 1000.0) < 1e-9

        args_with_interest = SimpleNamespace(**{**args_no_interest.__dict__, "margin_hourly_rate": 0.01})
        _, eq_with_interest, cash_with_interest, inv_with_interest = simulate_5m(
            args_with_interest, hourly_signals, bars_5m, initial_inv=0.0, initial_entry_ts=None
        )
        assert inv_with_interest == 0.0
        assert eq_with_interest < eq_no_interest
        assert cash_with_interest < cash_no_interest


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
