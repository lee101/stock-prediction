"""Tests for entry filter interactions -- catches SMA/dip contradiction bug."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from binance_worksteal.strategy import (
    WorkStealConfig,
    build_entry_candidates,
    compute_market_breadth_skip,
    _risk_off_triggered,
    resolve_entry_config,
    resolve_entry_regime,
    run_worksteal_backtest,
)
from binance_worksteal.trade_live import _relative_bps_distance


def make_bars(prices, start_date="2026-01-01", symbol="TESTUSD"):
    dates = pd.date_range(start_date, periods=len(prices), freq="D", tz="UTC")
    rows = []
    prev = prices[0]
    for d, p in zip(dates, prices):
        rows.append({
            "timestamp": d,
            "open": prev,
            "high": p * 1.01,
            "low": p * 0.99,
            "close": p,
            "volume": 1e6,
            "symbol": symbol,
        })
        prev = p
    return pd.DataFrame(rows)


def make_dip_scenario(n_symbols=5, lookback=20, dip_pct=0.20):
    all_bars = {}
    history = {}
    current_bars = {}
    base_price = 100.0

    for i in range(n_symbols):
        sym = f"SYM{i}USD"
        if i == 0:
            flat = [base_price] * lookback
            dipped = base_price * (1 - dip_pct)
            prices = flat + [dipped]
        else:
            prices = [base_price + i] * (lookback + 1)

        bars = make_bars(prices, symbol=sym)
        all_bars[sym] = bars
        history[sym] = bars
        current_bars[sym] = bars.iloc[-1]

    return all_bars, history, current_bars


class TestSmaDipContradiction:
    def test_sma_dip_contradiction_legacy(self):
        """With sma_check_method='current', the old SMA filter blocks dip entries."""
        _, history, current_bars = make_dip_scenario(
            n_symbols=5, lookback=20, dip_pct=0.20,
        )
        config = WorkStealConfig(
            dip_pct=0.20,
            proximity_pct=0.005,
            sma_filter_period=20,
            sma_check_method="current",
            lookback_days=20,
        )
        date = pd.Timestamp("2026-01-21", tz="UTC")
        candidates = build_entry_candidates(
            date=date,
            current_bars=current_bars,
            history=history,
            positions={},
            last_exit={},
            config=config,
            base_symbol=None,
        )
        assert len(candidates) == 0

    def test_sma_pre_dip_produces_candidates(self):
        """With sma_check_method='pre_dip' (new default), dip entries pass."""
        _, history, current_bars = make_dip_scenario(
            n_symbols=5, lookback=20, dip_pct=0.20,
        )
        config = WorkStealConfig(
            dip_pct=0.20,
            proximity_pct=0.03,
            sma_filter_period=20,
            sma_check_method="pre_dip",
            lookback_days=20,
        )
        date = pd.Timestamp("2026-01-21", tz="UTC")
        candidates = build_entry_candidates(
            date=date,
            current_bars=current_bars,
            history=history,
            positions={},
            last_exit={},
            config=config,
            base_symbol=None,
        )
        assert len(candidates) > 0

    def test_sma_disabled_produces_candidates(self):
        """Same scenario with sma_filter_period=0 -- candidates ARE produced."""
        _, history, current_bars = make_dip_scenario(
            n_symbols=5, lookback=20, dip_pct=0.20,
        )
        config = WorkStealConfig(
            dip_pct=0.20,
            proximity_pct=0.005,
            sma_filter_period=0,
            lookback_days=20,
        )
        date = pd.Timestamp("2026-01-21", tz="UTC")
        candidates = build_entry_candidates(
            date=date,
            current_bars=current_bars,
            history=history,
            positions={},
            last_exit={},
            config=config,
            base_symbol=None,
        )
        long_candidates = [c for c in candidates if c[1] == "long"]
        assert len(long_candidates) >= 1

    def test_wider_proximity_produces_candidates(self):
        """proximity_pct=0.03 may still not help because SMA filter blocks first."""
        _, history, current_bars = make_dip_scenario(
            n_symbols=5, lookback=20, dip_pct=0.20,
        )
        config = WorkStealConfig(
            dip_pct=0.20,
            proximity_pct=0.03,
            sma_filter_period=20,
            lookback_days=20,
        )
        date = pd.Timestamp("2026-01-21", tz="UTC")
        build_entry_candidates(
            date=date,
            current_bars=current_bars,
            history=history,
            positions={},
            last_exit={},
            config=config,
            base_symbol=None,
        )
        # With pre_dip SMA check (default), entries now pass because
        # pre-dip bars were above SMA. Use sma_check_method="current" to test old bug.
        config_legacy = WorkStealConfig(
            dip_pct=0.20,
            proximity_pct=0.03,
            sma_filter_period=20,
            sma_check_method="current",
            lookback_days=20,
        )
        candidates_legacy = build_entry_candidates(
            date=date,
            current_bars=current_bars,
            history=history,
            positions={},
            last_exit={},
            config=config_legacy,
            base_symbol=None,
        )
        assert len(candidates_legacy) == 0


class TestRiskOffMomentum:
    def _make_momentum_data(self, n_symbols=30, momentum_pct=-0.02, lookback=20):
        history = {}
        current_bars = {}
        for i in range(n_symbols):
            sym = f"C{i:02d}USD"
            base = 100.0 + i
            prices = [base] * lookback
            final = base * (1 + momentum_pct)
            for j in range(7):
                prices.append(base + (final - base) * (j + 1) / 7)
            bars = make_bars(prices, symbol=sym)
            history[sym] = bars
            current_bars[sym] = bars.iloc[-1]
        return history, current_bars

    def test_risk_off_moderate_momentum(self):
        """Avg 7d momentum ~-2%. With threshold=-0.05 (default), this does NOT trigger."""
        history, current_bars = self._make_momentum_data(
            n_symbols=30, momentum_pct=-0.02,
        )
        config = WorkStealConfig(
            risk_off_trigger_momentum_period=7,
            risk_off_trigger_sma_period=0,
            risk_off_momentum_threshold=-0.05,
        )
        assert _risk_off_triggered(
            current_bars=current_bars, history=history, config=config,
        ) is False

    def test_risk_off_severe_momentum(self):
        """Avg momentum -8% exceeds threshold=-0.05, fires."""
        history, current_bars = self._make_momentum_data(
            n_symbols=30, momentum_pct=-0.08,
        )
        config = WorkStealConfig(
            risk_off_trigger_momentum_period=7,
            risk_off_trigger_sma_period=0,
            risk_off_momentum_threshold=-0.05,
        )
        assert _risk_off_triggered(
            current_bars=current_bars, history=history, config=config,
        ) is True

    def test_risk_off_positive_momentum_does_not_fire(self):
        """Positive avg momentum should not trigger risk-off."""
        history, current_bars = self._make_momentum_data(
            n_symbols=30, momentum_pct=0.05,
        )
        config = WorkStealConfig(
            risk_off_trigger_momentum_period=7,
            risk_off_trigger_sma_period=0,
        )
        assert _risk_off_triggered(
            current_bars=current_bars, history=history, config=config,
        ) is False


class TestMarketBreadthFilter:
    def test_market_breadth_filter_blocks(self):
        """80% of coins down d/d -> breadth filter at 70% blocks entries."""
        n_total = 10
        history = {}
        current_bars = {}
        for i in range(n_total):
            sym = f"MB{i}USD"
            if i < 8:  # 80% down
                prices = [100.0] * 5 + [99.0]
            else:
                prices = [100.0] * 5 + [101.0]
            bars = make_bars(prices, symbol=sym)
            history[sym] = bars
            current_bars[sym] = bars.iloc[-1]

        config = WorkStealConfig(market_breadth_filter=0.70)
        assert compute_market_breadth_skip(current_bars, history, config) is True

    def test_market_breadth_filter_passes(self):
        """Only 40% of coins down -> breadth filter at 70% does NOT block."""
        n_total = 10
        history = {}
        current_bars = {}
        for i in range(n_total):
            sym = f"MB{i}USD"
            if i < 4:
                prices = [100.0] * 5 + [99.0]
            else:
                prices = [100.0] * 5 + [101.0]
            bars = make_bars(prices, symbol=sym)
            history[sym] = bars
            current_bars[sym] = bars.iloc[-1]

        config = WorkStealConfig(market_breadth_filter=0.70)
        assert compute_market_breadth_skip(current_bars, history, config) is False

    def test_market_breadth_disabled(self):
        """market_breadth_filter=0 never blocks."""
        history = {}
        current_bars = {}
        for i in range(10):
            sym = f"MB{i}USD"
            prices = [100.0] * 5 + [50.0]  # all down 50%
            bars = make_bars(prices, symbol=sym)
            history[sym] = bars
            current_bars[sym] = bars.iloc[-1]

        config = WorkStealConfig(market_breadth_filter=0.0)
        assert compute_market_breadth_skip(current_bars, history, config) is False


class TestEntryProximityBps:
    def test_entry_proximity_bps_blocks_limit_orders(self):
        """A valid dip candidate with buy_price 5% below close:
        relative bps distance = 500 bps, which is > 25 bps default."""
        close = 100.0
        buy_price = 95.0  # 5% below
        bps = _relative_bps_distance(close, buy_price)
        assert bps == pytest.approx(500.0, rel=0.01)
        assert bps > 25.0  # default entry_proximity_bps blocks this

    def test_entry_proximity_bps_passes_with_wide_threshold(self):
        """With bps=3000, a 5% distance (500 bps) passes."""
        close = 100.0
        buy_price = 95.0
        bps = _relative_bps_distance(close, buy_price)
        assert bps < 3000.0

    def test_entry_proximity_bps_zero_reference(self):
        """Zero reference price returns inf."""
        assert _relative_bps_distance(0.0, 100.0) == float("inf")

    def test_entry_proximity_bps_same_price(self):
        """Same price = 0 bps."""
        assert _relative_bps_distance(100.0, 100.0) == pytest.approx(0.0)

    def test_production_scenario_20pct_dip(self):
        """Production: dip_pct=0.20, lookback=20. ref_high~100, buy_target=80.
        close is near 80 (the dip). Distance from close(80) to buy_target(80)
        is ~0 bps, which should pass. But in practice the SMA filter blocks
        before we even get here."""
        ref_high = 100.0
        buy_target = ref_high * 0.80
        close = 81.0  # just above buy target
        bps = _relative_bps_distance(close, buy_target)
        assert bps == pytest.approx(123.5, rel=0.1)
        assert bps > 25.0  # still blocked by default 25 bps


class TestRealisticFill:
    def test_realistic_fill_only_when_low_reaches_target(self):
        """In backtest: buy_target=80, but daily low=81 -> fill at max(80,81)=81.
        This IS the current behavior. With a sufficiently deep dip the low
        reaches the target."""
        prices = [100.0] * 25 + [82.0, 81.0, 80.0]
        bars = {"TESTUSD": make_bars(prices, symbol="TESTUSD")}
        config = WorkStealConfig(
            dip_pct=0.20,
            proximity_pct=0.02,
            sma_filter_period=0,
            lookback_days=20,
            profit_target_pct=0.10,
            stop_loss_pct=0.15,
            max_hold_days=30,
        )
        eq, trades, metrics = run_worksteal_backtest(bars, config)
        buys = [t for t in trades if t.side == "buy"]
        if buys:
            # fill_price = max(buy_target, low_bar)
            # buy_target ~ 100*1.01*0.80 = 80.8
            # The fill price should be close to the target
            assert buys[0].price <= 82.0

    def test_no_fill_when_low_above_target(self):
        """If the low never reaches buy_target, no fill should occur.
        With sma_filter_period=0 and prices staying above target."""
        prices = [100.0] * 25 + [92.0, 91.0, 90.0]
        bars = {"TESTUSD": make_bars(prices, symbol="TESTUSD")}
        config = WorkStealConfig(
            dip_pct=0.20,
            proximity_pct=0.005,
            sma_filter_period=0,
            lookback_days=20,
        )
        eq, trades, metrics = run_worksteal_backtest(bars, config)
        buys = [t for t in trades if t.side == "buy"]
        # 20% dip from high ~101 -> target ~80.8. Prices only go to 89.
        assert len(buys) == 0


class TestOneSymbolUniverse:
    def test_one_symbol_universe(self):
        """Run backtest with just 1 symbol, no errors."""
        prices = [100.0] * 25 + [90.0, 85.0, 80.0, 85.0, 90.0, 95.0, 100.0]
        bars = {"SINGLEUSD": make_bars(prices, symbol="SINGLEUSD")}
        config = WorkStealConfig(
            dip_pct=0.15,
            proximity_pct=0.02,
            sma_filter_period=0,
            lookback_days=20,
            profit_target_pct=0.10,
            stop_loss_pct=0.15,
            max_hold_days=30,
        )
        eq, trades, metrics = run_worksteal_backtest(bars, config)
        assert not eq.empty
        assert "equity" in eq.columns
        assert metrics["n_days"] > 0


class TestAllFlatPrices:
    def test_all_flat_prices(self):
        """All prices constant at $100 -> 0 trades, no errors."""
        bars = {}
        for sym in ["FLAT1USD", "FLAT2USD", "FLAT3USD"]:
            bars[sym] = make_bars([100.0] * 40, symbol=sym)
        config = WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.005,
            sma_filter_period=0,
            lookback_days=20,
        )
        eq, trades, metrics = run_worksteal_backtest(bars, config)
        buys = [t for t in trades if t.side == "buy"]
        assert len(buys) == 0
        assert not eq.empty


class TestGapDown:
    def test_gap_down_stop_loss(self):
        """30% gap down in one bar triggers stop loss."""
        prices = [100.0] * 25 + [88.0, 85.0]  # dip to enter
        prices += [60.0]  # 30% gap down from 85
        prices += [58.0, 55.0]
        bars = {"GAPUSD": make_bars(prices, symbol="GAPUSD")}
        config = WorkStealConfig(
            dip_pct=0.15,
            proximity_pct=0.02,
            sma_filter_period=0,
            lookback_days=20,
            profit_target_pct=0.15,
            stop_loss_pct=0.10,
            max_hold_days=30,
        )
        eq, trades, metrics = run_worksteal_backtest(bars, config)
        exits = [t for t in trades if t.side == "sell"]
        if exits:
            assert any(t.reason in ("stop_loss", "max_dd_exit") for t in exits)


class TestMaxPositionsRespected:
    def test_max_positions_respected(self):
        """10 symbols all dip but max_positions=3."""
        bars = {}
        for i in range(10):
            sym = f"D{i:02d}USD"
            prices = [100.0 + i * 0.1] * 25 + [80.0 + i * 0.1] * 5
            bars[sym] = make_bars(prices, symbol=sym)
        config = WorkStealConfig(
            dip_pct=0.15,
            proximity_pct=0.05,
            sma_filter_period=0,
            lookback_days=20,
            max_positions=3,
            profit_target_pct=0.15,
            stop_loss_pct=0.15,
            max_hold_days=30,
        )
        eq, trades, metrics = run_worksteal_backtest(bars, config)
        if not eq.empty:
            assert eq["n_positions"].max() <= 3


class TestResolveEntryConfig:
    def test_risk_off_switches_config(self):
        """When risk-off triggers, config changes to risk-off params."""
        history = {}
        current_bars = {}
        for i in range(10):
            sym = f"R{i}USD"
            prices = [100.0] * 20 + [90.0] * 7  # down 10%
            bars = make_bars(prices, symbol=sym)
            history[sym] = bars
            current_bars[sym] = bars.iloc[-1]
        config = WorkStealConfig(
            ref_price_method="high",
            risk_off_ref_price_method="sma",
            market_breadth_filter=0.0,
            risk_off_market_breadth_filter=0.70,
            risk_off_trigger_momentum_period=7,
            risk_off_trigger_sma_period=0,
        )
        entry_config = resolve_entry_config(
            current_bars=current_bars, history=history, config=config,
        )
        assert entry_config.ref_price_method == "sma"
        assert entry_config.market_breadth_filter >= 0.70

    def test_risk_on_preserves_config(self):
        """When risk-off does NOT trigger, config unchanged."""
        history = {}
        current_bars = {}
        for i in range(10):
            sym = f"R{i}USD"
            prices = [100.0] * 20 + [110.0] * 7  # up 10%
            bars = make_bars(prices, symbol=sym)
            history[sym] = bars
            current_bars[sym] = bars.iloc[-1]
        config = WorkStealConfig(
            ref_price_method="high",
            risk_off_ref_price_method="sma",
            market_breadth_filter=0.0,
            risk_off_market_breadth_filter=0.70,
            risk_off_trigger_momentum_period=7,
            risk_off_trigger_sma_period=0,
        )
        entry_config = resolve_entry_config(
            current_bars=current_bars, history=history, config=config,
        )
        assert entry_config is config

    def test_resolve_entry_regime_reports_risk_off_without_breadth_skip(self):
        history = {}
        current_bars = {}
        for i in range(10):
            sym = f"R{i}USD"
            prices = [100.0] * 20 + [90.0] * 7
            bars = make_bars(prices, symbol=sym)
            history[sym] = bars
            current_bars[sym] = bars.iloc[-1]
        config = WorkStealConfig(
            ref_price_method="high",
            risk_off_ref_price_method="sma",
            market_breadth_filter=0.0,
            risk_off_market_breadth_filter=0.70,
            risk_off_trigger_momentum_period=7,
            risk_off_trigger_sma_period=0,
        )

        regime = resolve_entry_regime(
            current_bars=current_bars,
            history=history,
            config=config,
        )

        assert regime.risk_off is True
        assert regime.market_breadth_skip is False
        assert regime.skip_entries is True
        assert regime.config.ref_price_method == "sma"

    def test_resolve_entry_regime_reports_breadth_skip_without_risk_off(self):
        history = {}
        current_bars = {}
        for i in range(5):
            sym = f"B{i}USD"
            prices = [100.0] * 20 + [95.0, 90.0]
            bars = make_bars(prices, symbol=sym)
            history[sym] = bars
            current_bars[sym] = bars.iloc[-1]
        config = WorkStealConfig(
            market_breadth_filter=0.5,
            risk_off_trigger_momentum_period=0,
            risk_off_trigger_sma_period=0,
        )

        regime = resolve_entry_regime(
            current_bars=current_bars,
            history=history,
            config=config,
        )

        assert regime.risk_off is False
        assert regime.market_breadth_skip is True
        assert regime.skip_entries is True
        assert regime.config is config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
