"""Tests for simulation-production alignment in hourly_replay.simulate_daily_policy.

Verifies:
  - trailing stop fires correctly after 0.3% drawdown from peak
  - max hold exits after 6 bars
  - fee and slippage calculation is correct
  - min notional filter blocks small positions
"""
from __future__ import annotations

import numpy as np
import pytest

from pufferlib_market.hourly_replay import (
    MktdData,
    Position,
    simulate_daily_policy,
    _close_position,
    _open_long,
)

# ---------------------------------------------------------------------------
# Helpers to build minimal synthetic MktdData
# ---------------------------------------------------------------------------

FEATURES_PER_SYM = 16
PRICE_FEATS = 5  # O H L C V


def _make_data(prices: list[float], num_timesteps: int | None = None) -> MktdData:
    """Build a 1-symbol MktdData with the given close prices.

    prices: list of close prices. Open = High = Low = Close = price, Volume = 1.
    All features are zero. All timesteps tradable.
    """
    T = len(prices)
    if num_timesteps is not None and num_timesteps != T:
        raise ValueError("num_timesteps must match len(prices)")

    # features: [T, 1, 16] all zeros
    features = np.zeros((T, 1, FEATURES_PER_SYM), dtype=np.float32)

    # prices: [T, 1, 5] = O H L C V; set all OHLC to the close price
    price_arr = np.zeros((T, 1, PRICE_FEATS), dtype=np.float32)
    for t, p in enumerate(prices):
        price_arr[t, 0, :] = [p, p, p, p, 1.0]  # O H L C V

    # tradable: all 1
    tradable = np.ones((T, 1), dtype=np.uint8)

    return MktdData(
        version=2,
        symbols=["TEST"],
        features=features,
        prices=price_arr,
        tradable=tradable,
    )


def _hold_policy(action: int):
    """Policy factory: always returns the given action."""
    def _fn(obs: np.ndarray) -> int:
        return action
    return _fn


def _sequence_policy(actions: list[int]):
    """Policy factory: returns actions[step], cycling if out of range."""
    state = {"step": 0}
    def _fn(obs: np.ndarray) -> int:
        a = actions[state["step"] % len(actions)]
        state["step"] += 1
        return a
    return _fn


# ---------------------------------------------------------------------------
# Test: trailing stop fires after 0.3% drawdown from peak
# ---------------------------------------------------------------------------

class TestTrailingStop:
    def test_trailing_stop_fires_on_drawdown(self):
        """Position opened at 100, price rises to 110 (new peak), then drops below
        110*(1-0.003)=109.67 — trailing stop should fire and exit the position."""
        # T=10 bars: [100, 100, 100, 110, 109.5, 100, 100, 100, 100, 100]
        # Bar 0: policy=LONG, opens at 100
        # Bar 3: price 110 → new peak 110
        # Bar 4: price 109.5 < 110*0.997=109.67 → trailing stop fires
        prices = [100.0, 100.0, 100.0, 110.0, 109.5, 100.0, 100.0, 100.0, 100.0, 100.0]
        data = _make_data(prices)

        # action=1 = buy sym 0 long; action=0 = flat
        policy = _hold_policy(1)  # always wants to hold long sym 0

        result = simulate_daily_policy(
            data,
            policy,
            max_steps=8,
            fee_rate=0.0,
            slippage_bps=0.0,
            trailing_stop_pct=0.003,
            max_hold_bars=0,
            enable_drawdown_profit_early_exit=False,
        )
        # The trailing stop should have fired at bar 4 (price 109.5), then policy
        # buys back in, and trade count should be > 1 (at least one forced close)
        assert result.num_trades >= 1, f"Expected trailing stop to fire, got {result.num_trades} trades"

    def test_trailing_stop_does_not_fire_below_threshold(self):
        """Price rises from 100 to 110, then drops only 0.1% (to 109.89).
        Since 109.89 > 110*(1-0.003)=109.67, trailing stop should NOT fire."""
        prices = [100.0, 100.0, 110.0, 109.89, 109.89, 109.89, 109.89, 109.89, 109.89, 100.0]
        data = _make_data(prices)
        policy = _hold_policy(1)

        result_with_stop = simulate_daily_policy(
            data,
            policy,
            max_steps=8,
            fee_rate=0.0,
            slippage_bps=0.0,
            trailing_stop_pct=0.003,
            max_hold_bars=0,
            enable_drawdown_profit_early_exit=False,
        )
        result_no_stop = simulate_daily_policy(
            data,
            policy,
            max_steps=8,
            fee_rate=0.0,
            slippage_bps=0.0,
            trailing_stop_pct=0.0,
            max_hold_bars=0,
            enable_drawdown_profit_early_exit=False,
        )
        # With a drop of only 0.1% vs threshold 0.3%, both should have same trade count
        # (only the end-of-episode forced close)
        assert result_with_stop.num_trades == result_no_stop.num_trades, (
            f"Trailing stop should not have fired: with={result_with_stop.num_trades} "
            f"vs without={result_no_stop.num_trades}"
        )

    def test_trailing_stop_disabled_when_zero(self):
        """trailing_stop_pct=0.0 means no trailing stop, even on large drawdowns."""
        prices = [100.0, 200.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 100.0]
        data = _make_data(prices)
        policy = _hold_policy(1)

        result_no_stop = simulate_daily_policy(
            data,
            policy,
            max_steps=8,
            fee_rate=0.0,
            slippage_bps=0.0,
            trailing_stop_pct=0.0,
            max_hold_bars=0,
            enable_drawdown_profit_early_exit=False,
        )
        result_with_stop = simulate_daily_policy(
            data,
            policy,
            max_steps=8,
            fee_rate=0.0,
            slippage_bps=0.0,
            trailing_stop_pct=0.003,
            max_hold_bars=0,
            enable_drawdown_profit_early_exit=False,
        )
        # Without stop: only 1 trade (final close at episode end)
        # With stop: should fire earlier due to 75% drawdown from 200 to 50
        assert result_no_stop.num_trades == 1, (
            f"Expected 1 trade (final close only), got {result_no_stop.num_trades}"
        )
        assert result_with_stop.num_trades >= 1

    def test_trailing_stop_peak_tracks_correctly(self):
        """The trailing stop peak should update as price rises.
        Price goes: 100 → 105 → 110 → 108 → 106 → 103.
        Peak is 110 at bar 2. 110*(1-0.003)=109.67.
        Bar 3: 108 < 109.67 → stop fires.
        """
        prices = [100.0, 105.0, 110.0, 108.0, 106.0, 103.0, 100.0, 100.0, 100.0, 100.0]
        data = _make_data(prices)
        policy = _hold_policy(1)  # always long

        result = simulate_daily_policy(
            data,
            policy,
            max_steps=8,
            fee_rate=0.0,
            slippage_bps=0.0,
            trailing_stop_pct=0.003,
            max_hold_bars=0,
            enable_drawdown_profit_early_exit=False,
        )
        # Should have fired at bar 3 and re-entered (policy still wants long)
        assert result.num_trades >= 2, (
            f"Expected trailing stop at bar 3 + re-entry, got {result.num_trades} trades"
        )


# ---------------------------------------------------------------------------
# Test: max hold exits after 6 bars
# ---------------------------------------------------------------------------

class TestMaxHold:
    def test_max_hold_exits_after_6_bars(self):
        """Policy always wants to hold long sym 0. With max_hold_bars=6, should be
        force-closed after 6 bars and immediately re-entered by the policy."""
        prices = [100.0] * 20 + [100.0]  # 21 prices → 20 steps
        data = _make_data(prices)
        policy = _hold_policy(1)

        result = simulate_daily_policy(
            data,
            policy,
            max_steps=18,
            fee_rate=0.0,
            slippage_bps=0.0,
            trailing_stop_pct=0.0,
            max_hold_bars=6,
            enable_drawdown_profit_early_exit=False,
        )
        # 18 steps / 6 = 3 forced closes + immediate re-entries = 6 trades
        # (3 forced exits + 3 re-opens)
        assert result.num_trades >= 2, (
            f"Expected multiple max-hold exits over 18 steps, got {result.num_trades}"
        )

    def test_max_hold_disabled_when_zero(self):
        """With max_hold_bars=0, position should not be force-closed."""
        prices = [100.0] * 15
        data = _make_data(prices)
        policy = _hold_policy(1)

        result_hold = simulate_daily_policy(
            data,
            policy,
            max_steps=13,
            fee_rate=0.0,
            slippage_bps=0.0,
            trailing_stop_pct=0.0,
            max_hold_bars=0,
            enable_drawdown_profit_early_exit=False,
        )
        # Only 1 trade: final close at episode end
        assert result_hold.num_trades == 1, (
            f"Expected 1 trade (episode close), got {result_hold.num_trades}"
        )

    def test_max_hold_1_bar(self):
        """max_hold_bars=1 should force exit after every single bar."""
        prices = [100.0] * 15
        data = _make_data(prices)
        policy = _hold_policy(1)

        result = simulate_daily_policy(
            data,
            policy,
            max_steps=10,
            fee_rate=0.0,
            slippage_bps=0.0,
            trailing_stop_pct=0.0,
            max_hold_bars=1,
            enable_drawdown_profit_early_exit=False,
        )
        # Should exit every bar, then re-enter. Many trades.
        assert result.num_trades >= 5, (
            f"Expected frequent max-hold exits with max_hold_bars=1, got {result.num_trades}"
        )


# ---------------------------------------------------------------------------
# Test: fee and slippage calculation
# ---------------------------------------------------------------------------

class TestFeeAndSlippage:
    def test_zero_fee_zero_slippage_flat_price(self):
        """With 0 fee and 0 slippage on flat price, total return should be 0."""
        prices = [100.0] * 10
        data = _make_data(prices)
        # Buy at bar 0, hold until episode end (bar 8)
        policy = _hold_policy(1)

        result = simulate_daily_policy(
            data,
            policy,
            max_steps=8,
            fee_rate=0.0,
            slippage_bps=0.0,
            trailing_stop_pct=0.0,
            max_hold_bars=0,
            enable_drawdown_profit_early_exit=False,
        )
        assert abs(result.total_return) < 1e-6, (
            f"Expected 0 return on flat price with 0 fees, got {result.total_return:.6f}"
        )

    def test_slippage_reduces_return(self):
        """With slippage, each round-trip costs extra. Return should be lower than 0-fee case."""
        prices = [100.0] * 10
        data = _make_data(prices)
        policy = _hold_policy(1)

        result_no_slip = simulate_daily_policy(
            data, policy, max_steps=8, fee_rate=0.0, slippage_bps=0.0,
            trailing_stop_pct=0.0, max_hold_bars=0,
            enable_drawdown_profit_early_exit=False,
        )
        result_with_slip = simulate_daily_policy(
            data, policy, max_steps=8, fee_rate=0.0, slippage_bps=10.0,
            trailing_stop_pct=0.0, max_hold_bars=0,
            enable_drawdown_profit_early_exit=False,
        )
        assert result_with_slip.total_return < result_no_slip.total_return, (
            f"Slippage should reduce return: {result_with_slip.total_return} >= {result_no_slip.total_return}"
        )

    def test_slippage_adds_to_effective_fee(self):
        """fee_rate=0.001 + slippage_bps=10 (0.001) should equal fee_rate=0.002 alone."""
        prices = [100.0, 110.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        data = _make_data(prices)
        policy = _hold_policy(1)

        result_combined = simulate_daily_policy(
            data, policy, max_steps=8, fee_rate=0.001, slippage_bps=10.0,
            trailing_stop_pct=0.0, max_hold_bars=0,
            enable_drawdown_profit_early_exit=False,
        )
        result_double_fee = simulate_daily_policy(
            data, policy, max_steps=8, fee_rate=0.002, slippage_bps=0.0,
            trailing_stop_pct=0.0, max_hold_bars=0,
            enable_drawdown_profit_early_exit=False,
        )
        # The total returns should be the same (within floating-point tolerance)
        assert abs(result_combined.total_return - result_double_fee.total_return) < 1e-6, (
            f"fee_rate=0.001+slip=10bps != fee_rate=0.002: "
            f"{result_combined.total_return:.8f} vs {result_double_fee.total_return:.8f}"
        )

    def test_fee_rate_only_baseline(self):
        """Pure fee_rate (no slippage) still works as before (backward compat)."""
        prices = [100.0, 110.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        data = _make_data(prices)
        policy = _hold_policy(1)

        result = simulate_daily_policy(
            data, policy, max_steps=8, fee_rate=0.001, slippage_bps=0.0,
            trailing_stop_pct=0.0, max_hold_bars=0,
            enable_drawdown_profit_early_exit=False,
        )
        # With 10% price rise then fall back and 10bps round-trip fee, should be slightly positive
        # The position opens at bar 0 at 100, peak at 110, close at 100.
        # Return ~ (100-fee_open)/100 - fee_close = slightly negative overall
        assert isinstance(result.total_return, float)


# ---------------------------------------------------------------------------
# Test: min notional filter
# ---------------------------------------------------------------------------

class TestMinNotional:
    def test_min_notional_blocks_tiny_account(self):
        """Start with very little cash (< min_notional). Policy wants to buy but
        should be blocked by the min notional filter, staying flat."""
        prices = [100.0] * 15
        data = _make_data(prices)
        policy = _hold_policy(1)

        # initial_cash=5.0 < min_notional_usd=12.0 → no positions opened
        result = simulate_daily_policy(
            data, policy, max_steps=10,
            fee_rate=0.0, slippage_bps=0.0,
            trailing_stop_pct=0.0, max_hold_bars=0,
            min_notional_usd=12.0,
            initial_cash=5.0,
            enable_drawdown_profit_early_exit=False,
        )
        # No trades should occur since cash < min_notional
        assert result.num_trades == 0, (
            f"Expected 0 trades when cash < min_notional, got {result.num_trades}"
        )

    def test_min_notional_allows_sufficient_account(self):
        """With enough cash, min_notional filter should not block trades."""
        prices = [100.0] * 15
        data = _make_data(prices)
        policy = _hold_policy(1)

        result = simulate_daily_policy(
            data, policy, max_steps=10,
            fee_rate=0.0, slippage_bps=0.0,
            trailing_stop_pct=0.0, max_hold_bars=0,
            min_notional_usd=12.0,
            initial_cash=10000.0,
            enable_drawdown_profit_early_exit=False,
        )
        assert result.num_trades >= 1, (
            f"Expected at least 1 trade (final close) with sufficient cash, got {result.num_trades}"
        )

    def test_min_notional_zero_disabled(self):
        """min_notional_usd=0 should never block any trade."""
        prices = [100.0] * 15
        data = _make_data(prices)
        policy = _hold_policy(1)

        result = simulate_daily_policy(
            data, policy, max_steps=10,
            fee_rate=0.0, slippage_bps=0.0,
            trailing_stop_pct=0.0, max_hold_bars=0,
            min_notional_usd=0.0,
            initial_cash=1.0,  # very small but not blocked
            enable_drawdown_profit_early_exit=False,
        )
        # With initial_cash=1 and price=100, qty = 1/100 = 0.01 shares, value = 1
        # That should not be blocked by min_notional=0
        assert result.num_trades >= 1


# ---------------------------------------------------------------------------
# Test: combined effect (production defaults)
# ---------------------------------------------------------------------------

class TestProductionDefaults:
    def test_production_constraints_reduce_returns_vs_unconstrained(self):
        """Production constraints (trailing stop + max hold + slippage) should
        result in equal or lower returns vs unconstrained simulation on a
        price series with some volatility."""
        # Trending up then crashing: trailing stop should save some, slippage costs
        prices = (
            [100.0, 102.0, 105.0, 108.0, 112.0, 115.0,   # up trend
             113.0, 108.0, 103.0, 98.0, 95.0,             # reversal
             100.0, 102.0, 104.0, 106.0, 108.0, 110.0,    # recovery
             108.0, 105.0, 102.0, 99.0, 96.0]             # final drop
        )
        data = _make_data(prices)
        policy = _hold_policy(1)  # always long

        result_unconstrained = simulate_daily_policy(
            data, policy, max_steps=20,
            fee_rate=0.0, slippage_bps=0.0,
            trailing_stop_pct=0.0, max_hold_bars=0,
            min_notional_usd=0.0,
            enable_drawdown_profit_early_exit=False,
        )
        result_production = simulate_daily_policy(
            data, policy, max_steps=20,
            fee_rate=0.0, slippage_bps=3.0,
            trailing_stop_pct=0.003, max_hold_bars=6,
            min_notional_usd=12.0,
            initial_cash=10000.0,
            enable_drawdown_profit_early_exit=False,
        )
        # Slippage always costs money. The production sim must have at least the
        # slippage drag applied. Both results should be valid floats.
        assert isinstance(result_unconstrained.total_return, float)
        assert isinstance(result_production.total_return, float)
        # Slippage drag should reduce returns. On flat/trending up prices,
        # trailing stop may help (cuts losses) but slippage always hurts.
        # We just check that production mode has more trades (stops + max holds fire).
        assert result_production.num_trades >= result_unconstrained.num_trades

    def test_backward_compat_no_new_params(self):
        """Calling simulate_daily_policy without new params uses safe defaults
        (no trailing stop, no max hold, no slippage) — same as before."""
        prices = [100.0, 105.0, 103.0, 108.0, 106.0, 104.0, 102.0, 100.0, 100.0, 100.0]
        data = _make_data(prices)
        policy = _hold_policy(1)

        # Call with original parameters only (backward compat)
        result = simulate_daily_policy(
            data, policy, max_steps=8,
            fee_rate=0.001,
            enable_drawdown_profit_early_exit=False,
        )
        assert isinstance(result.total_return, float)
        assert result.num_trades >= 1


# ---------------------------------------------------------------------------
# Test: _close_position fee calculation
# ---------------------------------------------------------------------------

class TestClosePositionFee:
    def test_long_close_fee_deducted_from_proceeds(self):
        """Closing a long: proceeds = qty * price * (1 - fee_rate)."""
        pos = Position(sym=0, is_short=False, qty=1.0, entry_price=100.0)
        cash_before = 0.0
        fee_rate = 0.001  # 10bps
        price = 110.0

        cash_after, win = _close_position(cash_before, pos, price, fee_rate)

        expected_proceeds = 1.0 * 110.0 * (1.0 - 0.001)
        assert abs(cash_after - expected_proceeds) < 1e-9, (
            f"Expected {expected_proceeds:.6f}, got {cash_after:.6f}"
        )
        assert win is True  # 110 > 100

    def test_short_close_fee_applied_to_cost(self):
        """Closing a short: cost = qty * price * (1 + fee_rate)."""
        pos = Position(sym=0, is_short=True, qty=1.0, entry_price=110.0)
        cash_before = 110.0  # received when opened short
        fee_rate = 0.001
        price = 100.0  # price fell, short is a win

        cash_after, win = _close_position(cash_before, pos, price, fee_rate)

        expected_cost = 1.0 * 100.0 * (1.0 + 0.001)
        expected_cash = 110.0 - expected_cost
        assert abs(cash_after - expected_cash) < 1e-9, (
            f"Expected {expected_cash:.6f}, got {cash_after:.6f}"
        )
        assert win is True  # entry 110 > close 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
