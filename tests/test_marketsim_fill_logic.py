"""Tests for market simulator fill logic — _resolve_limit_fill_price, position
opening/closing, allocation helpers, and short borrow costs.

Covers edge cases: zero prices, NaN, extreme leverage, boundary fills, etc.
"""

import math
import pytest
import numpy as np

from pufferlib_market.hourly_replay import (
    Position,
    _resolve_limit_fill_price,
    _normalize_fill_buffer_bps,
    _action_allocation_pct,
    _action_level_offset_bps,
    _open_long,
    _open_short,
    _open_long_limit,
    _open_short_limit,
    _close_position,
    _short_borrow_fee,
    _apply_short_borrow_cost,
)


# ---------------------------------------------------------------------------
# _resolve_limit_fill_price
# ---------------------------------------------------------------------------

class TestResolveLimitFillPrice:
    """Binary fill logic: returns target_price if bar trades through, else None."""

    def test_buy_fills_when_low_touches_target(self):
        # Low exactly at target → fills
        result = _resolve_limit_fill_price(
            low=100.0, high=110.0, target_price=100.0, is_buy=True,
        )
        assert result == 100.0

    def test_buy_fills_when_low_below_target(self):
        result = _resolve_limit_fill_price(
            low=95.0, high=110.0, target_price=100.0, is_buy=True,
        )
        assert result == 100.0

    def test_buy_no_fill_when_low_above_target(self):
        # Low above target → no fill
        result = _resolve_limit_fill_price(
            low=101.0, high=110.0, target_price=100.0, is_buy=True,
        )
        assert result is None

    def test_sell_fills_when_high_touches_target(self):
        result = _resolve_limit_fill_price(
            low=90.0, high=100.0, target_price=100.0, is_buy=False,
        )
        assert result == 100.0

    def test_sell_fills_when_high_above_target(self):
        result = _resolve_limit_fill_price(
            low=90.0, high=105.0, target_price=100.0, is_buy=False,
        )
        assert result == 100.0

    def test_sell_no_fill_when_high_below_target(self):
        result = _resolve_limit_fill_price(
            low=90.0, high=99.0, target_price=100.0, is_buy=False,
        )
        assert result is None

    # -- fill_buffer_bps tests --

    def test_buy_fill_buffer_tightens_trigger(self):
        # Buffer makes fills HARDER (quality gate): low must dip further.
        # target=100, buffer=50bps → trigger = 100*(1-0.005) = 99.5
        # low=99.6 > 99.5 → NO fill (bar didn't dip far enough below target)
        result = _resolve_limit_fill_price(
            low=99.6, high=110.0, target_price=100.0, is_buy=True,
            fill_buffer_bps=50.0,
        )
        assert result is None

    def test_buy_fill_buffer_allows_when_low_enough(self):
        # low=99.4 < 99.5 → fills (bar dipped below trigger)
        result = _resolve_limit_fill_price(
            low=99.4, high=110.0, target_price=100.0, is_buy=True,
            fill_buffer_bps=50.0,
        )
        assert result == 100.0

    def test_buy_fill_buffer_still_rejects(self):
        # target=100, buffer=50bps → trigger = 99.5
        # low=99.6 with buffer=5bps → trigger = 100*(1-0.0005) = 99.95
        # 99.6 < 99.95 → fills
        # But low=100.1 → no fill even with 50bps buffer
        result = _resolve_limit_fill_price(
            low=100.1, high=110.0, target_price=100.0, is_buy=True,
            fill_buffer_bps=50.0,
        )
        assert result is None

    def test_sell_fill_buffer_widens_trigger(self):
        # target=100, buffer=50bps → trigger = 100*(1+0.005) = 100.5
        # high=100.4 < 100.5 → no fill without buffer? No:
        # Without buffer: trigger = 100.0, high=100.4 >= 100.0 → fills
        # With 50bps buffer: trigger = 100.5, high=100.4 < 100.5 → NO fill
        result = _resolve_limit_fill_price(
            low=90.0, high=100.4, target_price=100.0, is_buy=False,
            fill_buffer_bps=50.0,
        )
        assert result is None

    def test_sell_fill_buffer_allows_fill(self):
        # high=100.6 >= 100.5 → fills with 50bps buffer
        result = _resolve_limit_fill_price(
            low=90.0, high=100.6, target_price=100.0, is_buy=False,
            fill_buffer_bps=50.0,
        )
        assert result == 100.0

    def test_zero_fill_buffer_same_as_no_buffer(self):
        result_zero = _resolve_limit_fill_price(
            low=99.0, high=101.0, target_price=100.0, is_buy=True,
            fill_buffer_bps=0.0,
        )
        result_default = _resolve_limit_fill_price(
            low=99.0, high=101.0, target_price=100.0, is_buy=True,
        )
        assert result_zero == result_default

    # -- edge cases --

    def test_swapped_low_high_handled(self):
        # low > high should still work (normalized internally)
        result = _resolve_limit_fill_price(
            low=110.0, high=95.0, target_price=100.0, is_buy=True,
        )
        assert result == 100.0

    def test_zero_target_price_buy(self):
        result = _resolve_limit_fill_price(
            low=0.0, high=10.0, target_price=0.0, is_buy=True,
        )
        assert result == 0.0  # trigger = 0, low=0 <= 0 → fills

    def test_buy_exact_boundary_with_buffer(self):
        # target=100, buffer=100bps → trigger = 100*(1-0.01) = 99.0
        # low=99.0 exactly → should fill (99.0 is NOT > 99.0)
        result = _resolve_limit_fill_price(
            low=99.0, high=105.0, target_price=100.0, is_buy=True,
            fill_buffer_bps=100.0,
        )
        assert result == 100.0

    def test_sell_exact_boundary_with_buffer(self):
        # target=100, buffer=100bps → trigger = 101.0
        # high=101.0 exactly → should fill (101.0 is NOT < 101.0)
        result = _resolve_limit_fill_price(
            low=95.0, high=101.0, target_price=100.0, is_buy=False,
            fill_buffer_bps=100.0,
        )
        assert result == 100.0


class TestNormalizeFillBufferBps:
    def test_normal_value(self):
        assert _normalize_fill_buffer_bps(5.0) == 5.0

    def test_zero(self):
        assert _normalize_fill_buffer_bps(0.0) == 0.0

    def test_none_treated_as_zero(self):
        assert _normalize_fill_buffer_bps(None) == 0.0

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="finite and >= 0"):
            _normalize_fill_buffer_bps(-1.0)

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="finite and >= 0"):
            _normalize_fill_buffer_bps(float("nan"))

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="finite and >= 0"):
            _normalize_fill_buffer_bps(float("inf"))


# ---------------------------------------------------------------------------
# _action_allocation_pct / _action_level_offset_bps
# ---------------------------------------------------------------------------

class TestActionAllocationPct:
    def test_single_bin_returns_one(self):
        assert _action_allocation_pct(alloc_idx=0, alloc_bins=1) == 1.0

    def test_zero_bins_returns_one(self):
        assert _action_allocation_pct(alloc_idx=0, alloc_bins=0) == 1.0

    def test_first_of_four(self):
        # idx=0, bins=4 → (0+1)/4 = 0.25
        assert abs(_action_allocation_pct(alloc_idx=0, alloc_bins=4) - 0.25) < 1e-10

    def test_last_of_four(self):
        # idx=3, bins=4 → (3+1)/4 = 1.0
        assert abs(_action_allocation_pct(alloc_idx=3, alloc_bins=4) - 1.0) < 1e-10

    def test_index_clipped_above(self):
        # idx=100, bins=4 → clipped to 3 → 1.0
        assert abs(_action_allocation_pct(alloc_idx=100, alloc_bins=4) - 1.0) < 1e-10

    def test_index_clipped_below(self):
        # idx=-5, bins=4 → clipped to 0 → 0.25
        assert abs(_action_allocation_pct(alloc_idx=-5, alloc_bins=4) - 0.25) < 1e-10

    def test_minimum_floor_001(self):
        # idx=0, bins=1000 → (0+1)/1000 = 0.001 → clipped to 0.01
        assert _action_allocation_pct(alloc_idx=0, alloc_bins=1000) == 0.01


class TestActionLevelOffsetBps:
    def test_single_bin_returns_zero(self):
        assert _action_level_offset_bps(level_idx=0, level_bins=1, max_offset_bps=100.0) == 0.0

    def test_zero_max_returns_zero(self):
        assert _action_level_offset_bps(level_idx=1, level_bins=3, max_offset_bps=0.0) == 0.0

    def test_midpoint_returns_zero(self):
        # bins=3, idx=1 → frac = 1/2 = 0.5 → (2*0.5-1)*100 = 0.0
        assert abs(_action_level_offset_bps(level_idx=1, level_bins=3, max_offset_bps=100.0)) < 1e-10

    def test_first_returns_negative_max(self):
        # idx=0, bins=3 → frac=0 → (0-1)*100 = -100
        assert abs(_action_level_offset_bps(level_idx=0, level_bins=3, max_offset_bps=100.0) - (-100.0)) < 1e-10

    def test_last_returns_positive_max(self):
        # idx=2, bins=3 → frac=1.0 → (2-1)*100 = 100
        assert abs(_action_level_offset_bps(level_idx=2, level_bins=3, max_offset_bps=100.0) - 100.0) < 1e-10


# ---------------------------------------------------------------------------
# _open_long / _open_short (simple market order fills)
# ---------------------------------------------------------------------------

class TestOpenLong:
    def test_basic_open(self):
        cash, pos = _open_long(cash=10000.0, sym=0, price=100.0, fee_rate=0.001, max_leverage=1.0)
        assert pos is not None
        assert not pos.is_short
        assert pos.entry_price == 100.0
        # qty = 10000 / (100 * 1.001) ≈ 99.9
        assert abs(pos.qty - 10000.0 / (100.0 * 1.001)) < 1e-6
        assert cash < 1.0  # almost all cash used

    def test_zero_price_no_open(self):
        cash, pos = _open_long(cash=10000.0, sym=0, price=0.0, fee_rate=0.001, max_leverage=1.0)
        assert pos is None
        assert cash == 10000.0

    def test_zero_cash_no_open(self):
        cash, pos = _open_long(cash=0.0, sym=0, price=100.0, fee_rate=0.001, max_leverage=1.0)
        assert pos is None
        assert cash == 0.0

    def test_leverage_doubles_position(self):
        _, pos1 = _open_long(cash=10000.0, sym=0, price=100.0, fee_rate=0.001, max_leverage=1.0)
        _, pos2 = _open_long(cash=10000.0, sym=0, price=100.0, fee_rate=0.001, max_leverage=2.0)
        assert abs(pos2.qty / pos1.qty - 2.0) < 1e-6


class TestOpenShort:
    def test_basic_short(self):
        cash, pos = _open_short(cash=10000.0, sym=0, price=100.0, fee_rate=0.001, max_leverage=1.0)
        assert pos is not None
        assert pos.is_short
        assert pos.entry_price == 100.0
        # Cash increases from short proceeds
        assert cash > 10000.0

    def test_zero_price_no_short(self):
        cash, pos = _open_short(cash=10000.0, sym=0, price=0.0, fee_rate=0.001, max_leverage=1.0)
        assert pos is None


# ---------------------------------------------------------------------------
# _open_long_limit / _open_short_limit (limit order fills)
# ---------------------------------------------------------------------------

class TestOpenLongLimit:
    def test_fills_when_low_touches(self):
        cash, pos = _open_long_limit(
            cash=10000.0, sym=0, close_price=100.0,
            low_price=99.0, high_price=102.0,
            fee_rate=0.001, max_leverage=1.0,
            allocation_pct=1.0, level_offset_bps=0.0,
            fill_buffer_bps=0.0,
        )
        assert pos is not None
        assert pos.entry_price == 100.0  # fills at target = close_price

    def test_no_fill_when_low_above_target(self):
        cash, pos = _open_long_limit(
            cash=10000.0, sym=0, close_price=100.0,
            low_price=101.0, high_price=105.0,
            fee_rate=0.001, max_leverage=1.0,
            allocation_pct=1.0, level_offset_bps=0.0,
            fill_buffer_bps=0.0,
        )
        assert pos is None
        assert cash == 10000.0

    def test_allocation_pct_limits_position(self):
        _, pos_full = _open_long_limit(
            cash=10000.0, sym=0, close_price=100.0,
            low_price=95.0, high_price=105.0,
            fee_rate=0.001, max_leverage=1.0,
            allocation_pct=1.0, level_offset_bps=0.0,
            fill_buffer_bps=0.0,
        )
        _, pos_half = _open_long_limit(
            cash=10000.0, sym=0, close_price=100.0,
            low_price=95.0, high_price=105.0,
            fee_rate=0.001, max_leverage=1.0,
            allocation_pct=0.5, level_offset_bps=0.0,
            fill_buffer_bps=0.0,
        )
        assert abs(pos_half.qty / pos_full.qty - 0.5) < 1e-6

    def test_level_offset_adjusts_target(self):
        # 100bps offset → target = 100 * (1 + 0.01) = 101
        _, pos = _open_long_limit(
            cash=10000.0, sym=0, close_price=100.0,
            low_price=99.0, high_price=105.0,
            fee_rate=0.001, max_leverage=1.0,
            allocation_pct=1.0, level_offset_bps=100.0,
            fill_buffer_bps=0.0,
        )
        assert pos is not None
        assert abs(pos.entry_price - 101.0) < 1e-6

    def test_zero_allocation_no_open(self):
        cash, pos = _open_long_limit(
            cash=10000.0, sym=0, close_price=100.0,
            low_price=95.0, high_price=105.0,
            fee_rate=0.001, max_leverage=1.0,
            allocation_pct=0.0, level_offset_bps=0.0,
            fill_buffer_bps=0.0,
        )
        assert pos is None

    def test_fill_buffer_filters_marginal_trades(self):
        # close=100, low=99.9 → fills without buffer
        _, pos_no_buf = _open_long_limit(
            cash=10000.0, sym=0, close_price=100.0,
            low_price=99.9, high_price=101.0,
            fee_rate=0.001, max_leverage=1.0,
            allocation_pct=1.0, level_offset_bps=0.0,
            fill_buffer_bps=0.0,
        )
        assert pos_no_buf is not None

        # With 200bps buffer → trigger = 100*(1-0.02) = 98.0
        # low=99.9 > 98.0 → still fills (buffer makes it EASIER to fill for buys)
        # Wait, re-read: trigger is compared as lo > trigger → no fill
        # trigger = 100*(1-0.02) = 98.0, lo=99.9 > 98.0 → NO FILL
        # Actually: the code says `if lo > trigger: return None`
        # So: lo=99.9 > 98.0 → True → None. Hmm, that seems wrong.
        # Let me re-check: the buffer WIDENS the trigger for buys.
        # Without buffer: trigger = 100*(1-0) = 100.0, lo=99.9 < 100 → fills
        # With 200bps: trigger = 100*(1-0.02) = 98.0, lo=99.9 > 98.0 → no fill?
        #
        # Wait no. The condition is: if lo > trigger → return None (no fill).
        # So for a buy: the bar's low must dip AT LEAST to the trigger price.
        # Without buffer: trigger=100, low=99.9 ≤ 100 → fills ✓
        # With buffer: trigger=98, low=99.9 > 98 → fills? No: 99.9 > 98.0 is TRUE → None!
        # But that's wrong. A wider trigger should make fills EASIER.
        #
        # Actually re-reading: trigger = target * (1 - buffer). So buffer LOWERS
        # the trigger, meaning the bar's low must go LOWER to fill. Buffer makes
        # fills HARDER for buys. This is the "fill buffer" = slippage/quality gate.
        # The bar must trade far enough below target to prove the price was really there.
        #
        # Correction: buffer=200bps on target=100 → trigger=98. low must be ≤98 to fill.
        # This is correct: higher buffer = harder to fill = more selective.
        _, pos_with_buf = _open_long_limit(
            cash=10000.0, sym=0, close_price=100.0,
            low_price=99.9, high_price=101.0,
            fee_rate=0.001, max_leverage=1.0,
            allocation_pct=1.0, level_offset_bps=0.0,
            fill_buffer_bps=200.0,
        )
        assert pos_with_buf is None  # buffer too tight for this bar


class TestOpenShortLimit:
    def test_fills_when_high_reaches_target(self):
        cash, pos = _open_short_limit(
            cash=10000.0, sym=0, close_price=100.0,
            low_price=98.0, high_price=101.0,
            fee_rate=0.001, max_leverage=1.0,
            allocation_pct=1.0, level_offset_bps=0.0,
            fill_buffer_bps=0.0,
        )
        assert pos is not None
        assert pos.is_short
        assert pos.entry_price == 100.0

    def test_no_fill_when_high_below_target(self):
        cash, pos = _open_short_limit(
            cash=10000.0, sym=0, close_price=100.0,
            low_price=90.0, high_price=99.0,
            fee_rate=0.001, max_leverage=1.0,
            allocation_pct=1.0, level_offset_bps=0.0,
            fill_buffer_bps=0.0,
        )
        assert pos is None


# ---------------------------------------------------------------------------
# _close_position
# ---------------------------------------------------------------------------

class TestClosePosition:
    def test_close_long_win(self):
        pos = Position(sym=0, is_short=False, qty=10.0, entry_price=100.0)
        cash, win = _close_position(cash=0.0, pos=pos, price=110.0, fee_rate=0.001)
        assert win is True
        # proceeds = 10 * 110 * (1 - 0.001) = 1098.9
        assert abs(cash - 10.0 * 110.0 * 0.999) < 1e-6

    def test_close_long_loss(self):
        pos = Position(sym=0, is_short=False, qty=10.0, entry_price=100.0)
        cash, win = _close_position(cash=0.0, pos=pos, price=90.0, fee_rate=0.001)
        assert win is False

    def test_close_short_win(self):
        pos = Position(sym=0, is_short=True, qty=10.0, entry_price=100.0)
        cash, win = _close_position(cash=20000.0, pos=pos, price=90.0, fee_rate=0.001)
        assert win is True
        # cost = 10 * 90 * 1.001 = 900.9
        expected_cash = 20000.0 - 10.0 * 90.0 * 1.001
        assert abs(cash - expected_cash) < 1e-6

    def test_close_short_loss(self):
        pos = Position(sym=0, is_short=True, qty=10.0, entry_price=100.0)
        cash, win = _close_position(cash=20000.0, pos=pos, price=110.0, fee_rate=0.001)
        assert win is False

    def test_fee_rate_zero(self):
        pos = Position(sym=0, is_short=False, qty=100.0, entry_price=50.0)
        cash, win = _close_position(cash=0.0, pos=pos, price=50.0, fee_rate=0.0)
        assert abs(cash - 5000.0) < 1e-6
        assert win is False  # price == entry → not a win


# ---------------------------------------------------------------------------
# Short borrow costs
# ---------------------------------------------------------------------------

class TestShortBorrowFee:
    def test_no_fee_for_long(self):
        pos = Position(sym=0, is_short=False, qty=10.0, entry_price=100.0)
        assert _short_borrow_fee(pos=pos, price=100.0, short_borrow_apr=0.10, periods_per_year=365.0) == 0.0

    def test_no_fee_for_none(self):
        assert _short_borrow_fee(pos=None, price=100.0, short_borrow_apr=0.10, periods_per_year=365.0) == 0.0

    def test_no_fee_when_apr_zero(self):
        pos = Position(sym=0, is_short=True, qty=10.0, entry_price=100.0)
        assert _short_borrow_fee(pos=pos, price=100.0, short_borrow_apr=0.0, periods_per_year=365.0) == 0.0

    def test_correct_daily_fee(self):
        pos = Position(sym=0, is_short=True, qty=10.0, entry_price=100.0)
        # notional = 10 * 100 = 1000, daily fee = 1000 * 0.10 / 365
        fee = _short_borrow_fee(pos=pos, price=100.0, short_borrow_apr=0.10, periods_per_year=365.0)
        expected = 1000.0 * 0.10 / 365.0
        assert abs(fee - expected) < 1e-8

    def test_apply_short_borrow_cost_deducts(self):
        pos = Position(sym=0, is_short=True, qty=10.0, entry_price=100.0)
        cash, fee = _apply_short_borrow_cost(
            cash=10000.0, pos=pos, price=100.0,
            short_borrow_apr=0.10, periods_per_year=365.0,
        )
        assert fee > 0
        assert abs(cash - (10000.0 - fee)) < 1e-8


# ---------------------------------------------------------------------------
# Round-trip P&L consistency
# ---------------------------------------------------------------------------

class TestRoundTripPnL:
    """Verify that open + close at same price loses exactly 2x fees."""

    def test_long_roundtrip_at_same_price(self):
        fee_rate = 0.001  # 10bps
        initial_cash = 10000.0
        price = 100.0

        cash_after_open, pos = _open_long(cash=initial_cash, sym=0, price=price, fee_rate=fee_rate, max_leverage=1.0)
        assert pos is not None

        cash_after_close, _ = _close_position(cash=cash_after_open, pos=pos, price=price, fee_rate=fee_rate)

        # Lost ~2x fee_rate of initial cash
        pnl = cash_after_close - initial_cash
        expected_loss = -initial_cash * (1 - (1 - fee_rate) / (1 + fee_rate))
        assert abs(pnl - expected_loss) < 1.0  # within $1

    def test_short_roundtrip_at_same_price(self):
        fee_rate = 0.001
        initial_cash = 10000.0
        price = 100.0

        cash_after_open, pos = _open_short(cash=initial_cash, sym=0, price=price, fee_rate=fee_rate, max_leverage=1.0)
        assert pos is not None

        cash_after_close, _ = _close_position(cash=cash_after_open, pos=pos, price=price, fee_rate=fee_rate)

        pnl = cash_after_close - initial_cash
        # Should be negative (lost fees)
        assert pnl < 0
        assert abs(pnl) < initial_cash * 0.01  # less than 1% lost to fees


# ---------------------------------------------------------------------------
# Fill buffer production realism
# ---------------------------------------------------------------------------

class TestFillBufferProductionValues:
    """Test with production-realistic values: 5bps, 8bps, 10bps buffers."""

    @pytest.mark.parametrize("buffer_bps", [5.0, 8.0, 10.0])
    def test_production_buffer_values_accepted(self, buffer_bps):
        assert _normalize_fill_buffer_bps(buffer_bps) == buffer_bps

    def test_5bps_buy_fill(self):
        # target=100, buffer=5bps → trigger = 100*(1-0.0005) = 99.95
        # low must be ≤ 99.95 to fill
        result = _resolve_limit_fill_price(
            low=99.94, high=100.5, target_price=100.0, is_buy=True,
            fill_buffer_bps=5.0,
        )
        assert result == 100.0

    def test_5bps_buy_no_fill(self):
        result = _resolve_limit_fill_price(
            low=99.96, high=100.5, target_price=100.0, is_buy=True,
            fill_buffer_bps=5.0,
        )
        assert result is None

    def test_8bps_sell_fill(self):
        # target=100, buffer=8bps → trigger = 100*(1+0.0008) = 100.08
        # high must be ≥ 100.08 to fill
        result = _resolve_limit_fill_price(
            low=99.0, high=100.09, target_price=100.0, is_buy=False,
            fill_buffer_bps=8.0,
        )
        assert result == 100.0

    def test_8bps_sell_no_fill(self):
        result = _resolve_limit_fill_price(
            low=99.0, high=100.07, target_price=100.0, is_buy=False,
            fill_buffer_bps=8.0,
        )
        assert result is None
