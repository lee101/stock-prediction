"""Tests for market simulator evaluation correctness.

Covers determinism, fee accounting, leverage, short borrow costs,
annualization, no-trade policy, and early-exit monkey-patching.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from pufferlib_market.hourly_replay import (
    MktdData,
    Position,
    simulate_daily_policy,
    _open_long,
    _short_borrow_fee,
)
from pufferlib_market.evaluate_tail import _slice_tail
from pufferlib_market.metrics import annualize_total_return
from src.market_sim_early_exit import EarlyExitDecision


# ---------------------------------------------------------------------------
# Helpers — reusable policy functions and synthetic MKTD data builders
# ---------------------------------------------------------------------------


def _always_flat(_obs: np.ndarray) -> int:
    return 0


def _always_long(_obs: np.ndarray) -> int:
    return 1


def _always_short(_obs: np.ndarray) -> int:
    return 2


def _make_mktd(
    close: np.ndarray,
    *,
    num_symbols: int = 1,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    open_: np.ndarray | None = None,
) -> MktdData:
    """Build a minimal MktdData from a 1-D close array (single symbol).

    For multi-symbol, pass close shaped [T, S] and num_symbols > 1.
    """
    F = 16
    P = 5
    S = num_symbols

    if close.ndim == 1:
        close = close[:, None]  # [T, 1]
    T = close.shape[0]
    assert close.shape == (T, S)

    features = np.zeros((T, S, F), dtype=np.float32)
    prices = np.zeros((T, S, P), dtype=np.float32)
    prices[:, :, 3] = close  # P_CLOSE
    prices[:, :, 0] = close if open_ is None else (open_ if open_.ndim == 2 else open_[:, None])  # P_OPEN
    prices[:, :, 1] = close if high is None else (high if high.ndim == 2 else high[:, None])  # P_HIGH
    prices[:, :, 2] = close if low is None else (low if low.ndim == 2 else low[:, None])  # P_LOW
    tradable = np.ones((T, S), dtype=np.uint8)

    symbols = [f"SYM{i}" for i in range(S)]
    return MktdData(
        version=2,
        symbols=symbols,
        features=features,
        prices=prices,
        tradable=tradable,
    )


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------


def test_simulate_daily_policy_deterministic():
    """Run simulate_daily_policy twice with the same data and policy; results
    must be bit-identical across all scalar metrics."""
    close = np.array([100.0, 105.0, 110.0, 108.0, 112.0, 115.0], dtype=np.float32)
    data = _make_mktd(close)

    call_count = [0]

    def deterministic_policy(obs: np.ndarray) -> int:
        call_count[0] += 1
        # Alternate between long and flat every other step.
        return 1 if (call_count[0] % 2 == 1) else 0

    kwargs = dict(
        max_steps=5,
        fee_rate=0.001,
        max_leverage=1.0,
        periods_per_year=365.0,
    )

    call_count[0] = 0
    r1 = simulate_daily_policy(data, deterministic_policy, **kwargs)

    call_count[0] = 0
    r2 = simulate_daily_policy(data, deterministic_policy, **kwargs)

    assert r1.total_return == r2.total_return
    assert r1.num_trades == r2.num_trades
    assert r1.win_rate == r2.win_rate
    assert r1.sortino == r2.sortino
    assert r1.max_drawdown == r2.max_drawdown
    np.testing.assert_array_equal(r1.actions, r2.actions)


# ---------------------------------------------------------------------------
# 2. Fee calculation
# ---------------------------------------------------------------------------


def test_fee_calculation():
    """Buy then sell with fee_rate > 0 should produce worse return than
    fee_rate=0.  The difference should match the expected fee deductions."""
    # Flat price so PnL comes only from fees.
    close = np.array([100.0, 100.0, 100.0], dtype=np.float32)
    data = _make_mktd(close)

    # Policy: step 0 -> long (action=1), step 1 -> flat (action=0 closes).
    step_counter = [0]

    def buy_then_sell(obs: np.ndarray) -> int:
        s = step_counter[0]
        step_counter[0] += 1
        return 1 if s == 0 else 0

    # Zero-fee run.
    step_counter[0] = 0
    r_free = simulate_daily_policy(
        data, buy_then_sell, max_steps=2, fee_rate=0.0, max_leverage=1.0, periods_per_year=365.0,
    )
    assert r_free.total_return == pytest.approx(0.0, abs=1e-12)
    assert r_free.num_trades >= 1

    # With fees.
    fee_rate = 0.001
    step_counter[0] = 0
    r_fee = simulate_daily_policy(
        data, buy_then_sell, max_steps=2, fee_rate=fee_rate, max_leverage=1.0, periods_per_year=365.0,
    )

    # Should lose money from fees alone.
    assert r_fee.total_return < 0.0
    # Two fee events: one on buy, one on sell. Each ~0.1% of notional.
    # On buy: cost = qty * price * (1 + fee_rate), and on sell: proceeds = qty * price * (1 - fee_rate).
    # Net return = (1 - fee_rate)/(1 + fee_rate) - 1 ~ -2 * fee_rate for small fee_rate.
    expected_approx = (1.0 - fee_rate) / (1.0 + fee_rate) - 1.0
    assert r_fee.total_return == pytest.approx(expected_approx, rel=1e-6)

    # Difference should be strictly positive (free > fee).
    assert r_free.total_return > r_fee.total_return


# ---------------------------------------------------------------------------
# 3. Leverage budget
# ---------------------------------------------------------------------------


def test_leverage_budget():
    """Verify that _open_long with max_leverage=2.0 produces 2x the position
    size vs 1x.  Also verify simulate_daily_policy's limit-order path clamps
    cost to available cash (so 2x leverage has no effect when cost > cash)."""
    # _open_long (market-order style) allows cash to go negative (margin).
    cash = 10000.0
    cash_1x, pos_1x = _open_long(cash, sym=0, price=100.0, fee_rate=0.0, max_leverage=1.0)
    cash_2x, pos_2x = _open_long(cash, sym=0, price=100.0, fee_rate=0.0, max_leverage=2.0)
    assert pos_1x is not None and pos_2x is not None
    # 2x leverage produces 2x the position size.
    assert pos_2x.qty == pytest.approx(pos_1x.qty * 2.0, rel=1e-9)
    # 1x uses all cash; 2x goes negative (margin borrowed).
    assert cash_1x == pytest.approx(0.0, abs=1e-9)
    assert cash_2x == pytest.approx(-10000.0, abs=1e-9)

    # simulate_daily_policy uses _open_long_limit which clamps cost to cash.
    # Therefore, in the sim, 2x leverage has no extra effect when the limit
    # order cost exceeds available cash.
    close = np.array([100.0, 110.0, 110.0], dtype=np.float32)
    data = _make_mktd(close)

    r_1x = simulate_daily_policy(
        data, _always_long, max_steps=2, fee_rate=0.0, max_leverage=1.0, periods_per_year=365.0,
    )
    r_2x = simulate_daily_policy(
        data, _always_long, max_steps=2, fee_rate=0.0, max_leverage=2.0, periods_per_year=365.0,
    )

    # At 1x leverage, return should be 10%.
    assert r_1x.total_return == pytest.approx(0.10, abs=1e-9)
    # With limit-order clamping, 2x still produces 10% (cost clamped to cash).
    assert r_2x.total_return == pytest.approx(0.10, abs=1e-9)
    # Both runs produce the same result due to the cost clamp.
    assert r_1x.total_return == pytest.approx(r_2x.total_return, abs=1e-12)


# ---------------------------------------------------------------------------
# 4. Short borrow cost
# ---------------------------------------------------------------------------


def test_short_borrow_cost():
    """A short position held while prices are flat should lose money from
    the borrow fee, while a flat position should not."""
    # Constant price: any PnL must come from the borrow cost.
    close = np.array([100.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float32)
    data = _make_mktd(close)

    short_borrow_apr = 0.0625  # 6.25%
    periods_per_year = 365.0

    r_short = simulate_daily_policy(
        data,
        _always_short,
        max_steps=4,
        fee_rate=0.0,
        max_leverage=1.0,
        periods_per_year=periods_per_year,
        short_borrow_apr=short_borrow_apr,
    )

    # With flat prices and no fees, return should be negative due to borrow cost.
    assert r_short.total_return < 0.0

    # Compare to flat policy which should have zero return.
    r_flat = simulate_daily_policy(
        data,
        _always_flat,
        max_steps=4,
        fee_rate=0.0,
        max_leverage=1.0,
        periods_per_year=periods_per_year,
        short_borrow_apr=short_borrow_apr,
    )
    assert r_flat.total_return == pytest.approx(0.0, abs=1e-12)

    # Verify cost is in the right ballpark: each step costs
    # notional * apr / periods_per_year.
    # The borrow fee per step is applied to (qty * price), where qty ~ cash/price = 100 shares.
    # Expected fee per step: 10000 * 0.0625 / 365 ~ 1.7123.
    expected_fee_per_step = _short_borrow_fee(
        pos=Position(sym=0, is_short=True, qty=100.0, entry_price=100.0),
        price=100.0,
        short_borrow_apr=short_borrow_apr,
        periods_per_year=periods_per_year,
    )
    assert expected_fee_per_step > 0.0
    # Over 4 steps, total loss should be roughly 4 * fee / initial_equity,
    # but compounding makes it slightly different. Just check direction and order of magnitude.
    assert r_short.total_return < -expected_fee_per_step * 3.0 / 10000.0


# ---------------------------------------------------------------------------
# 5. Annualization: periods_per_year=252 vs 365
# ---------------------------------------------------------------------------


def test_periods_per_year_annualization():
    """annualize_total_return with periods_per_year=252 must give a different
    (and correct) result vs periods_per_year=365 for the same raw return."""
    total_return = 0.10  # 10% over 30 periods
    periods = 30.0

    ann_252 = annualize_total_return(total_return, periods=periods, periods_per_year=252.0)
    ann_365 = annualize_total_return(total_return, periods=periods, periods_per_year=365.0)

    # They should differ.
    assert ann_252 != ann_365

    # Correct formula: (1 + total_return)^(periods_per_year / periods) - 1
    expected_252 = (1.0 + total_return) ** (252.0 / periods) - 1.0
    expected_365 = (1.0 + total_return) ** (365.0 / periods) - 1.0
    assert ann_252 == pytest.approx(expected_252, rel=1e-9)
    assert ann_365 == pytest.approx(expected_365, rel=1e-9)

    # 365 periods_per_year gives a larger exponent (365/30 > 252/30), so
    # annualized return is higher with 365 than 252.
    assert ann_365 > ann_252

    # Edge cases.
    assert annualize_total_return(0.10, periods=0.0, periods_per_year=252.0) == 0.0
    assert annualize_total_return(0.10, periods=30.0, periods_per_year=0.0) == 0.0
    # Total loss.
    assert annualize_total_return(-1.0, periods=30.0, periods_per_year=252.0) == -1.0


# ---------------------------------------------------------------------------
# 6. No-trade policy
# ---------------------------------------------------------------------------


def test_no_trade_policy_returns_zero():
    """A policy that always returns action=0 (flat) should yield zero return
    and zero trades, regardless of price movements."""
    # Prices move around -- but no position is ever opened.
    close = np.array([100.0, 110.0, 90.0, 120.0, 80.0, 100.0], dtype=np.float32)
    data = _make_mktd(close)

    result = simulate_daily_policy(
        data,
        _always_flat,
        max_steps=5,
        fee_rate=0.001,
        max_leverage=1.0,
        periods_per_year=365.0,
    )

    assert result.total_return == pytest.approx(0.0, abs=1e-12)
    assert result.num_trades == 0
    assert result.win_rate == 0.0
    assert result.max_drawdown == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 7. Early-exit monkey-patch
# ---------------------------------------------------------------------------


def test_early_exit_monkey_patch():
    """Monkey-patching evaluate_drawdown_vs_profit_early_exit to always return
    should_stop=False must disable early stopping, allowing the simulation to
    run to completion even with a large drawdown."""
    # Build a series that would normally trigger early exit:
    # strong rise then a crash well past 50% progress.
    prices_list = (
        [100.0]
        + [100.0 + 5.0 * i for i in range(1, 11)]  # rise to 150
        + [150.0 - 10.0 * i for i in range(1, 11)]  # crash to 50
        + [50.0]
    )
    close = np.array(prices_list, dtype=np.float32)
    data = _make_mktd(close)
    max_steps = len(prices_list) - 1

    # First, run without monkey-patch -- expect early exit.
    r_normal = simulate_daily_policy(
        data,
        _always_long,
        max_steps=max_steps,
        fee_rate=0.0,
        max_leverage=1.0,
        periods_per_year=365.0,
    )
    # The early exit should cause actions to be partially zero-filled.
    # We can check that not all steps were used by looking at the result.
    # (early exit is triggered when drawdown > profit past 50% progress)

    # Now monkey-patch to disable early exit.
    def no_early_exit(equity_values, *, total_steps, label, **kwargs):
        return EarlyExitDecision(
            should_stop=False,
            progress_fraction=0.0,
            total_return=0.0,
            max_drawdown=0.0,
        )

    with patch(
        "pufferlib_market.hourly_replay.evaluate_drawdown_vs_profit_early_exit",
        new=no_early_exit,
    ):
        r_patched = simulate_daily_policy(
            data,
            _always_long,
            max_steps=max_steps,
            fee_rate=0.0,
            max_leverage=1.0,
            periods_per_year=365.0,
        )

    # The patched run must complete all steps.
    non_zero_normal = int(np.count_nonzero(r_normal.actions))
    non_zero_patched = int(np.count_nonzero(r_patched.actions))
    # The patched run should have used more (or equal) action steps, since
    # no early termination occurred. All actions should be 1 (always long).
    assert non_zero_patched >= non_zero_normal
    # Verify the patched run ran to the end: all actions should be 1.
    assert np.all(r_patched.actions == 1)


# ---------------------------------------------------------------------------
# Additional edge-case: _slice_tail
# ---------------------------------------------------------------------------


def test_slice_tail_preserves_tail_window():
    """_slice_tail should return exactly `steps + 1` timesteps from the end."""
    close = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], dtype=np.float32)
    data = _make_mktd(close)

    tail = _slice_tail(data, steps=3)
    assert tail.num_timesteps == 4  # steps + 1
    # Should be the last 4 entries.
    np.testing.assert_array_equal(
        tail.prices[:, 0, 3],
        np.array([50.0, 60.0, 70.0, 80.0], dtype=np.float32),
    )

    # Too many steps should raise.
    with pytest.raises(ValueError, match="too short"):
        _slice_tail(data, steps=8)
