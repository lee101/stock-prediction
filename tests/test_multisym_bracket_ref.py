"""Spec tests for the multi-symbol daily bracket env reference.

These tests pin the env semantics so the (forthcoming) CUDA kernel can be
golden-tested against this Python reference.
"""
from __future__ import annotations

import numpy as np
import pytest

from pufferlib_cpp_market_sim.python.market_sim_py.multisym_bracket_ref import (
    MultiSymBracketConfig,
    step,
)


def _tradable_all_true(B: int, S: int) -> np.ndarray:
    return np.ones((B, S), dtype=bool)


def test_no_op_action_returns_zero_reward_when_flat():
    cfg = MultiSymBracketConfig()
    B, S = 1, 3
    cash = np.array([100_000.0])
    positions = np.zeros((B, S))
    actions = np.zeros((B, S, 4))   # nothing happens
    prev_close = np.full((B, S), 100.0)
    bar_open = np.full((B, S), 100.0)
    bar_high = np.full((B, S), 102.0)
    bar_low  = np.full((B, S), 98.0)
    bar_close = np.full((B, S), 101.0)

    new_cash, new_pos, reward, info = step(
        cash, positions, actions, prev_close,
        bar_open, bar_high, bar_low, bar_close,
        _tradable_all_true(B, S), cfg,
    )
    assert np.allclose(new_cash, cash)
    assert np.allclose(new_pos, 0.0)
    assert np.allclose(reward, 0.0)
    assert info["fees"][0] == 0.0
    assert info["margin_cost"][0] == 0.0


def test_simple_buy_fills_at_limit_and_pays_fee():
    cfg = MultiSymBracketConfig(fee_bps=10.0, fill_buffer_bps=0.0, max_leverage=1.0)
    B, S = 1, 1
    cash = np.array([10_000.0])
    positions = np.zeros((B, S))
    # buy at limit equal to prev_close, allocate full equity, no sell.
    actions = np.array([[[0.0, 0.0, 1.0, 0.0]]])
    prev_close = np.array([[100.0]])
    bar_open  = np.array([[100.5]])
    bar_high  = np.array([[101.0]])
    bar_low   = np.array([[99.5]])    # touches limit -> filled
    bar_close = np.array([[100.0]])   # MTM at exactly limit (no PnL on the position)

    new_cash, new_pos, reward, info = step(
        cash, positions, actions, prev_close,
        bar_open, bar_high, bar_low, bar_close,
        _tradable_all_true(B, S), cfg,
    )
    # Bought 100 shares at $100 = $10,000 notional. Fee = 10 bps * 10000 = $10.
    expected_fee = 10_000.0 * 10e-4
    assert info["buy_filled"][0, 0]
    assert pytest.approx(new_pos[0, 0], rel=1e-6) == 100.0
    assert pytest.approx(new_cash[0], rel=1e-6) == 10_000.0 - 10_000.0 - expected_fee
    # New equity = -$10 (fee) → reward ≈ -0.001.
    assert pytest.approx(reward[0], rel=1e-4) == -expected_fee / 10_000.0


def test_buy_doesnt_fill_when_low_above_limit():
    cfg = MultiSymBracketConfig(fee_bps=10.0, fill_buffer_bps=0.0)
    B, S = 1, 1
    cash = np.array([10_000.0])
    positions = np.zeros((B, S))
    # limit BELOW prev_close at -50bps = $99.50
    actions = np.array([[[-50.0, 0.0, 1.0, 0.0]]])
    prev_close = np.array([[100.0]])
    bar_open = np.array([[100.0]])
    bar_high = np.array([[101.0]])
    bar_low  = np.array([[99.51]])   # 99.51 > 99.50 limit -> NO fill
    bar_close = np.array([[100.5]])

    new_cash, new_pos, reward, info = step(
        cash, positions, actions, prev_close,
        bar_open, bar_high, bar_low, bar_close,
        _tradable_all_true(B, S), cfg,
    )
    assert not info["buy_filled"][0, 0]
    assert new_pos[0, 0] == 0.0
    assert new_cash[0] == 10_000.0


def test_leverage_clip_scales_oversized_position():
    cfg = MultiSymBracketConfig(fee_bps=0.0, fill_buffer_bps=0.0, max_leverage=2.0)
    B, S = 1, 2
    cash = np.array([10_000.0])
    positions = np.zeros((B, S))
    # Buy 1.5x equity in each of 2 syms = 3.0x total, which exceeds 2x cap.
    actions = np.array([[[0.0, 0.0, 1.5, 0.0],
                         [0.0, 0.0, 1.5, 0.0]]])
    prev_close = np.array([[100.0, 50.0]])
    bar_open  = np.array([[100.0, 50.0]])
    bar_high  = np.array([[100.0, 50.0]])
    bar_low   = np.array([[100.0, 50.0]])   # limit touches → filled
    bar_close = np.array([[100.0, 50.0]])

    new_cash, new_pos, reward, info = step(
        cash, positions, actions, prev_close,
        bar_open, bar_high, bar_low, bar_close,
        _tradable_all_true(B, S), cfg,
    )
    # Σ|new_position * close| should equal 2.0 * equity_prev = $20,000.
    notional = np.abs(new_pos) * bar_close
    assert pytest.approx(notional.sum(), rel=1e-6) == 20_000.0
    assert info["leverage_clip_active"][0]


def test_margin_interest_charged_on_borrowed_amount():
    cfg = MultiSymBracketConfig(
        fee_bps=0.0, fill_buffer_bps=0.0, max_leverage=2.0,
        annual_margin_rate=0.0625, trading_days_per_year=252,
    )
    B, S = 1, 1
    cash = np.array([10_000.0])
    positions = np.zeros((B, S))
    # Buy 1.5x equity = $15,000 → borrow $5,000.
    actions = np.array([[[0.0, 0.0, 1.5, 0.0]]])
    prev_close = np.array([[100.0]])
    bar_open  = np.array([[100.0]])
    bar_high  = np.array([[100.0]])
    bar_low   = np.array([[100.0]])
    bar_close = np.array([[100.0]])

    new_cash, new_pos, reward, info = step(
        cash, positions, actions, prev_close,
        bar_open, bar_high, bar_low, bar_close,
        _tradable_all_true(B, S), cfg,
    )
    expected_daily = 5_000.0 * (0.0625 / 252.0)
    assert pytest.approx(info["margin_cost"][0], rel=1e-6) == expected_daily
    assert pytest.approx(info["borrowed"][0], rel=1e-6) == 5_000.0
    # New equity = $10,000 - margin_interest, position fully MTM at 100.
    expected_equity = 10_000.0 - expected_daily
    assert pytest.approx(info["new_equity"][0], rel=1e-6) == expected_equity


def test_fill_buffer_blocks_marginal_fill():
    cfg = MultiSymBracketConfig(fee_bps=0.0, fill_buffer_bps=10.0, max_leverage=1.0)
    B, S = 1, 1
    cash = np.array([10_000.0])
    positions = np.zeros((B, S))
    actions = np.array([[[0.0, 0.0, 1.0, 0.0]]])
    prev_close = np.array([[100.0]])
    bar_open = np.array([[100.0]])
    bar_high = np.array([[101.0]])
    # Bar low equals limit exactly — fb=10bps requires bar_low <= 99.90.
    bar_low = np.array([[99.95]])
    bar_close = np.array([[100.0]])

    new_cash, new_pos, reward, info = step(
        cash, positions, actions, prev_close,
        bar_open, bar_high, bar_low, bar_close,
        _tradable_all_true(B, S), cfg,
    )
    assert not info["buy_filled"][0, 0]


def test_tradable_mask_blocks_fill():
    cfg = MultiSymBracketConfig(fee_bps=0.0, fill_buffer_bps=0.0, max_leverage=1.0)
    B, S = 1, 2
    cash = np.array([10_000.0])
    positions = np.zeros((B, S))
    actions = np.array([[[0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0]]])
    prev_close = np.array([[100.0, 100.0]])
    bar_open  = np.array([[100.0, 100.0]])
    bar_high  = np.array([[100.0, 100.0]])
    bar_low   = np.array([[100.0, 100.0]])
    bar_close = np.array([[100.0, 100.0]])
    mask = np.array([[True, False]])   # second sym not tradable

    _, new_pos, _, info = step(
        cash, positions, actions, prev_close,
        bar_open, bar_high, bar_low, bar_close,
        mask, cfg,
    )
    assert info["buy_filled"][0, 0]
    assert not info["buy_filled"][0, 1]
    assert new_pos[0, 1] == 0.0


def test_short_sell_is_supported_when_already_long():
    cfg = MultiSymBracketConfig(fee_bps=0.0, fill_buffer_bps=0.0, max_leverage=2.0)
    B, S = 1, 1
    cash = np.array([10_000.0])
    positions = np.array([[100.0]])  # long 100 shares from a prior step
    # Sell-only action — limit at prev_close, full equity sized
    actions = np.array([[[0.0, 0.0, 0.0, 1.0]]])
    prev_close = np.array([[100.0]])
    bar_open  = np.array([[100.0]])
    bar_high  = np.array([[100.0]])  # touches → filled
    bar_low   = np.array([[100.0]])
    bar_close = np.array([[100.0]])

    new_cash, new_pos, reward, info = step(
        cash, positions, actions, prev_close,
        bar_open, bar_high, bar_low, bar_close,
        _tradable_all_true(B, S), cfg,
    )
    # equity_prev = 10_000 + 100 * 100 = 20_000. Sell 100% of equity = 200 shares.
    # New position = 100 - 200 = -100 (short 100). Cash = 10_000 + 200*100 = 30_000.
    assert pytest.approx(new_pos[0, 0]) == -100.0
    assert pytest.approx(new_cash[0]) == 30_000.0


def test_dual_fill_cancellation_blocks_intrabar_scalping():
    """Same-bar buy-low + sell-high on the same instrument no longer captures
    the H-L spread risk-free. Exactly one side fills, the other is cancelled.
    """
    cfg = MultiSymBracketConfig(fee_bps=0.0, fill_buffer_bps=0.0, max_leverage=1.0)
    B, S = 1, 1
    cash = np.array([10_000.0])
    positions = np.zeros((B, S))
    # Buy at -50 bps (limit 99.5), sell at +50 bps (limit 100.5), both 50% size.
    actions = np.array([[[-50.0, 50.0, 0.5, 0.5]]])
    prev_close = np.array([[100.0]])
    bar_open  = np.array([[100.0]])
    bar_high  = np.array([[101.0]])  # touches sell limit
    bar_low   = np.array([[99.0]])   # touches buy limit
    bar_close = np.array([[100.0]])

    new_cash, new_pos, reward, info = step(
        cash, positions, actions, prev_close,
        bar_open, bar_high, bar_low, bar_close,
        _tradable_all_true(B, S), cfg,
    )
    # Pre-fix: BOTH would fill, capturing 100 bps - fees risk-free per bar.
    # Post-fix: exactly one side fills.
    assert int(info["buy_filled"][0, 0]) + int(info["sell_filled"][0, 0]) == 1, \
        "exactly one side must fill, never both"
    # One-side trade can move equity by at most ~50 bps (the limit-vs-close
    # gap on the side that filled), NOT the full 100 bps risk-free dual-fill.
    new_eq = new_cash[0] + new_pos[0, 0] * 100.0
    pnl_bps = abs(new_eq - 10_000.0) / 10_000.0 * 1e4
    assert pnl_bps <= 60.0, f"single-side PnL was {pnl_bps:.1f} bps, expected ≤60"


def test_dual_fill_buy_only_when_buy_closer_to_open():
    """Buy limit ($99.5) closer to bar_open ($99.7) than sell limit ($100.5)
    => buy fills, sell does not.
    """
    cfg = MultiSymBracketConfig(fee_bps=0.0, fill_buffer_bps=0.0, max_leverage=1.0)
    actions = np.array([[[-50.0, 50.0, 0.5, 0.5]]])
    res = step(np.array([10_000.0]), np.zeros((1, 1)), actions,
               np.array([[100.0]]), np.array([[99.7]]),
               np.array([[101.0]]), np.array([[99.0]]), np.array([[100.0]]),
               _tradable_all_true(1, 1), cfg)
    assert res[3]["buy_filled"][0, 0] and not res[3]["sell_filled"][0, 0]


def test_leverage_clip_does_not_leak_equity_on_flat_tape():
    """Reproduces the short-only free-money exploit:

    A policy that, each bar, sells ``pct`` of equity on every one of ``S``
    symbols at ``limit = prev_close`` on a perfectly flat tape must NEVER
    grow equity. The old clip semantic scaled the *existing* position along
    with the new trade — shrinking the short with no corresponding buy-back
    cash flow — which turned a flat tape into >100% per-bar log-return.
    Fix: clip only the new trade delta, leave existing positions untouched.
    """
    cfg = MultiSymBracketConfig(fee_bps=0.0, fill_buffer_bps=0.0,
                                max_leverage=2.0, annual_margin_rate=0.0)
    B, S = 1, 32
    cash = np.array([10_000.0])
    positions = np.zeros((B, S))
    prev_close = np.full((B, S), 100.0)
    bar_open   = np.full((B, S), 100.0)
    bar_high   = np.full((B, S), 100.0)
    bar_low    = np.full((B, S), 100.0)
    bar_close  = np.full((B, S), 100.0)
    actions = np.zeros((B, S, 4))
    actions[..., 3] = 0.12  # sell 12% of equity per symbol (32*12% = 384% requested)
    trad = np.ones((B, S), dtype=bool)

    for t in range(25):
        cash, positions, reward, info = step(
            cash, positions, actions, prev_close,
            bar_open, bar_high, bar_low, bar_close, trad, cfg)

    # After many bars on a zero-fee flat tape, equity must remain within
    # rounding noise of init. Pre-fix this grew to ~150× init.
    eq_final = cash[0] + (positions[0] * bar_close[0]).sum()
    assert abs(eq_final - 10_000.0) < 1e-3, \
        f"leverage-clip leaked equity: $10,000 -> ${eq_final:,.2f} on flat tape"
    # And notional must be capped at exactly 2x equity.
    notional = float(np.abs(positions[0] * bar_close[0]).sum())
    assert notional <= 2.0 * 10_000.0 + 1e-3, f"notional {notional} exceeds 2x cap"


def test_leverage_clip_only_affects_new_trades_not_existing_position():
    """When the candidate trade breaches max_leverage, scale only the new
    buy/sell shares. Existing positions stay untouched (a real broker
    doesn't force-close your existing positions — it just rejects the
    margin-breaching part of your new order).
    """
    cfg = MultiSymBracketConfig(fee_bps=0.0, fill_buffer_bps=0.0,
                                max_leverage=1.0, annual_margin_rate=0.0)
    B, S = 1, 1
    # Existing 60% long, then try to buy another 60% (total would be 120% > cap=100%).
    cash = np.array([4_000.0])
    positions = np.array([[60.0]])        # 60 shares × $100 = $6,000 long
    prev_close = np.array([[100.0]])
    bar_open   = np.array([[100.0]])
    bar_high   = np.array([[101.0]])
    bar_low    = np.array([[99.0]])
    bar_close  = np.array([[100.0]])
    actions = np.array([[[0.0, 0.0, 0.6, 0.0]]])  # try buy 60% of equity more
    trad = np.ones((B, S), dtype=bool)

    new_cash, new_pos, reward, info = step(
        cash, positions, actions, prev_close,
        bar_open, bar_high, bar_low, bar_close, trad, cfg)

    # equity_prev = 4,000 + 60 * 100 = 10,000. Cap = 1.0 * 10,000 = 10,000.
    # Existing notional = 6,000. Headroom = 4,000.
    # Requested trade delta notional = 60% * 10,000 = 6,000. alpha = 4,000 / 6,000 = 0.667.
    # Scaled buy_shares = 0.667 * 60 = 40. new_pos = 60 + 40 = 100 shares.
    assert pytest.approx(new_pos[0, 0], rel=1e-5) == 100.0
    # Notional_close = 100 * 100 = 10,000 = cap exactly.
    # Cash paid out = 40 * 100 = 4,000. new_cash = 4,000 - 4,000 = 0.
    assert pytest.approx(new_cash[0], abs=1e-3) == 0.0
    # Equity preserved (no free PnL).
    new_eq = new_cash[0] + new_pos[0, 0] * 100.0
    assert pytest.approx(new_eq, rel=1e-5) == 10_000.0


def test_dual_fill_sell_only_when_sell_closer_to_open():
    """Sell limit ($100.5) closer to bar_open ($100.3) than buy limit ($99.5)
    => sell fills, buy does not.
    """
    cfg = MultiSymBracketConfig(fee_bps=0.0, fill_buffer_bps=0.0, max_leverage=1.0)
    actions = np.array([[[-50.0, 50.0, 0.5, 0.5]]])
    res = step(np.array([10_000.0]), np.zeros((1, 1)), actions,
               np.array([[100.0]]), np.array([[100.3]]),
               np.array([[101.0]]), np.array([[99.0]]), np.array([[100.0]]),
               _tradable_all_true(1, 1), cfg)
    assert res[3]["sell_filled"][0, 0] and not res[3]["buy_filled"][0, 0]
