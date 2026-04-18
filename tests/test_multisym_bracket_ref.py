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
