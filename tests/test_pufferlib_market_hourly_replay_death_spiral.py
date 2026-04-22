"""Parity tests for the Python sim death-spiral guard.

Mirrors the prod semantics in ``src/alpaca_singleton.guard_sell_against_death_spiral``:
intraday tight tolerance; overnight wide tolerance; refused sells keep the
position on the books (as they would in prod after the daemon crashes).
"""
from __future__ import annotations

import numpy as np

from pufferlib_market.hourly_replay import (
    InitialPositionSpec,
    MktdData,
    _death_spiral_refused,
    simulate_daily_policy,
)


def _crash_data(num_bars: int = 12, entry_px: float = 100.0, crash_px: float = 98.0) -> MktdData:
    """Single-symbol data: bar 0 at ``entry_px`` (where initial_position opens),
    bar 1+ crashes to ``crash_px`` so every subsequent close-at-close sees the
    drop vs the remembered entry price.
    """
    T = int(num_bars)
    prices = np.zeros((T, 1, 5), dtype=np.float32)
    prices[0, 0, :] = [entry_px, entry_px, entry_px, entry_px, 0.0]
    for t in range(1, T):
        prices[t, 0, :] = [crash_px, crash_px, crash_px, crash_px, 0.0]
    features = np.zeros((T, 1, 16), dtype=np.float32)
    return MktdData(
        version=2,
        symbols=["AAA"],
        features=features,
        prices=prices,
        tradable=np.ones((T, 1), dtype=np.uint8),
    )


class _HoldThenSell:
    """Policy: action=1 (hold AAA long) for ``hold_for_steps``, then action=0."""

    def __init__(self, hold_for_steps: int) -> None:
        self.hold_for_steps = int(hold_for_steps)
        self._calls = 0

    def __call__(self, _obs: np.ndarray) -> int:
        action = 1 if self._calls < self.hold_for_steps else 0
        self._calls += 1
        return action


def test_guard_disabled_by_default_closes_at_crash() -> None:
    """Back-compat: default ``death_spiral_tolerance_bps=None`` keeps old behavior.

    Hold through bar 0 (entry=100), try to sell at bar 1+ where price=98.
    Without the guard the sell fires and realizes the -200bps drop.
    """
    data = _crash_data(crash_px=98.0)
    result = simulate_daily_policy(
        data,
        _HoldThenSell(hold_for_steps=1),
        max_steps=6,
        initial_position=InitialPositionSpec(symbol="AAA", side="long", allocation_pct=0.25),
        enable_drawdown_profit_early_exit=False,
    )
    assert result.num_trades >= 1
    # 25% of portfolio at -200bps ≈ -0.5% total return (plus round-trip fees).
    assert result.total_return < -0.003


def test_intraday_refuses_sell_below_tolerance() -> None:
    """Fresh buy (hold_steps < stale_after_bars), 200bps drop, tol=50bps → refused."""
    data = _crash_data(crash_px=98.0)

    # Hold at bar 0 (no crash yet), then try to sell at bar 1 when the price
    # is 200bps below entry. hold_steps at bar 1 = 1 < 8 → intraday tol=50bps
    # → refused. Further sell attempts also refused (age stays < 8).
    result = simulate_daily_policy(
        data,
        _HoldThenSell(hold_for_steps=1),
        max_steps=6,  # never crosses the 8-bar stale threshold
        initial_position=InitialPositionSpec(symbol="AAA", side="long", allocation_pct=0.25),
        enable_drawdown_profit_early_exit=False,
        death_spiral_tolerance_bps=50.0,
        death_spiral_overnight_tolerance_bps=500.0,
        death_spiral_stale_after_bars=8,
    )
    # Guard refused every close, including end-of-episode → 0 realized trades.
    assert result.num_trades == 0, (
        f"expected guard to refuse all closes, got {result.num_trades} trades"
    )


def test_overnight_allows_200bps_drop() -> None:
    """After ``stale_after_bars`` bars, overnight tol=500bps permits 200bps drop."""
    data = _crash_data(num_bars=15, crash_px=98.0)

    # Hold through the crash and well past the 8-bar staleness threshold, then
    # attempt to sell. By the sell attempt hold_steps >= 10 → overnight regime
    # (500bps tolerance allows a 200bps drop).
    result = simulate_daily_policy(
        data,
        _HoldThenSell(hold_for_steps=10),
        max_steps=14,
        initial_position=InitialPositionSpec(symbol="AAA", side="long", allocation_pct=0.25),
        enable_drawdown_profit_early_exit=False,
        death_spiral_tolerance_bps=50.0,
        death_spiral_overnight_tolerance_bps=500.0,
        death_spiral_stale_after_bars=8,
    )
    assert result.num_trades >= 1
    # Realized loss ≈ -0.5% on the 25% allocation plus fees.
    assert result.total_return < -0.003


def test_stale_after_zero_forces_overnight_always() -> None:
    """``stale_after_bars=0`` collapses to overnight-only regime (widest tol)."""
    data = _crash_data(crash_px=98.0)

    result = simulate_daily_policy(
        data,
        _HoldThenSell(hold_for_steps=1),
        max_steps=6,
        initial_position=InitialPositionSpec(symbol="AAA", side="long", allocation_pct=0.25),
        enable_drawdown_profit_early_exit=False,
        death_spiral_tolerance_bps=50.0,   # would refuse if intraday
        death_spiral_overnight_tolerance_bps=500.0,  # allows 200bps
        death_spiral_stale_after_bars=0,   # always overnight
    )
    # Overnight-always mode permits the 200bps close.
    assert result.num_trades >= 1


# ---------------------------------------------------------------------------
# Direct helper tests (no sim harness)
# ---------------------------------------------------------------------------


def test_helper_shorts_never_refused() -> None:
    """Guard is long-only — short-side helper should not be called; direct test
    here pins the numeric behavior at the helper layer."""
    # Shorts are filtered in the call sites (``not pos.is_short`` guard), so
    # _death_spiral_refused itself doesn't know about side. This test just
    # pins that the helper returns False when the guard is disabled.
    assert _death_spiral_refused(
        sell_price=50.0, entry_price=100.0, hold_steps=0,
        tolerance_bps=None, overnight_tolerance_bps=500.0, stale_after_bars=8,
    ) is False


def test_helper_rejects_invalid_inputs() -> None:
    # Zero/negative prices: guard can't reason about them → allow (False).
    assert _death_spiral_refused(
        sell_price=0.0, entry_price=100.0, hold_steps=0,
        tolerance_bps=50.0, overnight_tolerance_bps=500.0, stale_after_bars=8,
    ) is False
    assert _death_spiral_refused(
        sell_price=99.0, entry_price=0.0, hold_steps=0,
        tolerance_bps=50.0, overnight_tolerance_bps=500.0, stale_after_bars=8,
    ) is False
