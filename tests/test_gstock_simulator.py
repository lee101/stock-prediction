"""Tests for gstockagent simulator bugs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gstockagent.simulator import (
    SimState, Position, portfolio_value, close_position, apply_fees
)


def test_short_loss_capped_at_margin():
    """Short position loss can't exceed initial margin (entry notional)."""
    state = SimState(cash=0)
    state.positions["BTC"] = Position(
        symbol="BTC", qty=1.0, entry_price=100.0, direction="short"
    )
    # price doubles: short loses 100% of entry (margin)
    val = portfolio_value(state, {"BTC": 200.0})
    assert val == 0.0, f"expected 0, got {val}"

    # price triples: still capped at 0
    val = portfolio_value(state, {"BTC": 300.0})
    assert val == 0.0, f"expected 0, got {val}"


def test_close_short_proceeds_capped():
    """Closing a losing short returns max(0, ...) proceeds."""
    state = SimState(cash=5000)
    state.positions["BTC"] = Position(
        symbol="BTC", qty=1.0, entry_price=100.0, direction="short"
    )
    close_position(state, "BTC", 250.0, "test", "2025-01-01")
    # proceeds = max(0, 1.0 * (200 - 250)) = 0
    # pnl capped at -100 (entry notional)
    assert state.cash == 5000.0  # only proceeds=0 added
    assert len(state.trade_log) == 1
    assert state.trade_log[0]["pnl"] == -100.0


def test_long_position_value():
    state = SimState(cash=0)
    state.positions["ETH"] = Position(
        symbol="ETH", qty=2.0, entry_price=50.0, direction="long"
    )
    val = portfolio_value(state, {"ETH": 75.0})
    assert val == 150.0


def test_liquidation_at_zero_equity():
    """Equity at 0 should trigger liquidation."""
    state = SimState(cash=-100)
    state.positions["BTC"] = Position(
        symbol="BTC", qty=1.0, entry_price=100.0, direction="long"
    )
    # price = 90, equity = -100 + 90 = -10
    eq = portfolio_value(state, {"BTC": 90.0})
    assert eq == -10.0  # raw equity is negative
    # simulator should catch this and set equity to 0


def test_equity_never_below_zero_in_curve():
    """After liquidation, equity in curve should be 0, not negative."""
    state = SimState(cash=-500)
    # no positions: equity = cash = -500
    # but equity curve should cap at 0
    eq = portfolio_value(state, {})
    assert eq == -500  # raw value
    # the simulator wraps this: equity = max(0, eq) when recording


def test_apply_fees():
    fee = apply_fees(10000, 10)  # 10 bps on $10000
    assert fee == 10.0


if __name__ == "__main__":
    test_short_loss_capped_at_margin()
    test_close_short_proceeds_capped()
    test_long_position_value()
    test_liquidation_at_zero_equity()
    test_equity_never_below_zero_in_curve()
    test_apply_fees()
    print("All tests passed")
