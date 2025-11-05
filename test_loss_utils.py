import torch
from loss_utils import (
    TRADING_FEE,
    calculate_profit_torch_with_entry_buysell_profit_values,
    calculate_trading_profit_torch,
)


def test_basic_long_profit():
    """Simple long: buy 1x, price goes up 2%, should profit ~0.02"""
    y_test_pred = torch.tensor([1.0])  # position size: +1 (long)
    y_test = torch.tensor([0.02])  # return: +2%

    profit = calculate_trading_profit_torch(None, None, y_test, y_test_pred)

    expected = 1.0 * 0.02 - (1.0 * TRADING_FEE)  # position * return - fee
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-6), f"Expected {expected}, got {profit.item()}"


def test_basic_long_loss():
    """Simple long: buy 1x, price goes DOWN 2%, should LOSE ~0.02"""
    y_test_pred = torch.tensor([1.0])  # position size: +1 (long)
    y_test = torch.tensor([-0.02])  # return: -2%

    profit = calculate_trading_profit_torch(None, None, y_test, y_test_pred)

    expected = 1.0 * (-0.02) - (1.0 * TRADING_FEE)  # should be negative
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-6), f"Expected {expected}, got {profit.item()}"
    assert profit < 0, "Should lose money when long and price drops"


def test_basic_short_profit():
    """Simple short: sell 1x, price goes DOWN 2%, should profit ~0.02"""
    y_test_pred = torch.tensor([-1.0])  # position size: -1 (short)
    y_test = torch.tensor([-0.02])  # return: -2%

    profit = calculate_trading_profit_torch(None, None, y_test, y_test_pred)

    # Short profit: negative position * negative return = positive
    expected = (-1.0) * (-0.02) - (1.0 * TRADING_FEE)
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-6), f"Expected {expected}, got {profit.item()}"
    assert profit > 0, "Should profit when short and price drops"


def test_basic_short_loss():
    """Simple short: sell 1x, price goes UP 2%, should LOSE ~0.02"""
    y_test_pred = torch.tensor([-1.0])  # position size: -1 (short)
    y_test = torch.tensor([0.02])  # return: +2%

    profit = calculate_trading_profit_torch(None, None, y_test, y_test_pred)

    # Short loss: negative position * positive return = negative
    expected = (-1.0) * 0.02 - (1.0 * TRADING_FEE)
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-6), f"Expected {expected}, got {profit.item()}"
    assert profit < 0, "Should lose money when short and price rises"


def test_entry_exit_long_profit():
    """
    Entry/exit logic:
    - Predict entry at low=-0.01 (-1%), exit at high=+0.03 (+3%)
    - Actual: low=-0.015, high=+0.04, close=+0.02
    - Should enter (our low > actual low), should exit at high (+3% gain from -1% entry = 4% total)
    """
    y_test_pred = torch.tensor([1.0])  # position size
    y_test = torch.tensor([0.02])  # close return: +2%
    y_test_low_pred = torch.tensor([-0.01])  # entry target: -1%
    y_test_high_pred = torch.tensor([0.03])  # exit target: +3%
    y_test_low = torch.tensor([-0.015])  # actual low: -1.5%
    y_test_high = torch.tensor([0.04])  # actual high: +4%

    profit_vals = calculate_profit_torch_with_entry_buysell_profit_values(
        y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred
    )

    # Enter at -1%, exit at +3% = 4% gain
    # But current implementation uses abs(), so let's see what we actually get
    print(f"Entry/exit long profit: {profit_vals.item()}")

    # Expected: 0.03 - (-0.01) = 0.04 (4% gain from entry to exit)
    # With fee: 0.04 - TRADING_FEE
    expected_movement = 0.03 - (-0.01)
    print(f"Expected movement: {expected_movement}")


def test_entry_exit_short_loss_case():
    """
    Short that should LOSE:
    - Predict short entry at high=+0.02 (+2%), exit at low=-0.01 (-1%)
    - Actual: high=+0.03, low=-0.02, close=+0.04
    - Should enter short at +2%, but price keeps going UP to +4% at close
    - Should LOSE money because we're short and price went up
    """
    y_test_pred = torch.tensor([-1.0])  # short position
    y_test = torch.tensor([0.04])  # close return: +4%
    y_test_high_pred = torch.tensor([0.02])  # short entry target: +2%
    y_test_low_pred = torch.tensor([-0.01])  # exit target: -1%
    y_test_high = torch.tensor([0.03])  # actual high: +3%
    y_test_low = torch.tensor([-0.02])  # actual low: -2%

    profit_vals = calculate_profit_torch_with_entry_buysell_profit_values(
        y_test,
        y_test_high,
        y_test_high_pred,
        y_test_low,
        y_test_low_pred,
        y_test_pred,
        close_at_eod=True,  # force close at EOD to see the issue
    )

    print(f"Short loss case profit: {profit_vals.item()}")

    # We enter short at +2%, close at +4%
    # We lose: -(+4% - (+2%)) = -2%
    # With current abs() bug, this might show as positive!
    expected_loss = -(0.04 - 0.02)  # should be -0.02
    print(f"Expected loss: {expected_loss}")

    # The bug: if abs() is used, profit_vals might be positive when it should be negative
    # This test will reveal the sign issue


def test_entry_exit_short_profit_case():
    """
    Short that should PROFIT:
    - Predict short entry at high=+0.02, exit at low=-0.02
    - Actual: high=+0.03, low=-0.03, close=-0.01
    - Enter short at +2%, exit at -2% = 4% profit
    """
    y_test_pred = torch.tensor([-1.0])  # short position
    y_test = torch.tensor([-0.01])  # close: -1%
    y_test_high_pred = torch.tensor([0.02])  # entry: +2%
    y_test_low_pred = torch.tensor([-0.02])  # exit: -2%
    y_test_high = torch.tensor([0.03])  # actual high: +3%
    y_test_low = torch.tensor([-0.03])  # actual low: -3%

    profit_vals = calculate_profit_torch_with_entry_buysell_profit_values(
        y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred, close_at_eod=True
    )

    print(f"Short profit case: {profit_vals.item()}")

    # Enter at +2%, exit at -2%: profit = +2% - (-2%) = 4%
    expected_profit = 0.02 - (-0.02)  # should be 0.04
    print(f"Expected profit: {expected_profit}")


def test_detailed_short_breakdown():
    """
    Detailed breakdown showing EXACTLY what's wrong with the short calculation
    """
    print("\n" + "=" * 60)
    print("DETAILED SHORT CALCULATION BREAKDOWN")
    print("=" * 60)

    # Scenario: Short at +2%, price rises to +4% (we lose money)
    y_test_pred = torch.tensor([-1.0])  # short 1x
    y_test = torch.tensor([0.04])  # close at +4%
    y_test_high_pred = torch.tensor([0.02])  # we enter short at +2%
    y_test_low_pred = torch.tensor([-0.01])  # exit target (not hit)
    y_test_high = torch.tensor([0.03])  # actual high +3% (we enter)
    y_test_low = torch.tensor([-0.02])  # actual low -2%

    print("\nScenario: Short entry at +2%, close at +4%")
    print(f"Position size: {y_test_pred.item()}")
    print(f"Entry (high_pred): {y_test_high_pred.item():.4f} (+2%)")
    print(f"Close: {y_test.item():.4f} (+4%)")
    print(f"Entry condition met: {y_test_high_pred.item()} < {y_test_high.item()}")

    # What SHOULD happen:
    print("\n--- CORRECT CALCULATION ---")
    entry_price = y_test_high_pred.item()
    exit_price = y_test.item()
    position = y_test_pred.item()
    print(f"Entry price (normalized): {entry_price}")
    print(f"Exit price (normalized): {exit_price}")
    print(f"Movement: {exit_price} - {entry_price} = {exit_price - entry_price}")
    print(
        f"P&L: position * movement = {position} * {exit_price - entry_price} = {position * (exit_price - entry_price)}"
    )
    print(f"With fee: {position * (exit_price - entry_price) - abs(position) * TRADING_FEE}")
    expected = position * (exit_price - entry_price) - abs(position) * TRADING_FEE

    # What ACTUALLY happens:
    print("\n--- BUGGY IMPLEMENTATION (with abs) ---")
    pred_high_to_close = torch.abs(y_test_high_pred - y_test)
    print(
        f"pred_high_to_close_percent_movements = abs({y_test_high_pred.item()} - {y_test.item()}) = {pred_high_to_close.item()}"
    )
    buggy_sold = -1 * torch.clip(y_test_pred, -10, 0) * pred_high_to_close * (y_test_high_pred < y_test_high)
    print(
        f"sold_profits = -1 * ({torch.clip(y_test_pred, -10, 0).item()}) * {pred_high_to_close.item()} * {(y_test_high_pred < y_test_high).item()}"
    )
    print(f"sold_profits = {buggy_sold.item()}")
    print(f"With fee: {buggy_sold.item() - abs(position) * TRADING_FEE}")

    actual = calculate_profit_torch_with_entry_buysell_profit_values(
        y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred, close_at_eod=True
    )

    print("\n--- RESULTS ---")
    print(f"Expected P&L: {expected:.6f} (NEGATIVE = loss, which is correct)")
    print(f"Actual P&L:   {actual.item():.6f} (POSITIVE = profit, which is WRONG)")
    print(f"Sign error:   {'YES - BUG CONFIRMED' if (expected < 0 and actual.item() > 0) else 'No'}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing basic long/short logic")
    print("=" * 60)

    test_basic_long_profit()
    print("✓ Basic long profit works")

    test_basic_long_loss()
    print("✓ Basic long loss works")

    test_basic_short_profit()
    print("✓ Basic short profit works")

    test_basic_short_loss()
    print("✓ Basic short loss works")

    print("\n" + "=" * 60)
    print("Testing entry/exit logic (THIS IS WHERE BUGS APPEAR)")
    print("=" * 60)

    test_entry_exit_long_profit()
    print("\n")

    test_entry_exit_short_loss_case()
    print("\n")

    test_entry_exit_short_profit_case()
    print("\n")

    test_detailed_short_breakdown()
