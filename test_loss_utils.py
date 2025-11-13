import torch
from loss_utils import (
    CRYPTO_TRADING_FEE,
    TRADING_FEE,
    calculate_profit_torch_with_entry_buysell_profit_values,
    calculate_trading_profit_torch,
    calculate_trading_profit_torch_with_entry_buysell,
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


def test_multi_trade_portfolio_profit_mix():
    """Two trades (long win, short win) should sum expected PnL."""
    y_test_pred = torch.tensor([1.0, -1.0])
    y_test = torch.tensor([0.02, -0.03])

    profit = calculate_trading_profit_torch(None, None, y_test, y_test_pred)

    expected_long = 0.02 - TRADING_FEE
    expected_short = 0.03 - TRADING_FEE
    averaged = (expected_long + expected_short) / 2.0
    assert torch.isclose(profit, torch.tensor(averaged), atol=1e-7)


def test_multi_trade_portfolio_loss_mix():
    """Two trades (long loss, short loss) should sum expected negative PnL."""
    y_test_pred = torch.tensor([1.0, -1.0])
    y_test = torch.tensor([-0.01, 0.015])

    profit = calculate_trading_profit_torch(None, None, y_test, y_test_pred)

    expected_long = -0.01 - TRADING_FEE
    expected_short = -0.015 - TRADING_FEE
    averaged = (expected_long + expected_short) / 2.0
    assert torch.isclose(profit, torch.tensor(averaged), atol=1e-7)
    assert profit < 0


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

    expected = (y_test_high_pred - y_test_low_pred).item() - TRADING_FEE
    assert torch.isclose(profit_vals, torch.tensor(expected), atol=1e-7)


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

    expected_loss = (-1.0) * (y_test.item() - y_test_high_pred.item()) - TRADING_FEE
    assert torch.isclose(profit_vals, torch.tensor(expected_loss), atol=1e-7)
    assert profit_vals < 0


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
        y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred, close_at_eod=False
    )

    expected_profit = (y_test_low_pred - y_test_high_pred).item() * -1.0 - TRADING_FEE
    assert torch.isclose(profit_vals, torch.tensor(expected_profit), atol=1e-7)
    assert profit_vals > 0


def test_entry_exit_two_trade_matrix_close_eod():
    """One long + one short with EOD exits should sum predictable PnL."""
    y_test_pred = torch.tensor([1.0, -1.0])
    y_test = torch.tensor([0.01, -0.02])
    y_test_low_pred = torch.tensor([-0.01, -0.02])
    y_test_low = torch.tensor([-0.015, -0.025])
    y_test_high_pred = torch.tensor([0.02, 0.03])
    y_test_high = torch.tensor([0.025, 0.035])

    profit_vals = calculate_profit_torch_with_entry_buysell_profit_values(
        y_test,
        y_test_high,
        y_test_high_pred,
        y_test_low,
        y_test_low_pred,
        y_test_pred,
        close_at_eod=True,
    )

    expected_long = (0.01 - (-0.01)) - TRADING_FEE
    expected_short = (0.03 - (-0.02)) - TRADING_FEE
    assert torch.isclose(profit_vals.sum(), torch.tensor(expected_long + expected_short), atol=1e-7)


def test_detailed_short_breakdown():
    """
    Detailed breakdown showing EXACTLY what's wrong with the short calculation
    """
    y_test_pred = torch.tensor([-1.0])
    y_test = torch.tensor([0.04])
    y_test_high_pred = torch.tensor([0.02])
    y_test_low_pred = torch.tensor([-0.01])
    y_test_high = torch.tensor([0.03])
    y_test_low = torch.tensor([-0.02])

    position = y_test_pred.item()
    entry_price = y_test_high_pred.item()
    exit_price = y_test.item()
    expected = position * (exit_price - entry_price) - abs(position) * TRADING_FEE

    actual = calculate_profit_torch_with_entry_buysell_profit_values(
        y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred, close_at_eod=True
    )

    assert torch.isclose(actual, torch.tensor(expected), atol=1e-7)
    assert actual < 0


def test_no_fee_when_entry_not_triggered_for_predicted_direction():
    """Fees should not be charged if the predicted direction never enters."""
    y_test_pred = torch.tensor([-1.0])  # short intent
    y_test = torch.tensor([0.0])  # flat close
    y_test_high_pred = torch.tensor([0.02])  # short entry target
    y_test_high = torch.tensor([0.015])  # never hit short entry (actual high < target)
    y_test_low_pred = torch.tensor([-0.01])
    y_test_low = torch.tensor([-0.02])  # long side would have entered, but we stayed short

    profit = calculate_profit_torch_with_entry_buysell_profit_values(
        y_test,
        y_test_high,
        y_test_high_pred,
        y_test_low,
        y_test_low_pred,
        y_test_pred,
        close_at_eod=True,
    )

    assert torch.isclose(profit, torch.tensor(0.0), atol=1e-7)


def test_intraday_long_fee_applied_once_entry_hits():
    """Intraday logic should charge a single fee when a long entry executes."""
    y_test_pred = torch.tensor([1.0])  # go long
    y_test = torch.tensor([0.01])  # +1% close
    y_test_low_pred = torch.tensor([-0.01])  # enter at -1%
    y_test_low = torch.tensor([-0.02])  # entry triggered
    y_test_high_pred = torch.tensor([0.03])  # exit target +3%
    y_test_high = torch.tensor([0.04])  # target hit intraday

    profit = calculate_profit_torch_with_entry_buysell_profit_values(
        y_test,
        y_test_high,
        y_test_high_pred,
        y_test_low,
        y_test_low_pred,
        y_test_pred,
        close_at_eod=False,
    )

    movement = (y_test_high_pred - y_test_low_pred).item()  # 0.04 between entry and exit
    expected = movement - TRADING_FEE
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-7)


def test_short_fee_applied_only_when_short_entry_hits():
    """Fees for shorts apply only when the short entry condition is satisfied."""
    y_test_pred = torch.tensor([-1.0])  # short
    y_test = torch.tensor([0.02])  # close equals entry price
    y_test_high_pred = torch.tensor([0.02])  # entry at +2%
    y_test_high = torch.tensor([0.03])  # actual high crosses entry
    y_test_low_pred = torch.tensor([-0.01])
    y_test_low = torch.tensor([-0.02])

    profit = calculate_profit_torch_with_entry_buysell_profit_values(
        y_test,
        y_test_high,
        y_test_high_pred,
        y_test_low,
        y_test_low_pred,
        y_test_pred,
        close_at_eod=True,
    )

    expected = -TRADING_FEE  # no price move, only fee
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-7)


def test_intraday_long_loss_when_close_below_entry():
    y_test_pred = torch.tensor([1.0])
    y_test = torch.tensor([-0.015])
    y_test_low_pred = torch.tensor([-0.01])
    y_test_low = torch.tensor([-0.02])
    y_test_high_pred = torch.tensor([0.03])
    y_test_high = torch.tensor([0.02])

    profit = calculate_profit_torch_with_entry_buysell_profit_values(
        y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred, close_at_eod=False
    )

    expected = (y_test.item() - y_test_low_pred.item()) - TRADING_FEE
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-7)
    assert profit < 0


def test_intraday_short_profit_hits_low_target():
    y_test_pred = torch.tensor([-1.0])
    y_test = torch.tensor([-0.005])
    y_test_high_pred = torch.tensor([0.02])
    y_test_high = torch.tensor([0.03])
    y_test_low_pred = torch.tensor([-0.02])
    y_test_low = torch.tensor([-0.03])

    profit = calculate_profit_torch_with_entry_buysell_profit_values(
        y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred, close_at_eod=False
    )

    expected = (y_test_low_pred - y_test_high_pred).item() * -1.0 - TRADING_FEE
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-7)
    assert profit > 0


def test_eod_long_loss_when_close_below_entry():
    y_test_pred = torch.tensor([1.0])
    y_test = torch.tensor([-0.02])
    y_test_low_pred = torch.tensor([-0.01])
    y_test_low = torch.tensor([-0.03])
    y_test_high_pred = torch.tensor([0.02])
    y_test_high = torch.tensor([0.025])

    profit = calculate_profit_torch_with_entry_buysell_profit_values(
        y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred, close_at_eod=True
    )

    expected = (y_test.item() - y_test_low_pred.item()) - TRADING_FEE
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-7)
    assert profit < 0


def test_eod_short_profit_when_close_below_entry():
    y_test_pred = torch.tensor([-1.0])
    y_test = torch.tensor([-0.02])
    y_test_high_pred = torch.tensor([0.02])
    y_test_high = torch.tensor([0.03])
    y_test_low_pred = torch.tensor([-0.01])
    y_test_low = torch.tensor([-0.015])

    profit = calculate_profit_torch_with_entry_buysell_profit_values(
        y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred, close_at_eod=True
    )

    expected = (-1.0) * (y_test.item() - y_test_high_pred.item()) - TRADING_FEE
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-7)
    assert profit > 0


def test_intraday_short_loss_when_exit_not_hit():
    y_test_pred = torch.tensor([-1.0])
    y_test = torch.tensor([0.03])
    y_test_high_pred = torch.tensor([0.02])
    y_test_high = torch.tensor([0.04])
    y_test_low_pred = torch.tensor([-0.02])
    y_test_low = torch.tensor([-0.005])

    profit = calculate_profit_torch_with_entry_buysell_profit_values(
        y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred, close_at_eod=False
    )

    expected = (-1.0) * (y_test.item() - y_test_high_pred.item()) - TRADING_FEE
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-7)
    assert profit < 0


def test_entry_exit_totals_match_profit_values_sum():
    y_test = torch.tensor([0.02, -0.01])
    y_test_high = torch.tensor([0.035, 0.025])
    y_test_high_pred = torch.tensor([0.03, 0.02])
    y_test_low = torch.tensor([-0.02, -0.03])
    y_test_low_pred = torch.tensor([-0.01, -0.02])
    y_test_pred = torch.tensor([1.0, -1.0])

    profit_values = calculate_profit_torch_with_entry_buysell_profit_values(
        y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred, close_at_eod=False
    )

    total = calculate_trading_profit_torch_with_entry_buysell(
        None,
        None,
        y_test,
        y_test_pred,
        y_test_high,
        y_test_high_pred,
        y_test_low,
        y_test_low_pred,
        close_at_eod=False,
    )

    assert torch.isclose(total, profit_values.sum(), atol=1e-7)


def test_crypto_vs_equity_fees():
    """Crypto fees (0.15%) should result in lower profit than equity fees (0.05%)"""

    y_test_pred = torch.tensor([1.0])
    y_test = torch.tensor([0.02])  # 2% profit

    equity_profit = calculate_trading_profit_torch(None, None, y_test, y_test_pred, trading_fee=TRADING_FEE)
    crypto_profit = calculate_trading_profit_torch(None, None, y_test, y_test_pred, trading_fee=CRYPTO_TRADING_FEE)

    expected_equity = 1.0 * 0.02 - TRADING_FEE
    expected_crypto = 1.0 * 0.02 - CRYPTO_TRADING_FEE

    assert torch.isclose(equity_profit, torch.tensor(expected_equity), atol=1e-6)
    assert torch.isclose(crypto_profit, torch.tensor(expected_crypto), atol=1e-6)
    assert crypto_profit < equity_profit, "Crypto fees should reduce profit more than equity fees"
    print(f"✓ Equity profit: {equity_profit.item():.6f}, Crypto profit: {crypto_profit.item():.6f}")


def test_crypto_fee_constant_matches_expected_profit():
    """Single crypto trade should deduct CRYPTO_TRADING_FEE exactly once."""
    y_test_pred = torch.tensor([1.0])
    y_test = torch.tensor([0.05])

    profit = calculate_trading_profit_torch(None, None, y_test, y_test_pred, trading_fee=CRYPTO_TRADING_FEE)

    expected = 0.05 - CRYPTO_TRADING_FEE
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-7)


def test_entry_exit_custom_fee():
    """Entry/exit logic with custom fee (0.2%)"""
    CUSTOM_FEE = 0.002

    y_test_pred = torch.tensor([1.0])
    y_test = torch.tensor([0.02])
    y_test_low_pred = torch.tensor([-0.01])
    y_test_high_pred = torch.tensor([0.03])
    y_test_low = torch.tensor([-0.015])
    y_test_high = torch.tensor([0.04])

    profit = calculate_profit_torch_with_entry_buysell_profit_values(
        y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred, trading_fee=CUSTOM_FEE
    )

    movement = 0.03 - (-0.01)  # 4% from entry to exit
    expected = movement - CUSTOM_FEE

    assert torch.isclose(profit, torch.tensor(expected), atol=1e-6)
    print(f"✓ Custom fee profit: {profit.item():.6f} (expected: {expected:.6f})")


def test_zero_fee_scenario():
    """Zero fees for testing/simulation"""
    y_test_pred = torch.tensor([1.0])
    y_test = torch.tensor([0.02])

    profit = calculate_trading_profit_torch(None, None, y_test, y_test_pred, trading_fee=0.0)

    expected = 1.0 * 0.02  # no fee deduction
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-6)
    print(f"✓ Zero fee profit: {profit.item():.6f}")


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

    print("\n" + "=" * 60)
    print("Testing fee scenarios")
    print("=" * 60)

    test_crypto_vs_equity_fees()
    test_entry_exit_custom_fee()
    test_zero_fee_scenario()
    print("\n✓ All fee tests passed")
