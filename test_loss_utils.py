import torch
import pytest
from loss_utils import (
    CRYPTO_TRADING_FEE,
    DAILY_LEVERAGE_COST,
    DEFAULT_ANNUAL_LEVERAGE_COST,
    DEFAULT_TRADING_DAYS,
    TRADING_FEE,
    calculate_profit_torch_with_entry_buysell_profit_values,
    calculate_trading_profit_torch,
    calculate_trading_profit_torch_with_entry_buysell,
    get_trading_profits_list,
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


@pytest.mark.parametrize(
    "positions,returns,trading_fee",
    [
        pytest.param([1.0], [0.012], TRADING_FEE, id="single_long_gain"),
        pytest.param([1.0, -1.0], [0.02, 0.015], TRADING_FEE, id="two_day_long_gain_short_loss"),
        pytest.param([0.6, -0.4, 1.2], [0.015, 0.02, -0.01], TRADING_FEE, id="three_day_mixed_fracs"),
        pytest.param([0.8, -0.5, 0.3], [0.01, -0.025, 0.04], CRYPTO_TRADING_FEE, id="crypto_fee_sequence"),
    ],
)
def test_trading_profit_multi_day_sequences(positions, returns, trading_fee):
    """Validate averaged PnL across 1/2/3-day mixes of longs/shorts and fees, including leverage costs."""
    y_test_pred = torch.tensor(positions, dtype=torch.float32)
    y_test = torch.tensor(returns, dtype=torch.float32)

    profit = calculate_trading_profit_torch(
        None,
        None,
        y_test,
        y_test_pred,
        trading_fee=trading_fee,
    )

    # Expected calculation now includes leverage cost for positions > 1.0
    def calc_trade_profit(pos, ret, fee):
        pnl = pos * ret - abs(pos) * fee
        leverage_amount = max(abs(pos) - 1.0, 0.0)
        pnl -= leverage_amount * DAILY_LEVERAGE_COST
        return pnl

    expected = sum(calc_trade_profit(pos, ret, trading_fee) for pos, ret in zip(positions, returns)) / len(positions)
    assert torch.isclose(profit, torch.tensor(expected, dtype=profit.dtype), atol=1e-7)


def test_get_trading_profits_list_matches_per_trade_breakdown():
    """Per-trade breakdown should match manual PnL (including fees) across days."""
    y_test_pred = torch.tensor([1.0, -0.5, 0.8], dtype=torch.float32)
    y_test = torch.tensor([0.02, 0.015, -0.01], dtype=torch.float32)

    profits = get_trading_profits_list(None, None, y_test, y_test_pred)

    expected = torch.tensor(
        [pos * ret - abs(pos) * TRADING_FEE for pos, ret in zip(y_test_pred.tolist(), y_test.tolist())],
        dtype=profits.dtype,
    )
    assert torch.allclose(profits, expected, atol=1e-7)


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


def test_entry_exit_multi_day_sequence_hits_and_misses():
    """Three-sample mix: hit long TP, hit short TP, skip fee when entry never triggers."""
    y_test_pred = torch.tensor([1.0, -1.0, 1.0], dtype=torch.float32)
    y_test = torch.tensor([0.02, -0.015, 0.005], dtype=torch.float32)
    y_test_high_pred = torch.tensor([0.03, 0.02, 0.01], dtype=torch.float32)
    y_test_high = torch.tensor([0.035, 0.025, 0.02], dtype=torch.float32)
    y_test_low_pred = torch.tensor([-0.01, -0.015, -0.05], dtype=torch.float32)
    y_test_low = torch.tensor([-0.02, -0.02, -0.04], dtype=torch.float32)

    profits = calculate_profit_torch_with_entry_buysell_profit_values(
        y_test,
        y_test_high,
        y_test_high_pred,
        y_test_low,
        y_test_low_pred,
        y_test_pred,
    )

    expected = torch.tensor(
        [
            (0.03 - (-0.01)) - TRADING_FEE,
            ((-0.015) - 0.02) * -1 - TRADING_FEE,
            0.0,
        ],
        dtype=profits.dtype,
    )

    assert torch.allclose(profits, expected, atol=1e-7)


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


# ==============================================================================
# LEVERAGE COST TESTS (6.5% annual rate on borrowed capital)
# ==============================================================================


def test_leverage_constants():
    """Verify leverage cost constants are set correctly."""
    assert DEFAULT_ANNUAL_LEVERAGE_COST == 0.065, "Annual rate should be 6.5%"
    assert DEFAULT_TRADING_DAYS == 252, "Should use 252 trading days"
    expected_daily = 0.065 / 252
    assert abs(DAILY_LEVERAGE_COST - expected_daily) < 1e-10, f"Daily cost should be {expected_daily}"


def test_no_leverage_cost_at_1x():
    """Position size of 1.0 should incur NO leverage cost (no borrowed capital)."""
    y_test_pred = torch.tensor([1.0])
    y_test = torch.tensor([0.02])

    profit = calculate_trading_profit_torch(None, None, y_test, y_test_pred)

    # Only trading fee, no leverage cost
    expected = 1.0 * 0.02 - TRADING_FEE
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-7), f"Expected {expected}, got {profit.item()}"


def test_no_leverage_cost_under_1x():
    """Position size under 1.0 should incur NO leverage cost."""
    y_test_pred = torch.tensor([0.5])
    y_test = torch.tensor([0.02])

    profit = calculate_trading_profit_torch(None, None, y_test, y_test_pred)

    # Only trading fee, no leverage cost
    expected = 0.5 * 0.02 - (0.5 * TRADING_FEE)
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-7)


def test_leverage_cost_at_2x_long():
    """2x leverage should charge daily financing on 1.0 borrowed capital."""
    y_test_pred = torch.tensor([2.0])
    y_test = torch.tensor([0.02])

    profit = calculate_trading_profit_torch(None, None, y_test, y_test_pred)

    # P&L: 2x * 2% = 4%, minus fee on 2x, minus leverage cost on 1.0 borrowed
    leverage_amount = 2.0 - 1.0  # 1.0 borrowed
    leverage_penalty = leverage_amount * DAILY_LEVERAGE_COST * 1.0  # 1 day
    expected = 2.0 * 0.02 - (2.0 * TRADING_FEE) - leverage_penalty
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-7), f"Expected {expected}, got {profit.item()}"


def test_leverage_cost_at_2x_short():
    """2x short should also charge leverage on borrowed capital."""
    y_test_pred = torch.tensor([-2.0])
    y_test = torch.tensor([-0.02])  # Price drops 2%

    profit = calculate_trading_profit_torch(None, None, y_test, y_test_pred)

    # Short profit: -2x * -2% = 4%
    leverage_amount = abs(-2.0) - 1.0  # 1.0 borrowed
    leverage_penalty = leverage_amount * DAILY_LEVERAGE_COST * 1.0
    expected = (-2.0) * (-0.02) - (2.0 * TRADING_FEE) - leverage_penalty
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-7)
    assert profit > 0, "Short should profit when price drops"


def test_leverage_cost_at_5x():
    """5x leverage should charge on 4.0 borrowed capital."""
    y_test_pred = torch.tensor([5.0])
    y_test = torch.tensor([0.01])

    profit = calculate_trading_profit_torch(None, None, y_test, y_test_pred)

    leverage_amount = 5.0 - 1.0  # 4.0 borrowed
    leverage_penalty = leverage_amount * DAILY_LEVERAGE_COST
    expected = 5.0 * 0.01 - (5.0 * TRADING_FEE) - leverage_penalty
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-7)


def test_leverage_cost_at_10x():
    """10x leverage (max) should charge on 9.0 borrowed capital."""
    y_test_pred = torch.tensor([10.0])
    y_test = torch.tensor([0.01])

    profit = calculate_trading_profit_torch(None, None, y_test, y_test_pred)

    leverage_amount = 10.0 - 1.0  # 9.0 borrowed
    leverage_penalty = leverage_amount * DAILY_LEVERAGE_COST
    expected = 10.0 * 0.01 - (10.0 * TRADING_FEE) - leverage_penalty
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-7)


def test_leverage_cost_fractional_above_1x():
    """1.5x leverage should charge on 0.5 borrowed capital."""
    y_test_pred = torch.tensor([1.5])
    y_test = torch.tensor([0.02])

    profit = calculate_trading_profit_torch(None, None, y_test, y_test_pred)

    leverage_amount = 1.5 - 1.0  # 0.5 borrowed
    leverage_penalty = leverage_amount * DAILY_LEVERAGE_COST
    expected = 1.5 * 0.02 - (1.5 * TRADING_FEE) - leverage_penalty
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-7)


def test_leverage_cost_custom_daily_rate():
    """Custom daily leverage cost parameter should override default."""
    y_test_pred = torch.tensor([2.0])
    y_test = torch.tensor([0.02])
    custom_daily_cost = 0.001  # 0.1% per day

    profit = calculate_trading_profit_torch(
        None, None, y_test, y_test_pred, leverage_cost_daily=custom_daily_cost
    )

    leverage_penalty = 1.0 * custom_daily_cost  # 1.0 borrowed * custom rate
    expected = 2.0 * 0.02 - (2.0 * TRADING_FEE) - leverage_penalty
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-7)


def test_leverage_cost_multi_day_position():
    """Multi-day position should multiply leverage cost by days held."""
    y_test_pred = torch.tensor([2.0])
    y_test = torch.tensor([0.05])  # 5% return over 5 days
    position_days = 5.0

    profit = calculate_trading_profit_torch(
        None, None, y_test, y_test_pred, position_duration_days=position_days
    )

    leverage_penalty = 1.0 * DAILY_LEVERAGE_COST * position_days  # 5 days of financing
    expected = 2.0 * 0.05 - (2.0 * TRADING_FEE) - leverage_penalty
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-7)


def test_leverage_cost_zero_when_explicitly_disabled():
    """Zero leverage cost should allow bypassing the feature."""
    y_test_pred = torch.tensor([5.0])
    y_test = torch.tensor([0.01])

    profit = calculate_trading_profit_torch(
        None, None, y_test, y_test_pred, leverage_cost_daily=0.0
    )

    # No leverage penalty
    expected = 5.0 * 0.01 - (5.0 * TRADING_FEE)
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-7)


def test_leverage_cost_multi_trade_mixed_leverage():
    """Multiple trades with different leverage levels."""
    y_test_pred = torch.tensor([1.0, 2.0, 0.5, 3.0])
    y_test = torch.tensor([0.01, 0.02, -0.01, 0.015])

    profit = calculate_trading_profit_torch(None, None, y_test, y_test_pred)

    # Calculate expected for each trade
    expected_profits = []
    for pos, ret in zip(y_test_pred.tolist(), y_test.tolist()):
        pnl = pos * ret - abs(pos) * TRADING_FEE
        leverage_amount = max(abs(pos) - 1.0, 0.0)
        pnl -= leverage_amount * DAILY_LEVERAGE_COST
        expected_profits.append(pnl)

    expected_avg = sum(expected_profits) / len(expected_profits)
    assert torch.isclose(profit, torch.tensor(expected_avg), atol=1e-7)


def test_get_trading_profits_list_includes_leverage_cost():
    """Per-trade profit list should include leverage costs."""
    y_test_pred = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    y_test = torch.tensor([0.02, 0.02, 0.02], dtype=torch.float32)

    profits = get_trading_profits_list(None, None, y_test, y_test_pred)

    expected = []
    for pos, ret in zip(y_test_pred.tolist(), y_test.tolist()):
        pnl = pos * ret - abs(pos) * TRADING_FEE
        leverage_amount = max(abs(pos) - 1.0, 0.0)
        pnl -= leverage_amount * DAILY_LEVERAGE_COST
        expected.append(pnl)

    expected_tensor = torch.tensor(expected, dtype=profits.dtype)
    assert torch.allclose(profits, expected_tensor, atol=1e-7)


def test_leverage_cost_makes_high_leverage_less_profitable():
    """Higher leverage should reduce profit due to financing costs, even with same return."""
    y_test = torch.tensor([0.01])  # 1% return

    profit_1x = calculate_trading_profit_torch(None, None, y_test, torch.tensor([1.0]))
    profit_2x = calculate_trading_profit_torch(None, None, y_test, torch.tensor([2.0]))
    profit_5x = calculate_trading_profit_torch(None, None, y_test, torch.tensor([5.0]))

    # Raw P&L scales with leverage, but leverage costs offset some gains
    # For 1% return: 1x=1%, 2x=2%, 5x=5% raw P&L
    # After leverage costs: 2x loses ~0.026%/day, 5x loses ~0.10%/day
    # So per-unit return decreases with higher leverage
    per_unit_1x = profit_1x.item() / 1.0
    per_unit_2x = profit_2x.item() / 2.0
    per_unit_5x = profit_5x.item() / 5.0

    # Higher leverage = lower per-unit efficiency due to financing costs
    assert per_unit_1x > per_unit_2x, "1x should be more efficient than 2x"
    assert per_unit_2x > per_unit_5x, "2x should be more efficient than 5x"


def test_leverage_cost_annual_rate_sanity_check():
    """
    Sanity check: holding 2x leverage for 252 days should cost ~6.5% of borrowed.
    """
    y_test_pred = torch.tensor([2.0])
    y_test = torch.tensor([0.0])  # No price movement
    position_days = 252.0  # Full year

    profit = calculate_trading_profit_torch(
        None, None, y_test, y_test_pred, position_duration_days=position_days
    )

    # With no price movement, we only pay fees and leverage cost
    leverage_penalty = 1.0 * DAILY_LEVERAGE_COST * 252  # Should be ~6.5%
    trading_fee_cost = 2.0 * TRADING_FEE

    expected = -trading_fee_cost - leverage_penalty
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-6)

    # Verify annual leverage cost is approximately 6.5%
    assert abs(leverage_penalty - 0.065) < 0.001, f"Annual cost should be ~6.5%, got {leverage_penalty}"


@pytest.mark.parametrize(
    "position,returns,expected_leverage_amount",
    [
        pytest.param(1.0, 0.02, 0.0, id="1x_no_leverage"),
        pytest.param(0.5, 0.02, 0.0, id="0.5x_no_leverage"),
        pytest.param(2.0, 0.02, 1.0, id="2x_1_borrowed"),
        pytest.param(3.0, 0.02, 2.0, id="3x_2_borrowed"),
        pytest.param(-1.0, -0.02, 0.0, id="short_1x_no_leverage"),
        pytest.param(-2.0, -0.02, 1.0, id="short_2x_1_borrowed"),
        pytest.param(1.5, 0.02, 0.5, id="1.5x_0.5_borrowed"),
    ],
)
def test_leverage_amount_calculation(position, returns, expected_leverage_amount):
    """Parametrized test for leverage amount calculation."""
    y_test_pred = torch.tensor([position])
    y_test = torch.tensor([returns])

    profit = calculate_trading_profit_torch(None, None, y_test, y_test_pred)

    leverage_penalty = expected_leverage_amount * DAILY_LEVERAGE_COST
    expected = position * returns - abs(position) * TRADING_FEE - leverage_penalty
    assert torch.isclose(profit, torch.tensor(expected), atol=1e-7)


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
