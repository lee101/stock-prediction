"""
Unit tests for close_at_eod parameter in loss_utils

Tests the difference between:
1. Intraday exits (default): Can exit at high/low prices during the day
2. EOD exits (close_at_eod=True): Must hold until end-of-day close price

This is critical for realistic simulation of after-hours trading restrictions.
"""

import torch
import pytest
from loss_utils import calculate_profit_torch_with_entry_buysell_profit_values, TRADING_FEE


class TestCloseAtEOD:
    """Test close_at_eod parameter for realistic EOD trading simulation"""

    def test_intraday_exit_allows_high_exit(self):
        """Test that default behavior allows exiting at intraday high"""
        # Setup: Buy at low=-5%, can exit at high=+10%
        # close=+2% (lower than high)

        y_test = torch.tensor([0.02])  # Close at +2%
        y_test_high = torch.tensor([0.10])  # High at +10% (BEST exit)
        y_test_high_pred = torch.tensor([0.08])  # Predicted high: +8%
        y_test_low = torch.tensor([-0.06])  # Low hit
        y_test_low_pred = torch.tensor([-0.05])  # Entry at -5%
        y_test_pred = torch.tensor([1.0])  # Full buy signal

        # Default: close_at_eod=False (allow intraday exits)
        profit_intraday = calculate_profit_torch_with_entry_buysell_profit_values(
            y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred,
            close_at_eod=False
        )

        # Should exit at high_pred=+8% (your sell limit order)
        # Profit = high_pred - low_pred = 0.08 - (-0.05) = 0.13 = 13%
        # Expected: low_to_high_pred movement = 0.13 - fee
        expected = 0.13 - TRADING_FEE

        assert profit_intraday[0] > 0.10, \
            f"Intraday exit should capture high profit, got {profit_intraday[0]:.4f}"
        assert torch.isclose(profit_intraday, torch.tensor([expected]), atol=1e-3), \
            f"Expected {expected:.4f}, got {profit_intraday[0]:.4f}"

    def test_eod_exit_forces_close_price(self):
        """Test that close_at_eod=True forces exit at close price only"""
        # Same setup as above

        y_test = torch.tensor([0.02])  # Close at +2%
        y_test_high = torch.tensor([0.10])  # High at +10% (can't use this!)
        y_test_high_pred = torch.tensor([0.08])
        y_test_low = torch.tensor([-0.06])  # Low hit
        y_test_low_pred = torch.tensor([-0.05])  # Entry at -5%
        y_test_pred = torch.tensor([1.0])

        # Force EOD exit: close_at_eod=True
        profit_eod = calculate_profit_torch_with_entry_buysell_profit_values(
            y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred,
            close_at_eod=True
        )

        # Should exit at close=+2% (not high!)
        # Profit = close - low_pred = 0.02 - (-0.05) = 0.07 = 7%
        # Expected: low_to_close movement = 0.07 - fee
        expected = 0.07 - TRADING_FEE

        assert profit_eod[0] < 0.10, \
            f"EOD exit should NOT capture intraday high, got {profit_eod[0]:.4f}"
        assert torch.isclose(profit_eod, torch.tensor([expected]), atol=1e-3), \
            f"Expected {expected:.4f}, got {profit_eod[0]:.4f}"

    def test_eod_vs_intraday_comparison(self):
        """Compare intraday vs EOD exits side-by-side"""

        y_test = torch.tensor([0.01])  # Close at +1%
        y_test_high = torch.tensor([0.08])  # High at +8%
        y_test_high_pred = torch.tensor([0.05])  # Predicted high: +5%
        y_test_low = torch.tensor([-0.06])
        y_test_low_pred = torch.tensor([-0.04])  # Entry at -4%
        y_test_pred = torch.tensor([1.0])

        profit_intraday = calculate_profit_torch_with_entry_buysell_profit_values(
            y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred,
            close_at_eod=False
        )

        profit_eod = calculate_profit_torch_with_entry_buysell_profit_values(
            y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred,
            close_at_eod=True
        )

        # Intraday should be higher (exits at high)
        assert profit_intraday[0] > profit_eod[0], \
            f"Intraday {profit_intraday[0]:.4f} should > EOD {profit_eod[0]:.4f}"

        print(f"\nIntraday profit: {profit_intraday[0]:.4f} (exits at high)")
        print(f"EOD profit: {profit_eod[0]:.4f} (exits at close)")
        print(f"Difference: {(profit_intraday[0] - profit_eod[0]):.4f}")

    def test_short_trade_eod_exit(self):
        """Test short trades with EOD exit"""
        # Short at high=+5%, should cover at close=-2% (not low=-8%)

        y_test = torch.tensor([-0.02])  # Close at -2%
        y_test_high = torch.tensor([0.06])  # High hit
        y_test_high_pred = torch.tensor([0.05])  # Short entry at +5%
        y_test_low = torch.tensor([-0.08])  # Low at -8% (better exit, but can't use!)
        y_test_low_pred = torch.tensor([-0.06])  # Predicted low
        y_test_pred = torch.tensor([-1.0])  # Full short signal

        profit_eod = calculate_profit_torch_with_entry_buysell_profit_values(
            y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred,
            close_at_eod=True
        )

        # Profit from short: entry(+5%) to exit(-2%) = 7% gain
        # Expected: high_to_close = abs(0.05 - (-0.02)) = 0.07
        expected = 0.07 - TRADING_FEE

        assert torch.isclose(profit_eod, torch.tensor([expected]), atol=1e-3), \
            f"Expected {expected:.4f}, got {profit_eod[0]:.4f}"

    def test_missed_entry_eod_vs_intraday(self):
        """Test that missed entries behave the same for both modes"""
        # Low not hit - no entry in both cases

        y_test = torch.tensor([0.01])
        y_test_high = torch.tensor([0.05])
        y_test_high_pred = torch.tensor([0.05])
        y_test_low = torch.tensor([-0.02])  # Only goes to -2%
        y_test_low_pred = torch.tensor([-0.05])  # Wanted -5% (NOT HIT)
        y_test_pred = torch.tensor([1.0])

        profit_intraday = calculate_profit_torch_with_entry_buysell_profit_values(
            y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred,
            close_at_eod=False
        )

        profit_eod = calculate_profit_torch_with_entry_buysell_profit_values(
            y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred,
            close_at_eod=True
        )

        # Both should be 0 (no entry = no profit)
        assert torch.isclose(profit_intraday, torch.tensor([0.0]), atol=1e-4)
        assert torch.isclose(profit_eod, torch.tensor([0.0]), atol=1e-4)
        assert torch.isclose(profit_intraday, profit_eod, atol=1e-4), \
            "Missed entries should give same result for both modes"

    def test_multiple_days_eod(self):
        """Test multiple days with EOD exit mode"""
        # Mix of winning and losing days

        y_test = torch.tensor([0.01, -0.01, 0.02])  # EOD closes
        y_test_high = torch.tensor([0.10, 0.05, 0.08])
        y_test_high_pred = torch.tensor([0.05, 0.05, 0.05])
        y_test_low = torch.tensor([-0.06, -0.06, -0.06])
        y_test_low_pred = torch.tensor([-0.04, -0.04, -0.04])
        y_test_pred = torch.tensor([1.0, 1.0, 1.0])

        profits_eod = calculate_profit_torch_with_entry_buysell_profit_values(
            y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred,
            close_at_eod=True
        )

        # Day 1: buy at -0.04, close at +0.01 = 0.05 profit
        # Day 2: buy at -0.04, close at -0.01 = 0.03 profit
        # Day 3: buy at -0.04, close at +0.02 = 0.06 profit

        assert profits_eod[0] > 0, "Day 1 should be profitable"
        assert profits_eod[1] > 0, "Day 2 should be profitable (small)"
        assert profits_eod[2] > 0, "Day 3 should be profitable"

        # Day 1 should be biggest
        assert profits_eod[2] > profits_eod[1], "Day 3 > Day 2"

    def test_realistic_after_hours_scenario(self):
        """
        Realistic scenario: After-hours trading where you can't exit until next day's close.

        Enter: Buy signal at 4pm (market close)
        Intraday: Stock hits high of +12% at 2pm next day
        Close: Stock closes at +3% next day

        With intraday exit: Would capture +12% (unrealistic if after-hours restricted)
        With EOD exit: Can only get +3% (realistic)
        """

        y_test = torch.tensor([0.03])  # Next day close: +3%
        y_test_high = torch.tensor([0.12])  # Next day intraday high: +12%
        y_test_high_pred = torch.tensor([0.08])  # Predicted target
        y_test_low = torch.tensor([-0.06])  # Entry filled overnight
        y_test_low_pred = torch.tensor([-0.05])  # Buy signal at -5%
        y_test_pred = torch.tensor([1.0])

        # Unrealistic (intraday exit)
        profit_unrealistic = calculate_profit_torch_with_entry_buysell_profit_values(
            y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred,
            close_at_eod=False
        )

        # Realistic (EOD exit only)
        profit_realistic = calculate_profit_torch_with_entry_buysell_profit_values(
            y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred,
            close_at_eod=True
        )

        print(f"\nRealistic after-hours scenario:")
        print(f"Unrealistic (intraday): {profit_unrealistic[0]:.4f} ({profit_unrealistic[0]*100:.2f}%)")
        print(f"Realistic (EOD only): {profit_realistic[0]:.4f} ({profit_realistic[0]*100:.2f}%)")
        print(f"Overestimation: {(profit_unrealistic[0] - profit_realistic[0])*100:.2f}%")

        # Unrealistic should significantly overestimate
        assert profit_unrealistic[0] > profit_realistic[0] * 1.5, \
            "Intraday exit should capture much more profit in this scenario"


if __name__ == "__main__":
    print("Running close_at_eod unit tests...")
    pytest.main([__file__, "-v", "-s"])
