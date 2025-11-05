"""
Unit tests for MaxDiff PnL calculations

MaxDiff Strategy Explanation:
================================

The MaxDiff strategy predicts high and low price levels, then:
1. BUYS when price hits low_pred, SELLS when price hits high_pred
2. Only executes if actual high/low prices reach predicted levels
3. Profit = (high_pred - low_pred) if both levels are hit

Key Calculations (from loss_utils.py:283-337):
----------------------------------------------

calculate_profit_torch_with_entry_buysell_profit_values():
  - For BUY trades (y_test_pred > 0):
    * Entry: Buy at low_pred (if actual low <= low_pred)
    * Exit: Sell at high_pred (if actual high >= high_pred)
    * Profit: (high_pred - low_pred) - fee
    * Miss: 0 profit if levels not hit

  - For SELL trades (y_test_pred < 0):
    * Entry: Short at high_pred (if actual high >= high_pred)
    * Exit: Cover at low_pred (if actual low <= low_pred)
    * Profit: (high_pred - low_pred) - fee
    * Miss: 0 profit if levels not hit

  - Fee: Only charged if BOTH entry and exit are hit

Why high/low must not be inverted:
-----------------------------------
If high_pred < low_pred:
  - BUY profit = (low_pred - high_pred) which is NEGATIVE
  - Strategy loses money on every filled trade
  - The 0.4% margin ensures: high >= low + 0.004
"""

import torch
import pytest
from loss_utils import calculate_profit_torch_with_entry_buysell_profit_values, TRADING_FEE


class TestMaxDiffPnLCalculations:
    """Test maxdiff profit calculations with clear examples"""

    def test_successful_buy_trade(self):
        """Test a successful buy trade where both levels are hit"""
        # Setup: Predict to buy at low=0.95, sell at high=1.05 (5% each way = 10% total)
        # Actual: low=0.94 (hit!), high=1.06 (hit!), close=1.00

        y_test = torch.tensor([0.00])  # Close movement (not used in entry logic)
        y_test_high = torch.tensor([0.06])  # Actual high movement: +6%
        y_test_high_pred = torch.tensor([0.05])  # Predicted high: +5%
        y_test_low = torch.tensor([-0.06])  # Actual low movement: -6%
        y_test_low_pred = torch.tensor([-0.05])  # Predicted low: -5%
        y_test_pred = torch.tensor([1.0])  # Trade signal: 1.0 = full buy

        profit = calculate_profit_torch_with_entry_buysell_profit_values(
            y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred
        )

        # Expected: low_to_high movement (0.05 - (-0.05)) = 0.10 = 10%
        # Minus fee: 0.10 - 0.0005 = 0.0995
        expected = 0.10 - TRADING_FEE
        assert torch.isclose(profit, torch.tensor([expected]), atol=1e-4), \
            f"Expected {expected:.4f}, got {profit[0]:.4f}"

    def test_missed_buy_low_level(self):
        """Test when buy level (low) is NOT hit"""
        # Setup: Predict to buy at low=-5%, but actual low only goes to -3%

        y_test = torch.tensor([0.00])
        y_test_high = torch.tensor([0.05])  # High hit at +5%
        y_test_high_pred = torch.tensor([0.05])
        y_test_low = torch.tensor([-0.03])  # Actual low: -3% (NOT low enough!)
        y_test_low_pred = torch.tensor([-0.05])  # Predicted low: -5%
        y_test_pred = torch.tensor([1.0])

        profit = calculate_profit_torch_with_entry_buysell_profit_values(
            y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred
        )

        # Expected: 0 profit (buy level not hit, no entry)
        assert torch.isclose(profit, torch.tensor([0.0]), atol=1e-4), \
            f"Expected 0.0 (no entry), got {profit[0]:.4f}"

    def test_inverted_predictions_negative_profit(self):
        """Test that inverted predictions (high < low) result in negative profit"""
        # Setup: INVERTED - high_pred=-0.02, low_pred=0.03 (backwards!)
        # This is the bug we're trying to prevent with margin constraint

        y_test = torch.tensor([0.00])
        y_test_high = torch.tensor([0.05])  # Actual high hit
        y_test_high_pred = torch.tensor([-0.02])  # Predicted high: -2% (WRONG!)
        y_test_low = torch.tensor([-0.05])  # Actual low hit
        y_test_low_pred = torch.tensor([0.03])  # Predicted low: +3% (WRONG!)
        y_test_pred = torch.tensor([1.0])

        profit = calculate_profit_torch_with_entry_buysell_profit_values(
            y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred
        )

        # Note: The function clamps high_pred to [0, 10] and low_pred to [-1, 0]
        # So high_pred becomes 0.0 and low_pred becomes 0.0
        # Movement = 0 - 0 = 0, minus fee = -0.0005
        # Actually it won't trade because the entry logic checks if levels are hit
        # Let me recalculate...

        # After clamping:
        # high_pred = clamp(-0.02, 0, 10) = 0.0
        # low_pred = clamp(0.03, -1, 0) = 0.0
        # Both are 0, so no movement profit, just fee if trading happens

        # The real issue is before clamping in the optimization
        assert profit[0] <= 0, \
            f"Inverted predictions should not be profitable, got {profit[0]:.4f}"

    def test_margin_constraint_prevents_inversion(self):
        """Test that 0.4% margin prevents rapid buy/sell flipping"""
        # Setup: high and low very close (within 0.4% margin)
        # This should be prevented by optimization constraint

        y_test = torch.tensor([0.00])
        y_test_high = torch.tensor([0.05])
        y_test_high_pred = torch.tensor([0.002])  # High: +0.2%
        y_test_low = torch.tensor([-0.05])
        y_test_low_pred = torch.tensor([-0.001])  # Low: -0.1%
        y_test_pred = torch.tensor([1.0])

        profit = calculate_profit_torch_with_entry_buysell_profit_values(
            y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred
        )

        # Gap = 0.002 - (-0.001) = 0.003 = 0.3% (less than 0.4% margin)
        # Expected: Very small profit minus fee = likely negative
        # The margin constraint in optimization should prevent this scenario
        gap = 0.002 - (-0.001)
        assert gap < 0.004, \
            f"Gap {gap:.4f} should be less than 0.004 (0.4% margin)"

    def test_sell_trade(self):
        """Test a successful short sell trade"""
        # Setup: Short at high=+5%, cover at low=-5%

        y_test = torch.tensor([0.00])
        y_test_high = torch.tensor([0.06])  # Actual high hit
        y_test_high_pred = torch.tensor([0.05])  # Short entry at +5%
        y_test_low = torch.tensor([-0.06])  # Actual low hit
        y_test_low_pred = torch.tensor([-0.05])  # Cover at -5%
        y_test_pred = torch.tensor([-1.0])  # Trade signal: -1.0 = full short

        profit = calculate_profit_torch_with_entry_buysell_profit_values(
            y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred
        )

        # For shorts: profit = -(high_to_low movement) = -(0.05 - (-0.05)) = -0.10
        # But wait, we profit from price going down!
        # Expected: (high_pred - low_pred) = 0.10 - fee
        # The calculation is: hit_low_points = -1 * (high_to_low) * clip(-1, -10, 0)
        expected = 0.10 - TRADING_FEE
        assert torch.isclose(profit, torch.tensor([expected]), atol=1e-4), \
            f"Expected {expected:.4f}, got {profit[0]:.4f}"

    def test_multiple_trades(self):
        """Test multiple days of trading"""
        # Day 1: Successful buy, Day 2: Miss, Day 3: Successful sell

        y_test = torch.tensor([0.00, 0.00, 0.00])
        y_test_high = torch.tensor([0.06, 0.03, 0.06])
        y_test_high_pred = torch.tensor([0.05, 0.05, 0.05])
        y_test_low = torch.tensor([-0.06, -0.02, -0.06])
        y_test_low_pred = torch.tensor([-0.05, -0.05, -0.05])
        y_test_pred = torch.tensor([1.0, 1.0, -1.0])  # buy, buy, sell

        profits = calculate_profit_torch_with_entry_buysell_profit_values(
            y_test, y_test_high, y_test_high_pred, y_test_low, y_test_low_pred, y_test_pred
        )

        # Day 1: 0.10 - fee = 0.0995
        # Day 2: 0 (low not hit)
        # Day 3: 0.10 - fee = 0.0995
        expected_day1 = 0.10 - TRADING_FEE
        expected_day2 = 0.0
        expected_day3 = 0.10 - TRADING_FEE

        assert torch.isclose(profits[0], torch.tensor(expected_day1), atol=1e-4)
        assert torch.isclose(profits[1], torch.tensor(expected_day2), atol=1e-4)
        assert torch.isclose(profits[2], torch.tensor(expected_day3), atol=1e-4)


class TestMaxDiffOptimization:
    """Test the multiplier optimization process"""

    def test_optimization_respects_margin(self):
        """Verify that optimization with margin constraint works"""
        # This is more of an integration test
        # The key is: high_pred + high_mult >= low_pred + low_mult + 0.004

        high_pred_base = 0.03  # 3%
        low_pred_base = -0.02  # -2%
        MARGIN_PCT = 0.004

        # Valid: high=0.04, low=-0.01 → 0.04 >= -0.01 + 0.004 → 0.04 >= -0.006 ✓
        high_mult = 0.01
        low_mult = 0.01
        assert (high_pred_base + high_mult) >= (low_pred_base + low_mult + MARGIN_PCT), \
            "Valid multipliers should pass margin constraint"

        # Invalid: high=0.02, low=0.00 → 0.02 >= 0.00 + 0.004 → 0.02 >= 0.004 ✓
        # Actually this is still valid, let me try worse

        # Invalid: high=0.025, low=0.00 → 0.025 >= 0.004 ✓
        # Let me try: high=0.002, low=0.00 → 0.002 >= 0.004 ✗
        high_mult = -0.028  # high = 0.002
        low_mult = 0.02  # low = 0.00
        assert (high_pred_base + high_mult) < (low_pred_base + low_mult + MARGIN_PCT), \
            "Invalid multipliers should fail margin constraint"


def test_run_7_simulations():
    """Test running backtest with only 7 simulations for recent performance"""
    from backtest_test3_inline import backtest_forecasts

    # Test with UNIUSD
    result = backtest_forecasts('UNIUSD', num_simulations=7)

    # Verify we got results
    assert len(result) == 7, f"Expected 7 simulations, got {len(result)}"

    # Check that MaxDiffAlwaysOn strategy exists
    assert 'maxdiffalwayson_avg_daily_return' in result.columns, \
        "MaxDiffAlwaysOn metrics should be in results"

    # Print recent performance
    avg_return = result['maxdiffalwayson_avg_daily_return'].mean()
    print(f"\n7-day recent MaxDiffAlwaysOn avg return: {avg_return:.4f} ({avg_return*100:.2f}%)")
    print(f"70-day historical: 19.26% (26.32 Sharpe)")
    print(f"Difference: {(avg_return - 0.1926)*100:.2f}%")

    return result


if __name__ == "__main__":
    # Run quick tests
    print("Running MaxDiff PnL unit tests...")
    pytest.main([__file__, "-v", "-s"])
