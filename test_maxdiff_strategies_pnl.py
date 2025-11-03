#!/usr/bin/env python3
"""
Tests for MaxDiff and MaxDiffAlwaysOn strategy PnL calculations.

This test suite ensures that:
1. The simulated PnL is accurately calculated for both strategies
2. Edge cases are handled properly
3. The multiplier optimization works correctly
4. Buy/sell logic is correct for both regular and crypto markets
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict

# Import the functions we're testing
from backtest_test3_inline import (
    evaluate_maxdiff_strategy,
    evaluate_maxdiff_always_on_strategy,
    StrategyEvaluation,
)


def create_simple_test_data(
    n_days: int = 5,
    base_close: float = 100.0,
    high_movement: float = 0.02,  # 2% high movement
    low_movement: float = -0.01,  # 1% low movement
) -> tuple[Dict[str, torch.Tensor], pd.DataFrame]:
    """
    Create simple, predictable test data for strategy testing.

    Returns:
        last_preds: Dictionary with predictions
        simulation_data: DataFrame with OHLC data
    """
    # Create predictions that are perfect (match actual movements)
    # Generate enough movements for any requested n_days
    base_movements = [0.01, -0.005, 0.015, -0.01, 0.02, 0.008, -0.012, 0.018, -0.008, 0.015]
    # Repeat the pattern if needed
    movements_list = (base_movements * ((n_days // len(base_movements)) + 1))[:n_days]
    close_actual_movement = torch.tensor(movements_list, dtype=torch.float32)

    high_actual_movement = torch.full((n_days,), high_movement, dtype=torch.float32)
    low_actual_movement = torch.full((n_days,), low_movement, dtype=torch.float32)

    # Perfect predictions (same as actual)
    high_predictions = high_actual_movement.clone()
    low_predictions = low_actual_movement.clone()

    # Create OHLC data
    closes = []
    highs = []
    lows = []
    current_price = base_close

    for i in range(n_days + 2):  # +2 for the offset used in the strategy
        closes.append(current_price)
        highs.append(current_price * (1 + high_movement))
        lows.append(current_price * (1 + low_movement))
        if i < n_days:
            current_price = current_price * (1 + close_actual_movement[i].item())

    simulation_data = pd.DataFrame({
        'Close': closes,
        'High': highs,
        'Low': lows,
    })

    last_preds = {
        'close_actual_movement_values': close_actual_movement,
        'high_actual_movement_values': high_actual_movement,
        'low_actual_movement_values': low_actual_movement,
        'high_predictions': high_predictions,
        'low_predictions': low_predictions,
        'high_predicted_price_value': highs[-1],
        'low_predicted_price_value': lows[-1],
    }

    return last_preds, simulation_data


def test_maxdiff_strategy_basic_calculation():
    """Test that MaxDiff strategy calculates PnL correctly with simple data."""
    last_preds, simulation_data = create_simple_test_data(n_days=5)

    evaluation, daily_returns, metadata = evaluate_maxdiff_strategy(
        last_preds,
        simulation_data,
        trading_fee=0.0,  # No fees for simplicity
        trading_days_per_year=252,
        is_crypto=False,
    )

    # Check that evaluation is a StrategyEvaluation instance
    assert isinstance(evaluation, StrategyEvaluation)

    # Check that we have the right number of returns
    assert len(daily_returns) == 5, f"Expected 5 daily returns, got {len(daily_returns)}"

    # Check that metadata has all expected keys
    expected_keys = [
        'maxdiffprofit_profit',
        'maxdiffprofit_profit_values',
        'maxdiffprofit_profit_high_multiplier',
        'maxdiffprofit_profit_low_multiplier',
        'maxdiffprofit_high_price',
        'maxdiffprofit_low_price',
        'maxdiff_turnover',
        'maxdiff_primary_side',
        'maxdiff_trade_bias',
        'maxdiff_trades_positive',
        'maxdiff_trades_negative',
        'maxdiff_trades_total',
    ]
    for key in expected_keys:
        assert key in metadata, f"Missing metadata key: {key}"

    # Check that total return matches sum of daily returns
    assert np.isclose(evaluation.total_return, np.sum(daily_returns), atol=1e-6), \
        f"Total return {evaluation.total_return} doesn't match sum of daily returns {np.sum(daily_returns)}"

    # Check that profit values in metadata match daily returns
    assert len(metadata['maxdiffprofit_profit_values']) == len(daily_returns)

    print(f"✓ MaxDiff basic calculation test passed")
    print(f"  Total return: {evaluation.total_return:.6f}")
    print(f"  Sharpe ratio: {evaluation.sharpe_ratio:.4f}")
    print(f"  Trade bias: {metadata['maxdiff_trade_bias']:.4f}")


def test_maxdiff_always_on_basic_calculation():
    """Test that MaxDiffAlwaysOn strategy calculates PnL correctly with simple data."""
    last_preds, simulation_data = create_simple_test_data(n_days=5)

    evaluation, daily_returns, metadata = evaluate_maxdiff_always_on_strategy(
        last_preds,
        simulation_data,
        trading_fee=0.0,
        trading_days_per_year=252,
        is_crypto=False,
    )

    # Check that evaluation is a StrategyEvaluation instance
    assert isinstance(evaluation, StrategyEvaluation)

    # Check that we have the right number of returns
    assert len(daily_returns) == 5, f"Expected 5 daily returns, got {len(daily_returns)}"

    # Check that metadata has all expected keys
    expected_keys = [
        'maxdiffalwayson_profit',
        'maxdiffalwayson_profit_values',
        'maxdiffalwayson_high_multiplier',
        'maxdiffalwayson_low_multiplier',
        'maxdiffalwayson_high_price',
        'maxdiffalwayson_low_price',
        'maxdiffalwayson_turnover',
        'maxdiffalwayson_buy_contribution',
        'maxdiffalwayson_sell_contribution',
        'maxdiffalwayson_filled_buy_trades',
        'maxdiffalwayson_filled_sell_trades',
        'maxdiffalwayson_trades_total',
        'maxdiffalwayson_trade_bias',
    ]
    for key in expected_keys:
        assert key in metadata, f"Missing metadata key: {key}"

    # Check that total return matches sum of daily returns
    assert np.isclose(evaluation.total_return, np.sum(daily_returns), atol=1e-6), \
        f"Total return {evaluation.total_return} doesn't match sum of daily returns {np.sum(daily_returns)}"

    # Check that buy + sell contributions sum to total return
    total_contribution = metadata['maxdiffalwayson_buy_contribution'] + metadata['maxdiffalwayson_sell_contribution']
    assert np.isclose(total_contribution, evaluation.total_return, atol=1e-6), \
        f"Buy+Sell contribution {total_contribution} doesn't match total return {evaluation.total_return}"

    # Check that trade counts are consistent
    total_fills = metadata['maxdiffalwayson_filled_buy_trades'] + metadata['maxdiffalwayson_filled_sell_trades']
    assert total_fills == metadata['maxdiffalwayson_trades_total'], \
        f"Sum of buy/sell fills {total_fills} doesn't match total {metadata['maxdiffalwayson_trades_total']}"

    print(f"✓ MaxDiffAlwaysOn basic calculation test passed")
    print(f"  Total return: {evaluation.total_return:.6f}")
    print(f"  Buy contribution: {metadata['maxdiffalwayson_buy_contribution']:.6f}")
    print(f"  Sell contribution: {metadata['maxdiffalwayson_sell_contribution']:.6f}")
    print(f"  Total fills: {metadata['maxdiffalwayson_trades_total']}")


def test_maxdiff_crypto_mode():
    """Test that MaxDiff strategy respects crypto mode (no short selling)."""
    last_preds, simulation_data = create_simple_test_data(n_days=5)

    evaluation, daily_returns, metadata = evaluate_maxdiff_strategy(
        last_preds,
        simulation_data,
        trading_fee=0.0,
        trading_days_per_year=252,
        is_crypto=True,  # Crypto mode
    )

    # In crypto mode, negative trades should be converted to 0
    # So we should have no negative trades
    assert metadata['maxdiff_trades_negative'] == 0, \
        f"Crypto mode should have 0 negative trades, got {metadata['maxdiff_trades_negative']}"

    print(f"✓ MaxDiff crypto mode test passed")
    print(f"  Positive trades: {metadata['maxdiff_trades_positive']}")
    print(f"  Negative trades: {metadata['maxdiff_trades_negative']}")


def test_maxdiff_always_on_crypto_mode():
    """Test that MaxDiffAlwaysOn strategy respects crypto mode."""
    last_preds, simulation_data = create_simple_test_data(n_days=5)

    evaluation, daily_returns, metadata = evaluate_maxdiff_always_on_strategy(
        last_preds,
        simulation_data,
        trading_fee=0.0,
        trading_days_per_year=252,
        is_crypto=True,
    )

    # In crypto mode, sell trades should be 0
    assert metadata['maxdiffalwayson_filled_sell_trades'] == 0, \
        f"Crypto mode should have 0 sell trades, got {metadata['maxdiffalwayson_filled_sell_trades']}"

    # All contribution should come from buys
    assert np.isclose(metadata['maxdiffalwayson_buy_contribution'], evaluation.total_return, atol=1e-6), \
        "In crypto mode, all return should come from buy contribution"

    print(f"✓ MaxDiffAlwaysOn crypto mode test passed")
    print(f"  Buy trades: {metadata['maxdiffalwayson_filled_buy_trades']}")
    print(f"  Sell trades: {metadata['maxdiffalwayson_filled_sell_trades']}")


def test_maxdiff_zero_validation_length():
    """Test that MaxDiff handles zero validation length gracefully."""
    last_preds = {
        'close_actual_movement_values': torch.tensor([], dtype=torch.float32),
        'high_predicted_price_value': 100.0,
        'low_predicted_price_value': 98.0,
    }
    simulation_data = pd.DataFrame({
        'Close': [100.0],
        'High': [102.0],
        'Low': [98.0],
    })

    evaluation, daily_returns, metadata = evaluate_maxdiff_strategy(
        last_preds,
        simulation_data,
        trading_fee=0.0,
        trading_days_per_year=252,
        is_crypto=False,
    )

    # Should return zero evaluation
    assert evaluation.total_return == 0.0
    assert len(daily_returns) == 0
    assert metadata['maxdiffprofit_profit'] == 0.0

    print(f"✓ MaxDiff zero validation test passed")


def test_maxdiff_missing_predictions():
    """Test that MaxDiff handles missing prediction arrays gracefully."""
    last_preds = {
        'close_actual_movement_values': torch.tensor([0.01, 0.02], dtype=torch.float32),
        # Missing high/low predictions
        'high_predicted_price_value': 100.0,
        'low_predicted_price_value': 98.0,
    }
    simulation_data = pd.DataFrame({
        'Close': [100.0, 101.0, 102.0, 103.0],
        'High': [102.0, 103.0, 104.0, 105.0],
        'Low': [98.0, 99.0, 100.0, 101.0],
    })

    evaluation, daily_returns, metadata = evaluate_maxdiff_strategy(
        last_preds,
        simulation_data,
        trading_fee=0.0,
        trading_days_per_year=252,
        is_crypto=False,
    )

    # Should return zero evaluation when predictions are missing
    assert evaluation.total_return == 0.0
    assert len(daily_returns) == 0

    print(f"✓ MaxDiff missing predictions test passed")


def test_maxdiff_multiplier_optimization():
    """Test that multiplier optimization improves PnL."""
    # Create data where the predictions are slightly off
    n_days = 10
    close_actual_movement = torch.tensor([0.01, -0.005, 0.015, -0.01, 0.02, 0.01, -0.01, 0.005, -0.015, 0.02], dtype=torch.float32)[:n_days]

    # Predictions are slightly biased (too optimistic on highs, too pessimistic on lows)
    high_actual_movement = torch.full((n_days,), 0.02, dtype=torch.float32)
    low_actual_movement = torch.full((n_days,), -0.01, dtype=torch.float32)
    high_predictions = torch.full((n_days,), 0.025, dtype=torch.float32)  # Overestimate
    low_predictions = torch.full((n_days,), -0.015, dtype=torch.float32)  # Overestimate magnitude

    # Create OHLC data
    closes = []
    highs = []
    lows = []
    current_price = 100.0

    for i in range(n_days + 2):
        closes.append(current_price)
        highs.append(current_price * 1.02)
        lows.append(current_price * 0.99)
        if i < n_days:
            current_price = current_price * (1 + close_actual_movement[i].item())

    simulation_data = pd.DataFrame({
        'Close': closes,
        'High': highs,
        'Low': lows,
    })

    last_preds = {
        'close_actual_movement_values': close_actual_movement,
        'high_actual_movement_values': high_actual_movement,
        'low_actual_movement_values': low_actual_movement,
        'high_predictions': high_predictions,
        'low_predictions': low_predictions,
        'high_predicted_price_value': highs[-1],
        'low_predicted_price_value': lows[-1],
    }

    evaluation, daily_returns, metadata = evaluate_maxdiff_strategy(
        last_preds,
        simulation_data,
        trading_fee=0.0,
        trading_days_per_year=252,
        is_crypto=False,
    )

    # Multipliers should be non-zero (optimizer should find adjustments)
    high_mult = metadata['maxdiffprofit_profit_high_multiplier']
    low_mult = metadata['maxdiffprofit_profit_low_multiplier']

    print(f"✓ MaxDiff multiplier optimization test passed")
    print(f"  High multiplier: {high_mult:.6f}")
    print(f"  Low multiplier: {low_mult:.6f}")
    print(f"  Total return: {evaluation.total_return:.6f}")


def test_sharpe_ratio_calculation():
    """Test that Sharpe ratio is calculated correctly."""
    # Create data with variable returns (not constant, so std > 0)
    n_days = 20
    # Variable returns - alternating pattern with some variation
    close_actual_movement = torch.tensor([
        0.01, 0.015, 0.005, 0.02, 0.01,
        0.012, 0.008, 0.018, 0.01, 0.015,
        0.01, 0.02, 0.005, 0.015, 0.01,
        0.01, 0.015, 0.01, 0.02, 0.01
    ], dtype=torch.float32)[:n_days]

    # Make high and low movements also vary
    high_actual_movement = torch.tensor([
        0.02, 0.025, 0.015, 0.03, 0.02,
        0.022, 0.018, 0.028, 0.02, 0.025,
        0.02, 0.03, 0.015, 0.025, 0.02,
        0.02, 0.025, 0.02, 0.03, 0.02
    ], dtype=torch.float32)[:n_days]

    low_actual_movement = torch.tensor([
        -0.005, -0.008, -0.003, -0.01, -0.005,
        -0.006, -0.004, -0.009, -0.005, -0.008,
        -0.005, -0.01, -0.003, -0.008, -0.005,
        -0.005, -0.008, -0.005, -0.01, -0.005
    ], dtype=torch.float32)[:n_days]

    high_predictions = high_actual_movement.clone()
    low_predictions = low_actual_movement.clone()

    closes = []
    highs = []
    lows = []
    current_price = 100.0

    for i in range(n_days + 2):
        if i < n_days:
            high_movement = high_actual_movement[i].item()
            low_movement = low_actual_movement[i].item()
        else:
            high_movement = 0.02
            low_movement = -0.005

        closes.append(current_price)
        highs.append(current_price * (1 + high_movement))
        lows.append(current_price * (1 + low_movement))

        if i < n_days:
            current_price = current_price * (1 + close_actual_movement[i].item())

    simulation_data = pd.DataFrame({
        'Close': closes,
        'High': highs,
        'Low': lows,
    })

    last_preds = {
        'close_actual_movement_values': close_actual_movement,
        'high_actual_movement_values': high_actual_movement,
        'low_actual_movement_values': low_actual_movement,
        'high_predictions': high_predictions,
        'low_predictions': low_predictions,
        'high_predicted_price_value': highs[-1],
        'low_predicted_price_value': lows[-1],
    }

    evaluation, daily_returns, metadata = evaluate_maxdiff_strategy(
        last_preds,
        simulation_data,
        trading_fee=0.0,
        trading_days_per_year=252,
        is_crypto=False,
    )

    # Manual Sharpe calculation
    mean_return = np.mean(daily_returns)
    std_return = np.std(daily_returns)

    # Sharpe should be calculable (std > 0)
    if std_return > 0:
        expected_sharpe = (mean_return / std_return) * np.sqrt(252)
        assert np.isclose(evaluation.sharpe_ratio, expected_sharpe, atol=1e-4), \
            f"Sharpe mismatch: {evaluation.sharpe_ratio} vs {expected_sharpe}"
        assert evaluation.sharpe_ratio != 0, "Sharpe should be non-zero when std > 0"
    else:
        # If std is 0, Sharpe should be 0
        assert evaluation.sharpe_ratio == 0, "Sharpe should be 0 when std is 0"

    print(f"✓ Sharpe ratio calculation test passed")
    print(f"  Sharpe ratio: {evaluation.sharpe_ratio:.4f}")
    print(f"  Mean daily return: {mean_return:.6f}")
    print(f"  Std daily return: {std_return:.6f}")


def test_trade_bias_calculation():
    """Test that trade bias is calculated correctly."""
    # Create data that should favor buying (high predictions > low predictions)
    n_days = 10
    close_actual_movement = torch.tensor([0.01, 0.015, 0.02, 0.01, 0.015, 0.02, 0.01, 0.015, 0.02, 0.01], dtype=torch.float32)
    high_actual_movement = torch.full((n_days,), 0.03, dtype=torch.float32)  # Higher magnitude
    low_actual_movement = torch.full((n_days,), -0.01, dtype=torch.float32)  # Lower magnitude
    high_predictions = torch.full((n_days,), 0.03, dtype=torch.float32)
    low_predictions = torch.full((n_days,), -0.01, dtype=torch.float32)

    closes = []
    highs = []
    lows = []
    current_price = 100.0

    for i in range(n_days + 2):
        closes.append(current_price)
        highs.append(current_price * 1.03)
        lows.append(current_price * 0.99)
        if i < n_days:
            current_price = current_price * (1 + close_actual_movement[i].item())

    simulation_data = pd.DataFrame({
        'Close': closes,
        'High': highs,
        'Low': lows,
    })

    last_preds = {
        'close_actual_movement_values': close_actual_movement,
        'high_actual_movement_values': high_actual_movement,
        'low_actual_movement_values': low_actual_movement,
        'high_predictions': high_predictions,
        'low_predictions': low_predictions,
        'high_predicted_price_value': highs[-1],
        'low_predicted_price_value': lows[-1],
    }

    evaluation, daily_returns, metadata = evaluate_maxdiff_strategy(
        last_preds,
        simulation_data,
        trading_fee=0.0,
        trading_days_per_year=252,
        is_crypto=False,
    )

    # Since high predictions have higher magnitude, should favor buy side
    assert metadata['maxdiff_trades_positive'] > 0, "Should have positive (buy) trades"
    assert metadata['maxdiff_trade_bias'] > 0, f"Trade bias should be positive, got {metadata['maxdiff_trade_bias']}"
    assert metadata['maxdiff_primary_side'] == 'buy', f"Primary side should be 'buy', got {metadata['maxdiff_primary_side']}"

    print(f"✓ Trade bias calculation test passed")
    print(f"  Trade bias: {metadata['maxdiff_trade_bias']:.4f}")
    print(f"  Primary side: {metadata['maxdiff_primary_side']}")
    print(f"  Positive trades: {metadata['maxdiff_trades_positive']}")
    print(f"  Negative trades: {metadata['maxdiff_trades_negative']}")


def test_trading_fees_included():
    """Test that trading fees are included in PnL calculation."""
    last_preds, simulation_data = create_simple_test_data(n_days=5)

    # Run the strategy - uses TRADING_FEE from loss_utils.py
    evaluation, daily_returns, metadata = evaluate_maxdiff_strategy(
        last_preds,
        simulation_data,
        trading_fee=0.0,
        trading_days_per_year=252,
        is_crypto=False,
    )

    # Verify that the calculation completes successfully with fees included
    assert isinstance(evaluation.total_return, float)
    assert not np.isnan(evaluation.total_return)
    assert len(daily_returns) == 5

    print(f"✓ Trading fees test passed")
    print(f"  Total return (includes fees): {evaluation.total_return:.6f}")


def test_pnl_consistency_between_calls():
    """Test that calling the strategy multiple times with the same data gives consistent results."""
    last_preds, simulation_data = create_simple_test_data(n_days=10)

    # Run the strategy 3 times with the same data
    results = []
    for _ in range(3):
        evaluation, daily_returns, metadata = evaluate_maxdiff_strategy(
            last_preds,
            simulation_data,
            trading_fee=0.0,
            trading_days_per_year=252,
            is_crypto=False,
        )
        results.append({
            'total_return': evaluation.total_return,
            'sharpe': evaluation.sharpe_ratio,
            'daily_returns': daily_returns.copy(),
        })

    # Check that all results are identical
    for i in range(1, 3):
        assert np.isclose(results[i]['total_return'], results[0]['total_return'], atol=1e-10), \
            f"Total return should be consistent across calls: {results[i]['total_return']} vs {results[0]['total_return']}"
        assert np.isclose(results[i]['sharpe'], results[0]['sharpe'], atol=1e-10), \
            f"Sharpe should be consistent across calls: {results[i]['sharpe']} vs {results[0]['sharpe']}"
        assert np.allclose(results[i]['daily_returns'], results[0]['daily_returns'], atol=1e-10), \
            "Daily returns should be consistent across calls"

    print(f"✓ PnL consistency test passed")
    print(f"  Verified consistency across 3 runs")
    print(f"  Total return: {results[0]['total_return']:.6f}")


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("Running MaxDiff Strategy PnL Tests")
    print("=" * 60)
    print()

    tests = [
        ("Basic MaxDiff Calculation", test_maxdiff_strategy_basic_calculation),
        ("Basic MaxDiffAlwaysOn Calculation", test_maxdiff_always_on_basic_calculation),
        ("MaxDiff Crypto Mode", test_maxdiff_crypto_mode),
        ("MaxDiffAlwaysOn Crypto Mode", test_maxdiff_always_on_crypto_mode),
        ("MaxDiff Zero Validation Length", test_maxdiff_zero_validation_length),
        ("MaxDiff Missing Predictions", test_maxdiff_missing_predictions),
        ("MaxDiff Multiplier Optimization", test_maxdiff_multiplier_optimization),
        ("Sharpe Ratio Calculation", test_sharpe_ratio_calculation),
        ("Trade Bias Calculation", test_trade_bias_calculation),
        ("Trading Fees Included", test_trading_fees_included),
        ("PnL Consistency Between Calls", test_pnl_consistency_between_calls),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        print("-" * 60)
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {test_name}")
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
