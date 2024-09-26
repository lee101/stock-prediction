import os
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

# Set the environment variable for testing
os.environ['TESTING'] = 'True'

# Import the function to test
from backtest_test3_inline import backtest_forecasts, simple_buy_sell_strategy, all_signals_strategy, \
    evaluate_strategy, buy_hold_strategy, unprofit_shutdown_buy_hold, CRYPTO_TRADING_FEE, ETH_SPREAD


@pytest.fixture
def mock_stock_data():
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 102,
        'Low': np.random.randn(100).cumsum() + 98,
        'Close': np.random.randn(100).cumsum() + 101,
    }, index=dates)


@pytest.fixture
def mock_pipeline():
    mock_forecast = MagicMock()
    mock_forecast.numpy.return_value = np.random.randn(20, 1)
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.predict.return_value = [mock_forecast]
    return mock_pipeline_instance


@patch('backtest_test3_inline.download_daily_stock_data')
@patch('backtest_test3_inline.ChronosPipeline.from_pretrained')
def test_backtest_forecasts(mock_pipeline_class, mock_download_data, mock_stock_data, mock_pipeline):
    mock_download_data.return_value = mock_stock_data
    mock_pipeline_class.return_value = mock_pipeline

    symbol = 'MOCKSYMBOL'
    num_simulations = 5
    results = backtest_forecasts(symbol, num_simulations)

    # Assertions
    assert isinstance(results, pd.DataFrame)
    assert len(results) == num_simulations
    assert 'buy_hold_return' in results.columns
    assert 'buy_hold_finalday' in results.columns

    # Check if the buy and hold strategy is calculated correctly
    for i in range(num_simulations):
        simulation_data = mock_stock_data.iloc[:-(i + 1)].copy()
        actual_returns = simulation_data['Close'].pct_change().iloc[-7:]

        # Calculate expected buy-and-hold return
        cumulative_return = (1 + actual_returns).prod() - 1
        expected_buy_hold_return = cumulative_return - CRYPTO_TRADING_FEE  # Apply fee once for initial buy

        assert pytest.approx(results['buy_hold_return'].iloc[i], rel=1e-4) == expected_buy_hold_return, \
            f"Expected buy hold return {expected_buy_hold_return}, but got {results['buy_hold_return'].iloc[i]}"

        # Check final day return
        expected_final_day_return = actual_returns.iloc[-1] - CRYPTO_TRADING_FEE
        assert pytest.approx(results['buy_hold_finalday'].iloc[i], rel=1e-4) == expected_final_day_return, \
            f"Expected final day return {expected_final_day_return}, but got {results['buy_hold_finalday'].iloc[i]}"

    # Check if the pipeline was called the correct number of times
    expected_pipeline_calls = num_simulations * 4 * 7  # 4 price types, 7 days each
    assert mock_pipeline.predict.call_count == expected_pipeline_calls


def test_simple_buy_sell_strategy():
    predictions = torch.tensor([-0.1, 0.2, 0, -0.3, 0.5])
    expected_output = torch.tensor([-1., 1., -1., -1., 1.])
    result = simple_buy_sell_strategy(predictions)
    assert torch.all(result.eq(expected_output)), f"Expected {expected_output}, but got {result}"


def test_all_signals_strategy():
    close_pred = torch.tensor([0.1, -0.2, 0.3, -0.4])
    high_pred = torch.tensor([0.2, -0.1, 0.4, -0.3])
    low_pred = torch.tensor([0.3, -0.3, 0.2, -0.2])
    open_pred = torch.tensor([0.4, -0.4, 0.1, -0.1])
    result = all_signals_strategy(close_pred, high_pred, low_pred, open_pred)

    expected_output = torch.tensor([1., -1., 0., -1.])
    assert torch.all(result.eq(expected_output)), f"Expected {expected_output}, but got {result}"


def test_evaluate_strategy_with_fees():
    strategy_signals = torch.tensor([1., 1., -1., -1., 1.])
    actual_returns = pd.Series([0.02, 0.01, -0.01, -0.02, 0.03])

    total_return, sharpe_ratio = evaluate_strategy(strategy_signals, actual_returns)

    # Calculate expected fees correctly
    expected_gains = [1.02 - (2 * CRYPTO_TRADING_FEE),
                      1.01 - (2 * CRYPTO_TRADING_FEE),
                      1.01 - (2 * CRYPTO_TRADING_FEE),
                      1.02 - (2 * CRYPTO_TRADING_FEE),
                      1.03 - (2 * CRYPTO_TRADING_FEE)]
    actual_gain = 1
    for gain in expected_gains:
        actual_gain *= gain
    actual_gain -= 1

    assert pytest.approx(total_return, rel=1e-4) == actual_gain, \
        f"Expected total return {actual_gain}, but got {total_return}"
    assert sharpe_ratio > 0, f"Sharpe ratio {sharpe_ratio} is not positive"


def test_buy_hold_strategy():
    predictions = torch.tensor([-0.1, 0.2, 0, -0.3, 0.5])
    expected_output = torch.tensor([0., 1., 0., 0., 1.])
    result = buy_hold_strategy(predictions)
    assert torch.all(result.eq(expected_output)), f"Expected {expected_output}, but got {result}"


def test_unprofit_shutdown_buy_hold():
    predictions = torch.tensor([0.1, 0.2, -0.1, 0.3, 0.5])
    actual_returns = pd.Series([0.02, 0.01, 0.01, 0.02, 0.03])

    result = unprofit_shutdown_buy_hold(predictions, actual_returns)
    expected_output = torch.tensor([1., 1., 1., 0., 1.])
    assert torch.all(result.eq(expected_output)), f"Expected {expected_output}, but got {result}"


def test_evaluate_buy_hold_strategy():
    predictions = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5])
    actual_returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.04])

    strategy_signals = buy_hold_strategy(predictions)
    total_return, sharpe_ratio = evaluate_strategy(strategy_signals, actual_returns)

    # Manual calculation
    expected_gains = [1.02 - (2 * CRYPTO_TRADING_FEE),
                      1.00,  # No trade
                      1.03 - (2 * CRYPTO_TRADING_FEE),
                      1.00,  # No trade
                      1.04 - (2 * CRYPTO_TRADING_FEE)]
    actual_gain = 1
    for gain in expected_gains:
        actual_gain *= gain
    actual_gain -= 1

    assert pytest.approx(total_return, rel=1e-4) == actual_gain, \
        f"Expected total return {actual_gain}, but got {total_return}"
    assert sharpe_ratio > 0, f"Sharpe ratio {sharpe_ratio} is not positive"


def test_evaluate_unprofit_shutdown_buy_hold():
    predictions = torch.tensor([0.1, 0.2, 0.1, 0.3, 0.5])
    actual_returns = pd.Series([0.02, 0.01, -0.01, 0.02, 0.03])

    strategy_signals = unprofit_shutdown_buy_hold(predictions, actual_returns)
    total_return, sharpe_ratio = evaluate_strategy(strategy_signals, actual_returns)

    # Manual calculation
    expected_gains = [1.02 - (2 * CRYPTO_TRADING_FEE),
                      1.01, #- (2 * CRYPTO_TRADING_FEE),
                      0.99, #- (2 * CRYPTO_TRADING_FEE),
                      1.00,  # No trade after shutdown
                      1.03 - (2 * CRYPTO_TRADING_FEE)
                      ]
    actual_gain = 1
    for gain in expected_gains:
        actual_gain *= gain
    actual_gain -= 1

    assert pytest.approx(total_return, rel=1e-4) == actual_gain, \
        f"Expected total return {actual_gain}, but got {total_return}"
    assert sharpe_ratio > 0, f"Sharpe ratio {sharpe_ratio} is not positive"


@patch('backtest_test3_inline.download_daily_stock_data')
@patch('backtest_test3_inline.ChronosPipeline.from_pretrained')
def test_backtest_forecasts_with_unprofit_shutdown(mock_pipeline_class, mock_download_data, mock_stock_data,
                                                   mock_pipeline):
    mock_download_data.return_value = mock_stock_data
    mock_pipeline_class.return_value = mock_pipeline

    symbol = 'MOCKSYMBOL'
    num_simulations = 5
    results = backtest_forecasts(symbol, num_simulations)

    # Assertions
    assert 'unprofit_shutdown_return' in results.columns
    assert 'unprofit_shutdown_sharpe' in results.columns
    assert 'unprofit_shutdown_finalday' in results.columns

    for i in range(num_simulations):
        simulation_data = mock_stock_data.iloc[:-(i + 1)].copy()
        actual_returns = simulation_data['Close'].pct_change().iloc[-7:]

        # Calculate expected unprofit shutdown return
        signals = [1]
        for j in range(1, len(actual_returns)):
            if actual_returns.iloc[j - 1] <= 0:
                signals.extend([0] * (len(actual_returns) - j))
                break
            signals.append(1)

        signals = np.array(signals)
        strategy_returns = signals * actual_returns.values - (
                np.abs(np.diff(np.concatenate(([0], signals)))) * (2 * CRYPTO_TRADING_FEE * ETH_SPREAD))
        expected_unprofit_shutdown_return = (1 + pd.Series(strategy_returns)).prod() - 1

        assert pytest.approx(results['unprofit_shutdown_return'].iloc[i],
                             rel=1e-4) == expected_unprofit_shutdown_return, \
            f"Expected unprofit shutdown return {expected_unprofit_shutdown_return}, but got {results['unprofit_shutdown_return'].iloc[i]}"

        # Check final day return
        expected_final_day_return = signals[-1] * actual_returns.iloc[-1] - (
            2 * CRYPTO_TRADING_FEE * ETH_SPREAD if signals[-1] != 0 else 0)
        assert pytest.approx(results['unprofit_shutdown_finalday'].iloc[i], rel=1e-4) == expected_final_day_return, \
            f"Expected final day return {expected_final_day_return}, but got {results['unprofit_shutdown_finalday'].iloc[i]}"
