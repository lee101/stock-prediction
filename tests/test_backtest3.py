import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta

# Import the function to test
from backtest_test3_inline import backtest_forecasts, ChronosPipeline, simple_buy_sell_strategy, all_signals_strategy, evaluate_strategy, buy_hold_strategy, CRYPTO_TRADING_FEE

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
    assert 'simple_strategy_return' in results.columns
    assert 'all_signals_strategy_return' in results.columns
    assert 'buy_hold_return' in results.columns
    assert 'buy_hold_sharpe' in results.columns

    # Check if the buy and hold strategy is calculated correctly
    for i in range(num_simulations):
        actual_returns = mock_stock_data['Close'].pct_change().iloc[-(7+i):-i] if i > 0 else mock_stock_data['Close'].pct_change().iloc[-7:]
        expected_buy_hold_return = (1 + actual_returns).prod() - 1
        assert pytest.approx(results['buy_hold_return'].iloc[i], rel=1e-4) == expected_buy_hold_return

    # Add a check to ensure buy_hold_signals are all ones
    buy_hold_signals = buy_hold_strategy(torch.tensor(mock_pipeline.predict.return_value[0].numpy()))
    assert torch.all(buy_hold_signals == 1), "Buy-hold strategy should always return 1 (buy)"

    # Check if the pipeline was called the correct number of times
    expected_pipeline_calls = num_simulations * 4 * 7  # 4 price types, 7 days each
    assert mock_pipeline.predict.call_count == expected_pipeline_calls

def test_simple_buy_sell_strategy():
    predictions = torch.tensor([-0.1, 0.2, 0, -0.3, 0.5])
    expected_output = torch.tensor([-1., 1., -1., -1., 1.])
    assert torch.all(simple_buy_sell_strategy(predictions).eq(expected_output))

def test_all_signals_strategy():
    close_pred = torch.tensor([0.1, -0.2, 0.3, -0.4])
    high_pred = torch.tensor([0.2, -0.1, 0.4, -0.3])
    low_pred = torch.tensor([0.3, -0.3, 0.2, -0.2])
    open_pred = torch.tensor([0.4, -0.4, 0.1, -0.1])
    result = all_signals_strategy(close_pred, high_pred, low_pred, open_pred)
    
    # Calculate expected output based on the actual implementation
    buy_signal = (close_pred > 0) & (high_pred > 0) & (low_pred > 0) & (open_pred > 0)
    sell_signal = (close_pred < 0) & (high_pred < 0) & (low_pred < 0) & (open_pred < 0)
    expected_output = buy_signal.float() - sell_signal.float()
    
    assert torch.all(result.eq(expected_output)), f"Expected {expected_output}, but got {result}"

def test_evaluate_strategy_with_fees():
    strategy_signals = torch.tensor([1., 1., -1., -1., 1.])
    actual_returns = pd.Series([0.02, 0.01, -0.01, -0.02, 0.03])
    
    total_return, sharpe_ratio = evaluate_strategy(strategy_signals, actual_returns)
    
    # Calculate expected return manually
    expected_returns = [0.02, 0.01, -0.01, -0.02, 0.03]
    expected_fees = [CRYPTO_TRADING_FEE, 0, CRYPTO_TRADING_FEE, 0, CRYPTO_TRADING_FEE]
    expected_strategy_returns = [r - f for r, f in zip(expected_returns, expected_fees)]
    expected_total_return = (1 + pd.Series(expected_strategy_returns)).prod() - 1
    
    assert pytest.approx(total_return, rel=1e-4) == expected_total_return, f"Expected total return {expected_total_return}, but got {total_return}"
    assert sharpe_ratio > 0, f"Sharpe ratio {sharpe_ratio} is not positive"

def test_buy_hold_strategy():
    predictions = torch.tensor([-0.1, 0.2, 0, -0.3, 0.5])
    expected_output = torch.ones_like(predictions)
    result = buy_hold_strategy(predictions)
    assert torch.all(result.eq(expected_output)), f"Expected {expected_output}, but got {result}"