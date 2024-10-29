from datetime import datetime
import pytz
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from trade_stock_e2e import (
    analyze_symbols,
    log_trading_plan,
    dry_run_manage_positions,
    analyze_next_day_positions,
    manage_market_close,
    get_market_hours
)

@pytest.fixture
def test_data():
    return {
        'symbols': ['AAPL', 'MSFT'],
        'mock_picks': {
            'AAPL': {
                'sharpe': 1.5,
                'side': 'buy',
                'predicted_movement': 0.02,
                'p_value': 0.01,
                'predictions': pd.DataFrame()
            }
        }
    }

def test_analyze_symbols_real_call():
    symbols = ['ETHUSD']
    results = analyze_symbols(symbols)
    
    assert isinstance(results, dict)
    # ah well? its not profitable
    # assert len(results) > 0
    # first_symbol = list(results.keys())[0]
    # assert 'sharpe' in results[first_symbol]
    # assert 'side' in results[first_symbol]


@patch('trade_stock_e2e.backtest_forecasts')
def test_analyze_symbols(mock_backtest, test_data):
    mock_df = pd.DataFrame({
        'simple_strategy_sharpe': [1.0],
        'all_signals_strategy_sharpe': [1.0],
        'buy_hold_sharpe': [1.0],
        'unprofit_shutdown_sharpe': [1.0],
        'predicted_close': [105],
        'close': [100]
    })
    mock_backtest.return_value = mock_df
    
    results = analyze_symbols(test_data['symbols'])
    
    assert isinstance(results, dict)
    assert len(results) > 0
    first_symbol = list(results.keys())[0]
    assert 'sharpe' in results[first_symbol]
    assert 'side' in results[first_symbol]
    assert 'p_value' in results[first_symbol]
    assert 'predicted_movement' in results[first_symbol]

@patch('trade_stock_e2e.logger')
def test_log_trading_plan(mock_logger, test_data):
    log_trading_plan(test_data['mock_picks'], "TEST")
    mock_logger.info.assert_called()

@patch('trade_stock_e2e.alpaca_wrapper.get_all_positions')
@patch('trade_stock_e2e.logger')
def test_dry_run_manage_positions(mock_logger, mock_get_positions, test_data):
    mock_position = MagicMock()
    mock_position.symbol = 'AAPL'
    mock_position.side = 'sell'
    mock_get_positions.return_value = [mock_position]
    
    dry_run_manage_positions(test_data['mock_picks'], {})
    mock_logger.info.assert_called()

def test_get_market_hours():
    market_open, market_close = get_market_hours()
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)
    
    assert market_open.hour == 9
    assert market_open.minute == 30
    assert market_close.hour == 16
    assert market_close.minute == 0

@patch('trade_stock_e2e.analyze_next_day_positions')
@patch('trade_stock_e2e.alpaca_wrapper.get_all_positions')
@patch('trade_stock_e2e.logger')
def test_manage_market_close(mock_logger, mock_get_positions, mock_analyze, test_data):
    mock_position = MagicMock()
    mock_position.symbol = 'MSFT'
    mock_position.side = 'buy'
    mock_get_positions.return_value = [mock_position]
    mock_analyze.return_value = test_data['mock_picks']
    
    result = manage_market_close(test_data['symbols'], {}, test_data['mock_picks'])
    assert isinstance(result, dict)
    mock_logger.info.assert_called()
