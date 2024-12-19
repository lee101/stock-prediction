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
                'avg_return': 0.03,
                'side': 'buy',
                'strategy': 'simple',
                'predicted_movement': 0.02,
                'predictions': pd.DataFrame()
            }
        }
    }

@patch('trade_stock_e2e.backtest_forecasts')
def test_analyze_symbols(mock_backtest, test_data):
    mock_df = pd.DataFrame({
        'simple_strategy_return': [0.02],
        'all_signals_strategy_return': [0.01],
        'entry_takeprofit_return': [0.005],
        'predicted_close': [105],
        'predicted_high': [106],
        'predicted_low': [104],
        'close': [100]
    })
    mock_backtest.return_value = mock_df
    
    results = analyze_symbols(test_data['symbols'])
    
    assert isinstance(results, dict)
    assert len(results) > 0
    first_symbol = list(results.keys())[0]
    assert 'avg_return' in results[first_symbol]
    assert 'side' in results[first_symbol]
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

@patch('trade_stock_e2e.backout_near_market')
@patch('trade_stock_e2e.alpaca_wrapper.get_all_positions')
@patch('trade_stock_e2e.logger')
def test_manage_positions_only_closes_on_opposite_forecast(mock_logger, mock_get_positions, mock_backout, test_data):
    """Test that positions are only closed when there's an opposite forecast."""
    
    # Setup test positions
    mock_positions = [
        MagicMock(symbol='AAPL', side='buy'),   # Should stay open - no forecast
        MagicMock(symbol='MSFT', side='buy'),   # Should stay open - matching forecast
        MagicMock(symbol='GOOG', side='buy'),   # Should close - opposite forecast
        MagicMock(symbol='TSLA', side='sell'),  # Should stay open - matching forecast
    ]
    mock_get_positions.return_value = mock_positions
    
    # Setup analysis results
    all_analyzed_results = {
        'MSFT': {
            'side': 'buy',
            'sharpe': 1.5,
            'avg_return': 0.05,
            'predicted_movement': 0.02,
            'predictions': pd.DataFrame(),
            'strategy': 'simple'
        },
        'GOOG': {
            'side': 'sell',
            'sharpe': 1.2,
            'avg_return': 0.01,
            'predicted_movement': -0.02,
            'predictions': pd.DataFrame(),
            'strategy': 'simple'
        },
        'TSLA': {
            'side': 'sell',
            'sharpe': 1.1,
            'avg_return': 0.02,
            'predicted_movement': -0.01,
            'predictions': pd.DataFrame(),
            'strategy': 'simple'
        }
    }
    
    current_picks = {k: v for k, v in all_analyzed_results.items() if v['sharpe'] > 0}
    
    from trade_stock_e2e import manage_positions
    manage_positions(current_picks, {}, all_analyzed_results)
    
    # Verify that backout was only called once for GOOG
    assert mock_backout.call_count == 1
    mock_backout.assert_called_once_with('GOOG')
    
    # Verify appropriate log messages
    log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
    warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
    
    assert any('Keeping MSFT position as forecast matches current buy direction' in call for call in log_calls)
    assert any('Keeping TSLA position as forecast matches current sell direction' in call for call in log_calls)
    assert any('No analysis data for AAPL - keeping position' in call for call in warning_calls)
    assert any('Closing position for GOOG due to direction change from buy to sell' in call for call in log_calls)

@patch('trade_stock_e2e.backtest_forecasts')
def test_analyze_symbols_strategy_selection(mock_backtest):
    """Test that analyze_symbols correctly selects and applies strategies based on performance."""
    
    # Test data with different strategy returns
    test_cases = [
        # Case 1: Simple strategy performs better
        {
            'simple_strategy_return': [0.06],  # single row
            'all_signals_strategy_return': [0.03],
            'entry_takeprofit_return': [0.01],
            'close': [100],
            'predicted_close': [105],
            'predicted_high': [106],
            'predicted_low': [104],
            'expected_strategy': 'simple'
        },
        # Case 2: All signals strategy performs better
        {
            'simple_strategy_return': [0.02],
            'all_signals_strategy_return': [0.06],
            'entry_takeprofit_return': [0.03],
            'close': [100],
            'predicted_close': [105],
            'predicted_high': [106],
            'predicted_low': [104],
            'expected_strategy': 'all_signals'
        },
        # Case 3: All signals but signals don't align
        {
            'simple_strategy_return': [0.02],
            'all_signals_strategy_return': [0.05],
            'entry_takeprofit_return': [0.01],
            'close': [100],
            'predicted_close': [105],   # Up
            'predicted_high': [99],     # Down
            'predicted_low': [104],     # Up
            'expected_strategy': None
        }
    ]
    
    symbols = ['TEST1', 'TEST2', 'TEST3']
    
    for symbol, test_case in zip(symbols, test_cases):
        mock_backtest.return_value = pd.DataFrame(test_case)
        
        results = analyze_symbols([symbol])
        
        if test_case['expected_strategy'] is None:
            assert symbol not in results
            continue
            
        assert symbol in results
        result = results[symbol]
        
        assert result['strategy'] == test_case['expected_strategy']
        
        if test_case['expected_strategy'] == 'simple':
            # For simple strategy, verify position based on close price only
            expected_side = 'buy' if test_case['predicted_close'] > test_case['close'] else 'sell'
            assert result['side'] == expected_side
            
        elif test_case['expected_strategy'] == 'all_signals':
            # For all signals strategy, verify all signals were considered
            movements = [
                test_case['predicted_close'] - test_case['close'],
                test_case['predicted_high'] - test_case['close'],
                test_case['predicted_low'] - test_case['close']
            ]
            if all(x > 0 for x in movements):
                assert result['side'] == 'buy'
            elif all(x < 0 for x in movements):
                assert result['side'] == 'sell'
            
        assert 'avg_return' in result
        assert 'predicted_movement' in result
        assert 'predictions' in result


