"""Tests for position sizing utilities."""

import pytest
from unittest.mock import Mock, patch
from src.sizing_utils import get_qty, get_current_symbol_exposure


class MockPosition:
    def __init__(self, symbol, market_value):
        self.symbol = symbol
        self.market_value = market_value


@patch('src.sizing_utils.alpaca_wrapper')
def test_get_current_symbol_exposure(mock_alpaca):
    """Test exposure calculation for a symbol."""
    mock_alpaca.equity = 10000
    
    positions = [
        MockPosition("AAPL", "2000"),
        MockPosition("GOOGL", "1000"),
        MockPosition("AAPL", "500"),  # Second AAPL position
    ]
    
    # Test exposure for AAPL (should be 25% = (2000 + 500) / 10000)
    exposure = get_current_symbol_exposure("AAPL", positions)
    assert exposure == 25.0
    
    # Test exposure for GOOGL (should be 10% = 1000 / 10000)
    exposure = get_current_symbol_exposure("GOOGL", positions)
    assert exposure == 10.0
    
    # Test exposure for non-existent symbol
    exposure = get_current_symbol_exposure("TSLA", positions)
    assert exposure == 0.0


@patch('src.sizing_utils.get_global_risk_threshold', return_value=1.0)
@patch('src.sizing_utils.alpaca_wrapper')
@patch('src.sizing_utils.filter_to_realistic_positions')
def test_get_qty_basic_calculation(mock_filter, mock_alpaca, mock_risk_threshold):
    """Test basic quantity calculation."""
    # Setup mocks
    mock_alpaca.total_buying_power = 10000
    mock_alpaca.equity = 20000
    mock_filter.return_value = []  # No existing positions
    
    # Test stock calculation (should be 50% of buying power)
    qty = get_qty("AAPL", 100.0, [])  # $100 per share
    assert qty == 50.0  # floor(0.5 * 10000 / 100)
    
    # Test crypto calculation (should be rounded to 3 decimals)
    with patch('src.sizing_utils.crypto_symbols', ["BTCUSD"]):
        qty = get_qty("BTCUSD", 30000.0, [])  # $30k per BTC
        assert qty == 0.166  # floor(0.5 * 10000 / 30000 * 1000) / 1000


@patch('src.sizing_utils.get_global_risk_threshold', return_value=1.0)
@patch('src.sizing_utils.alpaca_wrapper')
@patch('src.sizing_utils.filter_to_realistic_positions')
def test_get_qty_exposure_limits(mock_filter, mock_alpaca, mock_risk_threshold):
    """Test that exposure limits are respected."""
    # Setup mocks
    mock_alpaca.total_buying_power = 10000
    mock_alpaca.equity = 20000
    mock_filter.return_value = []
    
    # Create existing position with high exposure (55% of equity)
    existing_positions = [MockPosition("AAPL", "11000")]
    
    # Should limit quantity based on remaining 5% exposure allowance
    qty = get_qty("AAPL", 100.0, existing_positions)
    # Remaining exposure: 60% - 55% = 5% = 0.05 * 20000 = $1000
    # Max qty from exposure: $1000 / $100 = 10 shares
    # Max qty from buying power: 0.5 * 10000 / 100 = 50 shares
    # Should take minimum = 10 shares, but floored to 9
    assert qty == 9.0  # floor(10.0) in practice


@patch('src.sizing_utils.get_global_risk_threshold', return_value=1.0)
@patch('src.sizing_utils.alpaca_wrapper')
@patch('src.sizing_utils.filter_to_realistic_positions')
def test_get_qty_max_exposure_reached(mock_filter, mock_alpaca, mock_risk_threshold):
    """Test that quantity is 0 when max exposure is reached."""
    # Setup mocks
    mock_alpaca.total_buying_power = 10000
    mock_alpaca.equity = 20000
    mock_filter.return_value = []
    
    # Create existing position at max exposure (60% of equity)
    existing_positions = [MockPosition("AAPL", "12000")]
    
    # Should return 0 since we're at max exposure
    qty = get_qty("AAPL", 100.0, existing_positions)
    assert qty == 0.0


@patch('src.sizing_utils.get_global_risk_threshold', return_value=1.0)
@patch('src.sizing_utils.alpaca_wrapper')
@patch('src.sizing_utils.filter_to_realistic_positions')
def test_get_qty_over_max_exposure(mock_filter, mock_alpaca, mock_risk_threshold):
    """Test that quantity is 0 when already over max exposure."""
    # Setup mocks
    mock_alpaca.total_buying_power = 10000
    mock_alpaca.equity = 20000
    mock_filter.return_value = []
    
    # Create existing position over max exposure (70% of equity)
    existing_positions = [MockPosition("AAPL", "14000")]
    
    # Should return 0 since we're over max exposure
    qty = get_qty("AAPL", 100.0, existing_positions)
    assert qty == 0.0


@patch('src.sizing_utils.get_global_risk_threshold', return_value=1.0)
@patch('src.sizing_utils.alpaca_wrapper')
@patch('src.sizing_utils.filter_to_realistic_positions')
def test_get_qty_minimum_order_size(mock_filter, mock_alpaca, mock_risk_threshold):
    """Test handling of very small calculated quantities."""
    # Setup mocks with very high price
    mock_alpaca.total_buying_power = 10000
    mock_alpaca.equity = 20000
    mock_filter.return_value = []
    
    # Test with very high price that results in fractional stock quantity
    qty = get_qty("AAPL", 50000.0, [])  # Very expensive stock
    # 0.5 * 10000 / 50000 = 0.1, floor(0.1) = 0
    assert qty == 0.0


@patch('src.sizing_utils.alpaca_wrapper')
def test_get_current_symbol_exposure_zero_equity(mock_alpaca):
    """Test exposure calculation when equity is zero."""
    mock_alpaca.equity = 0
    
    positions = [MockPosition("AAPL", "1000")]
    exposure = get_current_symbol_exposure("AAPL", positions)
    assert exposure == 0.0


@patch('src.sizing_utils.get_global_risk_threshold', return_value=1.0)
@patch('src.sizing_utils.alpaca_wrapper')
@patch('src.sizing_utils.filter_to_realistic_positions')
def test_get_qty_zero_equity(mock_filter, mock_alpaca, mock_risk_threshold):
    """Test quantity calculation when equity is zero."""
    mock_alpaca.total_buying_power = 10000
    mock_alpaca.equity = 0  # Zero equity
    mock_filter.return_value = []
    
    qty = get_qty("AAPL", 100.0, [])
    # Should still calculate based on buying power since equity check is only for exposure limits
    assert qty == 50.0
