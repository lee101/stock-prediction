"""Position sizing utilities for trading operations."""

from math import floor
from typing import List, Optional

import alpaca_wrapper
from src.fixtures import crypto_symbols
from src.logging_utils import setup_logging
from src.trading_obj_utils import filter_to_realistic_positions

logger = setup_logging("sizing_utils.log")


def get_current_symbol_exposure(symbol: str, positions: List) -> float:
    """Calculate current exposure to a symbol as percentage of total equity."""
    total_exposure = 0
    equity = alpaca_wrapper.equity
    
    for position in positions:
        if position.symbol == symbol:
            market_value = float(position.market_value) if position.market_value else 0
            total_exposure += abs(market_value)  # Use abs to account for short positions
    
    return (total_exposure / equity) * 100 if equity > 0 else 0


def get_qty(symbol: str, entry_price: float, positions: Optional[List] = None) -> float:
    """
    Calculate quantity with 60% max exposure check per symbol.
    
    Args:
        symbol: Trading symbol
        entry_price: Price per unit for entry
        positions: Current positions (if None, will fetch from alpaca_wrapper)
        
    Returns:
        Quantity to trade (0 if exposure limits reached)
    """
    # Get current positions to check existing exposure if not provided
    if positions is None:
        positions = alpaca_wrapper.get_all_positions()
        positions = filter_to_realistic_positions(positions)
    
    # Check current exposure to this symbol
    current_exposure_pct = get_current_symbol_exposure(symbol, positions)
    
    # Maximum allowed exposure is 60%
    max_exposure_pct = 60.0
    
    if current_exposure_pct >= max_exposure_pct:
        logger.warning(f"Symbol {symbol} already at {current_exposure_pct:.1f}% exposure, max is {max_exposure_pct}%. Skipping position increase.")
        return 0
    
    # Calculate how much more we can add without exceeding 60%
    remaining_exposure_pct = max_exposure_pct - current_exposure_pct
    
    # Calculate qty as 50% of available buying power, but limit by remaining exposure
    buying_power = alpaca_wrapper.total_buying_power
    equity = alpaca_wrapper.equity
    
    # Calculate qty based on 50% of buying power
    qty_from_buying_power = 0.50 * buying_power / entry_price
    
    # Calculate max qty based on remaining exposure allowance (only if equity > 0)
    if equity > 0:
        max_additional_value = (remaining_exposure_pct / 100) * equity
        qty_from_exposure_limit = max_additional_value / entry_price
        # Use the smaller of the two
        qty = min(qty_from_buying_power, qty_from_exposure_limit)
    else:
        # If equity is 0 or negative, just use buying power
        qty = qty_from_buying_power
    
    # Round down to 3 decimal places for crypto
    if symbol in crypto_symbols:
        qty = floor(qty * 1000) / 1000.0
    else:
        # Round down to whole number for stocks
        qty = floor(qty)
    
    # Ensure qty is valid
    if qty <= 0:
        logger.warning(f"Calculated qty {qty} is invalid for {symbol} (current exposure: {current_exposure_pct:.1f}%)")
        return 0
    
    # Log the exposure calculation
    future_exposure_value = sum(abs(float(p.market_value)) for p in positions if p.symbol == symbol) + (qty * entry_price)
    future_exposure_pct = (future_exposure_value / equity) * 100 if equity > 0 else 0
    
    logger.info(f"Position sizing for {symbol}: current={current_exposure_pct:.1f}%, new position will be {future_exposure_pct:.1f}% of total equity")
    
    return qty