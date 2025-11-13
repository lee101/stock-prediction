"""Position sizing utilities for trading operations."""

import os
from collections.abc import Sequence
from math import floor
from typing import Any, Optional

from src.fixtures import crypto_symbols
from src.logging_utils import setup_logging, get_log_filename
from src.portfolio_risk import get_global_risk_threshold
from src.trading_obj_utils import filter_to_realistic_positions

# Detect if we're in hourly mode based on TRADE_STATE_SUFFIX env var
_is_hourly = os.getenv("TRADE_STATE_SUFFIX", "") == "hourly"
logger = setup_logging(get_log_filename("sizing_utils.log", is_hourly=_is_hourly))

PositionLike = Any
MAX_SYMBOL_EXPOSURE_PCT = 60.0

class _SimAlpacaWrapper:
    """Fallback context to let sizing math run without live Alpaca access."""

    equity: float = 100000.0
    total_buying_power: float = 100000.0

    @staticmethod
    def get_all_positions():
        return []


try:
    import alpaca_wrapper  # type: ignore
    _HAS_ALPACA = True
except Exception as exc:
    logger.warning(
        "Falling back to offline sizing because Alpaca wrapper failed to import: %s",
        exc,
    )
    alpaca_wrapper = _SimAlpacaWrapper()  # type: ignore
    _HAS_ALPACA = False


def get_current_symbol_exposure(symbol: str, positions: Sequence[PositionLike]) -> float:
    """Calculate current exposure to a symbol as percentage of total equity."""
    total_exposure = 0.0
    equity = alpaca_wrapper.equity
    
    for position in positions:
        if position.symbol == symbol:
            market_value = float(position.market_value) if position.market_value else 0
            total_exposure += abs(market_value)  # Use abs to account for short positions
    
    return (total_exposure / equity) * 100 if equity > 0 else 0


def get_qty(symbol: str, entry_price: float, positions: Optional[Sequence[PositionLike]] = None) -> float:
    """
    Calculate quantity with a 50% max exposure check per symbol.
    
    Args:
        symbol: Trading symbol
        entry_price: Price per unit for entry
        positions: Current positions (if None, will fetch from alpaca_wrapper)
        
    Returns:
        Quantity to trade (0 if exposure limits reached)
    """
    # Get current positions to check existing exposure if not provided
    if positions is None:
        raw_positions = alpaca_wrapper.get_all_positions()
        positions = list(filter_to_realistic_positions(raw_positions))
    
    # Check current exposure to this symbol
    current_exposure_pct = get_current_symbol_exposure(symbol, positions)
    
    # Maximum allowed exposure is 50%
    max_exposure_pct = MAX_SYMBOL_EXPOSURE_PCT
    
    if current_exposure_pct >= max_exposure_pct:
        logger.warning(f"Symbol {symbol} already at {current_exposure_pct:.1f}% exposure, max is {max_exposure_pct}%. Skipping position increase.")
        return 0
    
    # Calculate qty as 50% of available buying power, but limit by remaining exposure
    buying_power = float(getattr(alpaca_wrapper, "total_buying_power", 0.0) or 0.0)
    equity = float(getattr(alpaca_wrapper, "equity", 0.0) or 0.0)
    risk_multiplier = max(get_global_risk_threshold(), 1.0)
    if symbol in crypto_symbols:
        risk_multiplier = 1.0
    
    # Calculate qty based on 50% of buying power and risk multiplier
    qty_from_buying_power = 0.50 * buying_power * risk_multiplier / entry_price
    
    # Calculate max qty based on remaining exposure allowance (only if equity > 0)
    current_symbol_value = sum(
        abs(float(getattr(p, "market_value", 0))) for p in positions if getattr(p, "symbol", "") == symbol
    )

    if equity > 0:
        max_symbol_value = (max_exposure_pct / 100) * equity
        remaining_value = max(max_symbol_value - current_symbol_value - 1e-9, 0.0)
        leverage_cap = max(risk_multiplier, 1.0)
        if symbol in crypto_symbols:
            leverage_cap = 1.0
        max_additional_value = remaining_value * leverage_cap
        qty_from_exposure_limit = max_additional_value / entry_price if entry_price > 0 else 0.0
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
    future_exposure_value = current_symbol_value + (qty * entry_price)
    future_exposure_pct = (future_exposure_value / equity) * 100 if equity > 0 else 0
    
    logger.debug(
        "Position sizing for %s: current=%.1f%%, new=%.1f%% of equity with risk multiplier %.2f",
        symbol,
        current_exposure_pct,
        future_exposure_pct,
        risk_multiplier,
    )
    
    return qty
