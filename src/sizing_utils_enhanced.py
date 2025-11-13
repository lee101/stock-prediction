"""
Enhanced position sizing with Kelly_50pct @ 4x leverage strategy.

Implements the winning strategy from comprehensive testing:
- Crypto (BTCUSD, ETHUSD): Long only, no leverage
- Stocks: Up to 4x intraday leverage, 2x overnight max
- Uses Kelly 50% criterion with volatility adjustment
- Accounts for correlation and risk management
"""

import os
import numpy as np
from collections.abc import Sequence
from math import floor
from typing import Any, Optional, Dict
from pathlib import Path

from src.fixtures import crypto_symbols
from src.logging_utils import setup_logging, get_log_filename
from src.portfolio_risk import get_global_risk_threshold
from src.trading_obj_utils import filter_to_realistic_positions
from marketsimulator.sizing_strategies import (
    KellyStrategy,
    MarketContext,
)

# Detect if we're in hourly mode
_is_hourly = os.getenv("TRADE_STATE_SUFFIX", "") == "hourly"
logger = setup_logging(get_log_filename("sizing_utils.log", is_hourly=_is_hourly))

PositionLike = Any
MAX_SYMBOL_EXPOSURE_PCT = 60.0

# Leverage constraints
MAX_INTRADAY_LEVERAGE_STOCKS = 4.0
MAX_OVERNIGHT_LEVERAGE_STOCKS = 2.0
ANNUAL_INTEREST_RATE = 0.065  # 6.5%

# Initialize Kelly 50% strategy
kelly_strategy = KellyStrategy(fraction=0.5, cap=1.0)

# Lazy-load correlation data
_corr_data = None


class _SimAlpacaWrapper:
    """Fallback context for sizing calculations without live Alpaca."""
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
        "Falling back to offline sizing: %s", exc
    )
    alpaca_wrapper = _SimAlpacaWrapper()  # type: ignore
    _HAS_ALPACA = False


def _load_correlation_data() -> Optional[Dict]:
    """Lazy-load correlation matrix data."""
    global _corr_data
    if _corr_data is not None:
        return _corr_data

    try:
        from trainingdata.load_correlation_utils import load_correlation_matrix
        _corr_data = load_correlation_matrix()
        logger.info(f"Loaded correlation matrix with {len(_corr_data['symbols'])} symbols")
        return _corr_data
    except Exception as e:
        logger.warning(f"Could not load correlation data: {e}")
        return None


def get_current_symbol_exposure(symbol: str, positions: Sequence[PositionLike]) -> float:
    """Calculate current exposure to a symbol as percentage of total equity."""
    total_exposure = 0.0
    equity = alpaca_wrapper.equity

    for position in positions:
        if position.symbol == symbol:
            market_value = float(position.market_value) if position.market_value else 0
            total_exposure += abs(market_value)

    return (total_exposure / equity) * 100 if equity > 0 else 0


def get_enhanced_qty(
    symbol: str,
    entry_price: float,
    predicted_return: Optional[float] = None,
    predicted_volatility: Optional[float] = None,
    positions: Optional[Sequence[PositionLike]] = None,
    is_crypto: Optional[bool] = None,
) -> float:
    """
    Calculate position size using Kelly_50pct @ 4x strategy.

    Args:
        symbol: Trading symbol
        entry_price: Entry price per unit
        predicted_return: Forecasted return (optional, will estimate if not provided)
        predicted_volatility: Forecasted volatility (optional, will use historical if not provided)
        positions: Current positions
        is_crypto: Whether this is a crypto symbol (will auto-detect if None)

    Returns:
        Quantity to trade
    """
    # Get current positions
    if positions is None:
        raw_positions = alpaca_wrapper.get_all_positions()
        positions = list(filter_to_realistic_positions(raw_positions))

    # Auto-detect crypto
    if is_crypto is None:
        is_crypto = symbol in crypto_symbols

    # Check current exposure
    current_exposure_pct = get_current_symbol_exposure(symbol, positions)
    if current_exposure_pct >= MAX_SYMBOL_EXPOSURE_PCT:
        logger.warning(
            f"{symbol} already at {current_exposure_pct:.1f}% exposure "
            f"(max {MAX_SYMBOL_EXPOSURE_PCT}%). Skipping."
        )
        return 0

    # Get equity and buying power
    equity = float(getattr(alpaca_wrapper, "equity", 0.0) or 0.0)
    buying_power = float(getattr(alpaca_wrapper, "total_buying_power", 0.0) or 0.0)

    if equity <= 0:
        logger.warning(f"Invalid equity {equity}, using buying power fallback")
        equity = buying_power

    # Estimate predicted return and volatility if not provided
    if predicted_return is None or predicted_volatility is None:
        corr_data = _load_correlation_data()
        if corr_data and symbol in corr_data.get('volatility_metrics', {}):
            vol_metrics = corr_data['volatility_metrics'][symbol]
            if predicted_volatility is None:
                # Use daily volatility
                predicted_volatility = vol_metrics['annualized_volatility'] / np.sqrt(252)
            if predicted_return is None:
                # Estimate from Sharpe ratio (conservative)
                sharpe = vol_metrics.get('sharpe_ratio', 0.5)
                predicted_return = sharpe * predicted_volatility
        else:
            # Fallback defaults
            if predicted_volatility is None:
                predicted_volatility = 0.50 / np.sqrt(252) if is_crypto else 0.25 / np.sqrt(252)
            if predicted_return is None:
                predicted_return = 0.005  # Conservative daily return estimate

    # Build market context
    existing_position_value = 0.0
    for p in positions:
        if p.symbol == symbol:
            existing_position_value += float(p.market_value) if p.market_value else 0

    context = MarketContext(
        symbol=symbol,
        predicted_return=abs(predicted_return),  # Kelly uses absolute value
        predicted_volatility=predicted_volatility,
        current_price=entry_price,
        equity=equity,
        is_crypto=is_crypto,
        existing_position_value=existing_position_value,
    )

    # Calculate Kelly sizing
    try:
        sizing_result = kelly_strategy.calculate_size(context)
        base_fraction = sizing_result.position_fraction
    except Exception as e:
        logger.warning(f"Kelly calculation failed for {symbol}: {e}, using fallback")
        base_fraction = 0.25

    # Apply leverage multiplier
    if is_crypto:
        # Crypto: No leverage, long only
        target_fraction = max(base_fraction, 0)  # Ensure non-negative
        leverage_note = "no leverage (crypto)"
    else:
        # Stocks: Apply 4x intraday leverage
        # For overnight, we'll cap at 2x when closing positions
        target_fraction = base_fraction * MAX_INTRADAY_LEVERAGE_STOCKS
        leverage_note = f"{MAX_INTRADAY_LEVERAGE_STOCKS}x leverage (stock)"

    # Calculate target quantity
    target_value = target_fraction * equity
    target_qty = target_value / entry_price if entry_price > 0 else 0

    # Check exposure limits
    current_symbol_value = sum(
        abs(float(getattr(p, "market_value", 0)))
        for p in positions
        if getattr(p, "symbol", "") == symbol
    )

    if equity > 0:
        max_symbol_value = (MAX_SYMBOL_EXPOSURE_PCT / 100) * equity
        remaining_value = max(max_symbol_value - current_symbol_value, 0.0)

        # For stocks with leverage, buying power can support larger positions
        if not is_crypto:
            # Use buying power to support leveraged positions
            max_additional_value = min(remaining_value * MAX_INTRADAY_LEVERAGE_STOCKS, buying_power)
        else:
            max_additional_value = remaining_value

        qty_from_exposure_limit = max_additional_value / entry_price if entry_price > 0 else 0
        target_qty = min(target_qty, qty_from_exposure_limit)

    # Round appropriately
    if is_crypto:
        target_qty = floor(target_qty * 1000) / 1000.0
    else:
        target_qty = floor(target_qty)

    # Validate
    if target_qty <= 0:
        logger.debug(
            f"{symbol}: Calculated qty {target_qty} is invalid "
            f"(exposure: {current_exposure_pct:.1f}%)"
        )
        return 0

    # Log sizing details
    future_exposure_value = current_symbol_value + (target_qty * entry_price)
    future_exposure_pct = (future_exposure_value / equity) * 100 if equity > 0 else 0

    logger.info(
        f"{symbol}: Kelly sizing with {leverage_note} - "
        f"base_fraction={base_fraction:.3f}, target_qty={target_qty:.4f}, "
        f"exposure: {current_exposure_pct:.1f}% → {future_exposure_pct:.1f}%"
    )

    return target_qty


def get_qty(symbol: str, entry_price: float, positions: Optional[Sequence[PositionLike]] = None) -> float:
    """
    Backward-compatible wrapper for get_enhanced_qty.

    This maintains the existing API while using the enhanced Kelly strategy.
    """
    return get_enhanced_qty(
        symbol=symbol,
        entry_price=entry_price,
        positions=positions,
    )


def calculate_overnight_deleveraging(
    positions: Sequence[PositionLike],
    target_leverage: float = MAX_OVERNIGHT_LEVERAGE_STOCKS,
) -> Dict[str, float]:
    """
    Calculate position reductions needed to meet overnight leverage limits.

    Args:
        positions: Current positions
        target_leverage: Target overnight leverage (default 2.0x)

    Returns:
        Dict mapping symbol -> scale_factor (1.0 = no change, <1.0 = reduce)
    """
    equity = float(getattr(alpaca_wrapper, "equity", 0.0) or 0.0)
    if equity <= 0:
        logger.warning("Cannot calculate deleveraging with zero equity")
        return {}

    # Calculate total stock position value
    stock_position_value = 0.0
    stock_positions = []

    for p in positions:
        if p.symbol not in crypto_symbols:
            position_value = abs(float(p.market_value) if p.market_value else 0)
            stock_position_value += position_value
            stock_positions.append((p.symbol, position_value))

    # Calculate current leverage
    current_leverage = stock_position_value / equity if equity > 0 else 0

    if current_leverage <= target_leverage:
        # No deleveraging needed
        return {sym: 1.0 for sym, _ in stock_positions}

    # Scale down all stock positions proportionally
    scale_factor = target_leverage / current_leverage

    logger.info(
        f"Overnight deleveraging: current={current_leverage:.2f}x, "
        f"target={target_leverage:.2f}x, scale={scale_factor:.3f}"
    )

    return {sym: scale_factor for sym, _ in stock_positions}
