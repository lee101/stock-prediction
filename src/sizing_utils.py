"""Position sizing utilities for trading operations."""

import os
import numpy as np
from collections.abc import Sequence
from math import floor
from typing import Any, Optional, Dict

from src.fixtures import crypto_symbols
from src.logging_utils import setup_logging, get_log_filename
from src.portfolio_risk import get_global_risk_threshold
from src.trading_obj_utils import filter_to_realistic_positions

# Detect if we're in hourly mode based on TRADE_STATE_SUFFIX env var
_is_hourly = os.getenv("TRADE_STATE_SUFFIX", "") == "hourly"
logger = setup_logging(get_log_filename("sizing_utils.log", is_hourly=_is_hourly))

PositionLike = Any
MAX_SYMBOL_EXPOSURE_PCT = 60.0

# Enhanced sizing configuration
USE_ENHANCED_KELLY_SIZING = os.getenv("USE_ENHANCED_KELLY_SIZING", "true").lower() == "true"
MAX_INTRADAY_LEVERAGE_STOCKS = float(os.getenv("MAX_INTRADAY_LEVERAGE", "1.0"))
MAX_OVERNIGHT_LEVERAGE_STOCKS = float(os.getenv("MAX_OVERNIGHT_LEVERAGE", "1.0"))

# Lazy-load enhanced sizing components
_kelly_strategy = None
_corr_data = None

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


def _load_kelly_strategy():
    """Lazy-load Kelly strategy."""
    global _kelly_strategy
    if _kelly_strategy is not None:
        return _kelly_strategy

    try:
        from marketsimulator.sizing_strategies import KellyStrategy
        _kelly_strategy = KellyStrategy(fraction=0.5, cap=1.0)
        logger.info("Loaded Kelly_50pct strategy for enhanced sizing")
        return _kelly_strategy
    except Exception as e:
        logger.warning(f"Could not load Kelly strategy: {e}")
        return None


def _load_correlation_data():
    """Lazy-load correlation matrix."""
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
            total_exposure += abs(market_value)  # Use abs to account for short positions

    return (total_exposure / equity) * 100 if equity > 0 else 0


def get_qty(symbol: str, entry_price: float, positions: Optional[Sequence[PositionLike]] = None,
            predicted_return: Optional[float] = None, predicted_volatility: Optional[float] = None) -> float:
    """
    Calculate quantity with enhanced Kelly_50pct @ 4x strategy.

    Args:
        symbol: Trading symbol
        entry_price: Price per unit for entry
        positions: Current positions (if None, will fetch from alpaca_wrapper)
        predicted_return: Optional predicted return for Kelly calculation
        predicted_volatility: Optional predicted volatility for Kelly calculation

    Returns:
        Quantity to trade (0 if exposure limits reached)
    """
    # Get current positions
    if positions is None:
        raw_positions = alpaca_wrapper.get_all_positions()
        positions = list(filter_to_realistic_positions(raw_positions))

    # Check current exposure
    current_exposure_pct = get_current_symbol_exposure(symbol, positions)
    max_exposure_pct = MAX_SYMBOL_EXPOSURE_PCT

    if current_exposure_pct >= max_exposure_pct:
        logger.warning(f"{symbol} at {current_exposure_pct:.1f}% exposure (max {max_exposure_pct}%). Skipping.")
        return 0

    # Get equity and buying power
    equity = float(getattr(alpaca_wrapper, "equity", 0.0) or 0.0)
    buying_power = float(getattr(alpaca_wrapper, "total_buying_power", 0.0) or 0.0)
    is_crypto = symbol in crypto_symbols

    # Try enhanced Kelly sizing if enabled
    if USE_ENHANCED_KELLY_SIZING:
        kelly_strategy = _load_kelly_strategy()
        if kelly_strategy is not None:
            try:
                # Estimate volatility and return if not provided
                if predicted_return is None or predicted_volatility is None:
                    corr_data = _load_correlation_data()
                    if corr_data and symbol in corr_data.get('volatility_metrics', {}):
                        vol_metrics = corr_data['volatility_metrics'][symbol]
                        if predicted_volatility is None:
                            predicted_volatility = vol_metrics['annualized_volatility'] / np.sqrt(252)
                        if predicted_return is None:
                            sharpe = vol_metrics.get('sharpe_ratio', 0.5)
                            predicted_return = sharpe * predicted_volatility
                    else:
                        if predicted_volatility is None:
                            predicted_volatility = 0.50 / np.sqrt(252) if is_crypto else 0.25 / np.sqrt(252)
                        if predicted_return is None:
                            predicted_return = 0.005

                # Build market context
                from marketsimulator.sizing_strategies import MarketContext
                existing_position_value = sum(
                    abs(float(getattr(p, "market_value", 0)))
                    for p in positions
                    if getattr(p, "symbol", "") == symbol
                )

                context = MarketContext(
                    symbol=symbol,
                    predicted_return=abs(predicted_return),
                    predicted_volatility=predicted_volatility,
                    current_price=entry_price,
                    equity=equity,
                    is_crypto=is_crypto,
                    existing_position_value=existing_position_value,
                )

                # Calculate Kelly sizing
                sizing_result = kelly_strategy.calculate_size(context)
                base_fraction = sizing_result.position_fraction

                # Apply leverage for stocks
                if is_crypto:
                    target_fraction = max(base_fraction, 0)  # Long only
                    leverage_note = "no leverage (crypto)"
                else:
                    target_fraction = base_fraction * MAX_INTRADAY_LEVERAGE_STOCKS
                    leverage_note = f"{MAX_INTRADAY_LEVERAGE_STOCKS}x leverage"

                # Calculate target qty
                target_value = target_fraction * equity
                qty = target_value / entry_price if entry_price > 0 else 0

                # Apply exposure limits
                current_symbol_value = existing_position_value
                if equity > 0:
                    max_symbol_value = (max_exposure_pct / 100) * equity
                    remaining_value = max(max_symbol_value - current_symbol_value, 0.0)
                    if not is_crypto:
                        max_additional_value = min(remaining_value * MAX_INTRADAY_LEVERAGE_STOCKS, buying_power)
                    else:
                        max_additional_value = remaining_value
                    qty_from_exposure_limit = max_additional_value / entry_price if entry_price > 0 else 0
                    qty = min(qty, qty_from_exposure_limit)

                # Round appropriately
                if is_crypto:
                    qty = floor(qty * 1000) / 1000.0
                else:
                    qty = floor(qty)

                if qty > 0:
                    future_value = current_symbol_value + (qty * entry_price)
                    future_pct = (future_value / equity) * 100 if equity > 0 else 0
                    logger.info(
                        f"{symbol}: Enhanced Kelly sizing with {leverage_note} - "
                        f"base_fraction={base_fraction:.3f}, qty={qty:.4f}, "
                        f"exposure: {current_exposure_pct:.1f}% → {future_pct:.1f}%"
                    )
                    return qty
                else:
                    logger.debug(f"{symbol}: Enhanced Kelly sizing resulted in qty={qty}")
                    return 0

            except Exception as e:
                logger.warning(f"{symbol}: Enhanced sizing failed ({e}), falling back to legacy")

    # Fallback to legacy sizing
    risk_multiplier = max(get_global_risk_threshold(), 1.0)
    if is_crypto:
        risk_multiplier = 1.0

    qty_from_buying_power = 0.50 * buying_power * risk_multiplier / entry_price

    current_symbol_value = sum(
        abs(float(getattr(p, "market_value", 0)))
        for p in positions
        if getattr(p, "symbol", "") == symbol
    )

    if equity > 0:
        max_symbol_value = (max_exposure_pct / 100) * equity
        remaining_value = max(max_symbol_value - current_symbol_value - 1e-9, 0.0)
        leverage_cap = max(risk_multiplier, 1.0) if not is_crypto else 1.0
        max_additional_value = remaining_value * leverage_cap
        qty_from_exposure_limit = max_additional_value / entry_price if entry_price > 0 else 0.0
        qty = min(qty_from_buying_power, qty_from_exposure_limit)
    else:
        qty = qty_from_buying_power

    # Round
    if is_crypto:
        qty = floor(qty * 1000) / 1000.0
    else:
        qty = floor(qty)

    if qty <= 0:
        logger.warning(f"{symbol}: Legacy sizing gave qty={qty} (exposure: {current_exposure_pct:.1f}%)")
        return 0

    future_exposure_value = current_symbol_value + (qty * entry_price)
    future_exposure_pct = (future_exposure_value / equity) * 100 if equity > 0 else 0

    logger.debug(
        f"{symbol}: Legacy sizing - current={current_exposure_pct:.1f}%, "
        f"new={future_exposure_pct:.1f}%, risk_multiplier={risk_multiplier:.2f}"
    )

    return qty
