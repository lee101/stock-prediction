import json
import os
import re
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep

import cachetools
import math
import pandas as pd
import requests.exceptions
from alpaca.data import (
    StockBarsRequest,
    StockHistoricalDataClient,
    CryptoBarsRequest,
    CryptoHistoricalDataClient,
    CryptoLatestQuoteRequest,
    StockLatestQuoteRequest,
    TimeFrame,
    TimeFrameUnit,
)
from alpaca.data.enums import DataFeed
from alpaca.trading import OrderType, LimitOrderRequest, GetOrdersRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide
from alpaca.trading.requests import MarketOrderRequest
from alpaca_trade_api.rest import APIError
from loguru import logger
from retry import retry

from env_real import ALP_KEY_ID, ALP_SECRET_KEY, ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD, ALP_ENDPOINT, PAPER
from typing import Iterable, Dict, Any, List, Optional, Tuple
from types import SimpleNamespace
from src.comparisons import is_buy_side, is_sell_side
from src.crypto_loop import crypto_alpaca_looper_api
from src.fixtures import crypto_symbols, all_crypto_symbols
from src.logging_utils import setup_logging
from src.stock_utils import pairs_equal, remap_symbols
from src.trading_obj_utils import filter_to_realistic_positions

logger = setup_logging("alpaca_cli.log")


def _get_time_in_force_for_qty(qty: float) -> str:
    """
    Get appropriate time_in_force for Alpaca order based on quantity.

    Alpaca requires fractional orders to use time_in_force='day'.
    Whole number orders can use 'gtc' (good-til-cancelled).

    Args:
        qty: Order quantity

    Returns:
        'day' for fractional quantities, 'gtc' for whole numbers
    """
    try:
        is_fractional = float(qty) % 1 != 0
        return "day" if is_fractional else "gtc"
    except (TypeError, ValueError):
        # If we can't determine, default to 'day' (safer choice)
        logger.warning(f"Could not determine if qty={qty} is fractional, defaulting to day order")
        return "day"


# Market order spread threshold - don't use market orders if spread exceeds this
_MARKET_ORDER_SPREAD_CAP_RAW = os.getenv("MARKET_ORDER_MAX_SPREAD_PCT", "0.01")
try:
    MARKET_ORDER_MAX_SPREAD_PCT = max(float(_MARKET_ORDER_SPREAD_CAP_RAW), 0.0)
except ValueError:
    MARKET_ORDER_MAX_SPREAD_PCT = 0.01
    logger.warning(
        "Invalid MARKET_ORDER_MAX_SPREAD_PCT=%r; defaulting to %.2f%%",
        _MARKET_ORDER_SPREAD_CAP_RAW,
        MARKET_ORDER_MAX_SPREAD_PCT * 100,
    )

_PLACEHOLDER_TOKEN = "placeholder"

# Select credentials based on PAPER environment variable
_TRADING_KEY_ID = ALP_KEY_ID if PAPER else ALP_KEY_ID_PROD
_TRADING_SECRET_KEY = ALP_SECRET_KEY if PAPER else ALP_SECRET_KEY_PROD
_IS_PAPER = PAPER


def _missing_alpaca_credentials() -> bool:
    return (
        not _TRADING_KEY_ID
        or not _TRADING_SECRET_KEY
        or _PLACEHOLDER_TOKEN in _TRADING_KEY_ID
        or _PLACEHOLDER_TOKEN in _TRADING_SECRET_KEY
    )


def _is_unauthorized_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if "unauthorized" in message or "authentication" in message:
        return True
    status = getattr(exc, "status_code", None)
    if status == 401:
        return True
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            if getattr(response, "status_code", None) == 401:
                return True
        except Exception:
            pass
    return False


def _mock_clock() -> SimpleNamespace:
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        is_open=True,
        timestamp=now,
        next_open=now,
        next_close=now + timedelta(hours=6),
    )


alpaca_api = TradingClient(
    _TRADING_KEY_ID,
    _TRADING_SECRET_KEY,
    paper=_IS_PAPER,
)
logger.info(f"Initialized Alpaca Trading Client: {'PAPER' if _IS_PAPER else 'LIVE'} account")

data_client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)

TRAININGDATA_BASE_PATH = Path(__file__).resolve().parent / "trainingdata"
DEFAULT_HISTORY_DAYS = 365 * 4
DEFAULT_TEST_DAYS = 30
DEFAULT_SKIP_IF_RECENT_DAYS = 7

EXTENDED_CRYPTO_SYMBOLS: List[str] = [
    'ADAUSD', 'ALGOUSD', 'ATOMUSD', 'BNBUSD', 'BTCUSD', 'DOGEUSD', 'DOTUSD',
    'ETHUSD', 'LTCUSD', 'MATICUSD', 'PAXGUSD', 'SHIBUSD', 'TRXUSD',
    'UNIUSD', 'VETUSD', 'XLMUSD', 'XRPUSD',
]

EXTENDED_STOCK_SYMBOLS: List[str] = [
    'AA', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ADBE', 'ADI', 'ADSK', 'AEP', 'AFRM', 'AIV', 'ALLY', 'AMAT',
    'AMD', 'AMT', 'AMZN', 'APD', 'ARKG', 'ARKK', 'ARKQ', 'ARKW', 'ASML', 'ATVI', 'AVB', 'AVGO', 'AXP',
    'AZN', 'AZO', 'BA', 'BABA', 'BAC', 'BIIB', 'BKNG', 'BKR', 'BLK', 'BNTX', 'BP', 'BSX', 'BUD', 'BXP',
    'C', 'CAG', 'CAT', 'CCI', 'CCL', 'CHD', 'CHTR', 'CL', 'CLF', 'CLX', 'CMCSA', 'CME', 'CMG', 'CMI',
    'CNP', 'COF', 'COIN', 'COP', 'COST', 'COUR', 'CPB', 'CPT', 'CRM', 'CVS', 'CVX', 'D', 'DAL',
    'DASH', 'DDOG', 'DE', 'DEO', 'DHR', 'DIS', 'DISH', 'DOCU', 'DOV', 'DTE', 'DUK', 'EA', 'EBAY', 'ECL',
    'ED', 'EIX', 'EMR', 'ENB', 'ENPH', 'EOG', 'EPD', 'EQIX', 'EQR', 'ES', 'ESS', 'ESTC', 'ET', 'ETN',
    'ETR', 'ETSY', 'EW', 'EXC', 'EXR', 'F', 'FCX', 'FDX', 'GD', 'GE', 'GILD', 'GIS', 'GM', 'GOLD',
    'GOOG', 'GOOGL', 'GS', 'GSK', 'HAL', 'HCP', 'HD', 'HLT', 'HOLX', 'HON', 'HOOD', 'HSY', 'ICE', 'IFF',
    'ILMN', 'INTC', 'ISRG', 'ITW', 'JNJ', 'JPM', 'K', 'KHC', 'KLAC', 'KMB', 'KMI', 'KO', 'LC', 'LIN',
    'LLY', 'LMT', 'LOW', 'LRCX', 'LYFT', 'MA', 'MAA', 'MAR', 'MCD', 'MCO', 'MDB', 'MDT', 'MELI', 'META',
    'MGM', 'MLM', 'MMM', 'MNST', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRVL', 'MS', 'MSFT', 'MTCH', 'MU',
    'NDAQ', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW', 'NUE', 'NVDA', 'NVO', 'NVS', 'NXPI',
    'O', 'OIH', 'OKTA', 'ON', 'ORCL', 'ORLY', 'OXY', 'PANW', 'PCG', 'PEP', 'PFE', 'PG', 'PH', 'PINS',
    'PLD', 'PLTR', 'PNC', 'PPG', 'PPL', 'PSA', 'PSX', 'PTON', 'PYPL', 'QCOM', 'RBLX', 'RCL', 'REGN',
    'RHHBY', 'ROK', 'ROKU', 'RPM', 'RS', 'RTX', 'SAP', 'SBUX', 'SCHW', 'SE', 'SEDG', 'SHEL', 'SHOP',
    'SHW', 'SIRI', 'SJM', 'SLB', 'SNAP', 'SNOW', 'SNY', 'SO', 'SOFI', 'SONY', 'SPCE', 'SPGI', 'SPOT',
    'SQ', 'SRE', 'STLD', 'SYK', 'T', 'TEAM', 'TFC', 'TGT', 'TJX', 'TM', 'TMO', 'TMUS', 'TRP', 'TSLA',
    'TSM', 'TTWO', 'TWLO', 'TWTR', 'TXN', 'U', 'UAL', 'UBER', 'UDR', 'UL', 'UNH', 'UPS', 'UPST', 'USB',
    'V', 'VEEV', 'VLO', 'VMC', 'VRTX', 'VTR', 'VZ', 'WDAY', 'WEC', 'WELL', 'WFC', 'WMB', 'WMT', 'WYNN',
    'X', 'XEL', 'XOM', 'ZBH', 'ZM', 'ZS',
]

DEFAULT_CRYPTO_SYMBOLS: List[str] = sorted(set(crypto_symbols) | set(EXTENDED_CRYPTO_SYMBOLS))
DEFAULT_STOCK_SYMBOLS: List[str] = sorted(set(EXTENDED_STOCK_SYMBOLS))
DEFAULT_TRAINING_SYMBOLS: List[str] = DEFAULT_STOCK_SYMBOLS + DEFAULT_CRYPTO_SYMBOLS

force_open_the_clock = False


@cachetools.cached(cache=cachetools.TTLCache(maxsize=100, ttl=60 * 3)) # 3 mins
def get_clock(retries=3):
    clock = get_clock_internal(retries)
    if not clock.is_open and force_open_the_clock:
        clock.is_open = True
    return clock


def force_open_the_clock_func():
    global force_open_the_clock
    force_open_the_clock = True


def get_clock_internal(retries=3):
    try:
        return alpaca_api.get_clock()
    except Exception as e:
        logger.error(e)
        if _missing_alpaca_credentials() or _is_unauthorized_error(e):
            logger.warning("Alpaca clock unavailable; returning synthetic open clock.")
            return _mock_clock()
        if retries > 0:
            sleep(.1)
            logger.error("retrying get clock")
            return get_clock_internal(retries - 1)
        raise e


def _calculate_spread_pct(symbol: str) -> Optional[float]:
    """Calculate the current bid-ask spread percentage for a symbol.

    Args:
        symbol: The trading symbol

    Returns:
        Spread as a percentage (e.g., 0.01 for 1%) or None if data unavailable
    """
    try:
        quote = latest_data(symbol)
        ask_price = float(getattr(quote, "ask_price", 0) or 0)
        bid_price = float(getattr(quote, "bid_price", 0) or 0)

        if ask_price <= 0 or bid_price <= 0:
            return None

        mid_price = (ask_price + bid_price) / 2.0
        if mid_price <= 0:
            return None

        spread_pct = (ask_price - bid_price) / mid_price
        return spread_pct
    except Exception as e:
        logger.warning(f"Failed to calculate spread for {symbol}: {e}")
        return None


def _can_use_market_order(symbol: str, is_closing_position: bool = False) -> Tuple[bool, str]:
    """Check if a market order can be used for this symbol.

    Market orders are only allowed when:
    1. NOT crypto (Alpaca executes crypto market orders at bid/ask midpoint, not market price)
    2. Market is open (not during pre-market, after-hours, or overnight)
    3. If closing a position, spread must be <= MARKET_ORDER_MAX_SPREAD_PCT

    Args:
        symbol: The trading symbol
        is_closing_position: Whether this is closing an existing position

    Returns:
        Tuple of (allowed, reason) where reason explains why if not allowed
    """
    # NEVER use market orders for crypto - Alpaca executes them at the bid/ask midpoint
    # instead of actual market price, making the execution price unpredictable
    # Check against all_crypto_symbols for comprehensive coverage
    if symbol in all_crypto_symbols:
        return False, f"Crypto {symbol} - market orders execute at bid/ask midpoint, not market price (use limit orders for predictable fills)"

    # Check if market is open (regular hours only, not pre-market/after-hours/overnight)
    clock = get_clock()
    if not clock.is_open:
        return False, "Market is closed - market orders not allowed during pre-market, after-hours, or overnight (use limit orders)"

    # If closing a position, also check spread
    if is_closing_position:
        spread_pct = _calculate_spread_pct(symbol)
        if spread_pct is not None and spread_pct > MARKET_ORDER_MAX_SPREAD_PCT:
            return False, (
                f"Spread {spread_pct*100:.2f}% exceeds maximum {MARKET_ORDER_MAX_SPREAD_PCT*100:.2f}% "
                f"for market orders when closing positions (use limit orders)"
            )

    return True, ""


def get_all_positions(retries=3):
    try:
        return alpaca_api.get_all_positions()
    except Exception as e:
        logger.error(e)
        if _missing_alpaca_credentials() or _is_unauthorized_error(e):
            logger.warning("Alpaca positions unavailable; returning empty list.")
            return []
        if retries > 0:
            sleep(.1)
            logger.error("retrying get all positions")
            return get_all_positions(retries - 1)
        raise e


def cancel_all_orders(retries=3):
    result = None
    try:
        result = alpaca_api.cancel_orders()
        logger.info("canceled orders")
        logger.info(result)
    except Exception as e:
        logger.error(e)

        if retries > 0:
            sleep(.1)
            logger.error("retrying cancel all orders")
            return cancel_all_orders(retries - 1)
        logger.error("failed to cancel all orders")
        return None
    return result


# alpaca_api.submit_order(short_stock, qty, side, "market", "gtc")
def open_market_order_violently(symbol, qty, side, retries=3):
    """Submit a market order.

    Market orders are only allowed when the market is open. During pre-market
    or after-hours, this function will return None and log an error.

    Args:
        symbol: Trading symbol
        qty: Quantity to trade
        side: 'buy' or 'sell'
        retries: Number of retry attempts

    Returns:
        Order result or None if market order not allowed or failed
    """
    # Check if market orders are allowed
    can_use, reason = _can_use_market_order(symbol, is_closing_position=False)
    if not can_use:
        logger.error(f"Market order blocked for {symbol}: {reason}")
        logger.error(f"RETURNING None - Use limit orders instead for out-of-hours trading")
        return None

    result = None
    try:
        result = alpaca_api.submit_order(
            order_data=MarketOrderRequest(
                symbol=remap_symbols(symbol),
                qty=qty,
                side=side,
                type=OrderType.MARKET,
                time_in_force="gtc",
            )
        )
    except Exception as e:
        error_str = str(e)
        logger.error(f"Market order attempt failed for {symbol}: {error_str}")
        logger.error(f"Full exception object: {repr(e)}")
        logger.error(f"Exception type: {type(e)}")
        if hasattr(e, 'response'):
            logger.error(f"API response object: {e.response}")
        if hasattr(e, 'status_code'):
            logger.error(f"HTTP status code: {e.status_code}")
        if hasattr(e, '__dict__'):
            logger.error(f"Exception attributes: {e.__dict__}")
        if retries > 0:
            logger.info(f"Retrying market order for {symbol}, {retries} attempts left")
            return open_market_order_violently(symbol, qty, side, retries - 1)
        logger.error(f"RETURNING None - Market order failed after all retries for {symbol} {side} {qty}")
        return None
    print(result)
    return result


def _parse_available_balance(error_str: str) -> float:
    """Extract available balance from an error message."""
    try:
        data = json.loads(error_str)
        return float(data.get("available", 0))
    except Exception:
        pass

    match = re.search(r"available['\"]?:\s*([0-9]*\.?[0-9]+)", error_str)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            pass
    return 0.0


# er_stock:372 - LTCUSD buying 116.104 at 83.755

def has_current_open_position(symbol: str, side: str) -> bool:
    # normalize side out of paranoia
    if side == "long":
        side = "buy"
    if side == "short":
        side = "sell"
    current_positions = []
    for i in range(3):
        try:
            current_positions = get_all_positions()
            break
        except Exception as e:
            traceback.print_exc()
            logger.error(e)
            # sleep(.1)
    current_positions = filter_to_realistic_positions(current_positions)
    for position in current_positions:
        # if market value is significant
        if float(position.market_value) < 4:
            continue
        if pairs_equal(position.symbol, symbol):
            if is_buy_side(position.side) and is_buy_side(side):
                logger.info("position already open")
                return True
            if is_sell_side(position.side) and is_sell_side(side):
                logger.info("position already open")
                return True
    return False


def open_order_at_price(symbol, qty, side, price):
    result = None
    # todo: check if order is already open
    # cancel all other orders on this symbol
    current_open_orders = get_orders()
    for order in current_open_orders:
        if pairs_equal(order.symbol, symbol):
            cancel_order(order)
    # also check that there are not any open positions on this symbol
    has_current_position = has_current_open_position(symbol, side)
    if has_current_position:
        logger.info(f"position {symbol} already open")
        logger.error(f"RETURNING None - Position already open for {symbol} {side}")
        return None
    try:
        price = str(round(price, 2))
        time_in_force = _get_time_in_force_for_qty(qty)

        result = alpaca_api.submit_order(
            order_data=LimitOrderRequest(
                symbol=remap_symbols(symbol),
                qty=qty,
                side=side,
                type=OrderType.LIMIT,
                time_in_force=time_in_force,
                limit_price=price,
            )
        )
    except Exception as e:
        error_str = str(e)
        logger.error(f"Order placement failed for {symbol}: {error_str}")
        logger.error(f"Full exception object: {repr(e)}")
        logger.error(f"Exception type: {type(e)}")
        if hasattr(e, 'response'):
            logger.error(f"API response object: {e.response}")
        if hasattr(e, 'status_code'):
            logger.error(f"HTTP status code: {e.status_code}")
        if hasattr(e, '__dict__'):
            logger.error(f"Exception attributes: {e.__dict__}")
        logger.error(f"RETURNING None - Order placement failed for {symbol} {side} {qty} @ {price}")
        return None
    print(result)
    return result


def open_order_at_price_or_all(symbol, qty, side, price):
    result = None
    # Cancel existing orders for this symbol
    current_open_orders = get_orders()
    for order in current_open_orders:
        if pairs_equal(order.symbol, symbol):
            cancel_order(order)

    # Check for existing position
    has_current_position = has_current_open_position(symbol, side)
    if has_current_position:
        logger.info(f"position {symbol} already open")
        logger.error(f"RETURNING None - Position already open for {symbol} {side}")
        return None

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Keep price as float for calculations, only convert when submitting order
            price_rounded = round(price, 2)
            time_in_force = _get_time_in_force_for_qty(qty)

            result = alpaca_api.submit_order(
                order_data=LimitOrderRequest(
                    symbol=remap_symbols(symbol),
                    qty=qty,
                    side=side,
                    type=OrderType.LIMIT,
                    time_in_force=time_in_force,
                    limit_price=str(price_rounded),
                )
            )
            return result

        except Exception as e:
            error_str = str(e)
            logger.error(f"Order attempt {retry_count + 1} failed: {error_str}")
            logger.error(f"Full exception object: {repr(e)}")
            logger.error(f"Exception type: {type(e)}")
            if hasattr(e, 'response'):
                logger.error(f"API response object: {e.response}")
            if hasattr(e, 'status_code'):
                logger.error(f"HTTP status code: {e.status_code}")
            if hasattr(e, '__dict__'):
                logger.error(f"Exception attributes: {e.__dict__}")

            # Check if error indicates insufficient funds
            if "insufficient" in error_str.lower():
                logger.error(f"Detected insufficient funds error. Full error_str: '{error_str}'")
                available = _parse_available_balance(error_str)
                if available <= 0:
                    available = cash

                if available > 0:
                    # Calculate maximum quantity we can afford with available balance
                    # Use a small buffer to avoid repeated insufficient balance errors.
                    affordable_qty = 0.99 * available / price if price else 0

                    # Stocks require whole-share quantities while crypto can remain fractional.
                    is_stock_quantity = False
                    try:
                        is_stock_quantity = float(qty).is_integer()
                    except (TypeError, ValueError):
                        is_stock_quantity = False

                    if is_stock_quantity:
                        new_qty = math.floor(affordable_qty)
                    else:
                        new_qty = round(affordable_qty, 6)

                    if new_qty > 0 and new_qty != qty:
                        logger.info(f"Insufficient funds. Adjusting quantity from {qty} to {new_qty} (available: {available})")
                        qty = new_qty
                        continue  # Don't increment retry_count, just retry with new quantity
                    else:
                        logger.error(f"Cannot afford any quantity. Available: {available}, Price: {price}, Calculated qty: {new_qty}")
                        logger.error(f"RETURNING None - Insufficient funds for {symbol} {side} {qty} @ {price}")
                        return None  # Exit immediately if we can't afford any quantity

            retry_count += 1
            # if retry_count < max_retries:
            #     time.sleep(2)  # Wait before retry

    logger.error(f"Max retries reached, order failed for {symbol} {side} {qty} @ {price}")
    logger.error(f"RETURNING None - Max retries reached for {symbol}")
    return None


def open_order_at_price_allow_add_to_position(symbol, qty, side, price):
    """
    Similar to open_order_at_price_or_all but allows adding to existing positions.
    This is used when we want to increase position size to a target amount.
    """
    logger.info(f"Starting order placement for {symbol} {side} {qty} @ {price}")
    result = None
    # Cancel existing orders for this symbol
    current_open_orders = get_orders()
    for order in current_open_orders:
        if pairs_equal(order.symbol, symbol):
            cancel_order(order)
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Keep price as float for calculations, only convert when submitting order
            price_rounded = round(price, 2)
            time_in_force = _get_time_in_force_for_qty(qty)

            logger.debug(f"Submitting order: {symbol} {side} {qty} @ {price_rounded} (attempt {retry_count + 1}, tif={time_in_force})")
            result = alpaca_api.submit_order(
                order_data=LimitOrderRequest(
                    symbol=remap_symbols(symbol),
                    qty=qty,
                    side=side,
                    type=OrderType.LIMIT,
                    time_in_force=time_in_force,
                    limit_price=str(price_rounded),
                )
            )
            logger.info(f"Order placed successfully for {symbol}: {side} {qty} @ {price_rounded}, result: {result}")
            return result
        except Exception as e:
            error_str = str(e)
            logger.error(f"Order attempt {retry_count + 1} failed for {symbol}: {error_str}")
            logger.error(f"Full exception object: {repr(e)}")
            logger.error(f"Exception type: {type(e)}")
            if hasattr(e, 'response'):
                logger.error(f"API response object: {e.response}")
            if hasattr(e, 'status_code'):
                logger.error(f"HTTP status code: {e.status_code}")
            if hasattr(e, '__dict__'):
                logger.error(f"Exception attributes: {e.__dict__}")
            
            # Check if error indicates insufficient funds
            if "insufficient" in error_str.lower():
                logger.error(f"Detected insufficient funds error. Full error_str: '{error_str}'")
                available = _parse_available_balance(error_str)
                if available <= 0:
                    available = cash
                if available > 0:
                    # Calculate maximum quantity we can afford with available balance
                    # Use 0.99 buffer and round to 6 decimal places for crypto
                    new_qty = round(0.99 * available / price, 6)
                    if new_qty > 0 and new_qty != qty:
                        logger.info(f"Insufficient funds. Adjusting quantity from {qty} to {new_qty} (available: {available})")
                        qty = new_qty
                        continue  # Don't increment retry_count, just retry with new quantity
                    else:
                        logger.error(f"Cannot afford any quantity. Available: {available}, Price: {price}, Calculated qty: {new_qty}")
                        logger.error(f"RETURNING None - Insufficient funds for {symbol} {side} {qty} @ {price}")
                        return None  # Exit immediately if we can't afford any quantity
            
            retry_count += 1
            
    logger.error(f"Max retries reached, order failed for {symbol} {side} {qty} @ {price}")
    logger.error(f"RETURNING None - Max retries reached for {symbol}")
    return None


def execute_portfolio_orders(orders: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Execute multiple orders sequentially.

    Each order should be a mapping containing ``symbol``, ``qty``, ``side`` and
    ``price`` keys. If an order fails, the error is logged and execution
    continues with the remaining orders.

    Parameters
    ----------
    orders: Iterable[Dict[str, Any]]
        Iterable of order dictionaries.

    Returns
    -------
    Dict[str, Any]
        Mapping of symbol to the result returned by
        :func:`open_order_at_price_or_all` or ``None`` if the order failed.
    """
    results: Dict[str, Any] = {}
    for order in orders:
        symbol = order.get("symbol")
        qty = order.get("qty")
        side = order.get("side")
        price = order.get("price")

        try:
            results[symbol] = open_order_at_price_or_all(symbol, qty, side, price)
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Failed to execute order for {symbol}: {e}")
            results[symbol] = None

    return results


def close_position_violently(position):
    """Close a position using a market order, with fallback to limit order at midpoint.

    Market orders for closing positions are only allowed when:
    1. NOT crypto (Alpaca executes crypto market orders at bid/ask midpoint, not market price)
    2. Market is open (not during pre-market, after-hours, or overnight)
    3. Spread is <= MARKET_ORDER_MAX_SPREAD_PCT (default 1%)

    If market orders are blocked, automatically falls back to a limit order at the
    midpoint price, which works during overnight/extended hours and for crypto.

    Args:
        position: Position object with symbol, side, and qty

    Returns:
        Order result or None if both market and limit order attempts failed
    """
    # Check if market orders are allowed (includes spread check for closing)
    can_use_market, reason = _can_use_market_order(position.symbol, is_closing_position=True)

    if not can_use_market:
        logger.warning(f"Market order blocked for closing {position.symbol}: {reason}")
        logger.info(f"Falling back to limit order at midpoint price for {position.symbol}")

        # Fallback: Use limit order at midpoint price
        try:
            quote = latest_data(position.symbol)
            ask_price = float(getattr(quote, "ask_price", 0) or 0)
            bid_price = float(getattr(quote, "bid_price", 0) or 0)

            if ask_price <= 0 or bid_price <= 0:
                logger.error(f"Cannot get valid bid/ask for {position.symbol}")
                return None

            # Use midpoint price for the limit order
            midpoint_price = (ask_price + bid_price) / 2.0

            # For closing long, sell at midpoint (slightly favorable)
            # For closing short, buy at midpoint (slightly favorable)
            limit_price = round(midpoint_price, 2)

            logger.info(f"Placing limit order to close {position.symbol} at ${limit_price} (midpoint)")

            if position.side == "long":
                side = OrderSide.SELL
            else:
                side = OrderSide.BUY

            result = alpaca_api.submit_order(
                order_data=LimitOrderRequest(
                    symbol=remap_symbols(position.symbol),
                    qty=abs(float(position.qty)),
                    side=side,
                    type=OrderType.LIMIT,
                    time_in_force="gtc",
                    limit_price=str(limit_price),
                )
            )
            logger.info(f"Limit order placed successfully for {position.symbol}")
            print(result)
            return result

        except Exception as e:
            logger.error(f"Limit order fallback failed for {position.symbol}: {e}")
            traceback.print_exc()
            return None

    # Market orders are allowed - proceed with market order
    result = None
    try:
        if position.side == "long":
            result = alpaca_api.submit_order(
                order_data=MarketOrderRequest(
                    symbol=remap_symbols(position.symbol),
                    qty=abs(float(position.qty)),
                    side=OrderSide.SELL,
                    type=OrderType.MARKET,
                    time_in_force="gtc",
                )
            )
        else:
            result = alpaca_api.submit_order(
                order_data=MarketOrderRequest(
                    symbol=remap_symbols(position.symbol),
                    qty=abs(float(position.qty)),
                    side=OrderSide.BUY,
                    type=OrderType.MARKET,
                    time_in_force="gtc",
                )
            )
    except Exception as e:
        traceback.print_exc()
        logger.error(e)
        return None
    print(result)
    return result


def close_position_at_current_price(position, row):
    if not row["close_last_price_minute"]:
        logger.info(f"nan price - for {position.symbol} market likely closed")
        return False
    result = None
    try:
        if position.side == "long":
            if position.symbol in crypto_symbols:
                result = crypto_alpaca_looper_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=remap_symbols(position.symbol),
                        qty=abs(float(position.qty)),
                        side=OrderSide.SELL,
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(round(float(row["close_last_price_minute"]), 2)),
                    )
                )
            else:
                result = alpaca_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=remap_symbols(position.symbol),
                        qty=abs(math.floor(float(position.qty) * 1000) / 1000.0),
                        side="sell",
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(math.ceil(float(row["close_last_price_minute"]))),
                    )
                )
        else:
            if position.symbol in crypto_symbols:
                result = crypto_alpaca_looper_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=remap_symbols(position.symbol),
                        qty=abs(math.floor(float(position.qty) * 1000) / 1000.0),
                        side=OrderSide.BUY,
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(math.floor(float(row["close_last_price_minute"]))),
                    )
                )
            else:
                result = alpaca_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=remap_symbols(position.symbol),
                        qty=abs(math.floor(float(position.qty) * 1000) / 1000.0),
                        side="buy",
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(math.floor(float(row["close_last_price_minute"]))),
                    )
                )
    except Exception as e:
        logger.error(e)
        traceback.print_exc()
        return None
    print(result)
    return result


def backout_all_non_crypto_positions(positions, predictions):
    for position in positions:
        if position.symbol in crypto_symbols:
            continue
        current_row = None
        for pred in predictions:
            if pairs_equal(pred["symbol"], position.symbol):
                current_row = pred
                break
        logger.info(f"backing out {position.symbol}")
        close_position_at_almost_current_price(position, current_row)
    sleep(60 * 2)

    cancel_all_orders()
    for position in positions:
        if position.symbol in crypto_symbols:
            continue
        current_row = None
        for pred in predictions:
            if pairs_equal(pred["symbol"], position.symbol):
                current_row = pred
                break
        logger.info(f"backing out at market {position.symbol}")

        close_position_at_current_price(position, current_row)
    sleep(60 * 2)

    cancel_all_orders()
    for position in positions:
        if position.symbol in crypto_symbols:
            continue
        # don't violently close here as spreads can be high
        # logger.info(f"violently backing out {position.symbol}")
        # close_position_violently(position)
        current_row = None
        for pred in predictions:
            if pairs_equal(pred["symbol"], position.symbol):
                current_row = pred
                break
        logger.info(f"backing out at market {position.symbol}")

        close_position_at_current_price(position, current_row)


def close_position_at_almost_current_price(position, row):
    result = None
    try:
        if position.side == "long":
            if position.symbol in crypto_symbols:
                result = crypto_alpaca_looper_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=remap_symbols(position.symbol),
                        qty=abs(math.floor(float(position.qty) * 1000) / 1000.0),
                        side="sell",
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(round(row["close_last_price_minute"] * 1.0003, 1)),
                    )
                )
            else:
                result = alpaca_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=remap_symbols(position.symbol),
                        qty=abs(math.floor(float(position.qty) * 1000) / 1000.0),
                        side="sell",
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(round(row["close_last_price_minute"] * 1.0003, 1)),
                    )
                )
        else:
            if position.symbol in crypto_symbols:
                result = crypto_alpaca_looper_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=remap_symbols(position.symbol),
                        qty=abs(math.floor(float(position.qty) * 1000) / 1000.0),
                        side="buy",
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(round(row["close_last_price_minute"] * (1 - 0.0003), 1)),
                    )
                )
            else:
                result = alpaca_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=remap_symbols(position.symbol),
                        qty=abs(math.floor(float(position.qty) * 1000) / 1000.0),
                        side="buy",
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(round(row["close_last_price_minute"] * (1 - 0.0003), 1)),
                    )
                )
    except Exception as e:
        logger.error(e)
        return None
    print(result)
    return result


@retry(delay=.1, tries=3)
def get_orders():
    try:
        return alpaca_api.get_orders()
    except Exception as e:
        logger.error(e)
        if _missing_alpaca_credentials() or _is_unauthorized_error(e):
            logger.warning("Alpaca orders unavailable; returning empty list.")
            return []
        raise


def alpaca_order_stock(currentBuySymbol, row, price, margin_multiplier=1.95, side="long", bid=None, ask=None):
    result = None
    # trading at market to add more safety in high spread situations
    side = "buy" if is_buy_side(side) else "sell"
    if side == "buy" and bid:
        price = min(price, bid or price)
    else:
        price = max(price, ask or price)

    # skip crypto for now as its high fee
    # if currentBuySymbol in crypto_symbols and is_buy_side(side):
    #     logger.info(f"Skipping Buying Alpaca crypto order for {currentBuySymbol}")
    #     logger.info(f"TMp measure as fees are too high IMO move to binance")
    #     return False

    # poll untill we have closed all our positions
    # why we would wait here?
    # polls = 0
    # while True:
    #     positions = get_all_positions()
    #     if len(positions) == 0:
    #         break
    #     else:
    #         print('waiting for positions to close')
    #         sleep(.1)
    #         polls += 1
    #         if polls > 5:
    #             print('polling for too long, closing all positions again')
    #             # alpaca_api.close_all_positions() # todo respect manual orders
    #         if polls > 20:
    #             print('polling for too long, exiting, market is probably closed')
    #             break
    # notional_value = total_buying_power * 1.9 # trade with margin
    # notional_value = total_buying_power - 600 # trade with margin
    # non marginable
    if currentBuySymbol in ["BTCUSD", "ETHUSD", "LTCUSD", "PAXGUSD", "UNIUSD"]:

        margin_multiplier = min(margin_multiplier, 1)
        notional_value = cash * margin_multiplier  # todo predict margin/price
    else:
        notional_value = total_buying_power * margin_multiplier  # todo predict margin/price

    # side = 'buy'
    if row["close_predicted_price"] < 0:
        # side = 'sell'
        notional_value = (
                total_buying_power * margin_multiplier
        )  # trade with margin but not too much on the sell side
        # notional_value = total_buying_power - 2000
        # todo dont leave a short open over the weekend perhaps?

    try:
        current_price = float(row["close_last_price_minute"])

        amount_to_trade = notional_value / current_price
        if currentBuySymbol in ["BTCUSD"]:
            if amount_to_trade < 0.001:
                amount_to_trade = 0.001
        elif currentBuySymbol in ["ETHUSD"]:
            if amount_to_trade < 0.01:
                amount_to_trade = 0.01
        elif currentBuySymbol in ["LTCUSD"]:
            if amount_to_trade < 0.1:
                amount_to_trade = 0.1
                # too work out "PAXGUSD", "UNIUSD"
        elif amount_to_trade < 1:
            amount_to_trade = 1

        if currentBuySymbol not in ["BTCUSD", "ETHUSD", "LTCUSD", "PAXGUSD", "UNIUSD"]:
            # fractional orders are okay for crypto.
            amount_to_trade = int(amount_to_trade)
        else:
            amount_to_trade = abs(math.floor(float(amount_to_trade) * 1000) / 1000.0)

        # Cancel existing orders for this symbol
        current_orders = get_orders()
        for order in current_orders:
            if pairs_equal(order.symbol, currentBuySymbol):
                alpaca_api.cancel_order_by_id(order.id)

        # Submit the order
        if currentBuySymbol in crypto_symbols:
            result = crypto_alpaca_looper_api.submit_order(
                order_data=LimitOrderRequest(
                    symbol=remap_symbols(currentBuySymbol),
                    qty=amount_to_trade,
                    side=side,
                    type=OrderType.LIMIT,
                    time_in_force="gtc",
                    limit_price=str(math.floor(price) if is_buy_side(side) else math.ceil(price)),
                )
            )
        else:
            result = alpaca_api.submit_order(
                order_data=LimitOrderRequest(
                    symbol=remap_symbols(currentBuySymbol),
                    qty=amount_to_trade,
                    side=side,
                    type=OrderType.LIMIT,
                    time_in_force="gtc",
                    limit_price=str(math.floor(price) if is_buy_side(side) else math.ceil(price)),
                )
            )
        print(result)
        return True

    except APIError as e:
        logger.error(e)
        return False
    except Exception as e:
        logger.error(e)
        return False


def close_open_orders():
    alpaca_api.cancel_orders()


def re_setup_vars():
    global positions
    global account
    global alpaca_api
    global alpaca_clock
    global total_buying_power
    global equity
    global cash
    global margin_multiplier
    positions = get_all_positions()
    print(positions)
    account = get_account()
    print(account)
    # Figure out how much money we have to work with, accounting for margin
    equity = float(account.equity)
    cash = max(float(account.cash), 0)
    margin_multiplier = float(account.multiplier)
    total_buying_power = margin_multiplier * equity
    print(f"Initial total buying power = {total_buying_power}")
    alpaca_clock = get_clock()
    print(alpaca_clock)
    if not alpaca_clock.is_open:
        print("Market closed")


def open_take_profit_position(position, row, price, qty):
    result = None
    try:
        mapped_symbol = remap_symbols(position.symbol)
        if position.side == "long":
            if position.symbol in crypto_symbols:
                result = crypto_alpaca_looper_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=mapped_symbol,
                        qty=abs(math.floor(float(qty) * 1000) / 1000.0),
                        side="sell",
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(math.ceil(price)),
                    )
                )
            else:
                result = alpaca_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=mapped_symbol,
                        qty=abs(math.floor(float(qty) * 1000) / 1000.0),
                        side="sell",
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(math.ceil(price)),
                    )
                )
        else:
            if position.symbol in crypto_symbols:
                result = crypto_alpaca_looper_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=mapped_symbol,
                        qty=abs(math.floor(float(qty) * 1000) / 1000.0),
                        side="buy",
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(math.floor(price)),
                    )
                )
            else:
                result = alpaca_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=mapped_symbol,
                        qty=abs(math.floor(float(qty) * 1000) / 1000.0),
                        side="buy",
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(math.floor(price)),
                    )
                )
    except Exception as e:
        logger.error(e)
        traceback.print_exc()
        return None
    return result


def cancel_order(order):
    try:
        alpaca_api.cancel_order_by_id(order.id)
    except Exception as e:
        # Check if order is already pending cancellation (error 42210000)
        error_str = str(e)
        if "42210000" in error_str or "pending cancel" in error_str.lower():
            logger.info(f"Order {order.id} already pending cancellation, treating as success")
            return  # Treat as success - order is already being cancelled
        logger.error(e)
        # traceback
        traceback.print_exc()
        raise  # Re-raise other errors


def get_open_orders():
    # try:
    #     crypto_orders = crypto_alpaca_looper_api.get_orders()
    # except Exception as e:
    #     logger.error(e)
    #     crypto_orders = []
    #     traceback.print_exc()

    try:
        return get_orders()  # + crypto_orders
    except Exception as e:
        logger.error(e)
        traceback.print_exc()
        return []


crypto_client = CryptoHistoricalDataClient()


def latest_data(symbol):
    if symbol in crypto_symbols:
        symbol = remap_symbols(symbol)
        response = crypto_client.get_crypto_latest_quote(
            CryptoLatestQuoteRequest(symbol_or_symbols=[symbol])
        )
        return response[symbol]

    multisymbol_request_params = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
    latest_multisymbol_quotes = data_client.get_stock_latest_quote(multisymbol_request_params)

    return latest_multisymbol_quotes[symbol]


def _normalize_bar_frame(symbol: str, bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame()

    df = bars.copy()
    if isinstance(df.index, pd.MultiIndex):
        level_symbols = df.index.get_level_values(0)
        primary_symbol = remap_symbols(symbol) if symbol in DEFAULT_CRYPTO_SYMBOLS else symbol
        if primary_symbol in level_symbols:
            df = df.xs(primary_symbol, level=0, drop_level=True)
        elif symbol in level_symbols:
            df = df.xs(symbol, level=0, drop_level=True)
        else:
            df = df.xs(level_symbols[0], level=0, drop_level=True)

    df = df.reset_index()
    if "symbol" in df.columns:
        df = df.drop(columns=["symbol"])

    df = df.rename(columns=lambda c: c.lower() if isinstance(c, str) else c)
    if "timestamp" not in df.columns:
        for candidate in ("time", "date"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "timestamp"})
                break

    if "timestamp" not in df.columns:
        raise ValueError(f"Could not locate timestamp column for {symbol}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    df.set_index("timestamp", inplace=True)
    df.index.name = "timestamp"
    return df


def download_symbol_history(
    symbol: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    include_latest: bool = True,
    timeframe: Optional[TimeFrame] = None,
) -> pd.DataFrame:
    symbol = symbol.upper()
    is_crypto = symbol in DEFAULT_CRYPTO_SYMBOLS or symbol.endswith("USD")

    end_dt = end or datetime.now(timezone.utc)
    start_dt = start or (end_dt - timedelta(days=DEFAULT_HISTORY_DAYS))
    requested_timeframe = timeframe or TimeFrame(1, TimeFrameUnit.Day)

    if not is_crypto and requested_timeframe.unit != TimeFrameUnit.Day:
        raise ValueError("Stock history currently supports only daily timeframes.")

    try:
        if is_crypto:
            request = CryptoBarsRequest(
                symbol_or_symbols=remap_symbols(symbol),
                timeframe=requested_timeframe,
                start=start_dt,
                end=end_dt,
            )
            bars = crypto_client.get_crypto_bars(request).df
        else:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=requested_timeframe,
                start=start_dt,
                end=end_dt,
                adjustment="raw",
                feed=DataFeed.IEX,
            )
            bars = data_client.get_stock_bars(request).df
    except Exception as exc:
        logger.error(f"Failed to download historical bars for {symbol}: {exc}")
        raise

    df = _normalize_bar_frame(symbol, bars)
    if df.empty:
        return df

    if include_latest:
        try:
            quote = latest_data(symbol)
            ask_price = float(getattr(quote, "ask_price", 0) or 0)
            bid_price = float(getattr(quote, "bid_price", 0) or 0)
            if ask_price > 0 and bid_price > 0:
                mid_price = (ask_price + bid_price) / 2.0
                if "close" in df.columns:
                    df.iloc[-1, df.columns.get_loc("close")] = mid_price
                else:
                    df["close"] = mid_price
        except Exception as exc:
            logger.warning(f"Unable to augment latest quote for {symbol}: {exc}")

    df["symbol"] = symbol
    return df


def _split_train_test(df: pd.DataFrame, test_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df

    ordered = df.sort_index()
    if len(ordered) > test_days:
        train_df = ordered.iloc[:-test_days]
        test_df = ordered.iloc[-test_days:]
    else:
        split_idx = max(1, int(len(ordered) * 0.8))
        train_df = ordered.iloc[:split_idx]
        test_df = ordered.iloc[split_idx:]
    return train_df, test_df


def _persist_splits(symbol: str, train_df: pd.DataFrame, test_df: pd.DataFrame, base_path: Path) -> Tuple[Path, Path]:
    safe_symbol = symbol.replace("/", "-")
    train_dir = base_path / "train"
    test_dir = base_path / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df.index.name = "timestamp"
    test_df.index.name = "timestamp"

    train_path = train_dir / f"{safe_symbol}.csv"
    test_path = test_dir / f"{safe_symbol}.csv"
    train_df.to_csv(train_path)
    test_df.to_csv(test_path)
    return train_path, test_path


def _load_existing_summary(symbol: str, base_path: Path) -> Optional[Dict[str, Any]]:
    safe_symbol = symbol.replace("/", "-")
    train_file = base_path / "train" / f"{safe_symbol}.csv"
    test_file = base_path / "test" / f"{safe_symbol}.csv"

    if not train_file.exists() or not test_file.exists():
        return None

    try:
        train_df = pd.read_csv(train_file, index_col=0, parse_dates=True)
        test_df = pd.read_csv(test_file, index_col=0, parse_dates=True)
    except Exception:
        return None

    latest_values = []
    if not train_df.empty:
        latest_values.append(train_df.index.max())
    if not test_df.empty:
        latest_values.append(test_df.index.max())

    if not latest_values:
        return None

    latest_ts = max(latest_values)
    latest_ts = pd.to_datetime(latest_ts, utc=True, errors="coerce")
    if pd.isna(latest_ts):
        return None

    return {
        "symbol": symbol,
        "latest": latest_ts,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }


def _should_skip_symbol(symbol: str, base_path: Path, skip_if_recent_days: int) -> Optional[Dict[str, Any]]:
    if skip_if_recent_days <= 0:
        return None

    summary = _load_existing_summary(symbol, base_path)
    if not summary:
        return None

    latest_ts = summary["latest"]
    current_time = datetime.now(timezone.utc)
    days_old = (current_time - latest_ts).days
    if days_old < skip_if_recent_days:
        logger.info(f"Skipping {symbol} - latest data is {days_old} days old")
        summary.update(
            {
                "status": "skipped",
                "latest": latest_ts.isoformat(),
            }
        )
        return summary
    return None


def _write_training_summary(base_path: Path) -> None:
    train_dir = base_path / "train"
    if not train_dir.exists():
        return

    test_dir = base_path / "test"
    summary_rows = []
    for train_file in sorted(train_dir.glob("*.csv")):
        symbol = train_file.stem
        test_file = test_dir / f"{symbol}.csv"
        if not test_file.exists():
            continue

        try:
            train_df = pd.read_csv(train_file, index_col=0, parse_dates=True)
            test_df = pd.read_csv(test_file, index_col=0, parse_dates=True)
        except Exception as exc:
            logger.error(f"Unable to load training data for summary ({symbol}): {exc}")
            continue

        latest_candidates = []
        if not train_df.empty:
            latest_candidates.append(train_df.index.max())
        if not test_df.empty:
            latest_candidates.append(test_df.index.max())

        latest_ts = pd.to_datetime(max(latest_candidates), utc=True, errors="coerce") if latest_candidates else None
        summary_rows.append(
            {
                "symbol": symbol,
                "latest_date": latest_ts.strftime("%Y-%m-%d") if latest_ts is not None and not pd.isna(latest_ts) else "",
                "total_rows": len(train_df) + len(test_df),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_file": f"trainingdata/train/{symbol}.csv",
                "test_file": f"trainingdata/test/{symbol}.csv",
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("symbol")
    summary_path = base_path / "data_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Wrote training data summary to {summary_path}")


def download_training_pairs(
    symbols: Optional[Iterable[str]] = None,
    output_dir: Optional[Path] = None,
    test_days: int = DEFAULT_TEST_DAYS,
    history_days: int = DEFAULT_HISTORY_DAYS,
    skip_if_recent_days: int = DEFAULT_SKIP_IF_RECENT_DAYS,
    include_latest: bool = True,
    sleep_seconds: float = 0.0,
) -> List[Dict[str, Any]]:
    resolved_symbols = (
        sorted({s.upper().replace(" ", "") for s in DEFAULT_TRAINING_SYMBOLS})
        if symbols is None
        else sorted({s.upper().replace(" ", "") for s in symbols})
    )
    base_path = Path(output_dir) if output_dir else TRAININGDATA_BASE_PATH
    base_path.mkdir(parents=True, exist_ok=True)

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=history_days)

    results: List[Dict[str, Any]] = []
    for index, symbol in enumerate(resolved_symbols, start=1):
        skip_info = _should_skip_symbol(symbol, base_path, skip_if_recent_days)
        if skip_info:
            results.append(skip_info)
            continue

        try:
            df = download_symbol_history(symbol, start=start_dt, end=end_dt, include_latest=include_latest)
        except Exception as exc:
            logger.error(f"Download failed for {symbol}: {exc}")
            results.append({"symbol": symbol, "status": "error", "error": str(exc)})
            continue

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            results.append({"symbol": symbol, "status": "empty"})
            continue

        train_df, test_df = _split_train_test(df, test_days)
        train_path, test_path = _persist_splits(symbol, train_df, test_df, base_path)

        latest_candidates = []
        if not train_df.empty:
            latest_candidates.append(train_df.index.max())
        if not test_df.empty:
            latest_candidates.append(test_df.index.max())

        latest_ts = pd.to_datetime(max(latest_candidates), utc=True, errors="coerce") if latest_candidates else None

        results.append(
            {
                "symbol": symbol,
                "status": "ok",
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "latest": latest_ts.isoformat() if latest_ts is not None and not pd.isna(latest_ts) else None,
                "train_file": str(train_path.relative_to(base_path.parent)),
                "test_file": str(test_path.relative_to(base_path.parent)),
            }
        )

        if sleep_seconds and index < len(resolved_symbols):
            sleep(sleep_seconds)

    _write_training_summary(base_path)
    return results


@retry(delay=.1, tries=3)
def get_account():
    try:
        return alpaca_api.get_account()
    except Exception as e:
        logger.error(e)
        if _missing_alpaca_credentials() or _is_unauthorized_error(e):
            logger.warning("Alpaca account unavailable; returning synthetic account snapshot.")
            return SimpleNamespace(
                equity="0",
                cash="0",
                multiplier="1.0",
                buying_power="0",
            )
        raise


equity = 30000
cash = 30000
total_buying_power = 20000

try:
    positions = get_all_positions()
    print(positions)
    account = get_account()
    print(account)
    # Figure out how much money we have to work with, accounting for margin
    equity = float(account.equity)
    cash = max(float(account.cash), 0)
    margin_multiplier = float(account.multiplier)
    total_buying_power = margin_multiplier * equity
    print(f"Initial total buying power = {total_buying_power}")
    alpaca_clock = get_clock()
    print(alpaca_clock)
    if not alpaca_clock.is_open:
        print("Market closed")
except requests.exceptions.ConnectionError as e:
    logger.error("offline/connection error", e)
except APIError as e:
    logger.error("alpaca error", e)
except Exception as e:
    logger.error("exception", e)
    traceback.print_exc()


def close_position_near_market(position, pct_above_market=0.0):
    """Place a limit order at ``pct_above_market`` relative to the quote."""
    bids = {}
    asks = {}
    symbol = position.symbol
    very_latest_data = latest_data(position.symbol)
    # check if market closed
    ask_price = float(very_latest_data.ask_price)
    bid_price = float(very_latest_data.bid_price)
    if bid_price != 0 and ask_price != 0:
        bids[symbol] = bid_price
        asks[symbol] = ask_price

    ask_price = asks.get(position.symbol)
    bid_price = bids.get(position.symbol)

    if not ask_price or not bid_price:
        logger.error(f"error getting ask/bid price for {position.symbol}")
        return False

    if position.side == "long":
        # For long positions, reference the bid price when selling
        price = bid_price
    else:
        # For short positions, reference the ask price when buying back
        price = ask_price

    result = None
    try:
        if position.side == "long":
            sell_price = price * (1 + pct_above_market)
            sell_price = round(sell_price, 2)
            logger.info(f"selling {position.symbol} at {sell_price}")
            request = LimitOrderRequest(
                symbol=remap_symbols(position.symbol),
                qty=abs(float(position.qty)),
                side=OrderSide.SELL,
                type=OrderType.LIMIT,
                time_in_force="gtc",
                limit_price=sell_price,
            )
        else:
            buy_price = price * (1 + pct_above_market)
            buy_price = round(buy_price, 2)
            logger.info(f"buying {position.symbol} at {buy_price}")
            request = LimitOrderRequest(
                symbol=remap_symbols(position.symbol),
                qty=abs(float(position.qty)),
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                time_in_force="gtc",
                limit_price=buy_price,
            )

        result = alpaca_api.submit_order(order_data=request)

    except Exception as e:
        logger.error(f"Failed to submit close order for {position.symbol}: {e}")
        traceback.print_exc()
        return False

    return result


def get_executed_orders(alpaca_api):
    """
    Gets all historical orders that were executed.

    Args:
        alpaca_api: The Alpaca trading client instance

    Returns:
        List of executed orders
    """
    try:
        # Get all orders with status=filled filter
        orders = alpaca_api.get_orders(
            filter=GetOrdersRequest(
                status="filled"
            )
        )
        return orders

    except Exception as e:
        logger.error(f"Error getting executed orders: {e}")
        traceback.print_exc()
        return []


def get_account_activities(
        alpaca_api,
        activity_types=None,
        date=None,
        direction='desc',
        page_size=100,
        page_token=None
):
    """
    Retrieve account activities (trades, dividends, etc.) from the Alpaca API.
    Pagination is handled via page_token. The activity_types argument can be any of:
    'FILL', 'DIV', 'TRANS', 'MISC', etc.

    Args:
        alpaca_api: The Alpaca trading client instance.
        activity_types: List of activity type strings (e.g. ['FILL', 'DIV']).
        date: (Optional) The date for which you'd like to see activities.
        direction: 'asc' or 'desc' for sorting.
        page_size: The number of records to return per page (up to 100 if date is not set).
        page_token: Used for pagination.

    Returns:
        A list of account activity records, or an empty list on error.
    """
    query_params = {}
    if activity_types:
        # Convert single str to list if needed
        if isinstance(activity_types, str):
            activity_types = [activity_types]
        query_params["activity_types"] = ",".join(activity_types)

    if date:
        query_params["date"] = date
    if direction:
        query_params["direction"] = direction
    if page_size:
        query_params["page_size"] = str(page_size)
    if page_token:
        query_params["page_token"] = page_token

    try:
        # Directly use the TradingClient's underlying request method to access this endpoint
        response = alpaca_api._request("GET", "/account/activities", data=query_params)
        return response
    except Exception as e:
        logger.error(f"Error retrieving account activities: {e}")
        return []
