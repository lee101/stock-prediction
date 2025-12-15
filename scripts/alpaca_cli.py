import os
from datetime import datetime, timezone, timedelta
from time import sleep
import traceback
from typing import Optional

import alpaca_trade_api as tradeapi
import math
import pytz
import typer
from alpaca.data import StockHistoricalDataClient

import alpaca_wrapper
from data_curate_daily import download_exchange_latest_data, get_bid, get_ask
from env_real import ALP_KEY_ID, ALP_SECRET_KEY, ALP_ENDPOINT, ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
from jsonshelve import FlatShelf
from src.fixtures import crypto_symbols
from src.logging_utils import setup_logging, get_log_filename
from src.stock_utils import pairs_equal
from src.trading_obj_utils import filter_to_realistic_positions
from src.date_utils import is_nyse_open_on_date

# Import position sizing utilities
from src.sizing_utils import get_qty

# Import portfolio risk utilities
from src.portfolio_risk import get_global_risk_threshold

alpaca_api = tradeapi.REST(
    ALP_KEY_ID,
    ALP_SECRET_KEY,
    ALP_ENDPOINT,
    'v2')

# Detect if we're in hourly mode based on TRADE_STATE_SUFFIX env var
_is_hourly = os.getenv("TRADE_STATE_SUFFIX", "") == "hourly"
logger = setup_logging(get_log_filename("alpaca_cli.log", is_hourly=_is_hourly))

# We'll store strategy usage in a persistent shelf
positions_shelf = FlatShelf("positions_shelf.json")

BACKOUT_RAMP_MINUTES_DEFAULT = int(os.getenv("BACKOUT_RAMP_MINUTES", "30"))
BACKOUT_MARKET_AFTER_MINUTES_DEFAULT = int(os.getenv("BACKOUT_MARKET_AFTER_MINUTES", "50"))
_MARKET_SPREAD_CAP_RAW = os.getenv("BACKOUT_MARKET_MAX_SPREAD_PCT", "0.01")
try:
    BACKOUT_MARKET_MAX_SPREAD_PCT = max(float(_MARKET_SPREAD_CAP_RAW), 0.0)
except ValueError:
    BACKOUT_MARKET_MAX_SPREAD_PCT = 0.01
    logger.warning(
        "Invalid BACKOUT_MARKET_MAX_SPREAD_PCT=%r; defaulting to %.2f%%",
        _MARKET_SPREAD_CAP_RAW,
        BACKOUT_MARKET_MAX_SPREAD_PCT * 100,
    )


BACKOUT_INITIAL_OFFSET_PCT = float(os.getenv("BACKOUT_INITIAL_OFFSET_PCT", "0.003"))
BACKOUT_SOFT_CROSS_PCT = float(os.getenv("BACKOUT_SOFT_CROSS_PCT", "0.004"))
BACKOUT_FINAL_CROSS_PCT = float(os.getenv("BACKOUT_FINAL_CROSS_PCT", "0.02"))
BACKOUT_POLL_INTERVAL_SECONDS_DEFAULT = int(os.getenv("BACKOUT_POLL_INTERVAL_SECONDS", "45"))
BACKOUT_MARKET_CLOSE_BUFFER_MINUTES_DEFAULT = int(os.getenv("BACKOUT_MARKET_CLOSE_BUFFER_MINUTES", "30"))
BACKOUT_MARKET_CLOSE_FORCE_MINUTES_DEFAULT = int(os.getenv("BACKOUT_MARKET_CLOSE_FORCE_MINUTES", "3"))


def set_strategy_for_symbol(symbol: str, strategy: str) -> None:
    """Record that a symbol is traded under the given strategy for today's date."""
    day_key = datetime.now().strftime('%Y-%m-%d')
    shelf_key = f"{symbol}-{day_key}"
    positions_shelf[shelf_key] = strategy
    # positions_shelf.commit()


def get_strategy_for_symbol(symbol: str) -> str:
    """Retrieve the strategy for a symbol for today's date, if any."""
    day_key = datetime.now().strftime('%Y-%m-%d')
    # Reload the shelf to avoid race conditions
    positions_shelf.load()
    shelf_key = f"{symbol}-{day_key}"
    return positions_shelf.get(shelf_key, None)


def _calculate_total_exposure_value(positions) -> float:
    """Calculate total exposure value across all positions."""
    total_value = 0.0
    for position in positions:
        try:
            market_value = float(getattr(position, "market_value", 0.0) or 0.0)
        except Exception:
            market_value = 0.0
        total_value += abs(market_value)
    return total_value


def main(
    command: str,
    pair: Optional[str] = typer.Argument(None, help="Target symbol for single-symbol commands"),
    side: Optional[str] = typer.Option("buy", "--side", help="Order side for ramp/entry commands."),
    target_qty: Optional[float] = typer.Option(
        None, "--target-qty", help="Target quantity for ramp_into_position."
    ),
    maxdiff_overflow: bool = typer.Option(
        False, "--maxdiff-overflow", help="Check leverage before placing orders for maxdiff overflow trades."
    ),
    risk_threshold: Optional[float] = typer.Option(
        None, "--risk-threshold", help="Risk threshold to check against for maxdiff overflow."
    ),
    start_offset_minutes: int = typer.Option(
        0,
        "--start-offset-minutes",
        min=0,
        help="Extend backout ramp duration by this many minutes before forcing market.",
    ),
    ramp_minutes: Optional[int] = typer.Option(
        None,
        "--ramp-minutes",
        min=1,
        help="Base number of minutes used for the backout ramp before any start offset.",
    ),
    market_after_minutes: Optional[int] = typer.Option(
        None,
        "--market-after-minutes",
        min=1,
        help="Minutes before forcing a market order during backout (before start offset).",
    ),
    sleep_seconds: Optional[int] = typer.Option(
        None,
        "--sleep-seconds",
        min=0,
        help="Polling interval between backout iterations.",
    ),
    market_close_buffer_minutes: Optional[int] = typer.Option(
        None,
        "--market-close-buffer-minutes",
        min=0,
        help="Minutes from the close to keep tighter limit offsets before widening.",
    ),
    market_close_force_minutes: Optional[int] = typer.Option(
        None,
        "--market-close-force-minutes",
        min=0,
        help="Force market order when this close to market close even if ramp not completed.",
    ),
):
    """
    Alpaca CLI - Trade stocks with safety restrictions for out-of-hours trading.

    IMPORTANT: Market orders are NEVER used during pre-market or after-hours trading.
    When the market is closed, only limit orders are allowed. Market orders are also
    blocked when the spread exceeds 1% (configurable via MARKET_ORDER_MAX_SPREAD_PCT).

    Commands:
    ---------
    cancel_all_orders - cancel all orders

    close_all_positions - close all positions at near market price

    close_position_violently - close position with market order (only during market hours)

    backout_near_market BTCUSD - gradually backout of position, uses market orders only
                                  during market hours and when spread <= 1%

    backout_whole_account_near_market - iterate all open positions and run
                                        backout_near_market for each to flatten
                                        the account

    ramp_into_position BTCUSD buy - ramp into a position over time (works out-of-hours)

    show_account - display account summary, positions, and orders

    show_forecasts - display forecast predictions for a symbol

    debug_raw_data SYMBOL - print raw JSON data from Alpaca for the symbol

    Environment Variables:
    ----------------------
    MARKET_ORDER_MAX_SPREAD_PCT - Maximum spread (default: 0.01 = 1%) for market orders
                                   when closing positions
    BACKOUT_MARKET_MAX_SPREAD_PCT - Same as above, for backout operations

    :param pair: e.g. BTCUSD
    :param command:
    :param side: buy or sell (default: buy)
    :return:
    """
    if command == 'close_all_positions':
        close_all_positions()
    elif command == 'violently_close_all_positions':
        violently_close_all_positions()
    elif command == 'cancel_all_orders':
        alpaca_wrapper.cancel_all_orders()
    elif command == "backout_whole_account_near_market":
        backout_whole_account_near_market(
            ramp_minutes=ramp_minutes or BACKOUT_RAMP_MINUTES_DEFAULT,
            market_after=market_after_minutes or BACKOUT_MARKET_AFTER_MINUTES_DEFAULT,
            sleep_interval=sleep_seconds,
            start_offset_minutes=start_offset_minutes,
            market_close_buffer_minutes=(
                market_close_buffer_minutes
                if market_close_buffer_minutes is not None
                else BACKOUT_MARKET_CLOSE_BUFFER_MINUTES_DEFAULT
            ),
            market_close_force_minutes=(
                market_close_force_minutes
                if market_close_force_minutes is not None
                else BACKOUT_MARKET_CLOSE_FORCE_MINUTES_DEFAULT
            ),
        )
    elif command == "backout_near_market":
        # loop around until the order is closed at market
        if not pair:
            logger.error("Symbol is required for backout_near_market command")
            return
        now = datetime.now()
        backout_near_market(
            pair,
            start_time=now,
            ramp_minutes=ramp_minutes or BACKOUT_RAMP_MINUTES_DEFAULT,
            market_after=market_after_minutes or BACKOUT_MARKET_AFTER_MINUTES_DEFAULT,
            sleep_interval=sleep_seconds,
            start_offset_minutes=start_offset_minutes,
            market_close_buffer_minutes=(
                market_close_buffer_minutes
                if market_close_buffer_minutes is not None
                else BACKOUT_MARKET_CLOSE_BUFFER_MINUTES_DEFAULT
            ),
            market_close_force_minutes=(
                market_close_force_minutes
                if market_close_force_minutes is not None
                else BACKOUT_MARKET_CLOSE_FORCE_MINUTES_DEFAULT
            ),
        )
    elif command == "ramp_into_position":
        now = datetime.now()
        ramp_into_position(
            pair,
            side,
            start_time=now,
            target_qty=target_qty,
            maxdiff_overflow=maxdiff_overflow,
            risk_threshold=risk_threshold,
        )
    elif command == "close_position_at_takeprofit":
        close_position_at_takeprofit(pair, float(side))  # Use side param as target price
    elif command == 'show_account':
        show_account()
    elif command == 'show_forecasts':
        if not pair:
            logger.error("Symbol is required for show_forecasts command")
            return
        show_forecasts_for_symbol(pair)
    elif command == 'debug_raw_data':
        if not pair:
            logger.error("Symbol is required for debug_raw_data command")
            return
        debug_raw_data(pair)


_DATA_CLIENT: Optional[StockHistoricalDataClient] = None


def _get_data_client() -> Optional[StockHistoricalDataClient]:
    global _DATA_CLIENT
    if _DATA_CLIENT is not None:
        return _DATA_CLIENT
    try:
        _DATA_CLIENT = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
    except Exception as exc:
        logger.error("Failed to initialise StockHistoricalDataClient: %s", exc)
        _DATA_CLIENT = None
    return _DATA_CLIENT


def _current_spread_pct(symbol: str) -> Optional[float]:
    """Fetch latest bid/ask and compute relative spread."""
    client = _get_data_client()
    if client is None:
        return None
    try:
        download_exchange_latest_data(client, symbol)
    except Exception as exc:
        logger.warning("Unable to refresh quotes for %s: %s", symbol, exc)
    bid = get_bid(symbol)
    ask = get_ask(symbol)
    if bid is None or ask is None:
        return None
    if bid <= 0 or ask <= 0:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    return (ask - bid) / mid


def _minutes_until_market_close(now_utc: Optional[datetime] = None) -> Optional[float]:
    """Return minutes until NYSE close (16:00 ET) for the provided or current time.

    Uses exchange_calendars for accurate holiday detection.
    """
    try:
        eastern = pytz.timezone("US/Eastern")
    except Exception:
        return None

    now = now_utc or datetime.now(timezone.utc)
    aware_now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    now_et = aware_now.astimezone(eastern)

    # Use calendar-based check for holidays and weekends
    if not is_nyse_open_on_date(now_et):
        return None

    close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    if now_et > close:
        return 0.0

    delta = close - now_et
    return max(delta.total_seconds() / 60.0, 0.0)


def _resolve_offset_profile(is_long: bool, minutes_to_close: Optional[float], market_close_buffer_minutes: int):
    """
    Determine offset behaviour for the ramp.

    Returns (initial_offset, target_offset) already signed for the position side.
    """
    buffer_minutes = max(market_close_buffer_minutes, 0)
    initial = BACKOUT_INITIAL_OFFSET_PCT if is_long else -BACKOUT_INITIAL_OFFSET_PCT
    soft_cross = -BACKOUT_SOFT_CROSS_PCT if is_long else BACKOUT_SOFT_CROSS_PCT
    final_cross = -BACKOUT_FINAL_CROSS_PCT if is_long else BACKOUT_FINAL_CROSS_PCT

    if minutes_to_close is None:
        return initial, final_cross

    if buffer_minutes == 0 or minutes_to_close <= 0:
        return initial, final_cross

    if minutes_to_close >= buffer_minutes:
        return initial, soft_cross

    ratio = 1.0 - max(minutes_to_close, 0.0) / float(buffer_minutes)
    ratio = max(0.0, min(ratio, 1.0))
    dynamic_target = soft_cross + (final_cross - soft_cross) * ratio
    return initial, dynamic_target


def backout_whole_account_near_market(
    *,
    ramp_minutes: int = BACKOUT_RAMP_MINUTES_DEFAULT,
    market_after: int = BACKOUT_MARKET_AFTER_MINUTES_DEFAULT,
    sleep_interval: Optional[int] = None,
    start_offset_minutes: int = 0,
    market_close_buffer_minutes: int = BACKOUT_MARKET_CLOSE_BUFFER_MINUTES_DEFAULT,
    market_close_force_minutes: int = BACKOUT_MARKET_CLOSE_FORCE_MINUTES_DEFAULT,
) -> bool:
    """Sequentially back out every open position using backout_near_market.

    Intended for fast paper clean-ups: gathers open positions, deduplicates
    symbols, and applies the same ramp/market parameters to each. Returns True
    if all symbols completed without errors.
    """

    try:
        positions = filter_to_realistic_positions(alpaca_wrapper.get_all_positions())
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Unable to load positions for backout_whole_account_near_market: %s", exc)
        return False

    symbols = []
    seen = set()
    for pos in positions:
        sym = getattr(pos, "symbol", None)
        if not sym:
            continue
        if sym in seen:
            continue
        seen.add(sym)
        symbols.append(sym)

    if not symbols:
        logger.info("No open positions to back out; account already flat.")
        return True

    logger.info(
        "Starting backout_whole_account_near_market for %d symbols: %s",
        len(symbols),
        ", ".join(symbols),
    )

    overall_success = True
    for sym in symbols:
        logger.info("Initiating backout_near_market for %s", sym)
        try:
            result = backout_near_market(
                sym,
                start_time=datetime.now(),
                ramp_minutes=ramp_minutes,
                market_after=market_after,
                sleep_interval=sleep_interval,
                start_offset_minutes=start_offset_minutes,
                market_close_buffer_minutes=market_close_buffer_minutes,
                market_close_force_minutes=market_close_force_minutes,
            )
            if result is False:
                overall_success = False
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("backout_near_market failed for %s: %s", sym, exc)
            overall_success = False
    return overall_success


def backout_near_market(
    pair,
    start_time=None,
    ramp_minutes=BACKOUT_RAMP_MINUTES_DEFAULT,
    market_after=BACKOUT_MARKET_AFTER_MINUTES_DEFAULT,
    sleep_interval=None,
    *,
    start_offset_minutes: int = 0,
    market_close_buffer_minutes: int = BACKOUT_MARKET_CLOSE_BUFFER_MINUTES_DEFAULT,
    market_close_force_minutes: int = BACKOUT_MARKET_CLOSE_FORCE_MINUTES_DEFAULT,
):
    """Back out of an open position by progressively crossing the market.

    The function starts with a limit order slightly favourable to the
    current price and linearly ramps to the opposite side of the spread.
    The base ramp lasts ``ramp_minutes`` (default 30, configurable via
    ``BACKOUT_RAMP_MINUTES``) and can be extended by ``start_offset_minutes``
    to give more time before crossing. If the position is still open after
    ``market_after`` minutes (default 50, configurable via
    ``BACKOUT_MARKET_AFTER_MINUTES``) plus any start offset, or the market
    close is imminent, a market order is sent to guarantee the exit. During
    regular trading hours the limit offsets remain tight until within
    ``market_close_buffer_minutes`` of the close to minimise taker fees.

    **NOTE**: For crypto (24/7 trading), market orders are NEVER used due to
    wide spreads. Crypto positions are closed using limit orders only.

    Args:
        pair: The trading pair symbol, e.g. ``"META"``.
        start_time: ``datetime`` the ramp started. ``None`` means now.
        ramp_minutes: Minutes to complete the limit order ramp.
        market_after: Minutes before switching to a market order.
        sleep_interval: Seconds to wait between iterations.
    """
    if start_time is None:
        start_time = datetime.now()

    # Detect if this is a crypto symbol (24/7 trading)
    is_crypto = pair in crypto_symbols
    if is_crypto:
        logger.info(f"{pair} is crypto - will use limit orders only (no market order fallback)")
        # Disable market order fallback for crypto by setting to very high value
        effective_market_after = float('inf')
    else:
        effective_market_after = None  # Will be set below

    retries = 0
    max_retries = 5
    extra_minutes = max(int(start_offset_minutes), 0)
    effective_ramp_minutes = max(int(ramp_minutes) + extra_minutes, 1)
    # Only set effective_market_after for stocks; crypto already set to inf
    if effective_market_after is None:
        effective_market_after = max(int(market_after) + extra_minutes, effective_ramp_minutes)
    if sleep_interval is None:
        sleep_interval = BACKOUT_POLL_INTERVAL_SECONDS_DEFAULT
    sleep_interval = int(max(float(sleep_interval), 0.0))
    market_close_force_minutes = max(int(market_close_force_minutes), 0)
    market_close_buffer_minutes = max(int(market_close_buffer_minutes), 0)

    while True:
        try:
            all_positions = alpaca_wrapper.get_all_positions()
            logger.info(f"Retrieved {len(all_positions)} total positions")

            if len(all_positions) == 0:
                logger.info("no positions found, exiting")
                break

            positions = filter_to_realistic_positions(all_positions)
            logger.info(f"After filtering, {len(positions)} positions remain")

            # cancel all orders of pair
            orders = alpaca_wrapper.get_open_orders()
            logger.info(f"Found {len(orders)} open orders")

            for order in orders:
                if hasattr(order, 'symbol') and pairs_equal(order.symbol, pair):
                    logger.info(f"Cancelling order for {pair}")
                    alpaca_wrapper.cancel_order(order)
                    sleep(1)
                    break

            found_position = False
            for position in positions:
                if hasattr(position, 'symbol') and pairs_equal(position.symbol, pair):
                    logger.info(f"Found matching position for {pair}")
                    is_long = hasattr(position, 'side') and position.side == 'long'

                    minutes_since_start = (datetime.now() - start_time).seconds // 60
                    progress = min(minutes_since_start / effective_ramp_minutes, 1.0)

                    # Skip market close logic for crypto (24/7 trading)
                    if is_crypto:
                        minutes_to_close = None
                        force_market_due_to_time = False
                    else:
                        minutes_to_close = _minutes_until_market_close()
                        force_market_due_to_time = (
                            minutes_to_close is not None
                            and minutes_to_close <= market_close_force_minutes
                        )

                    pct_offset, pct_final_offset = _resolve_offset_profile(
                        is_long,
                        minutes_to_close,
                        market_close_buffer_minutes,
                    )
                    pct_above_market = pct_offset + (pct_final_offset - pct_offset) * progress

                    if minutes_since_start >= effective_market_after or force_market_due_to_time:
                        spread_pct = _current_spread_pct(pair)
                        if spread_pct is not None and spread_pct > BACKOUT_MARKET_MAX_SPREAD_PCT:
                            logger.info(
                                "Spread %.2f%% exceeds %.2f%% cap; holding limit order instead of market for %s",
                                spread_pct * 100.0,
                                BACKOUT_MARKET_MAX_SPREAD_PCT * 100.0,
                                pair,
                            )
                            pct_above_market = pct_final_offset
                        else:
                            if spread_pct is None:
                                logger.warning(
                                    "Spread unavailable for %s; proceeding with market order fallback",
                                    pair,
                                )
                            else:
                                logger.info(
                                    "Spread %.2f%% within %.2f%% cap; switching to market order for %s",
                                    spread_pct * 100.0,
                                    BACKOUT_MARKET_MAX_SPREAD_PCT * 100.0,
                                    pair,
                                )
                            succeeded = alpaca_wrapper.close_position_violently(position)
                            found_position = True
                            if succeeded:
                                logger.info("Market order fallback succeeded for %s", pair)
                            if not succeeded:
                                logger.info("Market order failed, will retry after delay")
                                retries += 1
                                if retries >= max_retries:
                                    logger.error("Max retries reached, exiting")
                                    return False
                                sleep(60)
                                continue
                            break
                    elif minutes_since_start >= effective_ramp_minutes:
                        # After ramp period, set price well beyond market to guarantee fill
                        pct_above_market = pct_final_offset

                    minutes_to_close_display = (
                        f"{minutes_to_close:.1f}" if minutes_to_close is not None else "n/a"
                    )
                    logger.info(
                        f"Position side: {'long' if is_long else 'short'}, "
                        f"pct_above_market: {pct_above_market:.4f}, "
                        f"minutes_since_start: {minutes_since_start}, "
                        f"progress: {progress:.2f}, "
                        f"minutes_to_close: {minutes_to_close_display}"
                    )

                    try:
                        succeeded = alpaca_wrapper.close_position_near_market(position,
                                                                            pct_above_market=pct_above_market)
                        found_position = True
                        if not succeeded:
                            logger.info("failed to close position, will retry after delay")
                            retries += 1
                            if retries >= max_retries:
                                logger.error("Max retries reached, exiting")
                                return False
                            sleep(60)
                            continue
                    except Exception as e:
                        logger.error(f"Error closing position: {e}")
                        retries += 1
                        if retries >= max_retries:
                            logger.error("Max retries reached, exiting")
                            return False
                        sleep(60)
                        continue

            if not found_position:
                logger.info(f"no position found or error closing for {pair}")
                return True
            retries = 0
            if sleep_interval > 0:
                dynamic_interval = sleep_interval
                sleep_minutes_to_close = _minutes_until_market_close()
                if sleep_minutes_to_close is not None:
                    scaled_interval = max(5, int(math.ceil(sleep_minutes_to_close * 6)))
                    dynamic_interval = min(sleep_interval, scaled_interval)
                sleep(dynamic_interval)

        except Exception as e:
            logger.error(f"Error in backout_near_market: {e}")
            retries += 1
            if retries >= max_retries:
                logger.error("Max retries reached, exiting")
                return False
            sleep(60)


def close_all_positions():
    positions = alpaca_wrapper.get_all_positions()

    for position in positions:
        if not hasattr(position, 'symbol'):
            continue

        symbol = position.symbol

        # get latest data then bid/ask
        data_client = _get_data_client()
        if data_client is not None:
            download_exchange_latest_data(data_client, symbol)
        bid = get_bid(symbol)
        ask = get_ask(symbol)

        current_price = ask if hasattr(position, 'side') and position.side == 'long' else bid
        # close a long with the ask price
        # close a short with the bid price
        # get bid/ask
        # get current price
        alpaca_wrapper.close_position_at_almost_current_price(
            position, {
                'close_last_price_minute': current_price
            }
        )
        # alpaca_order_stock(position.symbol, position.qty)


def violently_close_all_positions():
    """Close all positions using market orders.

    WARNING: Market orders are only allowed during market hours and when
    spread <= MARKET_ORDER_MAX_SPREAD_PCT (default 1%). If these conditions
    are not met, positions will NOT be closed. Use close_all_positions() for
    a safer alternative that uses limit orders.
    """
    positions = alpaca_wrapper.get_all_positions()
    for position in positions:
        result = alpaca_wrapper.close_position_violently(position)
        if result is None:
            logger.warning(
                f"Failed to close position {position.symbol} with market order - "
                f"may be due to market hours or high spread. Use close_all_positions() instead."
            )


def ramp_into_position(
    pair,
    side,
    start_time=None,
    target_qty=None,
    maxdiff_overflow=False,
    risk_threshold=None,
):
    """
    Ramp into a position with different strategies for crypto vs stocks:
    - Crypto: Start slightly worse than market price, ramp to opposite side over 1 hour
    - Stocks: More aggressive pricing starting at market, ramp over 1 hour
    - If target_qty is provided, will add to existing position to reach that target

    Args:
        pair: Trading symbol
        side: 'buy' or 'sell'
        start_time: Start time for the ramp
        target_qty: Optional target quantity
        maxdiff_overflow: If True, check leverage before placing orders
        risk_threshold: Risk threshold to check against (will be fetched if not provided)
    """
    if pair in crypto_symbols and side.lower() == "sell":
        logger.error(f"Cannot short crypto {pair}")
        return False

    if start_time is None:
        start_time = datetime.now()

    retries = 0
    max_retries = 5
    linear_ramp = 60  # 1 hour ramp for both crypto and stocks

    # Calculate target_qty ONCE at the start and cache it
    # This prevents recalculating on each iteration which causes multiple orders
    initial_target_qty = target_qty
    target_qty_cached = None

    while True:
        try:
            all_positions = alpaca_wrapper.get_all_positions()
            positions = filter_to_realistic_positions(all_positions)

            # Check current position size and calculate required quantity
            current_qty = 0
            existing_position = None
            for position in positions:
                if hasattr(position, 'symbol') and pairs_equal(position.symbol, pair):
                    current_qty = float(position.qty)
                    existing_position = position
                    logger.info(f"Existing position for {pair}: {current_qty} shares")
                    break

            # Calculate target_qty only on first iteration
            if target_qty_cached is None:
                if initial_target_qty is not None:
                    target_qty_cached = initial_target_qty
                    logger.info(f"Using provided target_qty: {target_qty_cached}")
                else:
                    # Get current market price to calculate target qty
                    data_client = _get_data_client()
                    if data_client is not None:
                        download_exchange_latest_data(data_client, pair)
                    bid_price = get_bid(pair)
                    ask_price = get_ask(pair)
                    if bid_price is None or ask_price is None:
                        logger.error(f"Failed to get bid/ask prices for {pair}")
                        return False
                    entry_price = ask_price if side == "buy" else bid_price

                    # Use the centralized get_qty function which includes exposure limits and risk management
                    target_qty_cached = get_qty(pair, entry_price, positions)
                    logger.info(f"Calculated target_qty from get_qty: {target_qty_cached}")

                    # If get_qty returns 0, we can't add more to this position
                    if target_qty_cached == 0:
                        logger.warning(f"Cannot add to position for {pair} - exposure limits reached or invalid quantity")
                        return True  # Return success since this is a risk management decision

            logger.info(f"Current position: {current_qty}, Target position: {target_qty_cached} (cached)")

            # Check if we already have the target position or more
            if current_qty >= target_qty_cached:
                logger.info(f"Position already at or above target for {pair} ({current_qty} >= {target_qty_cached})")
                return True

            # Calculate the quantity we need to add
            qty_to_add = target_qty_cached - current_qty
            logger.info(f"Need to add {qty_to_add} to reach target position")
            
            # Check for minimum order size to prevent tiny orders that fail
            min_order_size = 0.01 if pair in crypto_symbols else 1.0
            if abs(qty_to_add) < min_order_size:
                logger.info(f"Quantity to add ({qty_to_add}) is below minimum order size ({min_order_size}) for {pair}")
                return True  # Consider this a success since we're essentially at target

            # Cancel orders with retry logic
            cancel_attempts = 0
            max_cancel_attempts = 3
            orders_cancelled = False

            while cancel_attempts < max_cancel_attempts:
                try:
                    logger.info(f"Attempting to cancel orders for {pair}...")
                    # Get all open orders
                    orders = alpaca_wrapper.get_open_orders()
                    pair_orders = [order for order in orders if
                                   hasattr(order, 'symbol') and pairs_equal(order.symbol, pair)]

                    if not pair_orders:
                        orders_cancelled = True
                        logger.info(f"No existing orders found for {pair}")
                        break

                    # Cancel only orders for this pair
                    for order in pair_orders:
                        alpaca_wrapper.cancel_order(order)
                        sleep(1)  # Small delay between cancellations

                    # Verify cancellations
                    sleep(3)  # Let cancellations propagate
                    orders = alpaca_wrapper.get_open_orders()
                    pair_orders_all = [order for order in orders if
                                       hasattr(order, 'symbol') and pairs_equal(order.symbol, pair)]
                    # Filter out orders that are already pending cancellation - they're "good enough"
                    remaining_orders = [order for order in pair_orders_all
                                        if getattr(order, 'status', None) not in ['pending_cancel', 'cancelled']]

                    # Log order states for debugging
                    if pair_orders_all and len(pair_orders_all) != len(remaining_orders):
                        pending_count = len(pair_orders_all) - len(remaining_orders)
                        logger.info(f"{pending_count} orders for {pair} are pending_cancel/cancelled, treating as success")

                    if not remaining_orders:
                        orders_cancelled = True
                        logger.info(f"All orders for {pair} successfully cancelled or pending cancellation")
                        break
                    else:
                        logger.info(f"Found {len(remaining_orders)} remaining orders for {pair} (excluding pending_cancel), retrying cancellation")

                    cancel_attempts += 1
                    if not orders_cancelled:
                        sleep(5)  # Wait before retry

                except Exception as e:
                    logger.error(f"Error during order cancellation: {e}")
                    cancel_attempts += 1
                    sleep(5)

            if not orders_cancelled:
                logger.error("Failed to cancel orders after maximum attempts")
                retries += 1
                if retries >= max_retries:
                    logger.error("Max retries reached, exiting")
                    return False
                sleep(30)
                continue

            # Get current market prices
            try:
                data_client = _get_data_client()
                if data_client is None:
                    logger.error("Quote client unavailable; cannot fetch data for %s", pair)
                    retries += 1
                    if retries >= max_retries:
                        return False
                    sleep(30)
                    continue
                download_exchange_latest_data(data_client, pair)
                bid_price = get_bid(pair)
                ask_price = get_ask(pair)

                if bid_price is None or ask_price is None:
                    logger.error(f"Failed to get bid/ask prices for {pair}")
                    retries += 1
                    if retries >= max_retries:
                        return False
                    sleep(30)
                    continue

                minutes_since_start = (datetime.now() - start_time).seconds // 60

                # Calculate the price to place the order
                if pair in crypto_symbols:
                    # For crypto, start slightly worse than market and slowly move to other side
                    offset = 0.0004  # 0.04% initial offset from market
                    if side == "buy":
                        if minutes_since_start >= linear_ramp:
                            order_price = ask_price  # End at ask
                        else:
                            # Start slightly below bid, move to ask
                            progress = minutes_since_start / linear_ramp
                            start_price = bid_price * (1 - offset)  # Start worse than bid
                            price_range = ask_price - start_price
                            order_price = start_price + (price_range * progress)
                    else:  # sell
                        if minutes_since_start >= linear_ramp:
                            order_price = bid_price  # End at bid
                        else:
                            # Start slightly above ask, move to bid
                            progress = minutes_since_start / linear_ramp
                            start_price = ask_price * (1 + offset)  # Start worse than ask
                            price_range = bid_price - start_price
                            order_price = start_price + (price_range * progress)

                    logger.info(f"Crypto order: Starting at {'below bid' if side == 'buy' else 'above ask'}, "
                                f"progress {progress:.2%}, price {order_price:.2f}")
                else:
                    # For stocks, be more aggressive
                    if minutes_since_start >= linear_ramp:
                        order_price = ask_price if side == "buy" else bid_price
                    else:
                        # Start at market and move slightly away
                        progress = minutes_since_start / linear_ramp
                        if side == "buy":
                            price_range = ask_price - bid_price
                            order_price = bid_price + (price_range * progress)
                        else:
                            price_range = ask_price - bid_price
                            order_price = ask_price - (price_range * progress)

                # Use the quantity we calculated earlier (qty_to_add)
                qty = qty_to_add

                if qty <= 0:
                    logger.error(f"Calculated qty {qty} is invalid")
                    return False

                logger.info(f"Attempting to place order: {pair} {side} {qty} @ {order_price} (adding to existing {current_qty})")

                # Check account status before placing order
                try:
                    buying_power_check = alpaca_wrapper.cash
                    total_buying_power_check = alpaca_wrapper.total_buying_power
                    logger.info(f"Account status - Cash: ${buying_power_check:.2f}, Total buying power: ${total_buying_power_check:.2f}")
                    estimated_cost = qty * order_price
                    logger.info(f"Estimated order cost: ${estimated_cost:.2f}")
                except Exception as e:
                    logger.error(f"Error checking account status: {e}")

                # Check leverage constraint for maxdiff overflow trades
                if maxdiff_overflow:
                    try:
                        # Get current risk threshold
                        effective_risk_threshold = risk_threshold
                        if effective_risk_threshold is None:
                            effective_risk_threshold = get_global_risk_threshold()

                        # Calculate current total exposure
                        all_positions = alpaca_wrapper.get_all_positions()
                        current_positions = filter_to_realistic_positions(all_positions)
                        current_exposure = _calculate_total_exposure_value(current_positions)

                        # Calculate max allowed exposure and available room
                        equity = float(getattr(alpaca_wrapper, "equity", 0.0) or 0.0)
                        if equity <= 0:
                            logger.warning(f"Invalid equity ${equity:.2f} for {pair}, skipping leverage check")
                        else:
                            max_allowed_exposure = equity * effective_risk_threshold
                            available_room = max_allowed_exposure - current_exposure

                            # Calculate projected exposure including this order
                            order_value = abs(qty * order_price)
                            projected_exposure = current_exposure + order_value
                            projected_leverage = projected_exposure / equity

                            # Check if leverage would exceed threshold
                            if projected_leverage > effective_risk_threshold:
                                # Try to shrink the order to fit within leverage budget
                                if available_room > 0:
                                    # Calculate adjusted quantity to fit
                                    adjusted_order_value = available_room * 0.99  # Use 99% to leave buffer
                                    adjusted_qty = adjusted_order_value / abs(order_price)

                                    # Check minimum order size
                                    min_order_size = 0.01 if pair in crypto_symbols else 1.0

                                    if adjusted_qty >= min_order_size:
                                        logger.info(
                                            f"{pair} maxdiff overflow: shrinking order from {qty:.4f} to {adjusted_qty:.4f} "
                                            f"to fit within leverage threshold {effective_risk_threshold:.2f}x. "
                                            f"Available room: ${available_room:.2f}, Current exposure: ${current_exposure:.2f}, "
                                            f"Equity: ${equity:.2f}"
                                        )
                                        qty = adjusted_qty
                                        qty_to_add = adjusted_qty  # Keep qty_to_add in sync
                                        # Recalculate order value with adjusted qty
                                        order_value = abs(qty * order_price)
                                    else:
                                        logger.info(
                                            f"Skipping {pair} maxdiff overflow order: adjusted qty {adjusted_qty:.4f} "
                                            f"below minimum {min_order_size}. Available room: ${available_room:.2f}. "
                                            f"Will retry in next iteration."
                                        )
                                        # Reset retries since this is not an error
                                        retries = 0
                                        sleep_time = 5 * 60 if pair in crypto_symbols else 2 * 60
                                        sleep(sleep_time)
                                        continue
                                else:
                                    logger.info(
                                        f"Skipping {pair} maxdiff overflow order: no available leverage room. "
                                        f"Projected leverage {projected_leverage:.2f}x exceeds threshold {effective_risk_threshold:.2f}x. "
                                        f"Current exposure: ${current_exposure:.2f}, Max allowed: ${max_allowed_exposure:.2f}, "
                                        f"Equity: ${equity:.2f}. Will retry in next iteration."
                                    )
                                    # Reset retries since this is not an error
                                    retries = 0
                                    sleep_time = 5 * 60 if pair in crypto_symbols else 2 * 60
                                    sleep(sleep_time)
                                    continue
                            else:
                                logger.info(
                                    f"{pair} maxdiff overflow leverage check passed: projected leverage "
                                    f"{projected_leverage:.2f}x <= risk threshold {effective_risk_threshold:.2f}x. "
                                    f"Proceeding with order qty={qty:.4f}."
                                )
                    except Exception as e:
                        logger.warning(f"Error during leverage check for {pair}: {e}. Proceeding with order anyway.")

                # Place the order with error handling using new function that allows adding to positions
                try:
                    succeeded = alpaca_wrapper.open_order_at_price_allow_add_to_position(pair, qty, side, order_price)
                    logger.info(f"Order result: {succeeded} (type: {type(succeeded)})")
                    
                    if succeeded is None:
                        logger.error("Order placement returned None - check alpaca_wrapper logs for details")
                        retries += 1
                        if retries >= max_retries:
                            logger.error("Max retries reached, exiting")
                            return False
                        sleep(60)
                        continue
                    elif not succeeded:
                        logger.error("Order placement returned False")
                        retries += 1
                        if retries >= max_retries:
                            logger.error("Max retries reached, exiting")
                            return False
                        sleep(60)
                        continue
                    else:
                        logger.info(f"Order placed successfully: {succeeded} (ID: {succeeded.id if hasattr(succeeded, 'id') else 'N/A'})")
                    
                    # Order was successful, continue to reset retries and sleep
                    
                except Exception as e:
                    logger.error(f"Exception during order placement: {e}")
                    traceback.print_exc()
                    
                    # Check if it's an insufficient funds error and try to adjust quantity
                    error_str = str(e)
                    if "insufficient" in error_str.lower():
                        logger.warning("Insufficient funds detected, will retry with adjusted quantity on next iteration")
                    
                    retries += 1
                    if retries >= max_retries:
                        logger.error("Max retries reached, exiting")
                        return False
                    sleep(60)
                    continue

                # Reset retries on successful order placement  
                retries = 0

                # Longer sleep for crypto to reduce API calls
                sleep_time = 5 * 60 if pair in crypto_symbols else 2 * 60
                sleep(sleep_time)

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error during order placement: {e}")
                retries += 1
                if retries >= max_retries:
                    logger.error("Max retries reached, exiting")
                    return False
                sleep(60)
                continue

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in ramp_into_position main loop: {e}")
            retries += 1
            if retries >= max_retries:
                logger.error("Max retries reached, exiting")
                return False
            sleep(60)


def show_account():
    """Display account summary including positions, orders and market status"""
    # Get market clock using wrapper
    clock = alpaca_wrapper.get_clock()

    # Convert times to NZDT and EDT
    nz_tz = pytz.timezone('Pacific/Auckland')
    edt_tz = pytz.timezone('America/New_York')

    current_time_nz = datetime.now(timezone.utc).astimezone(nz_tz)
    current_time_edt = datetime.now(timezone.utc).astimezone(edt_tz)

    # Print market status and times
    logger.info("\n=== Market Status ===")
    logger.info(f"Market is {'OPEN' if clock.is_open else 'CLOSED'}")
    logger.info(f"Current time (NZDT): {current_time_nz.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"Current time (EDT): {current_time_edt.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    # Get account info
    logger.info("\n=== Account Summary ===")
    logger.info(f"Equity: ${alpaca_wrapper.equity:,.2f}")
    logger.info(f"Cash: ${alpaca_wrapper.cash:,.2f}")
    logger.info(f"Buying Power: ${alpaca_wrapper.total_buying_power:,.2f}")

    # Get and display positions
    positions = alpaca_wrapper.get_all_positions()
    logger.info("\n=== Open Positions ===")
    if not positions:
        logger.info("No open positions")
    else:
        for pos in positions:
            if hasattr(pos, 'symbol') and hasattr(pos, 'qty') and hasattr(pos, 'current_price'):
                side = "LONG" if hasattr(pos, 'side') and pos.side == 'long' else "SHORT"
                logger.info(f"{pos.symbol}: {side} {pos.qty} shares @ ${float(pos.current_price):,.2f}")

    # Get and display orders
    orders = alpaca_wrapper.get_open_orders()
    logger.info("\n=== Open Orders ===")
    if not orders:
        logger.info("No open orders")
    else:
        for order in orders:
            if hasattr(order, 'symbol') and hasattr(order, 'qty'):
                price_str = f"@ ${float(order.limit_price):,.2f}" if hasattr(order, 'limit_price') else "(market)"
                logger.info(f"{order.symbol}: {order.side.upper()} {order.qty} {price_str}")

    # Display maxdiff plans for today
    try:
        from trade_stock_e2e import _load_maxdiff_plans_for_today, _calculate_total_exposure_value
        from src.portfolio_risk import get_global_risk_threshold
        from src.trading_obj_utils import filter_to_realistic_positions

        maxdiff_plans = _load_maxdiff_plans_for_today()
        if maxdiff_plans:
            logger.info("\n=== Maxdiff Plans (Parallel Trading Layer) ===")

            # Calculate leverage info
            all_positions = alpaca_wrapper.get_all_positions()
            realistic_positions = filter_to_realistic_positions(all_positions)
            current_exposure = _calculate_total_exposure_value(realistic_positions)
            equity = float(getattr(alpaca_wrapper, "equity", 0.0) or 0.0)

            if equity > 0:
                current_leverage = current_exposure / equity
                risk_threshold = get_global_risk_threshold()
                max_allowed_exposure = equity * risk_threshold
                available_room = max(0, max_allowed_exposure - current_exposure)

                logger.info(f"Current Leverage: {current_leverage:.2f}x (${current_exposure:,.2f} / ${equity:,.2f})")
                logger.info(f"Global Risk Threshold: {risk_threshold:.2f}x")
                logger.info(f"Available for Maxdiff: ${available_room:,.2f}")

            # Sort plans by avg_return
            sorted_plans = sorted(
                maxdiff_plans.items(),
                key=lambda x: x[1].get("avg_return", 0.0),
                reverse=True
            )

            active_plans = [(sym, plan) for sym, plan in sorted_plans if plan.get("status") in {"identified", "listening", "spawned"}]
            filled_plans = [(sym, plan) for sym, plan in sorted_plans if plan.get("status") == "filled"]

            if active_plans:
                logger.info(f"\nActive Maxdiff Opportunities ({len(active_plans)}):")
                for symbol, plan in active_plans:
                    avg_ret = plan.get("avg_return", 0.0)
                    high = plan.get("maxdiffprofit_high_price") or plan.get("high_target", 0.0)
                    low = plan.get("maxdiffprofit_low_price") or plan.get("low_target", 0.0)
                    status = plan.get("status", "unknown")
                    logger.info(f"  {symbol}: high=${high:.2f} low=${low:.2f} avg_return={avg_ret:.4f} [{status}]")

            if filled_plans:
                logger.info(f"\nFilled Maxdiff Today ({len(filled_plans)}):")
                for symbol, plan in filled_plans:
                    avg_ret = plan.get("avg_return", 0.0)
                    logger.info(f"  {symbol}: avg_return={avg_ret:.4f} [filled]")
    except Exception as e:
        logger.warning(f"Failed to load maxdiff plans: {e}")


def close_position_at_takeprofit(pair: str, takeprofit_price: float, start_time=None):
    """
    Wait for up to 1 hour or 24 hours if symbol is under "highlow" strategy,
    then place a limit order to close that position at takeprofit_price.
    If no position is opened within the hour, or if something fails, exit.
    """
    from datetime import datetime
    from time import sleep

    if start_time is None:
        start_time = datetime.now()

    # Determine wait time by strategy
    strategy = get_strategy_for_symbol(pair)
    if strategy == "highlow":
        max_wait_minutes = 24 * 60
        logger.info(f"{pair} is traded with 'highlow' strategy, using 24-hour wait.")
    else:
        max_wait_minutes = 60  # default

    while True:
        elapsed_minutes = (datetime.now() - start_time).seconds // 60
        if elapsed_minutes >= max_wait_minutes:
            logger.error(f"Timed out waiting for position in {pair} under strategy={strategy}")
            return False

        all_positions = alpaca_wrapper.get_all_positions()
        positions = [p for p in all_positions if hasattr(p, 'symbol') and pairs_equal(p.symbol, pair)]
        if not positions:
            logger.info(f"No position for {pair} yet – waiting. Elapsed: {elapsed_minutes} min")
            sleep(30)
            continue

        # We have at least one matching position
        position = positions[0]
        logger.info(f"Position found for {pair}: side={position.side}, qty={position.qty}")

        # Cancel existing orders for this pair
        orders = alpaca_wrapper.get_open_orders()
        for order in orders:
            if hasattr(order, 'symbol') and pairs_equal(order.symbol, pair):
                logger.info(f"Cancelling order for {pair} before placing takeprofit limit")
                alpaca_wrapper.cancel_order(order)
                sleep(1)

        # Place the takeprofit order
        logger.info(f"Placing limit order to close {pair} at {takeprofit_price}")
        try:
            side = 'sell' if position.side == 'long' else 'buy'
            alpaca_wrapper.open_order_at_price(pair, position.qty, side, takeprofit_price)
            return True
        except Exception as e:
            logger.error(f"Failed to place takeprofit limit order: {e}")
            return False


def show_forecasts_for_symbol(symbol: str):
    """Display forecast predictions for a symbol, using cached data when markets are closed"""
    try:
        # Import here to avoid circular imports
        from show_forecasts import show_forecasts
        show_forecasts(symbol)
    except Exception as e:
        logger.error(f"Error showing forecasts for {symbol}: {e}")


def debug_raw_data(symbol: str):
    """Print raw JSON data from Alpaca for debugging bid/ask issues"""
    import json
    logger.info(f"=== DEBUG RAW DATA FOR {symbol} ===")
    
    try:
        # Get the raw data from alpaca_wrapper
        raw_data = alpaca_wrapper.latest_data(symbol)
        logger.info(f"Raw data type: {type(raw_data)}")
        logger.info(f"Raw data object: {raw_data}")
        
        # Try to convert to dict if it has attributes
        data_dict = {}
        for attr in dir(raw_data):
            if not attr.startswith('_'):
                try:
                    value = getattr(raw_data, attr)
                    if not callable(value):
                        data_dict[attr] = value
                except Exception:
                    pass
        
        logger.info(f"Raw data attributes as dict:")
        logger.info(json.dumps(data_dict, indent=2, default=str))
        
        # Extract specific fields we care about
        if hasattr(raw_data, 'ask_price'):
            logger.info(f"ask_price: {raw_data.ask_price} (type: {type(raw_data.ask_price)})")
        if hasattr(raw_data, 'bid_price'):
            logger.info(f"bid_price: {raw_data.bid_price} (type: {type(raw_data.bid_price)})")
        if hasattr(raw_data, 'ask_size'):
            logger.info(f"ask_size: {raw_data.ask_size}")
        if hasattr(raw_data, 'bid_size'):
            logger.info(f"bid_size: {raw_data.bid_size}")
        if hasattr(raw_data, 'timestamp'):
            logger.info(f"timestamp: {raw_data.timestamp}")
        
    except Exception as e:
        logger.error(f"Error getting raw data for {symbol}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    typer.run(main)
    # close_all_positions()
