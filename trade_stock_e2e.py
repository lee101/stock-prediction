from datetime import datetime
from math import floor
from time import sleep
from typing import List, Dict

import pytz
from loguru import logger

import alpaca_wrapper
from backtest_test3_inline import backtest_forecasts
from data_curate_daily import get_bid, get_ask, download_exchange_latest_data
from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
from src.comparisons import is_buy_side, is_same_side, is_sell_side
from src.date_utils import is_nyse_trading_day_now, is_nyse_trading_day_ending
from src.fixtures import crypto_symbols
from src.logging_utils import setup_logging
from src.trading_obj_utils import filter_to_realistic_positions
from src.process_utils import backout_near_market, ramp_into_position, spawn_close_position_at_takeprofit
from alpaca.data import StockHistoricalDataClient

# Configure logging
logger = setup_logging("trade_stock_e2e.log")


def get_market_hours() -> tuple:
    """Get market open and close times in EST."""
    est = pytz.timezone("US/Eastern")
    now = datetime.now(est)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open, market_close


def analyze_symbols(symbols: List[str]) -> Dict:
    """Run backtest analysis on symbols and return results sorted by average return."""
    results = {}

    for symbol in symbols:
        try:
            logger.info(f"Analyzing {symbol}")
            # not many because we need to adapt strats? eg the wierd spikes in uniusd are a big opportunity to trade w high/low
            # but then i bumped up because its not going to say buy crypto when its down, if its most recent based?
            num_simulations = 70 

            backtest_df = backtest_forecasts(symbol, num_simulations)
            # Get each strategy's average return
            simple_return = backtest_df["simple_strategy_return"].mean()
            all_signals_return = backtest_df["all_signals_strategy_return"].mean()
            takeprofit_return = backtest_df["entry_takeprofit_return"].mean()
            # Include highlow_return in our analysis
            highlow_return = backtest_df["highlow_return"].mean()

            # Compare all four strategy returns
            best_return = max(simple_return, all_signals_return, takeprofit_return, highlow_return)
            last_prediction = backtest_df.iloc[-1]

            if best_return == takeprofit_return:
                avg_return = takeprofit_return
                strategy = "takeprofit"
                predicted_movement = last_prediction["predicted_close"] - last_prediction["close"]
                position_side = "buy" if predicted_movement > 0 else "sell"
            elif best_return == all_signals_return:
                avg_return = all_signals_return
                strategy = "all_signals"
                # existing code to pick side from signals
                close_movement = last_prediction["predicted_close"] - last_prediction["close"]
                high_movement = last_prediction["predicted_high"] - last_prediction["close"]
                low_movement = last_prediction["predicted_low"] - last_prediction["close"]
                if all(x > 0 for x in [close_movement, high_movement, low_movement]):
                    position_side = "buy"
                elif all(x < 0 for x in [close_movement, high_movement, low_movement]):
                    position_side = "sell"
                else:
                    continue
                predicted_movement = close_movement
            elif best_return == highlow_return:
                avg_return = highlow_return
                strategy = "highlow"
                predicted_movement = last_prediction["predicted_close"] - last_prediction["close"]
                position_side = "buy" if predicted_movement > 0 else "sell"
            else:
                avg_return = simple_return
                strategy = "simple"
                predicted_movement = last_prediction["predicted_close"] - last_prediction["close"]
                position_side = "buy" if predicted_movement > 0 else "sell"

            results[symbol] = {
                "avg_return": avg_return,
                "predictions": backtest_df,
                "side": position_side,
                "predicted_movement": predicted_movement,
                "strategy": strategy,
                "predicted_high": float(last_prediction["predicted_high"]),
                "predicted_low": float(last_prediction["predicted_low"]),
            }

            logger.info(
                f"Analysis complete for {symbol}: best_strat={strategy}, avg_return={avg_return:.3f}, side={position_side}"
            )
            logger.info(f"Predicted movement: {predicted_movement:.3f}")
            logger.info(
                f"Predicted High: {last_prediction['predicted_high']:.3f}, "
                f"Predicted Low: {last_prediction['predicted_low']:.3f}, "
                f"Current Close: {last_prediction['close']:.3f}"
            )
            logger.info(f"Predicted Close: {last_prediction['predicted_close']:.3f}")

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            continue

    return dict(sorted(results.items(), key=lambda x: x[1]["avg_return"], reverse=True))


def log_trading_plan(picks: Dict[str, Dict], action: str):
    """Log the trading plan without executing trades."""
    logger.info(f"\n{'=' * 50}\nTRADING PLAN ({action})\n{'=' * 50}")

    for symbol, data in picks.items():
        logger.info(
            f"""
Symbol: {symbol}
Direction: {data['side']}
Avg Return: {data['avg_return']:.3f}
Predicted Movement: {data['predicted_movement']:.3f}
{'=' * 30}"""
        )


def manage_positions(
        current_picks: Dict[str, Dict],
        previous_picks: Dict[str, Dict],
        all_analyzed_results: Dict[str, Dict],
):
    """Execute actual position management."""
    positions = alpaca_wrapper.get_all_positions()
    positions = filter_to_realistic_positions(positions)
    logger.info("\nEXECUTING POSITION CHANGES:")

    if not positions:
        logger.info("No positions to analyze")

    if not all_analyzed_results and not current_picks:
        logger.warning(
            "No analysis results available - skipping position closure checks"
        )
        return

    # Handle position closures
    for position in positions:
        symbol = position.symbol
        should_close = False

        if symbol not in current_picks:
            # For crypto on weekends, only close if direction changed
            if symbol in crypto_symbols and not is_nyse_trading_day_now():
                if symbol in all_analyzed_results and not is_same_side(all_analyzed_results[symbol]["side"], position.side):
                    logger.info(f"Closing crypto position for {symbol} due to direction change (weekend)")
                    should_close = True
                else:
                    logger.info(f"Keeping crypto position for {symbol} on weekend - no direction change")
            # For stocks when market is closed, only close if direction changed
            elif symbol not in crypto_symbols and not is_nyse_trading_day_now():
                if symbol in all_analyzed_results and not is_same_side(all_analyzed_results[symbol]["side"], position.side):
                    logger.info(f"Closing stock position for {symbol} due to direction change (market closed)")
                    should_close = True
                else:
                    logger.info(f"Keeping stock position for {symbol} when market closed - no direction change")
            else:
                logger.info(f"Closing position for {symbol} as it's no longer in top picks")
                should_close = True
        elif symbol not in all_analyzed_results:
            # Only close positions when no analysis data if it's a short position and market is open
            if is_sell_side(position.side) and is_nyse_trading_day_now():
                logger.info(f"Closing short position for {symbol} as no analysis data available and market is open - reducing risk")
                should_close = True
            else:
                logger.info(f"No analysis data for {symbol} but keeping position (not a short or market not open)")
        elif not is_same_side(all_analyzed_results[symbol]["side"], position.side):
            logger.info(
                f"Closing position for {symbol} due to direction change from {position.side} to {all_analyzed_results[symbol]['side']}"
            )
            should_close = True

        if should_close:
            backout_near_market(symbol)

    # Enter new positions from current_picks
    if not current_picks:
        logger.warning("No current picks available - skipping new position entry")
        return

    logger.info(f"Current picks to attempt entering: {current_picks}")
    for symbol, data in current_picks.items():
        position_exists = any(p.symbol == symbol for p in positions)
        correct_side = any(
            p.symbol == symbol and is_same_side(p.side, data["side"]) for p in positions
        )
        
        # Calculate current position size and target size
        current_position_size = 0
        current_position_value = 0
        for p in positions:
            if p.symbol == symbol:
                current_position_size = float(p.qty)
                if hasattr(p, 'current_price'):
                    current_position_value = current_position_size * float(p.current_price)
                break
        
        # Calculate target position size
        client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
        download_exchange_latest_data(client, symbol)
        bid_price = get_bid(symbol)
        ask_price = get_ask(symbol)
        
        should_enter = False
        needs_size_increase = False
        
        if bid_price is not None and ask_price is not None:
            entry_price = ask_price if data["side"] == "buy" else bid_price
            target_qty = get_qty(symbol, entry_price, positions)
            target_value = target_qty * entry_price
            
            logger.info(f"{symbol}: Current position: {current_position_size} qty (${current_position_value:.2f}), Target: {target_qty} qty (${target_value:.2f})")
            
            # Check if we need to enter or increase position
            if symbol in crypto_symbols:
                should_enter = (not position_exists and is_buy_side(data["side"])) or (current_position_size < target_qty * 0.95)  # 5% tolerance
            else:
                should_enter = not position_exists or (current_position_size < target_qty * 0.95)
                
            needs_size_increase = current_position_size > 0 and current_position_size < target_qty * 0.95
        else:
            # Fallback to old logic if we can't get prices
            if symbol in crypto_symbols:
                should_enter = not position_exists and is_buy_side(data["side"])
            else:
                should_enter = not position_exists

        if should_enter or not correct_side:
            if needs_size_increase and bid_price is not None and ask_price is not None:
                entry_price = ask_price if data["side"] == "buy" else bid_price
                target_qty_for_log = get_qty(symbol, entry_price, positions)
                logger.info(f"Increasing existing {data['side']} position for {symbol} from {current_position_size} to {target_qty_for_log}")
            else:
                logger.info(f"Entering new {data['side']} position for {symbol}")
            
            # Use the prices and target_qty we already calculated
            if bid_price is not None and ask_price is not None:
                entry_price = ask_price if data["side"] == "buy" else bid_price
                target_qty = get_qty(symbol, entry_price, positions)  # Recalculate to be safe
                logger.info(f"Target quantity for {symbol}: {target_qty} at price {entry_price}")
                ramp_into_position(symbol, data["side"], target_qty=target_qty)
            else:
                logger.warning(f"Could not get bid/ask prices for {symbol}, using default sizing")
                ramp_into_position(symbol, data["side"])

            # If strategy is 'takeprofit', place a takeprofit limit later
            if data["strategy"] == "takeprofit" and is_buy_side(data["side"]):
                # e.g. call close_position_at_takeprofit with predicted_high
                tp_price = data["predicted_high"]
                logger.info(f"Scheduling a takeprofit at {tp_price:.3f} for {symbol}")
                # call the new function from alpaca_cli
                spawn_close_position_at_takeprofit(symbol, tp_price)
            elif data["strategy"] == "takeprofit" and is_sell_side(data["side"]):
                # If short, we might want to place a limit buy at predicted_low
                # (though you'd need to store predicted_low similarly)
                # For example:
                predicted_low = data["predictions"].iloc[-1]["predicted_low"]
                logger.info(f"Scheduling a takeprofit at {predicted_low:.3f} for short {symbol}")
                spawn_close_position_at_takeprofit(symbol, predicted_low)

            # If strategy is 'highlow', place a limit order at predicted_low (for buys)
            # or predicted_high (for shorts), and then schedule a takeprofit at the opposite predicted price.
            elif data["strategy"] == "highlow":
                if data["side"] == "buy":
                    entry_price = data["predicted_low"]
                    logger.info(
                        f"(Highlow) Placing limit BUY order for {symbol} at predicted_low={entry_price:.2f}"
                    )
                    qty = get_qty(symbol, entry_price, positions)
                    alpaca_wrapper.open_order_at_price_or_all(symbol, qty=qty, side="buy", price=entry_price)

                    tp_price = data["predicted_high"]
                    logger.info(f"(Highlow) Scheduling takeprofit at predicted_high={tp_price:.3f} for {symbol}")
                    spawn_close_position_at_takeprofit(symbol, tp_price)
                else:
                    entry_price = data["predicted_high"]
                    logger.info(
                        f"(Highlow) Placing limit SELL/short order for {symbol} at predicted_high={entry_price:.2f}"
                    )
                    qty = get_qty(symbol, entry_price, positions)
                    alpaca_wrapper.open_order_at_price_or_all(symbol, qty=qty, side="sell", price=entry_price)

                    tp_price = data["predicted_low"]
                    logger.info(f"(Highlow) Scheduling takeprofit at predicted_low={tp_price:.3f} for short {symbol}")
                    spawn_close_position_at_takeprofit(symbol, tp_price)

def get_current_symbol_exposure(symbol, positions):
    """Calculate current exposure to a symbol as percentage of total equity."""
    total_exposure = 0
    equity = alpaca_wrapper.equity
    
    for position in positions:
        if position.symbol == symbol:
            market_value = float(position.market_value) if position.market_value else 0
            total_exposure += abs(market_value)  # Use abs to account for short positions
    
    return (total_exposure / equity) * 100 if equity > 0 else 0


def get_qty(symbol, entry_price, positions=None):
    """Calculate quantity with 60% max exposure check per symbol."""
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
    
    # Calculate max qty based on remaining exposure allowance
    max_additional_value = (remaining_exposure_pct / 100) * equity
    qty_from_exposure_limit = max_additional_value / entry_price
    
    # Use the smaller of the two
    qty = min(qty_from_buying_power, qty_from_exposure_limit)
    
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

def manage_market_close(
        symbols: List[str],
        previous_picks: Dict[str, Dict],
        all_analyzed_results: Dict[str, Dict],
):
    """Execute market close position management."""
    logger.info("Managing positions for market close")

    if not all_analyzed_results:
        logger.warning("No analysis results available - keeping all positions open")
        return previous_picks

    positions = alpaca_wrapper.get_all_positions()
    positions = filter_to_realistic_positions(positions)
    if not positions:
        logger.info("No positions to manage for market close")
        return {
            symbol: data
            for symbol, data in list(all_analyzed_results.items())[:4]
            if data["avg_return"] > 0
        }

    # Close positions only when forecast shows opposite direction
    for position in positions:
        symbol = position.symbol
        should_close = False

        if symbol in all_analyzed_results:
            next_forecast = all_analyzed_results[symbol]
            if not is_same_side(next_forecast["side"], position.side):
                logger.info(
                    f"Closing position for {symbol} due to predicted direction change from {position.side} to {next_forecast['side']} tomorrow"
                )
                logger.info(
                    f"Predicted movement: {next_forecast['predicted_movement']:.3f}"
                )
                should_close = True
            else:
                logger.info(
                    f"Keeping {symbol} position as forecast matches current {position.side} direction"
                )
        else:
            logger.warning(f"No analysis data for {symbol} - keeping position")

        if should_close:
            backout_near_market(symbol)

    # Return top picks for next day
    return {
        symbol: data
        for symbol, data in list(all_analyzed_results.items())[:4]
        if data["avg_return"] > 0
    }


def analyze_next_day_positions(symbols: List[str]) -> Dict:
    """Analyze symbols for next day's trading session."""
    logger.info("Analyzing positions for next trading day")
    return analyze_symbols(symbols)  # Reuse existing analysis function


def dry_run_manage_positions(
        current_picks: Dict[str, Dict], previous_picks: Dict[str, Dict]
):
    """Simulate position management without executing trades."""
    positions = alpaca_wrapper.get_all_positions()
    positions = filter_to_realistic_positions(positions)

    logger.info("\nPLANNED POSITION CHANGES:")

    # Log position closures
    for position in positions:
        symbol = position.symbol
        should_close = False

        if symbol not in current_picks:
            # For crypto on weekends, only close if direction changed
            if symbol in crypto_symbols and not is_nyse_trading_day_now():
                logger.info(f"Would keep crypto position for {symbol} on weekend - no direction change check needed in dry run")
            # For stocks when market is closed, only close if direction changed  
            elif symbol not in crypto_symbols and not is_nyse_trading_day_now():
                logger.info(f"Would keep stock position for {symbol} when market closed - no direction change check needed in dry run")
            else:
                logger.info(
                    f"Would close position for {symbol} as it's no longer in top picks"
                )
                should_close = True
        elif symbol in current_picks and not is_same_side(current_picks[symbol]["side"], position.side):
            logger.info(
                f"Would close position for {symbol} to switch direction from {position.side} to {current_picks[symbol]['side']}"
            )
            should_close = True

    # Log new positions
    for symbol, data in current_picks.items():
        position_exists = any(p.symbol == symbol for p in positions)
        correct_side = any(
            p.symbol == symbol and is_same_side(p.side, data["side"]) for p in positions
        )

        if not position_exists or not correct_side:
            logger.info(f"Would enter new {data['side']} position for {symbol}")


def main():
    symbols = [
        "COUR",
        "GOOG",
        "TSLA",
        "NVDA",
        "AAPL",
        "U",
        "ADSK",
        "CRWD",
        "ADBE",
        "NET",
        "COIN",
        #"MSFT",
        # "NFLX",
        # adding more as we do quite well now with volatility
        "META",
        "AMZN",
        "AMD",
        "INTC",
        "LCID",
        "QUBT",

        "BTCUSD",
        "ETHUSD",
        "UNIUSD",
    ]
    previous_picks = {}

    # Track when each analysis was last run
    last_initial_run = None
    last_market_open_run = None
    last_market_open_hour2_run = None
    last_market_close_run = None

    while True:
        try:
            market_open, market_close = get_market_hours()
            now = datetime.now(pytz.timezone("US/Eastern"))
            today = now.date()

            # Initial analysis at NZ morning (22:00-22:30 EST)
            # run at start of program to check
            if last_initial_run is None or ((now.hour == 22 and 0 <= now.minute < 30) and (
                    last_initial_run is None or last_initial_run != today
            )):

                logger.info("\nINITIAL ANALYSIS STARTING...")
                all_analyzed_results = analyze_symbols(symbols)
                current_picks = {
                    symbol: data
                    for symbol, data in list(all_analyzed_results.items())[:4]
                    if data["avg_return"] > 0
                }
                log_trading_plan(current_picks, "INITIAL PLAN")
                dry_run_manage_positions(current_picks, previous_picks)
                manage_positions(current_picks, previous_picks, all_analyzed_results)

                previous_picks = current_picks
                last_initial_run = today

            # Market open analysis (9:30-10:00 EST)
            elif (
                    (
                            now.hour == market_open.hour
                            and market_open.minute <= now.minute < market_open.minute + 30
                    )
                    and (last_market_open_run is None or last_market_open_run != today)
                    and is_nyse_trading_day_now()
            ):

                logger.info("\nMARKET OPEN ANALYSIS STARTING...")
                all_analyzed_results = analyze_symbols(symbols)
                current_picks = {
                    symbol: data
                    for symbol, data in list(all_analyzed_results.items())[:4]
                    if data["avg_return"] > 0
                }
                log_trading_plan(current_picks, "MARKET OPEN PLAN")
                manage_positions(current_picks, previous_picks, all_analyzed_results)

                previous_picks = current_picks
                last_market_open_run = today

            # Market open hour 2 analysis (10:30-11:00 EST)
            elif (
                    (
                            now.hour == market_open.hour + 1
                            and market_open.minute <= now.minute < market_open.minute + 30
                    )
                    and (last_market_open_hour2_run is None or last_market_open_hour2_run != today)
                    and is_nyse_trading_day_now()
            ):

                logger.info("\nMARKET OPEN HOUR 2 ANALYSIS STARTING...")
                all_analyzed_results = analyze_symbols(symbols)
                current_picks = {
                    symbol: data
                    for symbol, data in list(all_analyzed_results.items())[:2]
                    if data["avg_return"] > 0
                }
                log_trading_plan(current_picks, "MARKET OPEN HOUR 2 PLAN")
                manage_positions(current_picks, previous_picks, all_analyzed_results)

                previous_picks = current_picks
                last_market_open_hour2_run = today

            # Market close analysis (15:45-16:00 EST)
            elif (
                    (now.hour == market_close.hour - 1 and now.minute >= 45)
                    and (last_market_close_run is None or last_market_close_run != today)
                    and is_nyse_trading_day_ending()
            ):

                logger.info("\nMARKET CLOSE ANALYSIS STARTING...")
                all_analyzed_results = analyze_symbols(symbols)
                previous_picks = manage_market_close(
                    symbols, previous_picks, all_analyzed_results
                )
                last_market_close_run = today

            sleep(60)

        except Exception as e:
            logger.exception(f"Error in main loop: {str(e)}")
            sleep(60)


if __name__ == "__main__":
    main()
