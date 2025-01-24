from datetime import datetime
from math import floor
from time import sleep
from typing import List, Dict

import pytz
from loguru import logger

import alpaca_wrapper
from backtest_test3_inline import backtest_forecasts
from src.comparisons import is_buy_side, is_same_side, is_sell_side
from src.date_utils import is_nyse_trading_day_now, is_nyse_trading_day_ending
from src.fixtures import crypto_symbols
from src.logging_utils import setup_logging
from src.trading_obj_utils import filter_to_realistic_positions
from src.process_utils import backout_near_market, ramp_into_position, spawn_close_position_at_takeprofit

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
        return

    if not all_analyzed_results:
        logger.warning(
            "No analysis results available - skipping position closure checks"
        )
        return

    # Handle position closures
    for position in positions:
        symbol = position.symbol
        should_close = False

        if symbol in all_analyzed_results:
            new_forecast = all_analyzed_results[symbol]
            if not is_same_side(new_forecast["side"], position.side):
                logger.info(
                    f"Closing position for {symbol} due to direction change from {position.side} to {new_forecast['side']}"
                )
                should_close = True
        else:
            logger.warning(f"No analysis data for {symbol} - keeping position")

        if should_close:
            backout_near_market(symbol)

    # Enter new positions from current_picks
    if not current_picks:
        logger.warning("No current picks available - skipping new position entry")
        return

    for symbol, data in current_picks.items():
        position_exists = any(p.symbol == symbol for p in positions)
        correct_side = any(
            p.symbol == symbol and is_same_side(p.side, data["side"]) for p in positions
        )

        if symbol in crypto_symbols:
            should_enter = not position_exists and is_buy_side(data["side"])
        else:
            should_enter = not position_exists

        if should_enter or not correct_side:
            logger.info(f"Entering new {data['side']} position for {symbol}")
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
                    qty = get_qty(symbol, entry_price)
                    alpaca_wrapper.open_order_at_price_or_all(symbol, qty=qty, side="buy", price=entry_price)

                    tp_price = data["predicted_high"]
                    logger.info(f"(Highlow) Scheduling takeprofit at predicted_high={tp_price:.3f} for {symbol}")
                    spawn_close_position_at_takeprofit(symbol, tp_price)
                else:
                    entry_price = data["predicted_high"]
                    logger.info(
                        f"(Highlow) Placing limit SELL/short order for {symbol} at predicted_high={entry_price:.2f}"
                    )
                    qty = get_qty(symbol, entry_price)
                    alpaca_wrapper.open_order_at_price_or_all(symbol, qty=qty, side="sell", price=entry_price)

                    tp_price = data["predicted_low"]
                    logger.info(f"(Highlow) Scheduling takeprofit at predicted_low={tp_price:.3f} for short {symbol}")
                    spawn_close_position_at_takeprofit(symbol, tp_price)

def get_qty(symbol, entry_price):
    # Calculate qty as 15% of available buying power
    buying_power = alpaca_wrapper.total_buying_power
    qty = 0.15 * buying_power / entry_price
    
    # Round down to 3 decimal places for crypto
    if symbol in crypto_symbols:
        qty = floor(qty * 1000) / 1000.0
    else:
        # Round down to whole number for stocks
        qty = floor(qty)
    
    # Ensure qty is valid
    if qty <= 0:
        logger.error(f"Calculated qty {qty} is invalid")
        return 0
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
        "MSFT",
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
