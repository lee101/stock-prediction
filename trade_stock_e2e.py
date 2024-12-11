import sys
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
from loguru import logger
import pytz
from time import sleep
import numpy as np
from scipy import stats

from backtest_test3_inline import backtest_forecasts
from src.process_utils import backout_near_market, ramp_into_position
from src.fixtures import crypto_symbols
import alpaca_wrapper
from src.date_utils import is_nyse_trading_day_now, is_nyse_trading_day_ending
from src.comparisons import is_same_side

from src.logging_utils import setup_logging

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
            num_simulations = 300

            backtest_df = backtest_forecasts(symbol, num_simulations)

            # Use different strategies for crypto vs stocks
            if symbol in crypto_symbols:
                # For crypto, only use buy and hold return and only allow long positions
                avg_return = backtest_df["buy_hold_return"].mean()
            else:
                # For stocks, continue using simple strategy return and allow both directions
                avg_return = backtest_df["simple_strategy_return"].mean()
            last_prediction = backtest_df.iloc[-1]
            predicted_movement = (
                last_prediction["predicted_close"] - last_prediction["close"]
            )
            position_side = "buy" if predicted_movement > 0 else "sell"

            # Only add to results if we have a valid position side
            results[symbol] = {
                "avg_return": avg_return,
                "predictions": backtest_df,
                "side": position_side,
                "predicted_movement": predicted_movement,
            }

            logger.info(
                f"Analysis complete for {symbol}: Avg Return={avg_return:.3f}, side={position_side}"
            )
            logger.info(f"Predicted movement: {predicted_movement:.3f}")
            logger.info(f"Current close: {last_prediction['close']:.3f}")
            logger.info(f"Predicted close: {last_prediction['predicted_close']:.3f}")

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            continue

    return dict(sorted(results.items(), key=lambda x: x[1]["avg_return"], reverse=True))


def log_trading_plan(picks: Dict[str, Dict], action: str):
    """Log the trading plan without executing trades."""
    logger.info(f"\n{'='*50}\nTRADING PLAN ({action})\n{'='*50}")

    for symbol, data in picks.items():
        logger.info(
            f"""
Symbol: {symbol}
Direction: {data['side']}
Avg Return: {data['avg_return']:.3f}
Predicted Movement: {data['predicted_movement']:.3f}
{'='*30}"""
        )


def manage_positions(
    current_picks: Dict[str, Dict],
    previous_picks: Dict[str, Dict],
    all_analyzed_results: Dict[str, Dict],
):
    """Execute actual position management."""
    positions = alpaca_wrapper.get_all_positions()

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
                logger.info(
                    f"Predicted movement: {new_forecast['predicted_movement']:.3f}"
                )
                should_close = True
        else:
            logger.warning(f"No analysis data for {symbol} - keeping position")

        if should_close:
            backout_near_market(symbol)

    # Enter new positions from current picks
    if not current_picks:
        logger.warning("No current picks available - skipping new position entry")
        return

    for symbol, data in current_picks.items():
        position_exists = any(p.symbol == symbol for p in positions)
        # For crypto, only check if position exists since we only do long positions
        correct_side = any(
            p.symbol == symbol and p.side == data["side"] for p in positions
        )

        if symbol in crypto_symbols:
            should_enter = not position_exists and data["side"] == "buy"
        else:
            should_enter = not position_exists

        if should_enter or not correct_side:
            logger.info(f"Entering new {data['side']} position for {symbol}")
            ramp_into_position(symbol, data["side"])


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
                    f"Keeping {symbol} position as tomorrow's forecast matches current {position.side} direction"
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
        elif symbol in current_picks and current_picks[symbol]["side"] != position.side:
            logger.info(
                f"Would close position for {symbol} to switch direction from {position.side} to {current_picks[symbol]['side']}"
            )
            should_close = True

    # Log new positions
    for symbol, data in current_picks.items():
        position_exists = any(p.symbol == symbol for p in positions)
        correct_side = any(
            p.symbol == symbol and p.side == data["side"] for p in positions
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
        "NFLX",
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
            if (now.hour == 22 and 0 <= now.minute < 30) and (
                last_initial_run is None or last_initial_run != today
            ):

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
