"""Backtest for two-stage portfolio optimization - HOURLY version.

Loads data from trainingdatahourly/ and simulates hourly trading.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence

import anthropic
import pandas as pd
from loguru import logger

from stockagent.constants import TRADING_FEE
from stockagent_pctline.data_formatter import PctLineData

from .portfolio_allocator import allocate_portfolio, async_allocate_portfolio, PortfolioAllocation, is_crypto
from .price_predictor import predict_prices, async_predict_prices, PricePrediction


def _get_api_key() -> str:
    """Get API key from environment or env_real.py fallback."""
    key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not key:
        try:
            from env_real import CLAUDE_API_KEY
            if CLAUDE_API_KEY:
                return CLAUDE_API_KEY
        except ImportError:
            pass
        raise RuntimeError("API key required: set ANTHROPIC_API_KEY or CLAUDE_API_KEY")
    return key


# Leverage cost: 6.5% annual, calculated hourly
LEVERAGE_ANNUAL_RATE = 0.065
LEVERAGE_HOURLY_RATE = LEVERAGE_ANNUAL_RATE / (365 * 24)  # ~0.00074% per hour

# Data directory for hourly data
HOURLY_DATA_DIR = Path("trainingdatahourly")


def _load_hourly_data(
    symbols: Sequence[str],
    end_timestamp: datetime,
    lookback_hours: int = 500,
) -> dict[str, pd.DataFrame]:
    """Load hourly OHLC data from trainingdatahourly/."""
    result = {}

    for symbol in symbols:
        # Try crypto first, then stocks
        crypto_path = HOURLY_DATA_DIR / "crypto" / f"{symbol}.csv"
        stock_path = HOURLY_DATA_DIR / "stocks" / f"{symbol}.csv"

        data_path = crypto_path if crypto_path.exists() else stock_path

        if not data_path.exists():
            logger.warning(f"No hourly data found for {symbol}")
            continue

        df = pd.read_csv(data_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        # Filter to end_timestamp
        df = df[df.index <= end_timestamp]

        # Take last lookback_hours
        if len(df) > lookback_hours:
            df = df.tail(lookback_hours)

        if len(df) > 10:
            result[symbol] = df
            logger.debug(f"Loaded {len(df)} hourly bars for {symbol}")

    return result


def _format_pctline_hourly(df: pd.DataFrame) -> PctLineData:
    """Convert hourly OHLC DataFrame to PctLineData format."""
    # Calculate hourly returns
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # close_pct: hourly close change as pct
    close_pct = (close[1:] - close[:-1]) / close[:-1] * 100

    # high_pct: how much high was above close that hour (as pct)
    high_pct = (high[1:] - close[1:]) / close[1:] * 100

    # low_pct: how much low was below close (as pct, typically negative)
    low_pct = (low[1:] - close[1:]) / close[1:] * 100

    # Build lines
    lines = []
    for i in range(len(close_pct)):
        lines.append(f"{close_pct[i]:.4f},{high_pct[i]:.4f},{low_pct[i]:.4f}")

    return PctLineData(
        lines="\n".join(lines),
        last_close=float(close[-1]),
    )


def _simulate_hour(
    allocations: PortfolioAllocation,
    predictions: dict[str, PricePrediction],
    next_hour_data: dict[str, dict],
    equity: float,
    trading_fee: float = TRADING_FEE,
    min_confidence: float = 0.3,
) -> tuple[float, list[dict]]:
    """Simulate one hour's trading based on two-stage predictions.

    Args:
        allocations: Portfolio allocation from Stage 1
        predictions: Price predictions from Stage 2
        next_hour_data: Actual OHLC for next hour
        equity: Current equity
        trading_fee: Fee per trade
        min_confidence: Minimum overall confidence to trade

    Returns:
        (pnl, trade_details)
    """
    pnl = 0.0
    trades = []

    # Check overall confidence
    if not allocations.should_trade(min_confidence):
        logger.debug(f"Skipping trades: overall confidence {allocations.overall_confidence:.2f} < {min_confidence}")
        return pnl, trades

    for symbol, alloc in allocations.allocations.items():
        if alloc.alloc <= 0.001:
            continue

        if symbol not in predictions:
            continue

        if symbol not in next_hour_data:
            continue

        # ENFORCE: Crypto cannot be shorted
        if is_crypto(symbol) and alloc.direction == "short":
            logger.warning(f"Crypto {symbol} cannot be shorted, skipping")
            continue

        pred = predictions[symbol]
        actual = next_hour_data[symbol]
        actual_open = actual["open"]
        actual_high = actual["high"]
        actual_low = actual["low"]
        actual_close = actual["close"]

        # Calculate notional with leverage (stocks only)
        leverage = alloc.leverage if not is_crypto(symbol) else 1.0
        leverage = min(2.0, max(1.0, leverage))

        notional = equity * alloc.alloc * leverage

        # Determine entry price
        entry_price = pred.entry_price
        exit_price = pred.exit_price
        stop_price = pred.stop_loss_price

        # Simulate trade execution
        trade_pnl = 0.0
        trade_status = "no_fill"
        fill_price = None
        close_price = None

        if alloc.direction == "long":
            # Long trade: buy at entry, sell at exit or stop
            if actual_low <= entry_price <= actual_high:
                # Entry filled
                fill_price = entry_price
                shares = notional / fill_price

                if actual_high >= exit_price:
                    # Exit hit
                    close_price = exit_price
                    trade_status = "profit"
                elif actual_low <= stop_price:
                    # Stop hit
                    close_price = stop_price
                    trade_status = "stopped"
                else:
                    # Close at end of hour
                    close_price = actual_close
                    trade_status = "held"

                trade_pnl = shares * (close_price - fill_price)
                # Subtract fees
                trade_pnl -= 2 * trading_fee * notional  # entry + exit

        else:  # short
            # Short trade: sell at entry, buy back at exit or stop
            if actual_low <= entry_price <= actual_high:
                fill_price = entry_price
                shares = notional / fill_price

                if actual_low <= exit_price:
                    close_price = exit_price
                    trade_status = "profit"
                elif actual_high >= stop_price:
                    close_price = stop_price
                    trade_status = "stopped"
                else:
                    close_price = actual_close
                    trade_status = "held"

                trade_pnl = shares * (fill_price - close_price)
                trade_pnl -= 2 * trading_fee * notional

        # Apply leverage cost if leveraged
        if leverage > 1.0:
            leveraged_amount = notional * (leverage - 1) / leverage
            leverage_cost = leveraged_amount * LEVERAGE_HOURLY_RATE
            trade_pnl -= leverage_cost

        pnl += trade_pnl

        if fill_price is not None:
            trades.append({
                "symbol": symbol,
                "direction": alloc.direction,
                "allocation": alloc.alloc,
                "leverage": leverage,
                "entry_price": fill_price,
                "exit_price": close_price,
                "pnl": trade_pnl,
                "status": trade_status,
                "confidence": alloc.confidence,
            })

    return pnl, trades


def run_backtest(
    symbols: Sequence[str],
    end_timestamp: datetime,
    num_hours: int = 120,  # 5 days
    initial_capital: float = 100_000.0,
    min_confidence: float = 0.3,
    max_lines: int = 200,
    use_thinking: bool = False,
) -> dict:
    """Run HOURLY two-stage backtest.

    Args:
        symbols: List of symbols to trade
        end_timestamp: End timestamp for backtest
        num_hours: Number of hours to backtest
        initial_capital: Starting capital
        min_confidence: Minimum confidence to execute trades
        max_lines: Max historical lines per symbol for prompts
        use_thinking: Enable extended thinking (Opus with 60000 token budget)

    Returns:
        Backtest results dict
    """
    logger.info(f"HOURLY Two-Stage Backtest: {num_hours} hours ending {end_timestamp}")
    if use_thinking:
        logger.info("Extended thinking ENABLED (Opus with 60000 token budget)")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Capital: ${initial_capital:,.0f}, min_confidence: {min_confidence}")

    # Load all hourly data
    all_data = _load_hourly_data(symbols, end_timestamp, lookback_hours=max_lines + num_hours + 100)
    for symbol, df in all_data.items():
        logger.info(f"Loaded {len(df)} hourly bars for {symbol}")

    if not all_data:
        logger.error("No data loaded")
        return {"error": "No data"}

    # Get trading hours
    sample_df = next(iter(all_data.values()))
    all_timestamps = sample_df.index.tolist()

    # Find the last num_hours timestamps
    if len(all_timestamps) < num_hours + 1:
        num_hours = len(all_timestamps) - 1
        logger.warning(f"Reduced to {num_hours} hours due to data availability")

    trading_timestamps = all_timestamps[-(num_hours + 1):]
    logger.info(f"Trading {len(trading_timestamps) - 1} hours")

    equity = initial_capital
    results = []
    all_trades = []

    for i in range(len(trading_timestamps) - 1):
        current_ts = trading_timestamps[i]
        next_ts = trading_timestamps[i + 1]

        # Get data up to current timestamp
        pct_data = {}
        next_hour_data = {}

        for symbol, df in all_data.items():
            current_df = df[df.index <= current_ts]
            if len(current_df) > 10:
                pct_data[symbol] = _format_pctline_hourly(current_df)

            # Get next hour's actual data
            if next_ts in df.index:
                row = df.loc[next_ts]
                next_hour_data[symbol] = {
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                }

        if not pct_data:
            continue

        # Stage 1: Portfolio allocation
        allocations = allocate_portfolio(
            pct_data,
            chronos_forecasts=None,  # Can add Chronos2 hourly forecasts later
            equity=equity,
            max_lines=max_lines,
            use_thinking=use_thinking,
        )

        logger.info(f"{current_ts}: Stage 1 - overall confidence {allocations.overall_confidence:.2f}, "
                   f"{len(allocations.allocations)} allocations")

        # Stage 2: Price predictions
        predictions = {}
        for symbol, alloc in allocations.allocations.items():
            if symbol in pct_data:
                pred = predict_prices(
                    symbol=symbol,
                    pct_data=pct_data[symbol],
                    allocation=alloc,
                    chronos_forecast=None,
                    max_lines=min(100, max_lines),
                )
                predictions[symbol] = pred

        # Simulate hour
        hour_pnl, hour_trades = _simulate_hour(
            allocations=allocations,
            predictions=predictions,
            next_hour_data=next_hour_data,
            equity=equity,
            min_confidence=min_confidence,
        )

        equity += hour_pnl
        all_trades.extend(hour_trades)

        results.append({
            "timestamp": current_ts,
            "pnl": hour_pnl,
            "equity": equity,
            "num_trades": len(hour_trades),
            "confidence": allocations.overall_confidence,
        })

        logger.info(f"{current_ts}: PnL ${hour_pnl:+,.2f} ({hour_pnl/equity*100:+.3f}%), "
                   f"equity ${equity:,.2f}, {len(hour_trades)} trades")

    # Summary
    total_return = (equity - initial_capital) / initial_capital
    # Annualized: assume 24*365 hours per year
    hours_tested = len(results)
    annualized = (1 + total_return) ** (24 * 365 / max(hours_tested, 1)) - 1 if hours_tested > 0 else 0

    winning_trades = [t for t in all_trades if t["pnl"] > 0]
    win_rate = len(winning_trades) / len(all_trades) if all_trades else 0
    avg_confidence = sum(r["confidence"] for r in results) / len(results) if results else 0

    summary = {
        "initial_capital": initial_capital,
        "final_equity": equity,
        "total_return": total_return,
        "annualized_return": annualized,
        "total_trades": len(all_trades),
        "win_rate": win_rate,
        "avg_confidence": avg_confidence,
        "hourly_results": results,
        "hours_tested": hours_tested,
    }

    logger.info("=" * 60)
    logger.info("HOURLY TWO-STAGE BACKTEST COMPLETE")
    logger.info(f"Total return: {total_return:.2%}")
    logger.info(f"Annualized: {annualized:.2%}")
    logger.info(f"Final equity: ${equity:,.2f}")
    logger.info(f"Hours tested: {hours_tested}")
    logger.info(f"Total trades: {len(all_trades)}, Win rate: {win_rate:.1%}")
    logger.info(f"Avg confidence: {avg_confidence:.2f}")
    logger.info("=" * 60)

    return summary


async def run_backtest_async(
    symbols: Sequence[str],
    end_timestamp: datetime,
    num_hours: int = 120,
    initial_capital: float = 100_000.0,
    min_confidence: float = 0.3,
    max_lines: int = 200,
    use_thinking: bool = False,
    max_parallel_hours: int = 5,
) -> dict:
    """Run HOURLY two-stage backtest with parallel hour processing.

    Args:
        symbols: List of symbols to trade
        end_timestamp: End timestamp for backtest
        num_hours: Number of hours to backtest
        initial_capital: Starting capital
        min_confidence: Minimum confidence to execute trades
        max_lines: Max historical lines per symbol for prompts
        use_thinking: Enable extended thinking (Opus with 60000 token budget)
        max_parallel_hours: Max hours to process in parallel

    Returns:
        Backtest results dict
    """
    logger.info(f"ASYNC HOURLY Two-Stage Backtest: {num_hours} hours ending {end_timestamp}")
    if use_thinking:
        logger.info("Extended thinking ENABLED (Opus with 60000 token budget)")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Capital: ${initial_capital:,.0f}, min_confidence: {min_confidence}")

    # Load all hourly data
    all_data = _load_hourly_data(symbols, end_timestamp, lookback_hours=max_lines + num_hours + 100)
    for symbol, df in all_data.items():
        logger.info(f"Loaded {len(df)} hourly bars for {symbol}")

    if not all_data:
        logger.error("No data loaded")
        return {"error": "No data"}

    # Get trading hours
    sample_df = next(iter(all_data.values()))
    all_timestamps = sample_df.index.tolist()

    if len(all_timestamps) < num_hours + 1:
        num_hours = len(all_timestamps) - 1
        logger.warning(f"Reduced to {num_hours} hours due to data availability")

    trading_timestamps = all_timestamps[-(num_hours + 1):]
    logger.info(f"Found {len(trading_timestamps) - 1} trading hours")

    # Create async client
    client = anthropic.AsyncAnthropic(api_key=_get_api_key())

    # Process hours
    equity = initial_capital
    results = []
    all_trades = []

    async def _process_single_hour(
        current_ts: datetime,
        next_ts: datetime,
        current_equity: float,
    ) -> tuple[datetime, float, list[dict], float]:
        """Process a single hour's trading."""
        pct_data = {}
        next_hour_data = {}

        for symbol, df in all_data.items():
            current_df = df[df.index <= current_ts]
            if len(current_df) > 10:
                pct_data[symbol] = _format_pctline_hourly(current_df)

            if next_ts in df.index:
                row = df.loc[next_ts]
                next_hour_data[symbol] = {
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                }

        if not pct_data:
            return current_ts, 0.0, [], 0.0

        # Stage 1: Portfolio allocation
        allocations = await async_allocate_portfolio(
            client=client,
            pct_data=pct_data,
            chronos_forecasts=None,
            equity=current_equity,
            max_lines=max_lines,
            use_thinking=use_thinking,
        )

        logger.info(f"{current_ts}: Stage 1 - confidence {allocations.overall_confidence:.2f}, "
                   f"{len(allocations.allocations)} allocations")

        # Stage 2: Price predictions (parallel)
        prediction_tasks = []
        for symbol, alloc in allocations.allocations.items():
            if symbol in pct_data:
                prediction_tasks.append(
                    async_predict_prices(
                        client=client,
                        symbol=symbol,
                        pct_data=pct_data[symbol],
                        allocation=alloc,
                        chronos_forecast=None,
                        max_lines=min(100, max_lines),
                    )
                )

        predictions = {}
        if prediction_tasks:
            pred_results = await asyncio.gather(*prediction_tasks, return_exceptions=True)
            for pred in pred_results:
                if isinstance(pred, PricePrediction):
                    predictions[pred.symbol] = pred

        # Simulate hour
        hour_pnl, hour_trades = _simulate_hour(
            allocations=allocations,
            predictions=predictions,
            next_hour_data=next_hour_data,
            equity=current_equity,
            min_confidence=min_confidence,
        )

        return current_ts, hour_pnl, hour_trades, allocations.overall_confidence

    # Process sequentially (P&L depends on previous equity)
    for i in range(len(trading_timestamps) - 1):
        current_ts = trading_timestamps[i]
        next_ts = trading_timestamps[i + 1]

        ts, hour_pnl, hour_trades, confidence = await _process_single_hour(
            current_ts, next_ts, equity
        )

        equity += hour_pnl
        all_trades.extend(hour_trades)

        results.append({
            "timestamp": ts,
            "pnl": hour_pnl,
            "equity": equity,
            "num_trades": len(hour_trades),
            "confidence": confidence,
        })

        logger.info(f"{ts}: PnL ${hour_pnl:+,.2f} ({hour_pnl/equity*100:+.3f}%), "
                   f"equity ${equity:,.2f}, {len(hour_trades)} trades")

    # Summary
    total_return = (equity - initial_capital) / initial_capital
    hours_tested = len(results)
    annualized = (1 + total_return) ** (24 * 365 / max(hours_tested, 1)) - 1 if hours_tested > 0 else 0

    winning_trades = [t for t in all_trades if t["pnl"] > 0]
    win_rate = len(winning_trades) / len(all_trades) if all_trades else 0
    avg_confidence = sum(r["confidence"] for r in results) / len(results) if results else 0

    summary = {
        "initial_capital": initial_capital,
        "final_equity": equity,
        "total_return": total_return,
        "annualized_return": annualized,
        "total_trades": len(all_trades),
        "win_rate": win_rate,
        "avg_confidence": avg_confidence,
        "hourly_results": results,
        "hours_tested": hours_tested,
    }

    logger.info("=" * 60)
    logger.info("ASYNC HOURLY TWO-STAGE BACKTEST COMPLETE")
    logger.info(f"Total return: {total_return:.2%}")
    logger.info(f"Annualized: {annualized:.2%}")
    logger.info(f"Final equity: ${equity:,.2f}")
    logger.info(f"Hours tested: {hours_tested}")
    logger.info(f"Total trades: {len(all_trades)}, Win rate: {win_rate:.1%}")
    logger.info(f"Avg confidence: {avg_confidence:.2f}")
    logger.info("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="HOURLY two-stage portfolio backtest")
    parser.add_argument(
        "--symbols", nargs="+",
        default=["BTCUSD", "ETHUSD", "LINKUSD"],
        help="Symbols to trade"
    )
    parser.add_argument("--hours", type=int, default=120, help="Hours to backtest (default: 120 = 5 days)")
    parser.add_argument("--capital", type=float, default=100_000, help="Initial capital")
    parser.add_argument("--min-confidence", type=float, default=0.3, help="Min confidence threshold")
    parser.add_argument("--max-lines", type=int, default=200, help="Max historical lines")
    parser.add_argument("--end-timestamp", type=str, default=None, help="End timestamp ISO format")
    parser.add_argument("--thinking", action="store_true", help="Enable extended thinking (Opus)")
    parser.add_argument("--async", dest="use_async", action="store_true", help="Use async backtest")

    args = parser.parse_args()

    # Calculate end timestamp
    if args.end_timestamp:
        end_timestamp = datetime.fromisoformat(args.end_timestamp)
    else:
        # Use a known good timestamp from the data
        end_timestamp = datetime(2024, 11, 1, 0, 0, 0, tzinfo=timezone.utc)

    if args.use_async:
        asyncio.run(run_backtest_async(
            symbols=args.symbols,
            end_timestamp=end_timestamp,
            num_hours=args.hours,
            initial_capital=args.capital,
            min_confidence=args.min_confidence,
            max_lines=args.max_lines,
            use_thinking=args.thinking,
        ))
    else:
        run_backtest(
            symbols=args.symbols,
            end_timestamp=end_timestamp,
            num_hours=args.hours,
            initial_capital=args.capital,
            min_confidence=args.min_confidence,
            max_lines=args.max_lines,
            use_thinking=args.thinking,
        )


if __name__ == "__main__":
    main()
