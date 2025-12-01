"""Backtest for two-stage portfolio optimization."""

from __future__ import annotations

import argparse
import asyncio
from datetime import date, datetime, timedelta, timezone
from typing import Mapping, Sequence

import anthropic
import pandas as pd
from loguru import logger

from stockagent.agentsimulator.market_data import fetch_latest_ohlc, MarketDataBundle
from stockagent.constants import TRADING_FEE
from stockagent_pctline.data_formatter import format_pctline_data, PctLineData

from .portfolio_allocator import allocate_portfolio, async_allocate_portfolio, PortfolioAllocation, is_crypto
from .price_predictor import predict_prices, async_predict_prices, PricePrediction


# Leverage cost: 6.5% annual, calculated daily
LEVERAGE_ANNUAL_RATE = 0.065
LEVERAGE_DAILY_RATE = LEVERAGE_ANNUAL_RATE / 365  # ~0.0178% per day


def _get_all_data(
    symbols: Sequence[str],
    end_date: date,
    lookback_days: int = 1000,
) -> dict[str, pd.DataFrame]:
    """Fetch historical data for all symbols."""
    as_of = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)

    bundle = fetch_latest_ohlc(
        symbols=symbols,
        lookback_days=lookback_days,
        as_of=as_of,
    )

    return bundle.bars


def _generate_chronos_forecasts(
    all_data: dict[str, pd.DataFrame],
    symbols: Sequence[str],
    trade_date: date,
    device: str = "cuda",
) -> Mapping[str, object] | None:
    """Generate Chronos2 forecasts for symbols."""
    try:
        from stockagentopus_chronos2.forecaster import generate_chronos2_forecasts
        from stockagent.agentsimulator.market_data import MarketDataBundle

        # Build a MarketDataBundle with data up to trade_date
        filtered_bars = {}
        for symbol, df in all_data.items():
            if symbol in symbols:
                # Skip DataFrames without DatetimeIndex (empty or invalid data)
                if not hasattr(df.index, 'date') or len(df) == 0:
                    continue
                df_up_to = df[df.index.date <= trade_date]
                if len(df_up_to) > 10:
                    filtered_bars[symbol] = df_up_to

        if not filtered_bars:
            return None

        # Create a minimal bundle with required arguments
        from datetime import timezone
        bundle = MarketDataBundle(
            bars=filtered_bars,
            lookback_days=512,
            as_of=datetime.combine(trade_date, datetime.max.time()).replace(tzinfo=timezone.utc),
        )

        forecasts = generate_chronos2_forecasts(
            market_data=bundle,
            symbols=list(filtered_bars.keys()),
            prediction_length=1,
            context_length=512,
            device_map=device,
        )
        return forecasts
    except ImportError as e:
        logger.warning(f"Chronos2 not available: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to generate Chronos2 forecasts: {e}")
        return None


def _simulate_day(
    allocations: PortfolioAllocation,
    predictions: dict[str, PricePrediction],
    next_day_data: dict[str, dict],
    equity: float,
    trading_fee: float = TRADING_FEE,
    min_confidence: float = 0.3,
) -> tuple[float, list[dict]]:
    """Simulate one day's trading based on two-stage predictions.

    Supports both long and short positions (stocks only for shorts).
    Supports leverage up to 2x for stocks (with daily interest cost).

    Args:
        allocations: Portfolio allocation from Stage 1
        predictions: Price predictions from Stage 2
        next_day_data: Actual OHLC for next day
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

        if symbol not in next_day_data:
            continue

        # ENFORCE: Crypto cannot be shorted (skip if somehow got through)
        if is_crypto(symbol) and alloc.direction == "short":
            logger.warning(f"Crypto {symbol} cannot be shorted, skipping")
            continue

        pred = predictions[symbol]
        actual = next_day_data[symbol]
        actual_open = actual["open"]
        actual_high = actual["high"]
        actual_low = actual["low"]
        actual_close = actual["close"]

        # Calculate notional with leverage (stocks only)
        leverage = alloc.leverage if not is_crypto(symbol) else 1.0
        leverage = min(2.0, max(1.0, leverage))  # Clamp to 1-2x

        base_notional = equity * alloc.alloc
        leveraged_notional = base_notional * leverage
        leveraged_amount = leveraged_notional - base_notional  # Amount borrowed

        is_long = alloc.direction == "long"

        # Entry logic: try to enter at predicted entry price
        entry_target = pred.entry_price

        if is_long:
            # For long: buy at entry_target if price drops to it, else buy at open
            if actual_low <= entry_target:
                entry_price = entry_target
            else:
                entry_price = actual_open  # market order at open
        else:
            # For short: sell at entry_target if price rises to it, else sell at open
            if actual_high >= entry_target:
                entry_price = entry_target
            else:
                entry_price = actual_open

        quantity = int(leveraged_notional / entry_price) if entry_price > 0 else 0
        if quantity <= 0:
            continue

        # Exit logic
        exit_target = pred.exit_price
        stop_loss = pred.stop_loss_price

        if is_long:
            # Long exit: sell at target if high reaches it, or stop if low hits it
            if actual_high >= exit_target:
                exit_price = exit_target
                exit_reason = "target"
            elif actual_low <= stop_loss:
                exit_price = stop_loss
                exit_reason = "stop"
            else:
                exit_price = actual_close
                exit_reason = "eod"

            # P&L for long: (exit - entry) * quantity
            gross_pnl = (exit_price - entry_price) * quantity
        else:
            # Short exit: cover at target if low reaches it, or stop if high hits it
            if actual_low <= exit_target:
                exit_price = exit_target
                exit_reason = "target"
            elif actual_high >= stop_loss:
                exit_price = stop_loss
                exit_reason = "stop"
            else:
                exit_price = actual_close
                exit_reason = "eod"

            # P&L for short: (entry - exit) * quantity
            gross_pnl = (entry_price - exit_price) * quantity

        # Fees on both legs
        fees = (entry_price * quantity + exit_price * quantity) * trading_fee

        # Leverage interest cost (daily rate on borrowed amount)
        leverage_cost = leveraged_amount * LEVERAGE_DAILY_RATE if leverage > 1.0 else 0.0

        net_pnl = gross_pnl - fees - leverage_cost

        pnl += net_pnl

        trades.append({
            "symbol": symbol,
            "direction": alloc.direction,
            "quantity": quantity,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "target": exit_target,
            "stop": stop_loss,
            "gross_pnl": gross_pnl,
            "fees": fees,
            "leverage": leverage,
            "leverage_cost": leverage_cost,
            "net_pnl": net_pnl,
            "alloc": alloc.alloc,
            "confidence": pred.confidence,
        })

    return pnl, trades


def run_backtest(
    symbols: Sequence[str],
    start_date: date,
    end_date: date,
    initial_capital: float = 100_000.0,
    min_confidence: float = 0.3,
    max_lines: int = 200,
    use_chronos: bool = True,
    chronos_device: str = "cuda",
    use_thinking: bool = False,
) -> dict:
    """Run two-stage backtest over a date range.

    Args:
        symbols: List of symbols to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Starting capital
        min_confidence: Minimum confidence to execute trades
        max_lines: Max historical lines per symbol for prompts
        use_chronos: Whether to use Chronos2 forecasts
        chronos_device: Device for Chronos2 (cuda/cpu)
        use_thinking: Enable extended thinking (Opus with 63999 token budget)

    Returns:
        Backtest results dict
    """
    logger.info(f"Two-Stage Backtest: {start_date} to {end_date}")
    if use_thinking:
        logger.info("Extended thinking ENABLED (Opus with 63999 token budget)")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Capital: ${initial_capital:,.0f}, min_confidence: {min_confidence}")

    # Get all historical data
    all_data = _get_all_data(symbols, end_date, lookback_days=max_lines + 100)
    for symbol, df in all_data.items():
        logger.info(f"Loaded {len(df)} bars for {symbol}")

    if not all_data:
        logger.error("No data loaded")
        return {"error": "No data"}

    # Get trading days in range
    sample_df = list(all_data.values())[0]
    trading_days = sample_df[
        (sample_df.index.date >= start_date) &
        (sample_df.index.date <= end_date)
    ].index.date
    trading_days = sorted(set(trading_days))

    logger.info(f"Found {len(trading_days)} trading days")

    # Run simulation
    equity = initial_capital
    results = []

    for i, trade_date in enumerate(trading_days[:-1]):
        next_date = trading_days[i + 1]

        # Build pct-line data up to trade_date
        pct_data: dict[str, PctLineData] = {}
        for symbol, df in all_data.items():
            # Skip DataFrames without DatetimeIndex (empty or invalid data)
            if not hasattr(df.index, 'date') or len(df) == 0:
                continue
            df_up_to = df[df.index.date <= trade_date]
            if len(df_up_to) > 10:
                pct_data[symbol] = format_pctline_data(df_up_to, symbol, max_lines)

        if not pct_data:
            continue

        # Generate Chronos2 forecasts if enabled
        chronos_forecasts = None
        if use_chronos:
            chronos_forecasts = _generate_chronos_forecasts(
                all_data, list(pct_data.keys()), trade_date, chronos_device
            )

        # Stage 1: Portfolio allocation
        try:
            allocations = allocate_portfolio(
                pct_data=pct_data,
                chronos_forecasts=chronos_forecasts,
                equity=equity,
                max_lines=max_lines,
                use_thinking=use_thinking,
            )
            logger.info(
                f"{trade_date}: Stage 1 - overall confidence {allocations.overall_confidence:.2f}, "
                f"{len(allocations.allocations)} allocations"
            )
        except Exception as e:
            logger.error(f"Stage 1 failed for {trade_date}: {e}")
            continue

        # Stage 2: Price predictions (only for allocated symbols)
        predictions: dict[str, PricePrediction] = {}
        if allocations.should_trade(min_confidence):
            try:
                predictions = predict_prices(
                    pct_data=pct_data,
                    allocations=allocations.allocations,
                    chronos_forecasts=chronos_forecasts,
                    max_lines=max_lines,
                )
                logger.info(f"{trade_date}: Stage 2 - {len(predictions)} price predictions")
            except Exception as e:
                logger.error(f"Stage 2 failed for {trade_date}: {e}")
                continue

        # Get next day's actual data
        next_day_data: dict[str, dict] = {}
        for symbol, df in all_data.items():
            next_df = df[df.index.date == next_date]
            if not next_df.empty:
                next_day_data[symbol] = {
                    "open": next_df["open"].iloc[0],
                    "high": next_df["high"].iloc[0],
                    "low": next_df["low"].iloc[0],
                    "close": next_df["close"].iloc[0],
                }

        # Simulate day
        pnl, trades = _simulate_day(
            allocations, predictions, next_day_data, equity,
            min_confidence=min_confidence
        )
        equity += pnl

        daily_return = pnl / (equity - pnl) if equity != pnl else 0

        results.append({
            "date": trade_date,
            "next_date": next_date,
            "pnl": pnl,
            "equity": equity,
            "daily_return": daily_return,
            "num_trades": len(trades),
            "trades": trades,
            "overall_confidence": allocations.overall_confidence,
        })

        # Log trade details
        for t in trades:
            leverage_str = f" {t['leverage']:.1f}x" if t.get('leverage', 1.0) > 1.0 else ""
            logger.info(
                f"  {t['symbol']} {t['direction']}{leverage_str}: "
                f"entry ${t['entry_price']:.2f} -> exit ${t['exit_price']:.2f} "
                f"({t['exit_reason']}), PnL ${t['net_pnl']:+,.2f}"
            )

        logger.info(
            f"{trade_date}: PnL ${pnl:+,.2f} ({daily_return:+.3%}), "
            f"equity ${equity:,.2f}, {len(trades)} trades"
        )

    # Summary
    total_return = (equity - initial_capital) / initial_capital
    num_days = len(results)

    # Use 365 for crypto (24/7 trading)
    trading_days_per_year = 365 if any("USD" in s for s in symbols) else 252
    annualized = ((1 + total_return) ** (trading_days_per_year / num_days) - 1) if num_days > 0 else 0

    # Calculate win rate
    all_trades = [t for r in results for t in r["trades"]]
    winning_trades = [t for t in all_trades if t["net_pnl"] > 0]
    win_rate = len(winning_trades) / len(all_trades) if all_trades else 0

    # Calculate avg confidence
    avg_confidence = sum(r["overall_confidence"] for r in results) / len(results) if results else 0

    summary = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "trading_days": num_days,
        "initial_capital": initial_capital,
        "final_equity": equity,
        "total_return": total_return,
        "annualized_return": annualized,
        "total_trades": len(all_trades),
        "win_rate": win_rate,
        "avg_confidence": avg_confidence,
        "daily_results": results,
    }

    logger.info("=" * 60)
    logger.info("TWO-STAGE BACKTEST COMPLETE")
    logger.info(f"Total return: {total_return:.2%}")
    logger.info(f"Annualized: {annualized:.2%}")
    logger.info(f"Final equity: ${equity:,.2f}")
    logger.info(f"Total trades: {len(all_trades)}, Win rate: {win_rate:.1%}")
    logger.info(f"Avg confidence: {avg_confidence:.2f}")
    logger.info("=" * 60)

    return summary


async def _process_single_day(
    client: anthropic.AsyncAnthropic,
    all_data: dict[str, pd.DataFrame],
    symbols: Sequence[str],
    trade_date: date,
    next_date: date,
    equity: float,
    max_lines: int,
    use_chronos: bool,
    chronos_device: str,
    use_thinking: bool,
    min_confidence: float,
) -> dict:
    """Process a single trading day asynchronously.

    Returns dict with date info, pnl, trades for later recombination.
    """
    # Build pct-line data up to trade_date
    pct_data: dict[str, PctLineData] = {}
    for symbol, df in all_data.items():
        # Skip DataFrames without DatetimeIndex (empty or invalid data)
        if not hasattr(df.index, 'date') or len(df) == 0:
            continue
        df_up_to = df[df.index.date <= trade_date]
        if len(df_up_to) > 10:
            pct_data[symbol] = format_pctline_data(df_up_to, symbol, max_lines)

    if not pct_data:
        return {"trade_date": trade_date, "next_date": next_date, "skip": True}

    # Generate Chronos2 forecasts if enabled (sync, happens once per day)
    chronos_forecasts = None
    if use_chronos:
        chronos_forecasts = _generate_chronos_forecasts(
            all_data, list(pct_data.keys()), trade_date, chronos_device
        )

    # Stage 1: Async portfolio allocation
    try:
        allocations = await async_allocate_portfolio(
            client=client,
            pct_data=pct_data,
            chronos_forecasts=chronos_forecasts,
            equity=equity,
            max_lines=max_lines,
            use_thinking=use_thinking,
        )
        logger.info(
            f"{trade_date}: Stage 1 - overall confidence {allocations.overall_confidence:.2f}, "
            f"{len(allocations.allocations)} allocations"
        )
    except Exception as e:
        logger.error(f"Stage 1 failed for {trade_date}: {e}")
        return {"trade_date": trade_date, "next_date": next_date, "skip": True, "error": str(e)}

    # Stage 2: Async price predictions (parallel for all symbols)
    predictions: dict[str, PricePrediction] = {}
    if allocations.should_trade(min_confidence):
        try:
            predictions = await async_predict_prices(
                client=client,
                pct_data=pct_data,
                allocations=allocations.allocations,
                chronos_forecasts=chronos_forecasts,
                max_lines=max_lines,
            )
            logger.info(f"{trade_date}: Stage 2 - {len(predictions)} price predictions")
        except Exception as e:
            logger.error(f"Stage 2 failed for {trade_date}: {e}")
            return {"trade_date": trade_date, "next_date": next_date, "skip": True, "error": str(e)}

    # Get next day's actual data
    next_day_data: dict[str, dict] = {}
    for symbol, df in all_data.items():
        next_df = df[df.index.date == next_date]
        if not next_df.empty:
            next_day_data[symbol] = {
                "open": next_df["open"].iloc[0],
                "high": next_df["high"].iloc[0],
                "low": next_df["low"].iloc[0],
                "close": next_df["close"].iloc[0],
            }

    # Simulate day (sync, fast)
    pnl, trades = _simulate_day(
        allocations, predictions, next_day_data, equity,
        min_confidence=min_confidence
    )

    return {
        "trade_date": trade_date,
        "next_date": next_date,
        "skip": False,
        "pnl": pnl,
        "trades": trades,
        "allocations": allocations,
    }


async def run_backtest_async(
    symbols: Sequence[str],
    start_date: date,
    end_date: date,
    initial_capital: float = 100_000.0,
    min_confidence: float = 0.3,
    max_lines: int = 200,
    use_chronos: bool = True,
    chronos_device: str = "cuda",
    use_thinking: bool = False,
    max_parallel_days: int = 5,
) -> dict:
    """Run two-stage backtest with parallel day processing.

    Note: Days are processed in parallel for API calls, but P&L is still
    calculated sequentially (each day's equity depends on previous day).

    Args:
        symbols: List of symbols to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Starting capital
        min_confidence: Minimum confidence to execute trades
        max_lines: Max historical lines per symbol for prompts
        use_chronos: Whether to use Chronos2 forecasts
        chronos_device: Device for Chronos2 (cuda/cpu)
        use_thinking: Enable extended thinking (Opus with 60000 token budget)
        max_parallel_days: Max days to process in parallel (limits API load)

    Returns:
        Backtest results dict
    """
    logger.info(f"ASYNC Two-Stage Backtest: {start_date} to {end_date}")
    if use_thinking:
        logger.info("Extended thinking ENABLED (Opus with 60000 token budget)")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Capital: ${initial_capital:,.0f}, min_confidence: {min_confidence}")

    # Get all historical data
    all_data = _get_all_data(symbols, end_date, lookback_days=max_lines + 100)
    for symbol, df in all_data.items():
        logger.info(f"Loaded {len(df)} bars for {symbol}")

    if not all_data:
        logger.error("No data loaded")
        return {"error": "No data"}

    # Get trading days in range - find a df with DatetimeIndex
    sample_df = None
    for df in all_data.values():
        if hasattr(df.index, 'date') and len(df) > 0:
            sample_df = df
            break

    if sample_df is None:
        logger.error("No DataFrames with DatetimeIndex found")
        return {"error": "No valid data"}

    trading_days = sample_df[
        (sample_df.index.date >= start_date) &
        (sample_df.index.date <= end_date)
    ].index.date
    trading_days = sorted(set(trading_days))

    logger.info(f"Found {len(trading_days)} trading days")

    # Create async client
    client = anthropic.AsyncAnthropic()

    # Process days in batches (to avoid overwhelming the API)
    equity = initial_capital
    results = []
    day_pairs = [(trading_days[i], trading_days[i + 1]) for i in range(len(trading_days) - 1)]

    for batch_start in range(0, len(day_pairs), max_parallel_days):
        batch = day_pairs[batch_start:batch_start + max_parallel_days]

        # Launch all days in batch in parallel
        tasks = [
            _process_single_day(
                client=client,
                all_data=all_data,
                symbols=symbols,
                trade_date=trade_date,
                next_date=next_date,
                equity=equity,  # Use current equity for all parallel days
                max_lines=max_lines,
                use_chronos=use_chronos,
                chronos_device=chronos_device,
                use_thinking=use_thinking,
                min_confidence=min_confidence,
            )
            for trade_date, next_date in batch
        ]

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results sequentially to update equity
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Day processing failed: {result}")
                continue

            if result.get("skip"):
                continue

            pnl = result["pnl"]
            trades = result["trades"]
            trade_date = result["trade_date"]
            next_date = result["next_date"]
            allocations = result["allocations"]

            equity += pnl
            daily_return = pnl / (equity - pnl) if equity != pnl else 0

            results.append({
                "date": trade_date,
                "next_date": next_date,
                "pnl": pnl,
                "equity": equity,
                "daily_return": daily_return,
                "num_trades": len(trades),
                "trades": trades,
                "overall_confidence": allocations.overall_confidence,
            })

            # Log trade details
            for t in trades:
                leverage_str = f" {t['leverage']:.1f}x" if t.get('leverage', 1.0) > 1.0 else ""
                logger.info(
                    f"  {t['symbol']} {t['direction']}{leverage_str}: "
                    f"entry ${t['entry_price']:.2f} -> exit ${t['exit_price']:.2f} "
                    f"({t['exit_reason']}), PnL ${t['net_pnl']:+,.2f}"
                )

            logger.info(
                f"{trade_date}: PnL ${pnl:+,.2f} ({daily_return:+.3%}), "
                f"equity ${equity:,.2f}, {len(trades)} trades"
            )

    # Summary
    total_return = (equity - initial_capital) / initial_capital
    num_days = len(results)

    # Use 365 for crypto (24/7 trading)
    trading_days_per_year = 365 if any("USD" in s for s in symbols) else 252
    annualized = ((1 + total_return) ** (trading_days_per_year / num_days) - 1) if num_days > 0 else 0

    # Calculate win rate
    all_trades = [t for r in results for t in r["trades"]]
    winning_trades = [t for t in all_trades if t["net_pnl"] > 0]
    win_rate = len(winning_trades) / len(all_trades) if all_trades else 0

    # Calculate avg confidence
    avg_confidence = sum(r["overall_confidence"] for r in results) / len(results) if results else 0

    summary = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "trading_days": num_days,
        "initial_capital": initial_capital,
        "final_equity": equity,
        "total_return": total_return,
        "annualized_return": annualized,
        "total_trades": len(all_trades),
        "win_rate": win_rate,
        "avg_confidence": avg_confidence,
        "daily_results": results,
    }

    logger.info("=" * 60)
    logger.info("ASYNC TWO-STAGE BACKTEST COMPLETE")
    logger.info(f"Total return: {total_return:.2%}")
    logger.info(f"Annualized: {annualized:.2%}")
    logger.info(f"Final equity: ${equity:,.2f}")
    logger.info(f"Total trades: {len(all_trades)}, Win rate: {win_rate:.1%}")
    logger.info(f"Avg confidence: {avg_confidence:.2f}")
    logger.info("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Two-stage portfolio backtest")
    parser.add_argument(
        "--symbols", nargs="+",
        default=["BTCUSD", "ETHUSD", "BNBUSD", "UNIUSD", "SKYUSD"],
        help="Symbols to trade"
    )
    parser.add_argument("--days", type=int, default=15, help="Days to backtest")
    parser.add_argument("--capital", type=float, default=100_000, help="Initial capital")
    parser.add_argument("--min-confidence", type=float, default=0.3, help="Min confidence threshold")
    parser.add_argument("--max-lines", type=int, default=200, help="Max historical lines")
    parser.add_argument("--no-chronos", action="store_true", help="Disable Chronos2 forecasts")
    parser.add_argument("--device", type=str, default="cuda", help="Chronos2 device")
    parser.add_argument("--end-date", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--thinking", action="store_true", help="Enable extended thinking (Opus with 60000 token budget)")
    parser.add_argument("--async", dest="use_async", action="store_true", help="Use async parallel backtest")
    parser.add_argument("--parallel-days", type=int, default=5, help="Max days to process in parallel (async mode)")

    args = parser.parse_args()

    # Calculate date range - use fixed historical period for consistent testing
    if args.end_date:
        end_date = date.fromisoformat(args.end_date)
    else:
        # Use a known good historical period
        end_date = date(2025, 10, 15)
    start_date = end_date - timedelta(days=args.days + 5)

    if args.use_async:
        asyncio.run(run_backtest_async(
            symbols=args.symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.capital,
            min_confidence=args.min_confidence,
            max_lines=args.max_lines,
            use_chronos=not args.no_chronos,
            chronos_device=args.device,
            use_thinking=args.thinking,
            max_parallel_days=args.parallel_days,
        ))
    else:
        run_backtest(
            symbols=args.symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.capital,
            min_confidence=args.min_confidence,
            max_lines=args.max_lines,
            use_chronos=not args.no_chronos,
            chronos_device=args.device,
            use_thinking=args.thinking,
        )


if __name__ == "__main__":
    main()
