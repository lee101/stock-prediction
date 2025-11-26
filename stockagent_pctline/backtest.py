"""Backtest for pct-line based trading agent."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from typing import Sequence

import pandas as pd
from loguru import logger

from stockagent.agentsimulator.market_data import fetch_latest_ohlc
from stockagent.constants import TRADING_FEE

from .data_formatter import format_pctline_data, PctLineData
from .agent import generate_allocation_plan, AllocationPrediction


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


def _simulate_day(
    predictions: dict[str, AllocationPrediction],
    next_day_data: dict[str, dict],
    equity: float,
    trading_fee: float = TRADING_FEE,
) -> tuple[float, list[dict]]:
    """Simulate one day's trading based on predictions.

    Args:
        predictions: Allocation predictions from agent
        next_day_data: Actual OHLC for next day {symbol: {open, high, low, close}}
        equity: Current equity
        trading_fee: Fee per trade as decimal

    Returns:
        (pnl, trade_details)
    """
    pnl = 0.0
    trades = []

    for symbol, pred in predictions.items():
        if pred.alloc <= 0.001:
            continue

        if symbol not in next_day_data:
            continue

        actual = next_day_data[symbol]
        actual_open = actual["open"]
        actual_high = actual["high"]
        actual_low = actual["low"]
        actual_close = actual["close"]

        # Calculate notional
        notional = equity * pred.alloc

        # Entry: we try to buy at open (market order simulation)
        entry_price = actual_open
        quantity = int(notional / entry_price) if entry_price > 0 else 0

        if quantity <= 0:
            continue

        # Exit: check if predicted exit price was hit during the day
        # Exit at predicted high (limit sell) if actual high reaches it
        exit_price = pred.pred_high

        if actual_high >= exit_price:
            # Limit order filled at predicted high
            sell_price = exit_price
        else:
            # Limit not hit, close at end of day
            sell_price = actual_close

        # Calculate P&L
        gross_pnl = (sell_price - entry_price) * quantity
        fees = (entry_price * quantity + sell_price * quantity) * trading_fee
        net_pnl = gross_pnl - fees

        pnl += net_pnl

        trades.append({
            "symbol": symbol,
            "quantity": quantity,
            "entry_price": entry_price,
            "exit_price": sell_price,
            "limit_hit": actual_high >= exit_price,
            "gross_pnl": gross_pnl,
            "fees": fees,
            "net_pnl": net_pnl,
            "alloc": pred.alloc,
        })

    return pnl, trades


def run_backtest(
    symbols: Sequence[str],
    start_date: date,
    end_date: date,
    initial_capital: float = 100_000.0,
    prompt_version: str = "v1",
    max_lines: int = 500,
) -> dict:
    """Run backtest over a date range.

    Args:
        symbols: List of symbols to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Starting capital
        prompt_version: Which prompt version to use
        max_lines: Max historical lines per symbol

    Returns:
        Backtest results dict
    """
    logger.info(f"Backtest: {start_date} to {end_date}, symbols: {symbols}")
    logger.info(f"Capital: ${initial_capital:,.0f}, prompt: {prompt_version}")

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

    for i, trade_date in enumerate(trading_days[:-1]):  # -1 because we need next day
        next_date = trading_days[i + 1]

        # Build pct-line data up to trade_date
        pct_data = {}
        for symbol, df in all_data.items():
            df_up_to = df[df.index.date <= trade_date]
            if len(df_up_to) > 1:
                pct_data[symbol] = format_pctline_data(df_up_to, symbol, max_lines)

        if not pct_data:
            continue

        # Get predictions
        try:
            predictions = generate_allocation_plan(pct_data, prompt_version, max_lines)
        except Exception as e:
            logger.error(f"Prediction failed for {trade_date}: {e}")
            continue

        # Get next day's actual data
        next_day_data = {}
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
        pnl, trades = _simulate_day(predictions, next_day_data, equity)
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
        })

        logger.info(f"{trade_date}: PnL ${pnl:+,.2f} ({daily_return:+.3%}), equity ${equity:,.2f}, {len(trades)} trades")

    # Summary
    total_return = (equity - initial_capital) / initial_capital
    num_days = len(results)
    annualized = ((1 + total_return) ** (252 / num_days) - 1) if num_days > 0 else 0

    summary = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "trading_days": num_days,
        "initial_capital": initial_capital,
        "final_equity": equity,
        "total_return": total_return,
        "annualized_return": annualized,
        "daily_results": results,
        "prompt_version": prompt_version,
    }

    logger.info("=" * 60)
    logger.info(f"BACKTEST COMPLETE")
    logger.info(f"Total return: {total_return:.2%}")
    logger.info(f"Annualized: {annualized:.2%}")
    logger.info(f"Final equity: ${equity:,.2f}")
    logger.info("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Pct-line backtest")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "NVDA"])
    parser.add_argument("--days", type=int, default=15)
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--prompt", type=str, default="v1", choices=["v1", "v2", "v3", "v4"])
    parser.add_argument("--max-lines", type=int, default=500)

    args = parser.parse_args()

    # Use same test period as Chronos2 for comparison
    end_date = date(2023, 7, 14)
    start_date = end_date - timedelta(days=args.days + 5)

    run_backtest(
        symbols=args.symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        prompt_version=args.prompt,
        max_lines=args.max_lines,
    )


if __name__ == "__main__":
    main()
