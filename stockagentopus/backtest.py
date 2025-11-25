#!/usr/bin/env python3
"""Backtest script for the Claude Opus Stock Trading Agent.

This script runs a simulated backtest using the Opus agent to generate
trading plans and evaluates them against historical market data.

Usage:
    python -m stockagentopus.backtest --symbols AAPL MSFT --days 5
    python -m stockagentopus.backtest --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stockagent.agentsimulator.data_models import AccountSnapshot
from stockagent.agentsimulator.market_data import MarketDataBundle


TRAINING_DATA_DIR = REPO_ROOT / "trainingdata"


def _load_from_csv(symbol: str, lookback_days: int, end_date: date) -> pd.DataFrame | None:
    """Load historical data from CSV files in trainingdata/."""
    csv_path = TRAINING_DATA_DIR / f"{symbol}.csv"
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)

        # Handle different column naming conventions
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ("date", "timestamp", "time"):
                col_map[col] = "date"
            elif col_lower == "open":
                col_map[col] = "open"
            elif col_lower == "high":
                col_map[col] = "high"
            elif col_lower == "low":
                col_map[col] = "low"
            elif col_lower == "close":
                col_map[col] = "close"

        df = df.rename(columns=col_map)

        if "date" not in df.columns:
            return None

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df.sort_index()

        # Make index timezone-aware
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        # Filter to lookback period
        end_dt = datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        df = df[df.index <= end_dt].tail(lookback_days)

        if len(df) < 5:
            return None

        return df[["open", "high", "low", "close"]]
    except Exception as e:
        logger.warning(f"Failed to load {symbol} from CSV: {e}")
        return None


def _load_historical_data(
    symbols: Sequence[str],
    lookback_days: int = 30,
    end_date: date | None = None,
) -> MarketDataBundle:
    """Load historical market data for the given symbols.

    Tries CSV files in trainingdata/ first, falls back to mock data.
    """
    end_date = end_date or date.today()

    bars = {}
    for symbol in symbols:
        df = _load_from_csv(symbol, lookback_days, end_date)
        if df is not None:
            bars[symbol] = df
            logger.info(f"Loaded {len(df)} bars for {symbol} from CSV")

    if bars:
        # Use the latest date from loaded data
        as_of = max(df.index.max() for df in bars.values())
        return MarketDataBundle(
            bars=bars,
            lookback_days=lookback_days,
            as_of=as_of.to_pydatetime(),
        )

    # Fallback to market data loader if no CSVs found
    try:
        from src.market_data_loader import load_market_bundle
        return load_market_bundle(
            symbols=list(symbols),
            lookback_days=lookback_days,
            as_of=datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc),
        )
    except Exception as e:
        logger.warning(f"Could not load real market data: {e}. Using mock data.")

    index = pd.date_range(
        end=datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc),
        periods=lookback_days,
        freq="B",  # Business days
    )

    bars = {}
    for symbol in symbols:
        base_price = 100.0 + hash(symbol) % 100
        noise = pd.Series(
            [0.01 * (hash(f"{symbol}{i}") % 100 - 50) for i in range(lookback_days)]
        ).cumsum()

        close_series = pd.Series(base_price + noise.values, index=index)
        high_series = close_series * (1 + 0.01 * abs((hash(f"{symbol}h") % 30) / 10))
        low_series = close_series * (1 - 0.01 * abs((hash(f"{symbol}l") % 30) / 10))
        open_series = (close_series.shift(1).fillna(close_series.iloc[0]) + close_series) / 2

        bars[symbol] = pd.DataFrame(
            {
                "open": open_series.values,
                "high": high_series.values,
                "low": low_series.values,
                "close": close_series.values,
            },
            index=index,
        )

    return MarketDataBundle(
        bars=bars,
        lookback_days=lookback_days,
        as_of=index[-1].to_pydatetime(),
    )


def _create_account_snapshot(
    starting_cash: float = 100_000.0,
) -> AccountSnapshot:
    """Create an initial account snapshot."""
    return AccountSnapshot(
        equity=starting_cash,
        cash=starting_cash,
        buying_power=starting_cash,
        timestamp=datetime.now(timezone.utc),
        positions=[],
    )


def run_backtest(
    symbols: Sequence[str],
    num_days: int = 5,
    starting_cash: float = 100_000.0,
    calibration_window: int = 14,
    use_mock_llm: bool = False,
) -> dict:
    """Run a backtest of the Opus trading agent.

    Args:
        symbols: List of symbols to trade
        num_days: Number of trading days to simulate
        starting_cash: Initial capital
        calibration_window: Days for signal calibration
        use_mock_llm: If True, use mock LLM responses for testing

    Returns:
        Dictionary containing backtest results
    """
    from stockagentopus.agent import simulate_opus_replanning

    logger.info(f"Starting Opus backtest: {symbols}, {num_days} days, ${starting_cash:,.0f} capital")

    bundle = _load_historical_data(symbols, lookback_days=30)
    snapshot = _create_account_snapshot(starting_cash)

    trading_dates = bundle.trading_days()
    if len(trading_dates) < num_days:
        logger.warning(f"Only {len(trading_dates)} trading days available, requested {num_days}")
        num_days = len(trading_dates)

    target_dates = trading_dates[-num_days:]

    market_data_by_date = {d: bundle for d in target_dates}

    if use_mock_llm:
        import stockagentopus.agent as opus_agent

        def _mock_chat(*args, **kwargs):
            return json.dumps({
                "target_date": str(target_dates[0]),
                "instructions": [
                    {
                        "symbol": symbols[0],
                        "action": "buy",
                        "quantity": 10,
                        "execution_session": "market_open",
                        "entry_price": 100.0,
                        "exit_price": 102.0,
                    },
                    {
                        "symbol": symbols[0],
                        "action": "exit",
                        "quantity": 10,
                        "execution_session": "market_close",
                        "exit_price": 102.0,
                    },
                ],
                "metadata": {"strategy": "mock_backtest"},
            })

        original_call = opus_agent.call_opus_chat
        opus_agent.call_opus_chat = _mock_chat

    try:
        result = simulate_opus_replanning(
            market_data_by_date=market_data_by_date,
            account_snapshot=snapshot,
            target_dates=target_dates,
            symbols=list(symbols),
            calibration_window=calibration_window,
        )
    finally:
        if use_mock_llm:
            opus_agent.call_opus_chat = original_call

    summary = {
        "agent": "OpusAgent (Claude Opus 4.5)",
        "symbols": list(symbols),
        "num_days": len(result.steps),
        "starting_equity": result.starting_equity,
        "ending_equity": result.ending_equity,
        "total_return_pct": result.total_return_pct,
        "annualized_return_pct": result.annualized_return_pct,
        "annualization_days": result.annualization_days,
        "daily_results": [
            {
                "date": str(step.date),
                "net_pnl": step.simulation.net_pnl,
                "realized_pnl": step.simulation.realized_pnl,
                "fees": step.simulation.total_fees,
                "daily_return_pct": step.daily_return_pct,
            }
            for step in result.steps
        ],
    }

    logger.info(f"\n{result.summary()}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Backtest the Claude Opus Stock Trading Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "MSFT"],
        help="Symbols to trade (default: AAPL MSFT)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=5,
        help="Number of trading days to simulate (default: 5)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Starting capital (default: 100000)",
    )
    parser.add_argument(
        "--calibration-window",
        type=int,
        default=14,
        help="Signal calibration window in days (default: 14)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM responses (for testing without API key)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results JSON",
    )

    args = parser.parse_args()

    results = run_backtest(
        symbols=args.symbols,
        num_days=args.days,
        starting_cash=args.capital,
        calibration_window=args.calibration_window,
        use_mock_llm=args.mock,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2))
        logger.info(f"Results saved to {output_path}")
    else:
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
