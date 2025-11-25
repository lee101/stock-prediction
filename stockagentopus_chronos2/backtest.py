"""CLI backtest runner for Claude Opus trading agent with Chronos2 forecasts."""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd
from loguru import logger

from stockagent.agentsimulator.data_models import AccountSnapshot
from stockagent.agentsimulator.market_data import MarketDataBundle
from stockagent.constants import DEFAULT_SYMBOLS

from .agent import simulate_opus_replanning


def _load_market_data_by_date(
    symbols: Sequence[str],
    start_date: date,
    end_date: date,
    lookback_days: int = 30,
    data_dir: Path = Path("trainingdata"),
) -> dict[date, MarketDataBundle]:
    """Load market data for each trading day in the backtest range."""
    results: dict[date, MarketDataBundle] = {}

    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        csv_path = data_dir / f"{symbol}.csv"
        if not csv_path.exists():
            logger.warning(f"No CSV file found for {symbol} at {csv_path}")
            continue
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        frames[symbol] = df

    if not frames:
        raise ValueError(f"No market data found in {data_dir}")

    # Find common trading days
    all_dates: set[date] = set()
    for df in frames.values():
        df_dates = set(df["timestamp"].dt.date.unique())
        if not all_dates:
            all_dates = df_dates
        else:
            all_dates &= df_dates

    trading_days = sorted([d for d in all_dates if start_date <= d <= end_date])
    if not trading_days:
        raise ValueError(f"No trading days found between {start_date} and {end_date}")

    logger.info(f"Found {len(trading_days)} trading days from {trading_days[0]} to {trading_days[-1]}")

    for target_date in trading_days:
        bars: dict[str, pd.DataFrame] = {}
        for symbol, df in frames.items():
            # Get data up to (but not including) target date for context
            # and include target date for simulation
            mask = df["timestamp"].dt.date <= target_date
            symbol_data = df[mask].tail(lookback_days + 1).copy()
            if symbol_data.empty:
                continue
            symbol_data = symbol_data.set_index("timestamp")
            bars[symbol] = symbol_data

        if bars:
            # as_of needs to be a datetime
            as_of = datetime.combine(target_date, datetime.min.time())
            bundle = MarketDataBundle(bars=bars, lookback_days=lookback_days, as_of=as_of)
            results[target_date] = bundle

    return results


def run_backtest(
    symbols: Sequence[str],
    num_days: int,
    starting_capital: float,
    data_dir: Path = Path("trainingdata"),
    lookback_days: int = 30,
    chronos2_device: str = "cuda",
) -> None:
    """Run a multi-day backtest with Opus + Chronos2."""
    # Determine date range from available data
    sample_csv = None
    for symbol in symbols:
        csv_path = data_dir / f"{symbol}.csv"
        if csv_path.exists():
            sample_csv = csv_path
            break

    if sample_csv is None:
        logger.error(f"No CSV files found in {data_dir}")
        sys.exit(1)

    sample_df = pd.read_csv(sample_csv, parse_dates=["timestamp"])
    all_dates = sorted(sample_df["timestamp"].dt.date.unique())

    if len(all_dates) < num_days + lookback_days:
        logger.warning(
            f"Only {len(all_dates)} days available, reducing backtest to "
            f"{max(1, len(all_dates) - lookback_days)} days"
        )
        num_days = max(1, len(all_dates) - lookback_days)

    # Use dates from the middle/end of available data
    end_idx = len(all_dates) - 1
    start_idx = max(lookback_days, end_idx - num_days + 1)

    start_date = all_dates[start_idx]
    end_date = all_dates[end_idx]

    logger.info(f"Backtest period: {start_date} to {end_date} ({num_days} days)")
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Starting capital: ${starting_capital:,.2f}")
    logger.info(f"Using Chronos2 device: {chronos2_device}")

    # Load market data
    market_data_by_date = _load_market_data_by_date(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        lookback_days=lookback_days,
        data_dir=data_dir,
    )

    target_dates = sorted(market_data_by_date.keys())[:num_days]
    logger.info(f"Simulating {len(target_dates)} trading days")

    # Create initial account snapshot
    initial_snapshot = AccountSnapshot(
        equity=starting_capital,
        cash=starting_capital,
        buying_power=starting_capital,
        timestamp=datetime.combine(target_dates[0], datetime.min.time()).replace(tzinfo=timezone.utc),
        positions=[],
    )

    # Run the backtest
    result = simulate_opus_replanning(
        market_data_by_date=market_data_by_date,
        account_snapshot=initial_snapshot,
        target_dates=target_dates,
        symbols=list(symbols),
        chronos2_device=chronos2_device,
    )

    # Print results
    print("\n" + "=" * 60)
    print("CLAUDE OPUS + CHRONOS2 BACKTEST RESULTS")
    print("=" * 60)
    print(f"Period: {target_dates[0]} to {target_dates[-1]}")
    print(f"Days simulated: {len(result.steps)}")
    print(f"Starting equity: ${result.starting_equity:,.2f}")
    print(f"Ending equity: ${result.ending_equity:,.2f}")
    print(f"Total return: {result.total_return_pct:.2%}")
    print(f"Annualized return ({result.annualization_days}d/yr): {result.annualized_return_pct:.2%}")
    print()

    print("Daily Results:")
    print("-" * 60)
    for step in result.steps:
        forecast_info = ""
        if step.chronos2_forecasts:
            forecast_info = f" (Chronos2: {len(step.chronos2_forecasts)} forecasts)"
        print(
            f"  {step.date}: PnL ${step.simulation.net_pnl:+,.2f} "
            f"({step.daily_return_pct:+.3%}), equity ${step.ending_equity:,.2f}{forecast_info}"
        )
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Claude Opus + Chronos2 trading backtest"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "MSFT", "NVDA"],
        help="Symbols to trade (default: AAPL MSFT NVDA)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=15,
        help="Number of trading days to simulate (default: 15)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Starting capital (default: $100,000)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("trainingdata"),
        help="Directory containing CSV files (default: trainingdata)",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=30,
        help="Days of historical context (default: 30)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for Chronos2 model (default: cuda)",
    )

    args = parser.parse_args()

    run_backtest(
        symbols=args.symbols,
        num_days=args.days,
        starting_capital=args.capital,
        data_dir=args.data_dir,
        lookback_days=args.lookback,
        chronos2_device=args.device,
    )


if __name__ == "__main__":
    main()
