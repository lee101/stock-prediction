#!/usr/bin/env python3
"""
Batch runner for market simulator across multiple stock pairs.
Downloads historical data, runs simulations, and saves PnL results.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from alpaca.data import StockBarsRequest, CryptoBarsRequest, TimeFrame, TimeFrameUnit
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.trading.client import TradingClient

from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
from src.fixtures import all_crypto_symbols
from marketsimulator.runner import simulate_strategy, SimulationReport


# Default symbols from trade_stock_e2e.py
DEFAULT_SYMBOLS = [
    # Top performing equities (high Sharpe, good win rates)
    "EQIX", "GS", "COST", "CRM", "AXP", "BA", "GE", "LLY", "AVGO", "SPY",
    "SHOP", "GLD", "PLTR", "MCD", "V", "VTI", "QQQ", "MA", "SAP",
    # Keep existing profitable ones
    "COUR", "ADBE", "INTC", "QUBT",
    # Top crypto performers
    "BTCUSD", "ETHUSD", "UNIUSD", "LINKUSD",
]


def get_all_alpaca_tradable_symbols(asset_class: str = "us_equity") -> List[str]:
    """
    Fetch all tradable symbols from Alpaca Markets API.

    Args:
        asset_class: "us_equity" or "crypto"

    Returns:
        List of tradable symbol strings
    """
    try:
        trading_client = TradingClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
        assets = trading_client.get_all_assets()

        tradable_symbols = []
        for asset in assets:
            # Filter by asset class and tradability
            if (asset.tradable and
                asset.status == 'active' and
                asset.asset_class == asset_class):
                tradable_symbols.append(asset.symbol)

        logger.info(f"Found {len(tradable_symbols)} tradable {asset_class} symbols from Alpaca")
        return sorted(tradable_symbols)

    except Exception as e:
        logger.error(f"Failed to fetch tradable symbols: {e}")
        return []


def download_historical_data(
    symbol: str,
    output_dir: Path,
    years: int = 10,
    is_crypto: bool = False
) -> Optional[pd.DataFrame]:
    """
    Download historical data for a symbol and save to trainingdata/ directory.

    Args:
        symbol: Stock/crypto symbol
        output_dir: Base directory for training data
        years: Number of years of historical data
        is_crypto: Whether this is a crypto symbol

    Returns:
        DataFrame with historical data or None on failure
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)

        if is_crypto:
            client = CryptoHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(1, TimeFrameUnit.Hour),
                start=start_date,
                end=end_date
            )
            bars = client.get_crypto_bars(request)
        else:
            client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(1, TimeFrameUnit.Hour),
                start=start_date,
                end=end_date,
                adjustment='raw'
            )
            bars = client.get_stock_bars(request)

        if bars and bars.df is not None and not bars.df.empty:
            df = bars.df

            # If multi-index with symbol, extract it
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level='symbol')

            # Reset index to make timestamp a column
            df = df.reset_index()

            # Normalize column names
            df.columns = [col.lower() if col != 'timestamp' else 'timestamp' for col in df.columns]

            # Ensure proper column names for simulator
            rename_map = {}
            for col in df.columns:
                col_lower = str(col).lower()
                if col_lower == 'open':
                    rename_map[col] = 'Open'
                elif col_lower == 'high':
                    rename_map[col] = 'High'
                elif col_lower == 'low':
                    rename_map[col] = 'Low'
                elif col_lower == 'close':
                    rename_map[col] = 'Close'
                elif col_lower == 'volume':
                    rename_map[col] = 'Volume'

            if rename_map:
                df = df.rename(columns=rename_map)

            # Save to trainingdata directory
            output_file = output_dir / f"{symbol}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Downloaded {len(df)} rows for {symbol} -> {output_file}")

            return df
        else:
            logger.warning(f"No data received for {symbol}")
            return None

    except Exception as e:
        logger.error(f"Error downloading {symbol}: {e}")
        return None


def run_simulation_for_symbol(
    symbol: str,
    simulation_days: int,
    initial_cash: float,
    output_dir: Path,
    force_kronos: bool = True
) -> Optional[Dict]:
    """
    Run market simulator for a single symbol and return results.

    Args:
        symbol: Symbol to simulate
        simulation_days: Number of trading days to simulate
        initial_cash: Starting cash for simulation
        output_dir: Directory to save results
        force_kronos: Use Kronos-only forecasting

    Returns:
        Dict with simulation results or None on failure
    """
    try:
        logger.info(f"Running simulation for {symbol}...")

        # Run simulation
        report = simulate_strategy(
            symbols=[symbol],
            days=simulation_days,
            step_size=24,  # 24 hourly steps = 1 day
            initial_cash=initial_cash,
            top_k=1,
            output_dir=output_dir / "plots" / symbol,
            force_kronos=force_kronos,
            flatten_end=True
        )

        # Extract key metrics
        result = {
            "symbol": symbol,
            "initial_cash": report.initial_cash,
            "final_equity": report.final_equity,
            "total_return": report.total_return,
            "total_return_pct": report.total_return_pct,
            "fees_paid": report.fees_paid,
            "trades_executed": report.trades_executed,
            "max_drawdown": report.max_drawdown,
            "max_drawdown_pct": report.max_drawdown_pct,
            "sharpe_ratio": _calculate_sharpe_ratio(report),
            "win_rate": _calculate_win_rate(report),
            "profit_factor": _calculate_profit_factor(report),
            "daily_snapshots": [
                {
                    "day": snap.day_index,
                    "phase": snap.phase,
                    "timestamp": snap.timestamp.isoformat(),
                    "equity": snap.equity,
                    "cash": snap.cash,
                    "positions": snap.positions
                }
                for snap in report.daily_snapshots
            ],
            "pnl_over_time": _extract_pnl_over_time(report)
        }

        return result

    except Exception as e:
        logger.error(f"Simulation failed for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _calculate_sharpe_ratio(report: SimulationReport) -> float:
    """Calculate Sharpe ratio from daily snapshots."""
    if not report.daily_snapshots:
        return 0.0

    # Extract daily returns
    equities = [snap.equity for snap in sorted(report.daily_snapshots, key=lambda s: s.timestamp)]
    if len(equities) < 2:
        return 0.0

    returns = pd.Series(equities).pct_change().dropna()
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    # Annualized Sharpe ratio (assuming ~252 trading days/year)
    sharpe = (returns.mean() / returns.std()) * (252 ** 0.5)
    return float(sharpe)


def _calculate_win_rate(report: SimulationReport) -> float:
    """Calculate win rate from trade executions."""
    if not report.trade_executions:
        return 0.0

    # Group trades by symbol to calculate P&L per trade
    winning_trades = 0
    total_trades = 0

    trades_by_symbol = {}
    for trade in report.trade_executions:
        if trade.symbol not in trades_by_symbol:
            trades_by_symbol[trade.symbol] = []
        trades_by_symbol[trade.symbol].append(trade)

    # Simple win rate: positive P&L trades
    for symbol, trades in trades_by_symbol.items():
        for trade in trades:
            if trade.cash_delta > 0:
                winning_trades += 1
            total_trades += 1

    if total_trades == 0:
        return 0.0

    return winning_trades / total_trades


def _calculate_profit_factor(report: SimulationReport) -> float:
    """Calculate profit factor (gross profit / gross loss)."""
    if not report.trade_executions:
        return 0.0

    gross_profit = sum(t.cash_delta for t in report.trade_executions if t.cash_delta > 0)
    gross_loss = abs(sum(t.cash_delta for t in report.trade_executions if t.cash_delta < 0))

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def _extract_pnl_over_time(report: SimulationReport) -> List[Dict]:
    """Extract PnL over time from snapshots."""
    pnl_timeline = []
    initial_cash = report.initial_cash

    for snap in sorted(report.daily_snapshots, key=lambda s: s.timestamp):
        pnl = snap.equity - initial_cash
        pnl_pct = (pnl / initial_cash * 100) if initial_cash > 0 else 0.0

        pnl_timeline.append({
            "timestamp": snap.timestamp.isoformat(),
            "day": snap.day_index,
            "phase": snap.phase,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "equity": snap.equity
        })

    return pnl_timeline


def save_results(
    results: List[Dict],
    output_dir: Path,
    run_name: str
) -> None:
    """
    Save simulation results to strategytraining/ directory.

    Args:
        results: List of simulation results
        output_dir: Output directory
        run_name: Name for this batch run
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results as JSON
    full_results_file = output_dir / f"{run_name}_full_results.json"
    with open(full_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved full results to {full_results_file}")

    # Create summary DataFrame
    summary_data = []
    for r in results:
        summary_data.append({
            "symbol": r["symbol"],
            "final_equity": r["final_equity"],
            "total_return": r["total_return"],
            "total_return_pct": r["total_return_pct"],
            "sharpe_ratio": r["sharpe_ratio"],
            "max_drawdown_pct": r["max_drawdown_pct"],
            "win_rate": r["win_rate"],
            "profit_factor": r["profit_factor"],
            "trades": r["trades_executed"],
            "fees": r["fees_paid"]
        })

    df = pd.DataFrame(summary_data)
    df = df.sort_values("total_return_pct", ascending=False)

    # Save summary as CSV
    summary_csv = output_dir / f"{run_name}_summary.csv"
    df.to_csv(summary_csv, index=False)
    logger.info(f"Saved summary to {summary_csv}")

    # Save PnL over time for each symbol
    pnl_dir = output_dir / "pnl_timeseries"
    pnl_dir.mkdir(exist_ok=True)

    for r in results:
        symbol = r["symbol"]
        pnl_data = r["pnl_over_time"]
        pnl_df = pd.DataFrame(pnl_data)
        pnl_file = pnl_dir / f"{symbol}_pnl.csv"
        pnl_df.to_csv(pnl_file, index=False)

    logger.info(f"Saved PnL timeseries to {pnl_dir}")

    # Print top performers
    print("\n" + "="*80)
    print(f"TOP 10 BEST PERFORMING SYMBOLS (by Return %)")
    print("="*80)
    print(df.head(10).to_string(index=False))
    print("\n" + "="*80)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch run market simulator across multiple stock pairs"
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Specific symbols to simulate (default: use symbols from trade_stock_e2e.py)"
    )

    parser.add_argument(
        "--use-all-alpaca",
        action="store_true",
        help="Use all tradable symbols from Alpaca Markets API"
    )

    parser.add_argument(
        "--asset-class",
        choices=["us_equity", "crypto"],
        default="us_equity",
        help="Asset class when using --use-all-alpaca (default: us_equity)"
    )

    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download data, skip simulation"
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip data download, use existing data"
    )

    parser.add_argument(
        "--simulation-days",
        type=int,
        default=30,
        help="Number of trading days to simulate (default: 30)"
    )

    parser.add_argument(
        "--data-years",
        type=int,
        default=10,
        help="Years of historical data to download (default: 10)"
    )

    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100_000.0,
        help="Initial cash for simulation (default: 100000)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("strategytraining/batch_results"),
        help="Output directory for results (default: strategytraining/batch_results)"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("trainingdata"),
        help="Directory for training data (default: trainingdata)"
    )

    parser.add_argument(
        "--run-name",
        default=None,
        help="Name for this batch run (default: auto-generated timestamp)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of symbols to process (useful for testing)"
    )

    parser.add_argument(
        "--force-kronos",
        action="store_true",
        default=True,
        help="Use Kronos-only forecasting (default: True)"
    )

    return parser.parse_args()


def main() -> int:
    """Main function."""
    args = parse_args()

    # Generate run name if not provided
    if args.run_name is None:
        args.run_name = f"batch_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info("="*80)
    logger.info(f"BATCH MARKET SIMULATOR - {args.run_name}")
    logger.info("="*80)

    # Determine which symbols to use
    if args.use_all_alpaca:
        logger.info(f"Fetching all tradable {args.asset_class} symbols from Alpaca...")
        symbols = get_all_alpaca_tradable_symbols(args.asset_class)
    elif args.symbols:
        symbols = args.symbols
        logger.info(f"Using {len(symbols)} symbols from command line")
    else:
        symbols = DEFAULT_SYMBOLS
        logger.info(f"Using {len(symbols)} default symbols from trade_stock_e2e.py")

    if args.limit:
        symbols = symbols[:args.limit]
        logger.info(f"Limited to first {args.limit} symbols")

    logger.info(f"Total symbols to process: {len(symbols)}")
    logger.info(f"Symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")

    # Create directories
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Download historical data
    if not args.skip_download:
        logger.info("\n" + "="*80)
        logger.info("DOWNLOADING HISTORICAL DATA")
        logger.info("="*80)

        successful_downloads = 0
        for i, symbol in enumerate(symbols, 1):
            is_crypto = symbol.upper() in all_crypto_symbols
            logger.info(f"[{i}/{len(symbols)}] Downloading {symbol} ({'crypto' if is_crypto else 'stock'})...")

            df = download_historical_data(
                symbol=symbol,
                output_dir=args.data_dir,
                years=args.data_years,
                is_crypto=is_crypto
            )

            if df is not None:
                successful_downloads += 1

        logger.info(f"\nSuccessfully downloaded data for {successful_downloads}/{len(symbols)} symbols")

        if args.download_only:
            logger.info("Download-only mode. Exiting.")
            return 0

    # Run simulations
    logger.info("\n" + "="*80)
    logger.info("RUNNING SIMULATIONS")
    logger.info("="*80)

    results = []
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"\n[{i}/{len(symbols)}] Simulating {symbol}...")

        result = run_simulation_for_symbol(
            symbol=symbol,
            simulation_days=args.simulation_days,
            initial_cash=args.initial_cash,
            output_dir=args.output_dir,
            force_kronos=args.force_kronos
        )

        if result:
            results.append(result)
            logger.info(
                f"  ✓ {symbol}: Return = {result['total_return_pct']:.2f}%, "
                f"Sharpe = {result['sharpe_ratio']:.2f}, "
                f"Trades = {result['trades_executed']}"
            )
        else:
            logger.warning(f"  ✗ {symbol}: Simulation failed")

    # Save results
    if results:
        logger.info("\n" + "="*80)
        logger.info("SAVING RESULTS")
        logger.info("="*80)

        save_results(results, args.output_dir, args.run_name)

        logger.info("\n" + "="*80)
        logger.info("BATCH RUN COMPLETE")
        logger.info("="*80)
        logger.info(f"Successfully simulated {len(results)}/{len(symbols)} symbols")
        logger.info(f"Results saved to {args.output_dir}")

        return 0
    else:
        logger.error("No successful simulations. Exiting.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
