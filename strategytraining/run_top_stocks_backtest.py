"""
Run backtests on top stocks and generate comprehensive reports

This script:
1. Fetches top N stocks from Alpaca (or uses provided list)
2. Downloads historical data for each stock
3. Runs all strategies across all stocks using the existing collector
4. Generates detailed CSV reports in strategytraining/reports/
"""

import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd
import argparse
from datetime import datetime
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fetch_top_stocks import fetch_top_stocks
from collect_strategy_pnl_dataset import StrategyPnLCollector, STRATEGIES
from alpaca_wrapper import download_training_pairs


def download_stock_data(symbols: List[str], data_dir: str = "trainingdata/train") -> List[str]:
    """
    Download historical data for stock symbols

    Returns:
        List of symbols that were successfully downloaded
    """
    logger.info(f"Downloading data for {len(symbols)} symbols...")

    # Download using existing infrastructure
    results = download_training_pairs(
        symbols=symbols,
        output_dir=data_dir,
        history_days=365 * 4,  # 4 years of data
        skip_if_recent_days=7,  # Skip if updated within 7 days
    )

    # Filter to successful downloads
    successful = [
        r['symbol'] for r in results
        if r.get('status') in ('ok', 'skipped')
    ]

    logger.info(f"Successfully downloaded/cached data for {len(successful)} symbols")
    return successful


def run_backtest_and_generate_reports(
    symbols: List[str],
    data_dir: str = "trainingdata/train",
    output_dir: str = "strategytraining/reports",
    window_days: int = 7,
    stride_days: int = 7,
) -> None:
    """
    Run backtest on all symbols and generate reports
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting backtests on {len(symbols)} symbols")
    logger.info(f"Strategies: {STRATEGIES}")
    logger.info(f"Window: {window_days} days, Stride: {stride_days} days")

    # Use existing collector
    collector = StrategyPnLCollector(
        data_dir=data_dir,
        output_dir=str(output_path),
        window_days=window_days,
        stride_days=stride_days,
    )

    # Collect data for all symbols
    all_performance = []
    all_trades = []

    for symbol in symbols:
        logger.info(f"Processing {symbol}...")
        try:
            result = collector.collect_symbol_data(symbol)
            if result:
                all_performance.extend(result.get('strategy_performance', []))
                all_trades.extend(result.get('trades', []))
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue

    if not all_performance:
        logger.error("No results collected!")
        return

    logger.info(f"Collected {len(all_performance)} strategy-window results")
    logger.info(f"Collected {len(all_trades)} trades")

    # Generate reports
    _generate_reports(all_performance, all_trades, output_path, timestamp, stride_days)


def _generate_reports(
    performance_data: List[dict],
    trades_data: List[dict],
    output_dir: Path,
    timestamp: str,
    stride_days: int,
) -> None:
    """Generate comprehensive CSV reports"""

    logger.info("Generating reports...")

    # Convert to DataFrames
    df_perf = pd.DataFrame(performance_data)
    df_trades = pd.DataFrame(trades_data) if trades_data else pd.DataFrame()

    # ===== 1. Strategy Performance by Stock =====
    # Aggregate across windows for each symbol-strategy pair
    agg_perf = df_perf.groupby(['symbol', 'strategy']).agg({
        'total_pnl': ['mean', 'sum', 'std', 'count'],
        'total_return': 'mean',
        'sharpe_ratio': 'mean',
        'win_rate': 'mean',
        'total_trades': 'sum',
        'max_drawdown': 'mean',
    }).reset_index()

    agg_perf.columns = [
        'symbol', 'strategy', 'avg_pnl_per_window', 'total_pnl', 'pnl_std', 'num_windows',
        'avg_return', 'avg_sharpe', 'avg_win_rate', 'total_trades', 'avg_max_drawdown'
    ]

    # Calculate annualized metrics
    windows_per_year = 365 / stride_days
    agg_perf['annualized_pnl'] = agg_perf['avg_pnl_per_window'] * windows_per_year

    # Sort by average PnL (descending)
    agg_perf = agg_perf.sort_values('avg_pnl_per_window', ascending=False)

    report_path = output_dir / f"strategy_performance_by_stock_{timestamp}.csv"
    agg_perf.to_csv(report_path, index=False)
    logger.info(f"✓ Saved: {report_path}")

    # ===== 2. Best Strategy Per Stock =====
    best_per_stock = agg_perf.loc[agg_perf.groupby('symbol')['avg_pnl_per_window'].idxmax()]
    best_per_stock = best_per_stock.sort_values('avg_pnl_per_window', ascending=False)

    best_path = output_dir / f"best_strategy_per_stock_{timestamp}.csv"
    best_per_stock.to_csv(best_path, index=False)
    logger.info(f"✓ Saved: {best_path}")

    # ===== 3. Strategy Rankings (Across All Stocks) =====
    strategy_rankings = agg_perf.groupby('strategy').agg({
        'avg_pnl_per_window': 'mean',
        'total_pnl': 'sum',
        'avg_sharpe': 'mean',
        'avg_win_rate': 'mean',
        'total_trades': 'sum',
        'num_windows': 'sum',
    }).reset_index()

    strategy_rankings = strategy_rankings.sort_values('avg_pnl_per_window', ascending=False)
    strategy_rankings['annualized_avg_pnl'] = strategy_rankings['avg_pnl_per_window'] * windows_per_year

    ranking_path = output_dir / f"strategy_rankings_{timestamp}.csv"
    strategy_rankings.to_csv(ranking_path, index=False)
    logger.info(f"✓ Saved: {ranking_path}")

    # ===== 4. Top Stocks Overall (Best Avg Performance) =====
    stock_performance = agg_perf.groupby('symbol').agg({
        'avg_pnl_per_window': 'mean',
        'total_pnl': 'sum',
        'avg_sharpe': 'mean',
        'avg_win_rate': 'mean',
        'total_trades': 'sum',
    }).reset_index()

    stock_performance = stock_performance.sort_values('avg_pnl_per_window', ascending=False)
    stock_performance['annualized_pnl'] = stock_performance['avg_pnl_per_window'] * windows_per_year

    stocks_path = output_dir / f"top_stocks_by_pnl_{timestamp}.csv"
    stock_performance.to_csv(stocks_path, index=False)
    logger.info(f"✓ Saved: {stocks_path}")

    # ===== 5. Day-by-Day PnL Breakdown (if available) =====
    # This would require equity curve data from each window
    # For now, we'll use the window-level data as a proxy

    # Window-level details
    window_details = df_perf.copy()
    window_details = window_details.sort_values(['symbol', 'strategy', 'window_idx'])

    detail_path = output_dir / f"window_level_details_{timestamp}.csv"
    window_details.to_csv(detail_path, index=False)
    logger.info(f"✓ Saved: {detail_path}")

    # ===== 6. Trade-Level Details =====
    if not df_trades.empty:
        trades_path = output_dir / f"all_trades_{timestamp}.csv"
        df_trades.to_csv(trades_path, index=False)
        logger.info(f"✓ Saved: {trades_path}")

    # ===== 7. Summary Statistics =====
    _print_summary(agg_perf, strategy_rankings, stock_performance, timestamp, output_dir)


def _print_summary(
    agg_perf: pd.DataFrame,
    strategy_rankings: pd.DataFrame,
    stock_performance: pd.DataFrame,
    timestamp: str,
    output_dir: Path,
) -> None:
    """Print and save summary statistics"""

    print("\n" + "="*80)
    print("BACKTEST SUMMARY")
    print("="*80)
    print(f"Timestamp: {timestamp}")
    print(f"Total Symbols: {agg_perf['symbol'].nunique()}")
    print(f"Total Strategies: {agg_perf['strategy'].nunique()}")
    print(f"Total Strategy-Window Combinations: {agg_perf['num_windows'].sum():.0f}")
    print(f"Total Trades: {agg_perf['total_trades'].sum():.0f}")
    print()

    print("TOP 10 STOCK-STRATEGY COMBINATIONS (by avg PnL per window):")
    print("-" * 80)
    for i, row in agg_perf.head(10).iterrows():
        print(f"{row['symbol']:8s} / {row['strategy']:20s}: "
              f"${row['avg_pnl_per_window']:8.2f} (Sharpe: {row['avg_sharpe']:6.2f}, "
              f"Win Rate: {row['avg_win_rate']:5.1%})")
    print()

    print("STRATEGY RANKINGS (by avg PnL across all stocks):")
    print("-" * 80)
    for i, row in strategy_rankings.iterrows():
        print(f"{row['strategy']:20s}: ${row['avg_pnl_per_window']:8.2f} avg "
              f"(${row['total_pnl']:10.2f} total, Sharpe: {row['avg_sharpe']:6.2f})")
    print()

    print("TOP 10 STOCKS (by avg PnL across all strategies):")
    print("-" * 80)
    for i, row in stock_performance.head(10).iterrows():
        print(f"{row['symbol']:8s}: ${row['avg_pnl_per_window']:8.2f} avg "
              f"(Sharpe: {row['avg_sharpe']:6.2f}, Win Rate: {row['avg_win_rate']:5.1%})")
    print()

    print("BOTTOM 10 STOCKS (by avg PnL):")
    print("-" * 80)
    for i, row in stock_performance.tail(10).iterrows():
        print(f"{row['symbol']:8s}: ${row['avg_pnl_per_window']:8.2f} avg "
              f"(Sharpe: {row['avg_sharpe']:6.2f}, Win Rate: {row['avg_win_rate']:5.1%})")

    print("="*80)
    print(f"\nReports saved to: {output_dir}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive backtest on top stocks from Alpaca"
    )
    parser.add_argument("--num-stocks", type=int, default=200,
                       help="Number of top stocks to fetch and test")
    parser.add_argument("--symbols-file", type=str, default=None,
                       help="Use specific symbols from CSV file (column: 'symbol')")
    parser.add_argument("--data-dir", type=str, default="trainingdata/train",
                       help="Directory for historical data")
    parser.add_argument("--output-dir", type=str, default="strategytraining/reports",
                       help="Directory for output reports")
    parser.add_argument("--window-days", type=int, default=7,
                       help="Window size in days")
    parser.add_argument("--stride-days", type=int, default=7,
                       help="Stride between windows in days")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip data download step (use existing data)")

    args = parser.parse_args()

    # Step 1: Get stock symbols
    if args.symbols_file and Path(args.symbols_file).exists():
        logger.info(f"Loading symbols from {args.symbols_file}")
        df = pd.read_csv(args.symbols_file)
        symbols = df['symbol'].tolist()
    else:
        logger.info(f"Fetching top {args.num_stocks} stocks from Alpaca")
        symbols = fetch_top_stocks(
            limit=args.num_stocks,
            output_file=Path(args.output_dir) / "fetched_stocks.csv"
        )

    logger.info(f"Testing {len(symbols)} symbols: {symbols[:10]}...")

    # Step 2: Download data (if not skipped)
    if not args.skip_download:
        successful_symbols = download_stock_data(symbols, args.data_dir)
        logger.info(f"Proceeding with {len(successful_symbols)} symbols that have data")
    else:
        logger.info("Skipping download, using existing data")
        successful_symbols = symbols

    # Step 3: Run backtests and generate reports
    run_backtest_and_generate_reports(
        symbols=successful_symbols,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_days=args.window_days,
        stride_days=args.stride_days,
    )

    logger.info("✓ All done!")


if __name__ == "__main__":
    main()
