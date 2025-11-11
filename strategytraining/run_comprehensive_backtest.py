"""
Comprehensive Strategy Backtesting Across Top Stocks

This script runs all strategies across the top N stocks from Alpaca,
collecting detailed PnL data and generating comprehensive reports.

Reports generated:
- Strategy performance by stock (CSV)
- Day-by-day PnL breakdown (CSV)
- Summary statistics (CSV)
- Trade-level details (Parquet)
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from fetch_top_stocks import fetch_top_stocks
from collect_strategy_pnl_dataset import StrategyPnLCollector, STRATEGIES
from marketsimulator.environment import activate_simulation
from marketsimulator.backtest_test3_inline import backtest_forecasts


class ComprehensiveBacktester:
    """Run comprehensive backtests across multiple stocks and strategies"""

    def __init__(
        self,
        symbols: List[str],
        data_dir: str = "trainingdata/train",
        output_dir: str = "strategytraining/reports",
        window_days: int = 7,
        stride_days: int = 7,
    ):
        self.symbols = symbols
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.window_days = window_days
        self.stride_days = stride_days

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Use the collector for the heavy lifting
        self.collector = StrategyPnLCollector(
            data_dir=str(data_dir),
            output_dir=str(output_dir),
            window_days=window_days,
            stride_days=stride_days,
        )

        # Storage for results
        self.all_results = []
        self.daily_pnl = []

    def run_backtest(self) -> Dict:
        """Run backtest on all symbols"""
        logger.info(f"Starting comprehensive backtest on {len(self.symbols)} symbols")
        logger.info(f"Strategies: {STRATEGIES}")

        activate_simulation()

        # Process each symbol
        for symbol in tqdm(self.symbols, desc="Backtesting stocks"):
            try:
                result = self._process_symbol(symbol)
                if result:
                    self.all_results.extend(result['strategy_performance'])
                    if 'daily_pnl' in result:
                        self.daily_pnl.extend(result['daily_pnl'])
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

        logger.info(f"Backtest complete. Processed {len(self.all_results)} strategy-window combinations")

        # Generate reports
        self._generate_reports()

        return {
            'total_results': len(self.all_results),
            'symbols_processed': len(set(r['symbol'] for r in self.all_results)),
            'strategies': len(STRATEGIES),
        }

    def _process_symbol(self, symbol: str) -> Optional[Dict]:
        """Process a single symbol across all strategies"""
        # Load data
        df = self.collector.load_symbol_data(symbol)
        if df is None or len(df) < self.collector.min_data_points:
            logger.debug(f"Skipping {symbol} - insufficient data")
            return None

        # Get time windows
        windows = self.collector.create_time_windows(df)
        if not windows:
            logger.debug(f"Skipping {symbol} - no valid windows")
            return None

        logger.debug(f"Processing {symbol}: {len(windows)} windows")

        strategy_performance = []
        daily_pnl = []

        # Process each window
        for window_idx, (start_date, end_date) in enumerate(windows):
            window_df = df[
                (df['timestamp'] >= start_date) &
                (df['timestamp'] <= end_date)
            ].copy()

            if len(window_df) < 5:
                continue

            # Run backtest with all strategies
            try:
                results = backtest_forecasts(
                    window_df,
                    symbol=symbol,
                    initial_cash=10000.0,
                )

                # Extract metrics for each strategy
                for strategy_name in STRATEGIES:
                    if strategy_name not in results:
                        continue

                    strategy_result = results[strategy_name]

                    # Window-level metrics
                    strategy_performance.append({
                        'symbol': symbol,
                        'strategy': strategy_name,
                        'window_idx': window_idx,
                        'start_date': start_date,
                        'end_date': end_date,
                        'total_pnl': strategy_result.get('total_pnl', 0),
                        'total_return': strategy_result.get('total_return', 0),
                        'sharpe_ratio': strategy_result.get('sharpe_ratio', 0),
                        'win_rate': strategy_result.get('win_rate', 0),
                        'total_trades': strategy_result.get('total_trades', 0),
                        'max_drawdown': strategy_result.get('max_drawdown', 0),
                        'avg_trade_pnl': strategy_result.get('avg_trade_pnl', 0),
                    })

                    # Daily PnL if available
                    if 'equity_curve' in strategy_result:
                        equity_curve = strategy_result['equity_curve']
                        for i in range(len(equity_curve)):
                            daily_pnl.append({
                                'symbol': symbol,
                                'strategy': strategy_name,
                                'window_idx': window_idx,
                                'day_idx': i,
                                'date': window_df.iloc[i]['timestamp'] if i < len(window_df) else None,
                                'equity': equity_curve[i],
                                'pnl': equity_curve[i] - (equity_curve[i-1] if i > 0 else 10000.0),
                            })

            except Exception as e:
                logger.error(f"Error backtesting {symbol} window {window_idx}: {e}")
                continue

        return {
            'strategy_performance': strategy_performance,
            'daily_pnl': daily_pnl,
        }

    def _generate_reports(self):
        """Generate comprehensive CSV reports"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Strategy performance by stock - sorted by PnL
        if self.all_results:
            df_performance = pd.DataFrame(self.all_results)

            # Aggregate by symbol and strategy (across all windows)
            df_agg = df_performance.groupby(['symbol', 'strategy']).agg({
                'total_pnl': ['mean', 'sum', 'std'],
                'total_return': 'mean',
                'sharpe_ratio': 'mean',
                'win_rate': 'mean',
                'total_trades': 'sum',
                'max_drawdown': 'mean',
            }).reset_index()

            df_agg.columns = ['symbol', 'strategy', 'avg_pnl', 'total_pnl', 'pnl_std',
                            'avg_return', 'avg_sharpe', 'avg_win_rate', 'total_trades',
                            'avg_max_drawdown']

            # Sort by average PnL
            df_agg = df_agg.sort_values('avg_pnl', ascending=False)

            # Calculate annualized metrics
            windows_per_year = 365 / self.stride_days
            df_agg['annualized_pnl'] = df_agg['avg_pnl'] * windows_per_year

            # Save main report
            report_path = self.output_dir / f"strategy_performance_{timestamp}.csv"
            df_agg.to_csv(report_path, index=False)
            logger.info(f"Saved strategy performance report to {report_path}")

            # 2. Best strategies per stock
            best_strategies = df_agg.loc[df_agg.groupby('symbol')['avg_pnl'].idxmax()]
            best_path = self.output_dir / f"best_strategies_per_stock_{timestamp}.csv"
            best_strategies.to_csv(best_path, index=False)
            logger.info(f"Saved best strategies report to {best_path}")

            # 3. Strategy rankings (aggregated across all stocks)
            strategy_rankings = df_agg.groupby('strategy').agg({
                'avg_pnl': 'mean',
                'total_pnl': 'sum',
                'avg_sharpe': 'mean',
                'avg_win_rate': 'mean',
                'total_trades': 'sum',
            }).sort_values('avg_pnl', ascending=False).reset_index()

            ranking_path = self.output_dir / f"strategy_rankings_{timestamp}.csv"
            strategy_rankings.to_csv(ranking_path, index=False)
            logger.info(f"Saved strategy rankings to {ranking_path}")

            # 4. Save full window-level details
            detail_path = self.output_dir / f"window_details_{timestamp}.csv"
            df_performance.to_csv(detail_path, index=False)
            logger.info(f"Saved window-level details to {detail_path}")

        # 5. Daily PnL breakdown
        if self.daily_pnl:
            df_daily = pd.DataFrame(self.daily_pnl)
            daily_path = self.output_dir / f"daily_pnl_{timestamp}.csv"
            df_daily.to_csv(daily_path, index=False)
            logger.info(f"Saved daily PnL breakdown to {daily_path}")

        # 6. Summary statistics
        self._generate_summary(timestamp)

    def _generate_summary(self, timestamp: str):
        """Generate summary statistics report"""
        if not self.all_results:
            return

        df = pd.DataFrame(self.all_results)

        summary = {
            'run_timestamp': timestamp,
            'total_symbols': len(df['symbol'].unique()),
            'total_strategies': len(df['strategy'].unique()),
            'total_windows': len(df),
            'date_range': {
                'start': df['start_date'].min().isoformat() if not df.empty else None,
                'end': df['end_date'].max().isoformat() if not df.empty else None,
            },
            'overall_metrics': {
                'avg_pnl_per_window': float(df['total_pnl'].mean()),
                'total_pnl': float(df['total_pnl'].sum()),
                'avg_sharpe': float(df['sharpe_ratio'].mean()),
                'avg_win_rate': float(df['win_rate'].mean()),
                'total_trades': int(df['total_trades'].sum()),
            },
            'best_symbol_strategy': df.nlargest(10, 'total_pnl')[
                ['symbol', 'strategy', 'total_pnl', 'sharpe_ratio', 'win_rate']
            ].to_dict('records'),
            'worst_symbol_strategy': df.nsmallest(10, 'total_pnl')[
                ['symbol', 'strategy', 'total_pnl', 'sharpe_ratio', 'win_rate']
            ].to_dict('records'),
        }

        summary_path = self.output_dir / f"summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Saved summary to {summary_path}")

        # Print summary to console
        print("\n" + "="*80)
        print("BACKTEST SUMMARY")
        print("="*80)
        print(f"Symbols: {summary['total_symbols']}")
        print(f"Strategies: {summary['total_strategies']}")
        print(f"Total windows: {summary['total_windows']}")
        print(f"Avg PnL per window: ${summary['overall_metrics']['avg_pnl_per_window']:.2f}")
        print(f"Total PnL: ${summary['overall_metrics']['total_pnl']:.2f}")
        print(f"Avg Sharpe: {summary['overall_metrics']['avg_sharpe']:.3f}")
        print(f"Avg Win Rate: {summary['overall_metrics']['avg_win_rate']:.1%}")
        print(f"Total Trades: {summary['overall_metrics']['total_trades']}")
        print("\nTop 5 Symbol-Strategy Combinations:")
        for i, combo in enumerate(summary['best_symbol_strategy'][:5], 1):
            print(f"  {i}. {combo['symbol']} / {combo['strategy']}: ${combo['total_pnl']:.2f} "
                  f"(Sharpe: {combo['sharpe_ratio']:.2f}, Win Rate: {combo['win_rate']:.1%})")
        print("="*80 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run comprehensive backtest across top stocks"
    )
    parser.add_argument("--stocks", type=int, default=200,
                       help="Number of top stocks to test")
    parser.add_argument("--data-dir", type=str, default="trainingdata/train",
                       help="Directory containing training data")
    parser.add_argument("--output-dir", type=str, default="strategytraining/reports",
                       help="Directory for output reports")
    parser.add_argument("--window-days", type=int, default=7,
                       help="Window size in days")
    parser.add_argument("--stride-days", type=int, default=7,
                       help="Stride between windows in days")
    parser.add_argument("--symbols-file", type=str, default=None,
                       help="Use specific symbols from CSV file instead of fetching")

    args = parser.parse_args()

    # Get stock symbols
    if args.symbols_file and Path(args.symbols_file).exists():
        logger.info(f"Loading symbols from {args.symbols_file}")
        df = pd.read_csv(args.symbols_file)
        symbols = df['symbol'].tolist()
    else:
        logger.info(f"Fetching top {args.stocks} stocks from Alpaca")
        symbols = fetch_top_stocks(limit=args.stocks)

    logger.info(f"Testing {len(symbols)} symbols")

    # Run backtest
    backtester = ComprehensiveBacktester(
        symbols=symbols,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_days=args.window_days,
        stride_days=args.stride_days,
    )

    results = backtester.run_backtest()

    logger.info("Backtest complete!")
    logger.info(f"Results: {results}")


if __name__ == "__main__":
    main()
