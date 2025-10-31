"""
Position Sizing Dataset Analysis Utilities

Provides tools to analyze and visualize the collected position sizing dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json


class DatasetAnalyzer:
    """Analyze collected position sizing dataset"""

    def __init__(self, dataset_base_path: str):
        """
        Initialize analyzer with dataset base path (without suffix)

        Example: 'strategytraining/datasets/position_sizing_dataset_20250131_120000'
        """
        self.base_path = Path(dataset_base_path)

        # Load datasets
        self.trades_df = pd.read_parquet(f"{dataset_base_path}_trades.parquet")
        self.summaries_df = pd.read_parquet(f"{dataset_base_path}_summaries.parquet")
        self.positions_df = pd.read_parquet(f"{dataset_base_path}_positions.parquet")

        # Load metadata
        with open(f"{dataset_base_path}_metadata.json", 'r') as f:
            self.metadata = json.load(f)

        print(f"Loaded dataset: {self.metadata['dataset_name']}")
        print(f"  Trades: {len(self.trades_df)}")
        print(f"  Windows: {len(self.summaries_df)}")
        print(f"  Symbols: {len(self.metadata['symbols'])}")

    def get_summary_statistics(self) -> Dict:
        """Get overall dataset statistics"""

        stats = {
            'total_trades': len(self.trades_df),
            'total_windows': len(self.summaries_df),
            'unique_symbols': len(self.summaries_df['symbol'].unique()),
            'date_range': {
                'start': self.summaries_df['start_time'].min(),
                'end': self.summaries_df['end_time'].max()
            }
        }

        # Trade statistics
        if len(self.trades_df) > 0:
            stats['trade_stats'] = {
                'avg_pnl': self.trades_df['pnl'].mean(),
                'median_pnl': self.trades_df['pnl'].median(),
                'std_pnl': self.trades_df['pnl'].std(),
                'win_rate': (self.trades_df['pnl'] > 0).mean(),
                'avg_pnl_pct': self.trades_df['pnl_pct'].mean(),
                'avg_duration_bars': self.trades_df['duration_bars'].mean(),
                'total_pnl': self.trades_df['pnl'].sum()
            }

        # Window statistics
        if len(self.summaries_df) > 0:
            stats['window_stats'] = {
                'avg_return': self.summaries_df['total_return'].mean(),
                'median_return': self.summaries_df['total_return'].median(),
                'std_return': self.summaries_df['total_return'].std(),
                'avg_sharpe': self.summaries_df['sharpe_ratio'].mean(),
                'avg_max_drawdown': self.summaries_df['max_drawdown'].mean(),
                'avg_trades_per_window': self.summaries_df['num_trades'].mean()
            }

        # Symbol breakdown
        stats['symbol_breakdown'] = self._get_symbol_breakdown()

        return stats

    def _get_symbol_breakdown(self) -> Dict:
        """Get per-symbol statistics"""

        breakdown = {}

        for symbol in self.summaries_df['symbol'].unique():
            symbol_summaries = self.summaries_df[self.summaries_df['symbol'] == symbol]
            symbol_trades = self.trades_df[self.trades_df['symbol'] == symbol]

            breakdown[symbol] = {
                'num_windows': len(symbol_summaries),
                'num_trades': len(symbol_trades),
                'avg_return': symbol_summaries['total_return'].mean(),
                'avg_sharpe': symbol_summaries['sharpe_ratio'].mean(),
                'total_pnl': symbol_trades['pnl'].sum() if len(symbol_trades) > 0 else 0,
                'win_rate': (symbol_trades['pnl'] > 0).mean() if len(symbol_trades) > 0 else 0,
                'is_crypto': symbol_summaries['is_crypto'].iloc[0]
            }

        return breakdown

    def analyze_by_symbol(self, symbol: str) -> Dict:
        """Get detailed analysis for a specific symbol"""

        symbol_summaries = self.summaries_df[self.summaries_df['symbol'] == symbol]
        symbol_trades = self.trades_df[self.trades_df['symbol'] == symbol]

        if len(symbol_summaries) == 0:
            return None

        analysis = {
            'symbol': symbol,
            'is_crypto': symbol_summaries['is_crypto'].iloc[0],
            'num_windows': len(symbol_summaries),
            'num_trades': len(symbol_trades),
            'window_stats': {
                'returns': {
                    'mean': symbol_summaries['total_return'].mean(),
                    'median': symbol_summaries['total_return'].median(),
                    'std': symbol_summaries['total_return'].std(),
                    'min': symbol_summaries['total_return'].min(),
                    'max': symbol_summaries['total_return'].max()
                },
                'sharpe': {
                    'mean': symbol_summaries['sharpe_ratio'].mean(),
                    'median': symbol_summaries['sharpe_ratio'].median(),
                    'std': symbol_summaries['sharpe_ratio'].std()
                },
                'drawdown': {
                    'mean': symbol_summaries['max_drawdown'].mean(),
                    'worst': symbol_summaries['max_drawdown'].min()
                }
            }
        }

        if len(symbol_trades) > 0:
            analysis['trade_stats'] = {
                'total_pnl': symbol_trades['pnl'].sum(),
                'avg_pnl': symbol_trades['pnl'].mean(),
                'median_pnl': symbol_trades['pnl'].median(),
                'win_rate': (symbol_trades['pnl'] > 0).mean(),
                'avg_pnl_pct': symbol_trades['pnl_pct'].mean(),
                'profit_factor': (
                    symbol_trades[symbol_trades['pnl'] > 0]['pnl'].sum() /
                    abs(symbol_trades[symbol_trades['pnl'] < 0]['pnl'].sum())
                    if (symbol_trades['pnl'] < 0).any() else float('inf')
                ),
                'avg_winner': symbol_trades[symbol_trades['pnl'] > 0]['pnl'].mean()
                    if (symbol_trades['pnl'] > 0).any() else 0,
                'avg_loser': symbol_trades[symbol_trades['pnl'] < 0]['pnl'].mean()
                    if (symbol_trades['pnl'] < 0).any() else 0,
                'avg_duration': symbol_trades['duration_bars'].mean()
            }

        return analysis

    def compare_stocks_vs_crypto(self) -> Dict:
        """Compare performance between stocks and crypto"""

        stocks = self.summaries_df[self.summaries_df['is_crypto'] == False]
        crypto = self.summaries_df[self.summaries_df['is_crypto'] == True]

        comparison = {
            'stocks': {
                'num_symbols': stocks['symbol'].nunique(),
                'num_windows': len(stocks),
                'avg_return': stocks['total_return'].mean(),
                'avg_sharpe': stocks['sharpe_ratio'].mean(),
                'avg_drawdown': stocks['max_drawdown'].mean(),
                'win_rate': (stocks['total_return'] > 0).mean()
            },
            'crypto': {
                'num_symbols': crypto['symbol'].nunique(),
                'num_windows': len(crypto),
                'avg_return': crypto['total_return'].mean(),
                'avg_sharpe': crypto['sharpe_ratio'].mean(),
                'avg_drawdown': crypto['max_drawdown'].mean(),
                'win_rate': (crypto['total_return'] > 0).mean()
            }
        }

        return comparison

    def get_best_worst_windows(self, n: int = 10) -> Dict:
        """Get best and worst performing windows"""

        sorted_summaries = self.summaries_df.sort_values('total_return', ascending=False)

        return {
            'best': sorted_summaries.head(n)[
                ['symbol', 'window_num', 'total_return', 'sharpe_ratio',
                 'num_trades', 'start_time', 'end_time']
            ].to_dict('records'),
            'worst': sorted_summaries.tail(n)[
                ['symbol', 'window_num', 'total_return', 'sharpe_ratio',
                 'num_trades', 'start_time', 'end_time']
            ].to_dict('records')
        }

    def export_summary_report(self, output_path: str):
        """Export comprehensive summary report"""

        report = {
            'metadata': self.metadata,
            'summary_statistics': self.get_summary_statistics(),
            'stocks_vs_crypto': self.compare_stocks_vs_crypto(),
            'best_worst_windows': self.get_best_worst_windows(),
            'per_symbol_analysis': {
                symbol: self.analyze_by_symbol(symbol)
                for symbol in self.summaries_df['symbol'].unique()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Exported summary report to {output_path}")

        return report


def quick_analysis(dataset_path: str):
    """Quick analysis and print summary"""

    analyzer = DatasetAnalyzer(dataset_path)

    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)

    stats = analyzer.get_summary_statistics()

    print(f"\nOverall Statistics:")
    print(f"  Total Trades: {stats['total_trades']:,}")
    print(f"  Total Windows: {stats['total_windows']:,}")
    print(f"  Unique Symbols: {stats['unique_symbols']}")

    if 'trade_stats' in stats:
        print(f"\nTrade Statistics:")
        print(f"  Average PnL: ${stats['trade_stats']['avg_pnl']:.2f}")
        print(f"  Win Rate: {stats['trade_stats']['win_rate']:.2%}")
        print(f"  Avg PnL %: {stats['trade_stats']['avg_pnl_pct']:.2%}")
        print(f"  Avg Duration: {stats['trade_stats']['avg_duration_bars']:.1f} bars")
        print(f"  Total PnL: ${stats['trade_stats']['total_pnl']:,.2f}")

    if 'window_stats' in stats:
        print(f"\nWindow Statistics:")
        print(f"  Avg Return: {stats['window_stats']['avg_return']:.2%}")
        print(f"  Avg Sharpe: {stats['window_stats']['avg_sharpe']:.2f}")
        print(f"  Avg Max Drawdown: {stats['window_stats']['avg_max_drawdown']:.2%}")
        print(f"  Avg Trades/Window: {stats['window_stats']['avg_trades_per_window']:.1f}")

    print(f"\n" + "="*80)
    print("STOCKS vs CRYPTO COMPARISON")
    print("="*80)

    comparison = analyzer.compare_stocks_vs_crypto()

    print(f"\nStocks:")
    print(f"  Symbols: {comparison['stocks']['num_symbols']}")
    print(f"  Avg Return: {comparison['stocks']['avg_return']:.2%}")
    print(f"  Avg Sharpe: {comparison['stocks']['avg_sharpe']:.2f}")
    print(f"  Win Rate: {comparison['stocks']['win_rate']:.2%}")

    print(f"\nCrypto:")
    print(f"  Symbols: {comparison['crypto']['num_symbols']}")
    print(f"  Avg Return: {comparison['crypto']['avg_return']:.2%}")
    print(f"  Avg Sharpe: {comparison['crypto']['avg_sharpe']:.2f}")
    print(f"  Win Rate: {comparison['crypto']['win_rate']:.2%}")

    print(f"\n" + "="*80)
    print("TOP 5 SYMBOLS BY TOTAL PnL")
    print("="*80)

    symbol_breakdown = stats['symbol_breakdown']
    top_symbols = sorted(
        symbol_breakdown.items(),
        key=lambda x: x[1]['total_pnl'],
        reverse=True
    )[:5]

    for symbol, data in top_symbols:
        print(f"\n{symbol}:")
        print(f"  Total PnL: ${data['total_pnl']:,.2f}")
        print(f"  Win Rate: {data['win_rate']:.2%}")
        print(f"  Avg Return: {data['avg_return']:.2%}")
        print(f"  Num Trades: {data['num_trades']}")


def main():
    """Main entry point"""

    import argparse

    parser = argparse.ArgumentParser(description="Analyze position sizing dataset")
    parser.add_argument('dataset_path', help='Path to dataset (without suffix)')
    parser.add_argument('--export', help='Export full report to JSON file')
    parser.add_argument('--symbol', help='Analyze specific symbol')

    args = parser.parse_args()

    # Run quick analysis
    quick_analysis(args.dataset_path)

    # Export if requested
    if args.export:
        analyzer = DatasetAnalyzer(args.dataset_path)
        analyzer.export_summary_report(args.export)

    # Symbol-specific analysis
    if args.symbol:
        analyzer = DatasetAnalyzer(args.dataset_path)
        analysis = analyzer.analyze_by_symbol(args.symbol)

        if analysis:
            print(f"\n" + "="*80)
            print(f"DETAILED ANALYSIS: {args.symbol}")
            print("="*80)
            print(json.dumps(analysis, indent=2, default=str))
        else:
            print(f"\nSymbol {args.symbol} not found in dataset")


if __name__ == '__main__':
    main()
