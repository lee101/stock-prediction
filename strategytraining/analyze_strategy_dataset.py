"""
Analyze Strategy PnL Dataset

Provides analysis of which (symbol, strategy) combinations work best over time.
Perfect for understanding position sizing opportunities.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List


class StrategyDatasetAnalyzer:
    """Analyze collected strategy PnL dataset"""

    def __init__(self, dataset_base_path: str):
        """
        Load dataset from base path (without suffix)

        Example: 'strategytraining/datasets/strategy_pnl_dataset_20250131_120000'
        """
        self.base_path = Path(dataset_base_path)

        # Load datasets
        self.performance_df = pd.read_parquet(f"{dataset_base_path}_strategy_performance.parquet")
        self.trades_df = pd.read_parquet(f"{dataset_base_path}_trades.parquet")

        with open(f"{dataset_base_path}_metadata.json", 'r') as f:
            self.metadata = json.load(f)

        print(f"Loaded: {self.metadata['dataset_name']}")
        print(f"  Symbols: {self.metadata['num_symbols']}")
        print(f"  Strategies: {self.metadata['num_strategies']}")
        print(f"  Records: {len(self.performance_df)}")

    def get_strategy_rankings(self) -> pd.DataFrame:
        """Rank strategies by overall performance"""

        rankings = []

        for strategy in self.performance_df['strategy'].unique():
            strat_data = self.performance_df[self.performance_df['strategy'] == strategy]

            rankings.append({
                'strategy': strategy,
                'avg_return': strat_data['total_return'].mean(),
                'median_return': strat_data['total_return'].median(),
                'avg_sharpe': strat_data['sharpe_ratio'].mean(),
                'win_rate': strat_data['win_rate'].mean(),
                'total_pnl': strat_data['total_pnl'].sum(),
                'avg_trades_per_window': strat_data['num_trades'].mean(),
                'num_windows': len(strat_data),
                'positive_windows_pct': (strat_data['total_pnl'] > 0).mean()
            })

        df = pd.DataFrame(rankings)
        return df.sort_values('avg_sharpe', ascending=False)

    def get_symbol_strategy_matrix(self) -> pd.DataFrame:
        """Create matrix showing best strategy per symbol"""

        matrix_data = []

        for symbol in self.performance_df['symbol'].unique():
            symbol_data = self.performance_df[self.performance_df['symbol'] == symbol]

            row = {'symbol': symbol}

            for strategy in self.performance_df['strategy'].unique():
                strat_data = symbol_data[symbol_data['strategy'] == strategy]

                if len(strat_data) > 0:
                    row[f"{strategy}_return"] = strat_data['total_return'].mean()
                    row[f"{strategy}_sharpe"] = strat_data['sharpe_ratio'].mean()
                    row[f"{strategy}_pnl"] = strat_data['total_pnl'].sum()
                else:
                    row[f"{strategy}_return"] = np.nan
                    row[f"{strategy}_sharpe"] = np.nan
                    row[f"{strategy}_pnl"] = np.nan

            # Find best strategy for this symbol
            sharpe_cols = [c for c in row.keys() if '_sharpe' in c]
            if sharpe_cols:
                sharpes = {c.replace('_sharpe', ''): row[c] for c in sharpe_cols}
                best_strat = max(sharpes, key=lambda k: sharpes[k] if not np.isnan(sharpes[k]) else -999)
                row['best_strategy'] = best_strat
                row['best_sharpe'] = sharpes[best_strat]
            else:
                row['best_strategy'] = None
                row['best_sharpe'] = np.nan

            matrix_data.append(row)

        return pd.DataFrame(matrix_data)

    def get_best_pairs(self, n: int = 20) -> pd.DataFrame:
        """Get top N (symbol, strategy) pairs by performance"""

        pair_performance = []

        for (symbol, strategy), group in self.performance_df.groupby(['symbol', 'strategy']):
            pair_performance.append({
                'symbol': symbol,
                'strategy': strategy,
                'avg_return': group['total_return'].mean(),
                'avg_sharpe': group['sharpe_ratio'].mean(),
                'total_pnl': group['total_pnl'].sum(),
                'win_rate': group['win_rate'].mean(),
                'num_windows': len(group),
                'consistency': (group['total_pnl'] > 0).mean()  # % of profitable windows
            })

        df = pd.DataFrame(pair_performance)
        return df.sort_values('avg_sharpe', ascending=False).head(n)

    def analyze_symbol(self, symbol: str) -> Dict:
        """Detailed analysis for specific symbol"""

        symbol_data = self.performance_df[self.performance_df['symbol'] == symbol]

        if len(symbol_data) == 0:
            return None

        analysis = {
            'symbol': symbol,
            'is_crypto': symbol_data['is_crypto'].iloc[0],
            'num_windows': symbol_data['window_num'].nunique(),
            'strategies': {}
        }

        for strategy in symbol_data['strategy'].unique():
            strat_data = symbol_data[symbol_data['strategy'] == strategy]

            analysis['strategies'][strategy] = {
                'avg_return': strat_data['total_return'].mean(),
                'median_return': strat_data['total_return'].median(),
                'std_return': strat_data['total_return'].std(),
                'avg_sharpe': strat_data['sharpe_ratio'].mean(),
                'win_rate': strat_data['win_rate'].mean(),
                'total_pnl': strat_data['total_pnl'].sum(),
                'best_window_return': strat_data['total_return'].max(),
                'worst_window_return': strat_data['total_return'].min(),
                'consistency': (strat_data['total_pnl'] > 0).mean()
            }

        # Find best strategy
        best_strategy = max(
            analysis['strategies'].items(),
            key=lambda x: x[1]['avg_sharpe']
        )
        analysis['best_strategy'] = best_strategy[0]
        analysis['best_strategy_sharpe'] = best_strategy[1]['avg_sharpe']

        return analysis

    def get_temporal_performance(self, symbol: str, strategy: str) -> pd.DataFrame:
        """Get time-series performance for a (symbol, strategy) pair"""

        data = self.performance_df[
            (self.performance_df['symbol'] == symbol) &
            (self.performance_df['strategy'] == strategy)
        ].sort_values('window_num')

        if len(data) == 0:
            return None

        return data[['window_num', 'start_time', 'end_time', 'total_return',
                     'sharpe_ratio', 'total_pnl', 'num_trades', 'win_rate']]

    def export_summary_report(self, output_path: str):
        """Export comprehensive analysis report"""

        report = {
            'metadata': self.metadata,
            'strategy_rankings': self.get_strategy_rankings().to_dict('records'),
            'top_pairs': self.get_best_pairs(n=50).to_dict('records'),
            'symbol_strategy_matrix': self.get_symbol_strategy_matrix().to_dict('records'),
            'per_symbol_analysis': {
                symbol: self.analyze_symbol(symbol)
                for symbol in self.performance_df['symbol'].unique()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Exported report to {output_path}")
        return report


def quick_analysis(dataset_path: str):
    """Quick analysis and print key insights"""

    analyzer = StrategyDatasetAnalyzer(dataset_path)

    print("\n" + "="*80)
    print("STRATEGY RANKINGS")
    print("="*80)

    rankings = analyzer.get_strategy_rankings()
    print(rankings.to_string(index=False))

    print("\n" + "="*80)
    print("TOP 10 (SYMBOL, STRATEGY) PAIRS")
    print("="*80)

    top_pairs = analyzer.get_best_pairs(n=10)
    print(top_pairs[['symbol', 'strategy', 'avg_return', 'avg_sharpe',
                     'win_rate', 'total_pnl']].to_string(index=False))

    print("\n" + "="*80)
    print("SYMBOL-STRATEGY MATRIX (Best Strategy per Symbol)")
    print("="*80)

    matrix = analyzer.get_symbol_strategy_matrix()
    print(matrix[['symbol', 'best_strategy', 'best_sharpe']].to_string(index=False))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze strategy PnL dataset")
    parser.add_argument('dataset_path', help='Path to dataset (without suffix)')
    parser.add_argument('--export', help='Export full report to JSON')
    parser.add_argument('--symbol', help='Analyze specific symbol')
    parser.add_argument('--strategy', help='Filter by strategy')

    args = parser.parse_args()

    # Quick analysis
    quick_analysis(args.dataset_path)

    # Symbol-specific analysis
    if args.symbol:
        analyzer = StrategyDatasetAnalyzer(args.dataset_path)
        analysis = analyzer.analyze_symbol(args.symbol)

        if analysis:
            print(f"\n" + "="*80)
            print(f"DETAILED ANALYSIS: {args.symbol}")
            print("="*80)
            print(json.dumps(analysis, indent=2, default=str))
        else:
            print(f"\nSymbol {args.symbol} not found")

    # Export if requested
    if args.export:
        analyzer = StrategyDatasetAnalyzer(args.dataset_path)
        analyzer.export_summary_report(args.export)


if __name__ == '__main__':
    main()
