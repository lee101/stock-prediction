#!/usr/bin/env python3
"""
Fast position sizing strategy testing using precomputed PnL data.

Uses strategytraining/ precomputed trades to quickly evaluate different
sizing strategies without re-running full market simulation.

Usage:
    python strategytraining/test_sizing_on_precomputed_pnl.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json
from datetime import datetime

from marketsimulator.sizing_strategies import (
    FixedFractionStrategy,
    KellyStrategy,
    VolatilityTargetStrategy,
    CorrelationAwareStrategy,
    VolatilityAdjustedStrategy,
    MarketContext,
)
from trainingdata.load_correlation_utils import load_correlation_matrix


@dataclass
class SizingStrategyResult:
    """Results for a sizing strategy tested on precomputed trades."""
    strategy_name: str
    total_pnl: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    num_trades: int
    avg_position_size: float
    volatility: float
    win_rate: float


class PrecomputedPnLSizingTester:
    """
    Fast sizing strategy tester using precomputed trade data.

    Takes trade-level PnL data and applies different sizing strategies
    to see how they would have performed.
    """

    def __init__(
        self,
        trades_df: pd.DataFrame,
        initial_capital: float = 100000,
        corr_data: Dict = None,
    ):
        """
        Args:
            trades_df: DataFrame with precomputed trades
            initial_capital: Starting capital
            corr_data: Correlation matrix data for advanced strategies
        """
        self.trades_df = trades_df.copy()
        self.initial_capital = initial_capital
        self.corr_data = corr_data

        # Sort by entry timestamp
        self.trades_df = self.trades_df.sort_values('entry_timestamp').reset_index(drop=True)

        # Add time index for chronological processing
        self.trades_df['time_idx'] = pd.to_datetime(self.trades_df['entry_timestamp'])

    def run_strategy(self, strategy, strategy_name: str) -> SizingStrategyResult:
        """
        Apply sizing strategy to precomputed trades.

        Key insight: The precomputed trades have a baseline position_size.
        We calculate what size the strategy would have used, then scale
        the PnL accordingly.

        Args:
            strategy: Sizing strategy instance
            strategy_name: Name for reporting

        Returns:
            SizingStrategyResult with performance metrics
        """
        capital = self.initial_capital
        capital_history = [capital]
        position_sizes = []
        scaled_pnls = []

        # Group trades by entry timestamp to handle concurrent positions
        for timestamp, trades_at_time in self.trades_df.groupby('time_idx'):

            # Current equity before these trades
            current_equity = capital

            # Calculate sizing for each concurrent trade
            for idx, trade in trades_at_time.iterrows():
                symbol = trade['symbol']
                is_crypto = trade['is_crypto']

                # Estimate the predicted return from the realized PnL
                # This is an approximation - in reality we'd use the forecast
                predicted_return = trade['pnl_pct']

                # Estimate volatility from historical data or use default
                if self.corr_data and symbol in self.corr_data.get('volatility_metrics', {}):
                    vol_metrics = self.corr_data['volatility_metrics'][symbol]
                    predicted_volatility = vol_metrics['annualized_volatility'] / np.sqrt(252)
                else:
                    # Default daily volatility estimate
                    predicted_volatility = 0.02

                # Build market context
                ctx = MarketContext(
                    symbol=symbol,
                    predicted_return=abs(predicted_return),  # Use abs since we want position size
                    predicted_volatility=predicted_volatility,
                    current_price=trade['entry_price'],
                    equity=current_equity,
                    is_crypto=is_crypto,
                    existing_position_value=0,
                )

                # Calculate sizing
                try:
                    sizing = strategy.calculate_size(ctx)
                    position_fraction = sizing.position_fraction
                except Exception as e:
                    # Fallback on error
                    position_fraction = 0.5 / len(trades_at_time)

                # Baseline position size from precomputed data
                baseline_position_size = trade['position_size']

                # Calculate how many shares we would have traded
                # Baseline uses some fixed allocation, we scale based on our strategy
                baseline_fraction = baseline_position_size * trade['entry_price'] / self.initial_capital

                if baseline_fraction > 0:
                    size_multiplier = position_fraction / baseline_fraction
                else:
                    size_multiplier = 1.0

                # Cap multiplier to reasonable range
                size_multiplier = np.clip(size_multiplier, 0.1, 10.0)

                # Scale the PnL by the size multiplier
                scaled_pnl = trade['pnl'] * size_multiplier

                # Update capital
                capital += scaled_pnl

                scaled_pnls.append(scaled_pnl)
                position_sizes.append(position_fraction)

            capital_history.append(capital)

        # Calculate metrics
        total_pnl = capital - self.initial_capital
        total_return_pct = total_pnl / self.initial_capital

        # Sharpe ratio
        if len(capital_history) > 1:
            returns = np.diff(capital_history) / capital_history[:-1]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
            volatility = np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0
            volatility = 0.0

        # Max drawdown
        peak = self.initial_capital
        max_dd = 0.0
        for eq in capital_history:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd

        # Win rate
        wins = sum(1 for pnl in scaled_pnls if pnl > 0)
        win_rate = wins / len(scaled_pnls) if scaled_pnls else 0.0

        avg_position_size = np.mean(position_sizes) if position_sizes else 0.0

        return SizingStrategyResult(
            strategy_name=strategy_name,
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            num_trades=len(scaled_pnls),
            avg_position_size=avg_position_size,
            volatility=volatility,
            win_rate=win_rate,
        )


def main():
    print("=" * 80)
    print("FAST POSITION SIZING TESTING ON PRECOMPUTED PnL DATA")
    print("=" * 80)
    print()

    # Load latest full dataset
    dataset_path = Path("strategytraining/datasets")
    latest_files = sorted(dataset_path.glob("full_strategy_dataset_*_trades.parquet"))

    if not latest_files:
        print("ERROR: No precomputed trade data found!")
        print("Run: python strategytraining/collect_strategy_pnl_dataset.py")
        return

    trades_file = latest_files[-1]
    print(f"Loading: {trades_file.name}")

    trades_df = pd.read_parquet(trades_file)
    print(f"✓ Loaded {len(trades_df):,} trades")

    # Filter to test symbols
    test_symbols = ['BTCUSD', 'ETHUSD', 'AAPL', 'MSFT', 'NVDA', 'SPY']

    # Map symbol names (dataset uses BTC-USD, we use BTCUSD)
    symbol_mapping = {
        'BTC-USD': 'BTCUSD',
        'ETH-USD': 'ETHUSD',
    }

    trades_df['symbol'] = trades_df['symbol'].replace(symbol_mapping)

    # Load performance data to filter to profitable windows
    perf_file = str(trades_file).replace('_trades.parquet', '_strategy_performance.parquet')
    perf_df = pd.read_parquet(perf_file)
    perf_df['symbol'] = perf_df['symbol'].replace(symbol_mapping)

    # Filter to test symbols
    available_symbols = [s for s in test_symbols if s in trades_df['symbol'].values]
    print(f"Available test symbols: {', '.join(available_symbols)}")

    trades_df = trades_df[trades_df['symbol'].isin(available_symbols)].copy()
    perf_df = perf_df[perf_df['symbol'].isin(available_symbols)].copy()

    print(f"Total trades on test symbols: {len(trades_df):,}")

    # Filter to profitable strategy windows only
    profitable_windows = perf_df[perf_df['total_return'] > 0][['symbol', 'strategy', 'window_num']]
    print(f"Profitable strategy windows: {len(profitable_windows)} / {len(perf_df)} ({100*len(profitable_windows)/len(perf_df):.1f}%)")

    # Merge to keep only trades from profitable windows
    trades_df = trades_df.merge(
        profitable_windows,
        on=['symbol', 'strategy', 'window_num'],
        how='inner'
    )

    print(f"✓ Filtered to {len(trades_df):,} trades from profitable windows")
    print()

    # Load correlation data
    print("Loading correlation and volatility data...")
    try:
        corr_data = load_correlation_matrix()
        print(f"✓ Loaded correlation matrix")
    except Exception as e:
        print(f"⚠️  Could not load correlation data: {e}")
        corr_data = None
    print()

    # Initialize tester
    tester = PrecomputedPnLSizingTester(trades_df, corr_data=corr_data)

    # Define strategies to test
    strategies = [
        # Baseline
        (FixedFractionStrategy(0.5), "Naive_50pct_Baseline"),

        # Fixed allocations
        (FixedFractionStrategy(0.25), "Fixed_25pct"),
        (FixedFractionStrategy(0.75), "Fixed_75pct"),
        (FixedFractionStrategy(1.0), "Fixed_100pct"),

        # Kelly variants
        (KellyStrategy(fraction=0.25, cap=1.0), "Kelly_25pct"),
        (KellyStrategy(fraction=0.5, cap=1.0), "Kelly_50pct"),
        (KellyStrategy(fraction=0.75, cap=1.0), "Kelly_75pct"),
        (KellyStrategy(fraction=1.0, cap=1.0), "Kelly_100pct"),

        # Volatility-based
        (VolatilityTargetStrategy(target_vol=0.10), "VolTarget_10pct"),
        (VolatilityTargetStrategy(target_vol=0.15), "VolTarget_15pct"),
        (VolatilityTargetStrategy(target_vol=0.20), "VolTarget_20pct"),

        # Volatility-adjusted
        (VolatilityAdjustedStrategy(corr_data=corr_data, target_vol_contribution=0.10), "VolAdjusted_10pct"),
        (VolatilityAdjustedStrategy(corr_data=corr_data, target_vol_contribution=0.15), "VolAdjusted_15pct"),

        # Correlation-aware
        (CorrelationAwareStrategy(corr_data=corr_data, uncertainty_penalty=2.0, fractional_kelly=0.25), "CorrAware_Conservative"),
        (CorrelationAwareStrategy(corr_data=corr_data, uncertainty_penalty=1.0, fractional_kelly=0.5), "CorrAware_Moderate"),
        (CorrelationAwareStrategy(corr_data=corr_data, uncertainty_penalty=0.5, fractional_kelly=0.75), "CorrAware_Aggressive"),
    ]

    # Run all strategies
    print("Running sizing strategies on precomputed trades...")
    print()

    results = []
    for strategy, name in strategies:
        result = tester.run_strategy(strategy, name)
        results.append(result)
        print(f"  ✓ Completed: {name}")

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    # Convert to DataFrame
    df_data = []
    for r in results:
        df_data.append({
            'Strategy': r.strategy_name,
            'Total PnL': f"${r.total_pnl:,.0f}",
            'Return': f"{r.total_return_pct:.2%}",
            'Sharpe': f"{r.sharpe_ratio:.2f}",
            'Max DD': f"{r.max_drawdown_pct:.2%}",
            'Volatility': f"{r.volatility:.2%}",
            'Win Rate': f"{r.win_rate:.2%}",
            'Avg Size': f"{r.avg_position_size:.2%}",
            'Trades': r.num_trades,
        })

    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))
    print()

    # Rank by Sharpe
    print("Top 5 by Sharpe Ratio:")
    print("-" * 80)
    sorted_results = sorted(results, key=lambda x: x.sharpe_ratio, reverse=True)
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"  {i}. {r.strategy_name:30s} Sharpe: {r.sharpe_ratio:6.2f}  "
              f"Return: {r.total_return_pct:7.2%}  DD: {r.max_drawdown_pct:6.2%}")
    print()

    # Rank by return
    print("Top 5 by Total Return:")
    print("-" * 80)
    sorted_results = sorted(results, key=lambda x: x.total_return_pct, reverse=True)
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"  {i}. {r.strategy_name:30s} Return: {r.total_return_pct:7.2%}  "
              f"Sharpe: {r.sharpe_ratio:6.2f}  DD: {r.max_drawdown_pct:6.2%}")
    print()

    # Save results
    output_file = Path("strategytraining/sizing_strategy_fast_test_results.json")
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'dataset': str(trades_file.name),
        'num_trades': len(trades_df),
        'symbols': available_symbols,
        'results': [
            {
                'strategy': r.strategy_name,
                'total_pnl': r.total_pnl,
                'return_pct': r.total_return_pct,
                'sharpe': r.sharpe_ratio,
                'max_dd_pct': r.max_drawdown_pct,
                'volatility': r.volatility,
                'win_rate': r.win_rate,
                'avg_size': r.avg_position_size,
                'num_trades': r.num_trades,
            }
            for r in results
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    # Compare to baseline
    baseline = next((r for r in results if 'Baseline' in r.strategy_name), None)
    if baseline:
        print("=" * 80)
        print("COMPARISON TO BASELINE")
        print("=" * 80)
        print(f"Baseline (Naive 50%): Return {baseline.total_return_pct:.2%}, Sharpe {baseline.sharpe_ratio:.2f}")
        print()

        # Show top improvements
        improvements = []
        for r in results:
            if r.strategy_name != baseline.strategy_name:
                return_delta = r.total_return_pct - baseline.total_return_pct
                sharpe_delta = r.sharpe_ratio - baseline.sharpe_ratio
                improvements.append((r, return_delta, sharpe_delta))

        # Sort by return improvement
        improvements.sort(key=lambda x: x[1], reverse=True)

        print("Top improvements vs baseline:")
        for i, (r, ret_delta, sharpe_delta) in enumerate(improvements[:5], 1):
            print(f"  {i}. {r.strategy_name:30s} "
                  f"Return: {r.total_return_pct:7.2%} ({ret_delta:+.2%})  "
                  f"Sharpe: {r.sharpe_ratio:6.2f} ({sharpe_delta:+.2f})")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
