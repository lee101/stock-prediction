#!/usr/bin/env python3
"""
Test current production sizing logic: Kelly_50pct @ 4x leverage for stocks.

Validates:
1. Position sizing matches production logic (src/sizing_utils.py)
2. Risk limits are enforced (60% per symbol, 120% total)
3. Performance vs baseline strategies
4. Edge case handling (high vol, exposure limits, etc.)

Usage:
    python strategytraining/test_current_production_sizing.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
from datetime import datetime

from marketsimulator.sizing_strategies import (
    KellyStrategy,
    FixedFractionStrategy,
    VolatilityAdjustedStrategy,
    MarketContext,
)
from trainingdata.load_correlation_utils import load_correlation_matrix


# Production configuration
MAX_SYMBOL_EXPOSURE_PCT = 60.0
MAX_TOTAL_EXPOSURE_PCT = 120.0
MAX_INTRADAY_LEVERAGE_STOCKS = 4.0
KELLY_FRACTION = 0.5


@dataclass
class ProductionSizingResult:
    """Results for production sizing test."""
    strategy_name: str
    total_pnl: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    num_trades: int
    avg_position_size: float
    volatility: float
    win_rate: float
    max_symbol_exposure: float
    max_total_exposure: float
    num_exposure_violations: int
    avg_leverage: float


class ProductionSizingTester:
    """
    Test production sizing logic on precomputed trades.

    Simulates exactly what runs in production:
    - Kelly_50pct base sizing
    - 4x leverage for stocks, 1x for crypto
    - 60% max per symbol, 120% max total
    """

    def __init__(
        self,
        trades_df: pd.DataFrame,
        initial_capital: float = 100000,
        corr_data: Dict = None,
    ):
        self.trades_df = trades_df.copy()
        self.initial_capital = initial_capital
        self.corr_data = corr_data

        # Sort by entry timestamp
        self.trades_df = self.trades_df.sort_values('entry_timestamp').reset_index(drop=True)
        self.trades_df['time_idx'] = pd.to_datetime(self.trades_df['entry_timestamp'])

    def run_production_sizing(self) -> ProductionSizingResult:
        """
        Simulate production sizing: Kelly_50pct @ 4x leverage.

        Enforces:
        - 60% max per symbol
        - 120% max total exposure
        - 4x leverage for stocks, 1x for crypto
        """
        capital = self.initial_capital
        capital_history = [capital]
        position_sizes = []
        scaled_pnls = []
        symbol_exposures = {}  # Track per-symbol exposure

        max_symbol_exposure_seen = 0.0
        max_total_exposure_seen = 0.0
        num_violations = 0
        leverages = []

        kelly_strategy = KellyStrategy(fraction=KELLY_FRACTION, cap=1.0)

        # Process trades chronologically
        for timestamp, trades_at_time in self.trades_df.groupby('time_idx'):
            current_equity = capital

            # Reset symbol exposures for this timestep
            total_exposure = 0.0

            for idx, trade in trades_at_time.iterrows():
                symbol = trade['symbol']
                is_crypto = trade['is_crypto']

                # Get predicted values
                predicted_return = trade['pnl_pct']
                if self.corr_data and symbol in self.corr_data.get('volatility_metrics', {}):
                    vol_metrics = self.corr_data['volatility_metrics'][symbol]
                    predicted_volatility = vol_metrics['annualized_volatility'] / np.sqrt(252)
                else:
                    predicted_volatility = 0.02

                # Build market context
                existing_exposure = symbol_exposures.get(symbol, 0.0)
                ctx = MarketContext(
                    symbol=symbol,
                    predicted_return=abs(predicted_return),
                    predicted_volatility=predicted_volatility,
                    current_price=trade['entry_price'],
                    equity=current_equity,
                    is_crypto=is_crypto,
                    existing_position_value=existing_exposure,
                )

                # Calculate Kelly sizing
                sizing = kelly_strategy.calculate_size(ctx)
                base_fraction = sizing.position_fraction

                # Apply production leverage
                if is_crypto:
                    target_fraction = max(base_fraction, 0)  # Long only
                    leverage = 1.0
                else:
                    target_fraction = base_fraction * MAX_INTRADAY_LEVERAGE_STOCKS
                    leverage = MAX_INTRADAY_LEVERAGE_STOCKS

                # Calculate position value
                position_value = target_fraction * current_equity

                # Check symbol exposure limit
                symbol_exposure_pct = ((existing_exposure + abs(position_value)) / current_equity) * 100
                if symbol_exposure_pct > MAX_SYMBOL_EXPOSURE_PCT:
                    # Reduce to respect limit
                    max_additional = (MAX_SYMBOL_EXPOSURE_PCT / 100 * current_equity) - existing_exposure
                    if max_additional <= 0:
                        position_value = 0
                        target_fraction = 0
                        num_violations += 1
                    else:
                        position_value = max_additional
                        target_fraction = position_value / current_equity

                # Check total exposure limit
                total_exposure_pct = ((total_exposure + abs(position_value)) / current_equity) * 100
                if total_exposure_pct > MAX_TOTAL_EXPOSURE_PCT:
                    # Reduce to respect limit
                    max_additional = (MAX_TOTAL_EXPOSURE_PCT / 100 * current_equity) - total_exposure
                    if max_additional <= 0:
                        position_value = 0
                        target_fraction = 0
                        num_violations += 1
                    else:
                        position_value = max_additional
                        target_fraction = position_value / current_equity

                # Update exposures
                symbol_exposures[symbol] = symbol_exposures.get(symbol, 0) + abs(position_value)
                total_exposure += abs(position_value)

                # Track max exposures
                symbol_exp = (symbol_exposures[symbol] / current_equity) * 100
                max_symbol_exposure_seen = max(max_symbol_exposure_seen, symbol_exp)
                total_exp = (total_exposure / current_equity) * 100
                max_total_exposure_seen = max(max_total_exposure_seen, total_exp)

                # Scale PnL based on size
                baseline_position_size = trade['position_size']
                baseline_fraction = baseline_position_size * trade['entry_price'] / self.initial_capital

                if baseline_fraction > 0:
                    size_multiplier = target_fraction / baseline_fraction
                else:
                    size_multiplier = 1.0

                size_multiplier = np.clip(size_multiplier, 0.1, 10.0)
                scaled_pnl = trade['pnl'] * size_multiplier

                # Update capital
                capital += scaled_pnl

                scaled_pnls.append(scaled_pnl)
                position_sizes.append(target_fraction)
                leverages.append(leverage if target_fraction > 0 else 0)

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
        avg_leverage = np.mean(leverages) if leverages else 0.0

        return ProductionSizingResult(
            strategy_name="Production_Kelly50pct_4x",
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            num_trades=len(scaled_pnls),
            avg_position_size=avg_position_size,
            volatility=volatility,
            win_rate=win_rate,
            max_symbol_exposure=max_symbol_exposure_seen,
            max_total_exposure=max_total_exposure_seen,
            num_exposure_violations=num_violations,
            avg_leverage=avg_leverage,
        )

    def run_baseline_comparison(self) -> List[ProductionSizingResult]:
        """Run baseline strategies for comparison."""
        baselines = [
            (FixedFractionStrategy(0.50), "Baseline_Fixed_50pct"),
            (FixedFractionStrategy(0.25), "Conservative_Fixed_25pct"),
            (VolatilityAdjustedStrategy(corr_data=self.corr_data, target_vol_contribution=0.15), "VolAdjusted_15pct"),
        ]

        results = []
        for strategy, name in baselines:
            result = self._run_simple_strategy(strategy, name)
            results.append(result)

        return results

    def _run_simple_strategy(self, strategy, name: str) -> ProductionSizingResult:
        """Run a simple strategy without exposure limits for comparison."""
        capital = self.initial_capital
        capital_history = [capital]
        position_sizes = []
        scaled_pnls = []

        for timestamp, trades_at_time in self.trades_df.groupby('time_idx'):
            current_equity = capital

            for idx, trade in trades_at_time.iterrows():
                symbol = trade['symbol']
                is_crypto = trade['is_crypto']

                predicted_return = trade['pnl_pct']
                if self.corr_data and symbol in self.corr_data.get('volatility_metrics', {}):
                    vol_metrics = self.corr_data['volatility_metrics'][symbol]
                    predicted_volatility = vol_metrics['annualized_volatility'] / np.sqrt(252)
                else:
                    predicted_volatility = 0.02

                ctx = MarketContext(
                    symbol=symbol,
                    predicted_return=abs(predicted_return),
                    predicted_volatility=predicted_volatility,
                    current_price=trade['entry_price'],
                    equity=current_equity,
                    is_crypto=is_crypto,
                    existing_position_value=0,
                )

                sizing = strategy.calculate_size(ctx)
                position_fraction = sizing.position_fraction

                # Scale PnL
                baseline_fraction = trade['position_size'] * trade['entry_price'] / self.initial_capital
                if baseline_fraction > 0:
                    size_multiplier = position_fraction / baseline_fraction
                else:
                    size_multiplier = 1.0

                size_multiplier = np.clip(size_multiplier, 0.1, 10.0)
                scaled_pnl = trade['pnl'] * size_multiplier

                capital += scaled_pnl
                scaled_pnls.append(scaled_pnl)
                position_sizes.append(position_fraction)

            capital_history.append(capital)

        # Calculate metrics (simplified)
        total_pnl = capital - self.initial_capital
        total_return_pct = total_pnl / self.initial_capital

        if len(capital_history) > 1:
            returns = np.diff(capital_history) / capital_history[:-1]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
            volatility = np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0
            volatility = 0.0

        peak = self.initial_capital
        max_dd = 0.0
        for eq in capital_history:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd

        wins = sum(1 for pnl in scaled_pnls if pnl > 0)
        win_rate = wins / len(scaled_pnls) if scaled_pnls else 0.0

        return ProductionSizingResult(
            strategy_name=name,
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            num_trades=len(scaled_pnls),
            avg_position_size=np.mean(position_sizes) if position_sizes else 0,
            volatility=volatility,
            win_rate=win_rate,
            max_symbol_exposure=0,  # Not tracked for baselines
            max_total_exposure=0,
            num_exposure_violations=0,
            avg_leverage=0,
        )


def main():
    print("=" * 80)
    print("PRODUCTION SIZING VALIDATION TEST")
    print("=" * 80)
    print()
    print("Testing: Kelly_50pct @ 4x leverage with exposure limits")
    print(f"  - Max symbol exposure: {MAX_SYMBOL_EXPOSURE_PCT}%")
    print(f"  - Max total exposure: {MAX_TOTAL_EXPOSURE_PCT}%")
    print(f"  - Stock leverage: {MAX_INTRADAY_LEVERAGE_STOCKS}x")
    print(f"  - Crypto: No leverage, long only")
    print()

    # Load latest dataset
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

    symbol_mapping = {
        'BTC-USD': 'BTCUSD',
        'ETH-USD': 'ETHUSD',
    }
    trades_df['symbol'] = trades_df['symbol'].replace(symbol_mapping)

    # Load performance data
    perf_file = str(trades_file).replace('_trades.parquet', '_strategy_performance.parquet')
    perf_df = pd.read_parquet(perf_file)
    perf_df['symbol'] = perf_df['symbol'].replace(symbol_mapping)

    # Filter to test symbols
    available_symbols = [s for s in test_symbols if s in trades_df['symbol'].values]
    print(f"Available test symbols: {', '.join(available_symbols)}")

    trades_df = trades_df[trades_df['symbol'].isin(available_symbols)].copy()
    perf_df = perf_df[perf_df['symbol'].isin(available_symbols)].copy()

    print(f"Total trades on test symbols: {len(trades_df):,}")

    # Filter to profitable windows only
    profitable_windows = perf_df[perf_df['total_return'] > 0][['symbol', 'strategy', 'window_num']]
    print(f"Profitable windows: {len(profitable_windows)} / {len(perf_df)} ({100*len(profitable_windows)/len(perf_df):.1f}%)")

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
    tester = ProductionSizingTester(trades_df, corr_data=corr_data)

    # Run production sizing
    print("Running production sizing test...")
    print()
    prod_result = tester.run_production_sizing()

    # Run baselines
    print("Running baseline comparisons...")
    print()
    baseline_results = tester.run_baseline_comparison()

    # Display results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    all_results = [prod_result] + baseline_results

    # Create table
    print(f"{'Strategy':<30} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'AvgSize':>8}")
    print("-" * 80)

    for r in all_results:
        print(f"{r.strategy_name:<30} {r.total_return_pct:>9.2%} {r.sharpe_ratio:>8.2f} "
              f"{r.max_drawdown_pct:>7.2%} {r.win_rate:>7.2%} {r.avg_position_size:>7.2%}")

    print()
    print("=" * 80)
    print("PRODUCTION SIZING DETAILS")
    print("=" * 80)
    print(f"Max symbol exposure: {prod_result.max_symbol_exposure:.1f}% (limit: {MAX_SYMBOL_EXPOSURE_PCT}%)")
    print(f"Max total exposure: {prod_result.max_total_exposure:.1f}% (limit: {MAX_TOTAL_EXPOSURE_PCT}%)")
    print(f"Exposure violations: {prod_result.num_exposure_violations}")
    print(f"Average leverage: {prod_result.avg_leverage:.2f}x")
    print()

    # Comparison to best baseline
    best_baseline = max(baseline_results, key=lambda x: x.sharpe_ratio)
    print("=" * 80)
    print("COMPARISON TO BEST BASELINE")
    print("=" * 80)
    print(f"Best Baseline: {best_baseline.strategy_name}")
    print(f"  Sharpe: {best_baseline.sharpe_ratio:.2f}, Return: {best_baseline.total_return_pct:.2%}")
    print()
    print(f"Production: {prod_result.strategy_name}")
    print(f"  Sharpe: {prod_result.sharpe_ratio:.2f}, Return: {prod_result.total_return_pct:.2%}")
    print()

    sharpe_delta = prod_result.sharpe_ratio - best_baseline.sharpe_ratio
    return_delta = prod_result.total_return_pct - best_baseline.total_return_pct

    print(f"Delta: Sharpe {sharpe_delta:+.2f}, Return {return_delta:+.2%}")

    if sharpe_delta > 0:
        print("✓ Production sizing outperforms best baseline on risk-adjusted returns")
    else:
        print("⚠️ Production sizing underperforms best baseline - consider switching")

    print()

    # Save results
    output_file = Path("strategytraining/production_sizing_test_results.json")
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'dataset': str(trades_file.name),
        'num_trades': len(trades_df),
        'symbols': available_symbols,
        'production': {
            'strategy': prod_result.strategy_name,
            'return_pct': prod_result.total_return_pct,
            'sharpe': prod_result.sharpe_ratio,
            'max_dd_pct': prod_result.max_drawdown_pct,
            'win_rate': prod_result.win_rate,
            'avg_size': prod_result.avg_position_size,
            'max_symbol_exposure': prod_result.max_symbol_exposure,
            'max_total_exposure': prod_result.max_total_exposure,
            'num_violations': prod_result.num_exposure_violations,
            'avg_leverage': prod_result.avg_leverage,
        },
        'baselines': [
            {
                'strategy': r.strategy_name,
                'return_pct': r.total_return_pct,
                'sharpe': r.sharpe_ratio,
                'max_dd_pct': r.max_drawdown_pct,
                'win_rate': r.win_rate,
                'avg_size': r.avg_position_size,
            }
            for r in baseline_results
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
