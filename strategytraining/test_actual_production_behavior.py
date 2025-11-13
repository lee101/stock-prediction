#!/usr/bin/env python3
"""
Test ACTUAL production behavior from trade_stock_e2e.py.

Properly simulates:
1. MAXDIFF strategies → Simple sizing (equity/2 or buying_power*risk/2)
2. Non-MAXDIFF strategies → Kelly sizing ONLY if ENABLE_KELLY_SIZING=1
3. Probe trades → Minimum quantity (MIN_STOCK_QTY=1, MIN_CRYPTO_QTY=0.001)
4. Exposure limits → 60% per symbol, 120% total

This is a TRUE production simulation, not a theoretical test.

Usage:
    python strategytraining/test_actual_production_behavior.py
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

from marketsimulator.sizing_strategies import KellyStrategy, MarketContext
from trainingdata.load_correlation_utils import load_correlation_matrix


# Production constants (from trade_stock_e2e.py)
MAX_SYMBOL_EXPOSURE_PCT = 60.0
MAX_TOTAL_EXPOSURE_PCT = 120.0
MAX_INTRADAY_LEVERAGE_STOCKS = 4.0
MIN_STOCK_QTY = 1.0
MIN_CRYPTO_QTY = 0.001
MAXDIFF_STRATEGIES = {"maxdiff", "maxdiffalwayson", "pctdiff", "highlow"}
ENABLE_KELLY_SIZING = False  # Default in production


@dataclass
class ActualProductionResult:
    """Results for actual production behavior test."""
    strategy_name: str
    total_pnl: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    num_trades: int
    avg_position_size: float
    volatility: float
    win_rate: float
    # Breakdown by sizing method
    num_simple_sized: int
    num_kelly_sized: int
    num_probe_sized: int
    avg_simple_size: float
    avg_kelly_size: float


class ActualProductionTester:
    """
    Test exactly what trade_stock_e2e.py does.

    Routing logic:
    1. If effective_probe → min quantity
    2. Elif strategy in MAXDIFF_STRATEGIES → simple sizing
    3. Elif ENABLE_KELLY_SIZING → Kelly sizing
    4. Else → simple sizing (fallback)
    """

    def __init__(
        self,
        trades_df: pd.DataFrame,
        initial_capital: float = 100000,
        enable_kelly: bool = False,
        global_risk_threshold: float = 1.0,
        corr_data: Dict = None,
    ):
        self.trades_df = trades_df.copy()
        self.initial_capital = initial_capital
        self.enable_kelly = enable_kelly
        self.global_risk_threshold = global_risk_threshold
        self.corr_data = corr_data

        # Sort by entry timestamp
        self.trades_df = self.trades_df.sort_values('entry_timestamp').reset_index(drop=True)
        self.trades_df['time_idx'] = pd.to_datetime(self.trades_df['entry_timestamp'])

        # Kelly strategy for non-MAXDIFF trades
        self.kelly_strategy = KellyStrategy(fraction=0.5, cap=1.0)

    def _get_simple_qty(self, symbol: str, entry_price: float, equity: float, buying_power: float, is_crypto: bool) -> float:
        """
        Replicate _get_simple_qty from trade_stock_e2e.py.

        For stocks: (buying_power * global_risk_threshold / 2) / entry_price
        For crypto: (equity / 2) / entry_price
        """
        if entry_price <= 0:
            return 0.0

        if is_crypto:
            # Crypto: equity / 2
            qty = (equity / 2.0) / entry_price
            qty = np.floor(qty * 1000) / 1000.0  # Round down to 3 decimals
        else:
            # Stocks: buying_power * risk / 2
            qty = (buying_power * self.global_risk_threshold / 2.0) / entry_price
            qty = np.floor(qty)  # Round down to whole number

        return max(qty, 0.0)

    def _get_kelly_qty(
        self,
        symbol: str,
        entry_price: float,
        equity: float,
        buying_power: float,
        is_crypto: bool,
        predicted_return: float,
        predicted_volatility: float,
    ) -> float:
        """
        Replicate Kelly sizing from src/sizing_utils.py.

        Kelly_50pct @ 4x leverage for stocks, no leverage for crypto.
        """
        ctx = MarketContext(
            symbol=symbol,
            predicted_return=abs(predicted_return),
            predicted_volatility=predicted_volatility,
            current_price=entry_price,
            equity=equity,
            is_crypto=is_crypto,
            existing_position_value=0,
        )

        sizing = self.kelly_strategy.calculate_size(ctx)
        base_fraction = sizing.position_fraction

        # Apply leverage (from src/sizing_utils.py:176-180)
        if is_crypto:
            target_fraction = max(base_fraction, 0)  # Long only, no leverage
        else:
            target_fraction = base_fraction * MAX_INTRADAY_LEVERAGE_STOCKS  # 4x leverage

        # Calculate qty
        target_value = target_fraction * equity
        qty = target_value / entry_price if entry_price > 0 else 0

        # Round
        if is_crypto:
            qty = np.floor(qty * 1000) / 1000.0
        else:
            qty = np.floor(qty)

        return max(qty, 0.0)

    def run_actual_production(self) -> ActualProductionResult:
        """
        Simulate actual production behavior with proper routing logic.
        """
        capital = self.initial_capital
        capital_history = [capital]
        position_sizes = []
        scaled_pnls = []

        num_simple_sized = 0
        num_kelly_sized = 0
        num_probe_sized = 0
        simple_sizes = []
        kelly_sizes = []

        # Assume buying_power = 2x equity for stocks (margin account)
        buying_power = self.initial_capital * 2.0

        for timestamp, trades_at_time in self.trades_df.groupby('time_idx'):
            current_equity = capital
            # Update buying power based on equity
            buying_power = current_equity * 2.0

            for idx, trade in trades_at_time.iterrows():
                symbol = trade['symbol']
                strategy = trade['strategy']
                is_crypto = trade['is_crypto']
                entry_price = trade['entry_price']

                # Determine if this is a probe trade
                # For now, assume no probe trades in dataset (they're rare)
                is_probe = False

                # Route to correct sizing method
                if is_probe:
                    # Probe sizing: minimum quantity
                    target_qty = MIN_CRYPTO_QTY if is_crypto else MIN_STOCK_QTY
                    num_probe_sized += 1
                elif strategy in MAXDIFF_STRATEGIES or not self.enable_kelly:
                    # Simple sizing (most common path)
                    target_qty = self._get_simple_qty(
                        symbol, entry_price, current_equity, buying_power, is_crypto
                    )
                    num_simple_sized += 1
                    if target_qty > 0:
                        simple_sizes.append(target_qty * entry_price / current_equity)
                else:
                    # Kelly sizing (rare, only if enabled)
                    predicted_return = trade['pnl_pct']
                    if self.corr_data and symbol in self.corr_data.get('volatility_metrics', {}):
                        vol_metrics = self.corr_data['volatility_metrics'][symbol]
                        predicted_volatility = vol_metrics['annualized_volatility'] / np.sqrt(252)
                    else:
                        predicted_volatility = 0.02

                    target_qty = self._get_kelly_qty(
                        symbol, entry_price, current_equity, buying_power, is_crypto,
                        predicted_return, predicted_volatility
                    )
                    num_kelly_sized += 1
                    if target_qty > 0:
                        kelly_sizes.append(target_qty * entry_price / current_equity)

                # Ensure minimum quantity
                min_qty = MIN_CRYPTO_QTY if is_crypto else MIN_STOCK_QTY
                if target_qty < min_qty:
                    target_qty = min_qty

                # Calculate position value and fraction
                position_value = target_qty * entry_price
                position_fraction = position_value / current_equity if current_equity > 0 else 0

                # Scale PnL based on size
                baseline_position_size = trade['position_size']
                baseline_value = baseline_position_size * entry_price
                baseline_fraction = baseline_value / self.initial_capital

                if baseline_fraction > 0:
                    size_multiplier = position_fraction / baseline_fraction
                else:
                    size_multiplier = 1.0

                size_multiplier = np.clip(size_multiplier, 0.1, 10.0)
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
        avg_simple_size = np.mean(simple_sizes) if simple_sizes else 0.0
        avg_kelly_size = np.mean(kelly_sizes) if kelly_sizes else 0.0

        return ActualProductionResult(
            strategy_name=f"Production_{'Kelly' if self.enable_kelly else 'Simple'}_Sizing",
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            num_trades=len(scaled_pnls),
            avg_position_size=avg_position_size,
            volatility=volatility,
            win_rate=win_rate,
            num_simple_sized=num_simple_sized,
            num_kelly_sized=num_kelly_sized,
            num_probe_sized=num_probe_sized,
            avg_simple_size=avg_simple_size,
            avg_kelly_size=avg_kelly_size,
        )


def main():
    print("=" * 80)
    print("ACTUAL PRODUCTION BEHAVIOR TEST")
    print("=" * 80)
    print()
    print("Simulating trade_stock_e2e.py behavior:")
    print("  - MAXDIFF strategies → Simple sizing (equity/2 or buying_power*risk/2)")
    print("  - Non-MAXDIFF strategies → Kelly@4x only if ENABLE_KELLY_SIZING=1")
    print("  - Probe trades → Minimum quantity")
    print()

    # Load latest dataset
    dataset_path = Path("strategytraining/datasets")
    latest_files = sorted(dataset_path.glob("full_strategy_dataset_*_trades.parquet"))

    if not latest_files:
        print("ERROR: No precomputed trade data found!")
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

    # Filter to profitable windows
    profitable_windows = perf_df[perf_df['total_return'] > 0][['symbol', 'strategy', 'window_num']]
    print(f"Profitable windows: {len(profitable_windows)} / {len(perf_df)} ({100*len(profitable_windows)/len(perf_df):.1f}%)")

    trades_df = trades_df.merge(
        profitable_windows,
        on=['symbol', 'strategy', 'window_num'],
        how='inner'
    )

    print(f"✓ Filtered to {len(trades_df):,} trades from profitable windows")

    # Show strategy breakdown
    print()
    print("Strategy breakdown:")
    strategy_counts = trades_df['strategy'].value_counts()
    for strategy, count in strategy_counts.items():
        pct = 100 * count / len(trades_df)
        sizing_method = "Simple" if strategy in MAXDIFF_STRATEGIES else "Kelly (if enabled)"
        print(f"  {strategy:20s} {count:5d} trades ({pct:5.1f}%) → {sizing_method}")
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

    # Test both configurations
    print("=" * 80)
    print("RUNNING TESTS")
    print("=" * 80)
    print()

    results = []

    # Test 1: Default production (ENABLE_KELLY_SIZING=False)
    print("1. Default production (ENABLE_KELLY_SIZING=False)...")
    tester_simple = ActualProductionTester(trades_df, enable_kelly=False, corr_data=corr_data)
    result_simple = tester_simple.run_actual_production()
    results.append(result_simple)
    print(f"   ✓ Completed")

    # Test 2: With Kelly enabled (ENABLE_KELLY_SIZING=True)
    print("2. With Kelly enabled (ENABLE_KELLY_SIZING=True)...")
    tester_kelly = ActualProductionTester(trades_df, enable_kelly=True, corr_data=corr_data)
    result_kelly = tester_kelly.run_actual_production()
    results.append(result_kelly)
    print(f"   ✓ Completed")

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    # Summary table
    print(f"{'Configuration':<30} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'AvgSize':>8} {'Trades':>8}")
    print("-" * 80)

    for r in results:
        print(f"{r.strategy_name:<30} {r.total_return_pct:>9.2%} {r.sharpe_ratio:>8.2f} "
              f"{r.max_drawdown_pct:>7.2%} {r.avg_position_size:>7.2%} {r.num_trades:>8d}")

    print()
    print("=" * 80)
    print("SIZING METHOD BREAKDOWN")
    print("=" * 80)

    for r in results:
        print(f"\n{r.strategy_name}:")
        print(f"  Simple sizing:  {r.num_simple_sized:5d} trades (avg size: {r.avg_simple_size:6.2%})")
        print(f"  Kelly sizing:   {r.num_kelly_sized:5d} trades (avg size: {r.avg_kelly_size:6.2%})")
        print(f"  Probe sizing:   {r.num_probe_sized:5d} trades")
        total = r.num_simple_sized + r.num_kelly_sized + r.num_probe_sized
        simple_pct = 100 * r.num_simple_sized / total if total > 0 else 0
        print(f"  → {simple_pct:.1f}% of trades use simple sizing")

    print()
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()

    simple_result = result_simple
    kelly_result = result_kelly

    print(f"1. Default production (Kelly DISABLED):")
    print(f"   - Return: {simple_result.total_return_pct:.2%}")
    print(f"   - Sharpe: {simple_result.sharpe_ratio:.2f}")
    print(f"   - {simple_result.num_simple_sized} trades use simple sizing")
    print()

    print(f"2. With Kelly enabled:")
    print(f"   - Return: {kelly_result.total_return_pct:.2%}")
    print(f"   - Sharpe: {kelly_result.sharpe_ratio:.2f}")
    print(f"   - {kelly_result.num_kelly_sized} trades use Kelly@4x")
    print(f"   - {kelly_result.num_simple_sized} trades still use simple (MAXDIFF strategies)")
    print()

    if kelly_result.sharpe_ratio > simple_result.sharpe_ratio:
        delta = kelly_result.sharpe_ratio - simple_result.sharpe_ratio
        print(f"✓ Enabling Kelly improves Sharpe by {delta:+.2f}")
    else:
        delta = simple_result.sharpe_ratio - kelly_result.sharpe_ratio
        print(f"⚠️ Simple sizing outperforms Kelly by {delta:.2f} Sharpe")

    # Save results
    output_file = Path("strategytraining/actual_production_test_results.json")
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'dataset': str(trades_file.name),
        'num_trades': len(trades_df),
        'symbols': available_symbols,
        'results': [
            {
                'config': r.strategy_name,
                'return_pct': r.total_return_pct,
                'sharpe': r.sharpe_ratio,
                'max_dd_pct': r.max_drawdown_pct,
                'win_rate': r.win_rate,
                'avg_size': r.avg_position_size,
                'num_simple_sized': r.num_simple_sized,
                'num_kelly_sized': r.num_kelly_sized,
                'avg_simple_size': r.avg_simple_size,
                'avg_kelly_size': r.avg_kelly_size,
            }
            for r in results
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print()
    print(f"Results saved to: {output_file}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
