#!/usr/bin/env python3
"""
Compare different position sizing strategies on market simulator.

Tests strategies on BTCUSD, ETHUSD, MSFT, NVDA, AAPL, SPY over 10 days.
Compares:
- Naive 50/50 split (baseline)
- Volatility-adjusted sizing
- Correlation-aware robust Kelly
- Other strategies

Usage:
    python experiments/test_sizing_strategies_comparison.py
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
import json
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from marketsimulator.sizing_strategies import (
    FixedFractionStrategy,
    KellyStrategy,
    VolatilityTargetStrategy,
    CorrelationAwareStrategy,
    VolatilityAdjustedStrategy,
    MarketContext,
    SizingResult,
)
from trainingdata.load_correlation_utils import (
    load_correlation_matrix,
    get_volatility_metrics,
)


@dataclass
class StrategyResult:
    """Results for a single strategy."""
    strategy_name: str
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    num_trades: int
    avg_position_size: float
    volatility: float


class SimplifiedBacktester:
    """
    Simplified backtester for position sizing comparison.

    Simulates trading multiple symbols with different sizing strategies
    over a fixed period.
    """

    def __init__(
        self,
        symbols: List[str],
        initial_cash: float = 100000,
        days: int = 10,
        corr_data: Dict = None,
    ):
        self.symbols = symbols
        self.initial_cash = initial_cash
        self.days = days
        self.corr_data = corr_data

        # Generate synthetic price paths and forecasts
        # In practice, this would come from your actual forecaster
        self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """Generate synthetic price data and forecasts for testing."""
        np.random.seed(42)  # Reproducible results

        self.prices = {}
        self.forecasts = {}

        for symbol in self.symbols:
            # Get volatility from correlation data if available
            if self.corr_data and symbol in self.corr_data.get('symbols', []):
                try:
                    vol_metrics = self.corr_data['volatility_metrics'][symbol]
                    annual_vol = vol_metrics['annualized_volatility']
                except Exception:
                    annual_vol = 0.3  # Default 30%
            else:
                annual_vol = 0.3

            # Daily volatility
            daily_vol = annual_vol / np.sqrt(252)

            # Generate price path (geometric Brownian motion)
            drift = 0.0005  # Small positive drift
            returns = np.random.normal(drift, daily_vol, self.days + 1)

            # Starting price
            start_price = 100.0 if not symbol.endswith('USD') else 2000.0
            prices = start_price * np.exp(np.cumsum(returns))

            self.prices[symbol] = prices

            # Generate forecasts (add noise to actual future returns)
            forecast_returns = []
            forecast_vols = []

            for i in range(self.days):
                # Actual next-day return
                actual_return = (prices[i+1] - prices[i]) / prices[i]

                # Noisy forecast (60% accuracy)
                forecast_return = actual_return * 0.6 + np.random.normal(0, daily_vol * 0.5)

                # Forecast volatility (slightly noisy)
                forecast_vol = daily_vol * np.random.uniform(0.9, 1.1)

                forecast_returns.append(forecast_return)
                forecast_vols.append(forecast_vol)

            self.forecasts[symbol] = {
                'returns': forecast_returns,
                'volatilities': forecast_vols,
            }

    def run_strategy(self, strategy, strategy_name: str) -> StrategyResult:
        """
        Run a single strategy through the backtest.

        Args:
            strategy: Sizing strategy instance
            strategy_name: Name for reporting

        Returns:
            StrategyResult with performance metrics
        """
        cash = self.initial_cash
        positions = {sym: 0.0 for sym in self.symbols}  # Number of shares
        equity_history = [self.initial_cash]
        position_sizes = []
        num_trades = 0

        for day in range(self.days):
            # Calculate current equity
            equity = cash
            for sym, qty in positions.items():
                equity += qty * self.prices[sym][day]

            equity_history.append(equity)

            # Build market context for each symbol
            contexts = {}
            for sym in self.symbols:
                is_crypto = sym.endswith('USD')
                contexts[sym] = MarketContext(
                    symbol=sym,
                    predicted_return=self.forecasts[sym]['returns'][day],
                    predicted_volatility=self.forecasts[sym]['volatilities'][day],
                    current_price=self.prices[sym][day],
                    equity=equity,
                    is_crypto=is_crypto,
                    existing_position_value=positions[sym] * self.prices[sym][day],
                )

            # Calculate sizing for each symbol
            for sym in self.symbols:
                ctx = contexts[sym]

                # Pass portfolio context if strategy supports it
                try:
                    if isinstance(strategy, CorrelationAwareStrategy):
                        sizing = strategy.calculate_size(ctx, portfolio_context=contexts)
                    else:
                        sizing = strategy.calculate_size(ctx)
                except Exception as e:
                    # Fallback on error
                    print(f"  Warning: {strategy_name} failed for {sym}: {e}")
                    sizing = MarketContext(sym, 0, 0.01, ctx.current_price, equity, ctx.is_crypto)
                    sizing = SizingResult(0, 0, 0, 1.0, "Error fallback")

                # Calculate target quantity
                target_qty = sizing.quantity

                # Execute trade (simplified - no slippage/fees)
                if target_qty != positions[sym]:
                    trade_value = (target_qty - positions[sym]) * ctx.current_price
                    cash -= trade_value
                    positions[sym] = target_qty
                    num_trades += 1
                    position_sizes.append(abs(sizing.position_fraction))

        # Final equity
        final_equity = cash
        for sym, qty in positions.items():
            final_equity += qty * self.prices[sym][-1]

        equity_history.append(final_equity)

        # Calculate metrics
        total_return_pct = (final_equity - self.initial_cash) / self.initial_cash

        # Max drawdown
        peak = self.initial_cash
        max_dd = 0.0
        for eq in equity_history:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (simplified)
        if len(equity_history) > 1:
            returns = np.diff(equity_history) / equity_history[:-1]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Volatility
        if len(equity_history) > 1:
            returns = np.diff(equity_history) / equity_history[:-1]
            volatility = np.std(returns) * np.sqrt(252)
        else:
            volatility = 0.0

        avg_position_size = np.mean(position_sizes) if position_sizes else 0.0

        return StrategyResult(
            strategy_name=strategy_name,
            final_equity=final_equity,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            num_trades=num_trades,
            avg_position_size=avg_position_size,
            volatility=volatility,
        )


def main():
    print("=" * 80)
    print("POSITION SIZING STRATEGY COMPARISON")
    print("=" * 80)
    print()

    # Test symbols
    symbols = ['BTCUSD', 'ETHUSD', 'MSFT', 'NVDA', 'AAPL', 'SPY']
    print(f"Testing on {len(symbols)} symbols: {', '.join(symbols)}")
    print(f"Simulation period: 10 days")
    print(f"Initial capital: $100,000")
    print()

    # Load correlation data
    print("Loading correlation and volatility data...")
    try:
        corr_data = load_correlation_matrix()
        print(f"✓ Loaded correlation matrix with {len(corr_data['symbols'])} symbols")
        print()
    except Exception as e:
        print(f"⚠️  Could not load correlation data: {e}")
        print("  Continuing with default volatility estimates")
        corr_data = None
        print()

    # Initialize backtester
    backtester = SimplifiedBacktester(symbols, corr_data=corr_data)

    # Define strategies to test
    strategies = [
        # Baseline: naive 50/50 split
        (FixedFractionStrategy(0.5 / len(symbols)), "Naive_Equal_Weight"),

        # Fixed allocations
        (FixedFractionStrategy(0.25), "Fixed_25pct"),
        (FixedFractionStrategy(0.5), "Fixed_50pct"),

        # Kelly variants
        (KellyStrategy(fraction=0.25, cap=1.0), "Kelly_25pct"),
        (KellyStrategy(fraction=0.5, cap=1.0), "Kelly_50pct"),

        # Volatility-based
        (VolatilityTargetStrategy(target_vol=0.10), "VolTarget_10pct"),
        (VolatilityTargetStrategy(target_vol=0.15), "VolTarget_15pct"),

        # New: Volatility-adjusted (uses correlation data)
        (VolatilityAdjustedStrategy(corr_data=corr_data, target_vol_contribution=0.10), "VolAdjusted_10pct"),
        (VolatilityAdjustedStrategy(corr_data=corr_data, target_vol_contribution=0.15), "VolAdjusted_15pct"),

        # New: Correlation-aware robust Kelly
        (CorrelationAwareStrategy(corr_data=corr_data, uncertainty_penalty=2.0, fractional_kelly=0.25), "CorrAware_Conservative"),
        (CorrelationAwareStrategy(corr_data=corr_data, uncertainty_penalty=1.0, fractional_kelly=0.5), "CorrAware_Moderate"),
        (CorrelationAwareStrategy(corr_data=corr_data, uncertainty_penalty=0.5, fractional_kelly=0.75), "CorrAware_Aggressive"),
    ]

    # Run all strategies
    print("Running simulations...")
    print()

    results = []
    for strategy, name in strategies:
        result = backtester.run_strategy(strategy, name)
        results.append(result)
        print(f"  ✓ Completed: {name}")

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    # Convert to DataFrame for easy viewing
    df_data = []
    for r in results:
        df_data.append({
            'Strategy': r.strategy_name,
            'Final Equity': f"${r.final_equity:,.0f}",
            'Return': f"{r.total_return_pct:.2%}",
            'Max DD': f"{r.max_drawdown_pct:.2%}",
            'Sharpe': f"{r.sharpe_ratio:.2f}",
            'Volatility': f"{r.volatility:.2%}",
            'Trades': r.num_trades,
            'Avg Size': f"{r.avg_position_size:.2%}",
        })

    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))
    print()

    # Rank by Sharpe ratio
    print("Ranking by Sharpe Ratio:")
    print("-" * 80)
    sorted_results = sorted(results, key=lambda x: x.sharpe_ratio, reverse=True)
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"  {i}. {r.strategy_name:30s} Sharpe: {r.sharpe_ratio:6.2f}  "
              f"Return: {r.total_return_pct:6.2%}  DD: {r.max_drawdown_pct:6.2%}")

    print()

    # Rank by return
    print("Ranking by Total Return:")
    print("-" * 80)
    sorted_results = sorted(results, key=lambda x: x.total_return_pct, reverse=True)
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"  {i}. {r.strategy_name:30s} Return: {r.total_return_pct:6.2%}  "
              f"Sharpe: {r.sharpe_ratio:6.2f}  DD: {r.max_drawdown_pct:6.2%}")

    print()

    # Save results
    output_file = Path("experiments/sizing_strategy_results.json")
    output_file.parent.mkdir(exist_ok=True)

    output_data = {
        'timestamp': datetime.now().isoformat(),
        'symbols': symbols,
        'initial_cash': backtester.initial_cash,
        'days': backtester.days,
        'results': [
            {
                'strategy': r.strategy_name,
                'final_equity': r.final_equity,
                'return_pct': r.total_return_pct,
                'max_dd_pct': r.max_drawdown_pct,
                'sharpe': r.sharpe_ratio,
                'volatility': r.volatility,
                'trades': r.num_trades,
                'avg_size': r.avg_position_size,
            }
            for r in results
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    # Summary
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    baseline = next((r for r in results if r.strategy_name == "Naive_Equal_Weight"), None)
    if baseline:
        print(f"Baseline (Naive Equal Weight):")
        print(f"  Return: {baseline.total_return_pct:.2%}")
        print(f"  Sharpe: {baseline.sharpe_ratio:.2f}")
        print(f"  Max DD: {baseline.max_drawdown_pct:.2%}")
        print()

        # Compare to best correlation-aware
        corr_aware_results = [r for r in results if 'CorrAware' in r.strategy_name]
        if corr_aware_results:
            best_corr = max(corr_aware_results, key=lambda x: x.sharpe_ratio)
            print(f"Best Correlation-Aware ({best_corr.strategy_name}):")
            print(f"  Return: {best_corr.total_return_pct:.2%} ({best_corr.total_return_pct - baseline.total_return_pct:+.2%} vs baseline)")
            print(f"  Sharpe: {best_corr.sharpe_ratio:.2f} ({best_corr.sharpe_ratio - baseline.sharpe_ratio:+.2f} vs baseline)")
            print(f"  Max DD: {best_corr.max_drawdown_pct:.2%} ({best_corr.max_drawdown_pct - baseline.max_drawdown_pct:+.2%} vs baseline)")
            print()

        # Compare to best vol-adjusted
        vol_adj_results = [r for r in results if 'VolAdjusted' in r.strategy_name]
        if vol_adj_results:
            best_vol = max(vol_adj_results, key=lambda x: x.sharpe_ratio)
            print(f"Best Vol-Adjusted ({best_vol.strategy_name}):")
            print(f"  Return: {best_vol.total_return_pct:.2%} ({best_vol.total_return_pct - baseline.total_return_pct:+.2%} vs baseline)")
            print(f"  Sharpe: {best_vol.sharpe_ratio:.2f} ({best_vol.sharpe_ratio - baseline.sharpe_ratio:+.2f} vs baseline)")
            print(f"  Max DD: {best_vol.max_drawdown_pct:.2%} ({best_vol.max_drawdown_pct - baseline.max_drawdown_pct:+.2%} vs baseline)")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
