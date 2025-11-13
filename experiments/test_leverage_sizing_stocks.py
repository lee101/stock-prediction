#!/usr/bin/env python3
"""
Test sizing strategies with leverage on stocks.

For stocks (not crypto), we have:
- Up to 4x intraday leverage
- Must reduce to 2x by end of day
- 6.5% annual interest on overnight leverage > 1x

Tests combinations of:
- Sizing strategies (Kelly, VolAdjusted, etc.)
- Leverage multipliers (1x, 1.5x, 2x, 2.5x, 3x, 4x)

Focus: Maximize total return while managing leverage costs.

Usage:
    python experiments/test_leverage_sizing_stocks.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime
import json

from marketsimulator.sizing_strategies import (
    FixedFractionStrategy,
    KellyStrategy,
    VolatilityAdjustedStrategy,
    MarketContext,
)
from trainingdata.load_correlation_utils import load_correlation_matrix


@dataclass
class LeverageStrategyResult:
    """Results for a leverage + sizing strategy combo."""
    strategy_name: str
    leverage_multiplier: float
    final_equity: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    num_trades: int
    avg_position_size: float
    volatility: float
    downside_volatility: float
    total_interest_paid: float
    avg_overnight_leverage: float


class LeveragedBacktester:
    """
    Backtester with leverage support for stocks.

    Key features:
    - Intraday leverage up to max_intraday_leverage (e.g., 4x)
    - Overnight leverage limited to max_overnight_leverage (e.g., 2x)
    - Interest charged on overnight leverage > 1x
    """

    def __init__(
        self,
        symbols: List[str],
        initial_cash: float = 100000,
        days: int = 10,
        corr_data: Dict = None,
        max_intraday_leverage: float = 4.0,
        max_overnight_leverage: float = 2.0,
        annual_interest_rate: float = 0.065,  # 6.5%
    ):
        self.symbols = symbols
        self.initial_cash = initial_cash
        self.days = days
        self.corr_data = corr_data
        self.max_intraday_leverage = max_intraday_leverage
        self.max_overnight_leverage = max_overnight_leverage
        self.daily_interest_rate = annual_interest_rate / 365

        # Generate synthetic price paths
        self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """Generate synthetic price data for stocks."""
        np.random.seed(42)

        self.prices = {}
        self.forecasts = {}

        for symbol in self.symbols:
            # Get volatility from correlation data if available
            if self.corr_data and symbol in self.corr_data.get('volatility_metrics', {}):
                vol_metrics = self.corr_data['volatility_metrics'][symbol]
                annual_vol = vol_metrics['annualized_volatility']
            else:
                annual_vol = 0.25  # 25% for stocks

            daily_vol = annual_vol / np.sqrt(252)

            # Generate price path
            drift = 0.0005  # Positive drift
            returns = np.random.normal(drift, daily_vol, self.days + 1)
            start_price = 150.0  # Typical stock price
            prices = start_price * np.exp(np.cumsum(returns))

            self.prices[symbol] = prices

            # Generate forecasts
            forecast_returns = []
            forecast_vols = []

            for i in range(self.days):
                actual_return = (prices[i+1] - prices[i]) / prices[i]
                forecast_return = actual_return * 0.6 + np.random.normal(0, daily_vol * 0.5)
                forecast_vol = daily_vol * np.random.uniform(0.9, 1.1)

                forecast_returns.append(forecast_return)
                forecast_vols.append(forecast_vol)

            self.forecasts[symbol] = {
                'returns': forecast_returns,
                'volatilities': forecast_vols,
            }

    def run_strategy(
        self,
        strategy,
        strategy_name: str,
        leverage_multiplier: float = 1.0,
    ) -> LeverageStrategyResult:
        """
        Run strategy with leverage multiplier.

        The leverage_multiplier scales the position sizes from the base strategy.

        Args:
            strategy: Base sizing strategy
            strategy_name: Name for reporting
            leverage_multiplier: Multiplier for position sizes (1x-4x)

        Returns:
            LeverageStrategyResult with performance metrics
        """
        cash = self.initial_cash
        positions = {sym: 0.0 for sym in self.symbols}
        equity_history = [cash]
        position_sizes = []
        num_trades = 0
        total_interest_paid = 0.0
        overnight_leverages = []

        for day in range(self.days):
            # Start of day: Calculate current equity with overnight positions
            equity = cash
            for sym, qty in positions.items():
                equity += qty * self.prices[sym][day]

            # Calculate overnight leverage and charge interest
            if day > 0:
                position_value = sum(abs(qty * self.prices[sym][day]) for sym, qty in positions.items())
                overnight_leverage = position_value / equity if equity > 0 else 0
                overnight_leverages.append(overnight_leverage)

                # Interest on borrowed amount (leverage > 1x)
                if overnight_leverage > 1.0:
                    borrowed_amount = (overnight_leverage - 1.0) * equity
                    daily_interest = borrowed_amount * self.daily_interest_rate
                    cash -= daily_interest
                    total_interest_paid += daily_interest
                    equity -= daily_interest  # Update equity after interest

            # Build market contexts
            contexts = {}
            for sym in self.symbols:
                contexts[sym] = MarketContext(
                    symbol=sym,
                    predicted_return=self.forecasts[sym]['returns'][day],
                    predicted_volatility=self.forecasts[sym]['volatilities'][day],
                    current_price=self.prices[sym][day],
                    equity=equity,
                    is_crypto=False,  # All stocks
                    existing_position_value=positions[sym] * self.prices[sym][day],
                )

            # INTRADAY: Establish positions with leverage multiplier
            # Calculate desired intraday positions (can use up to 4x)
            intraday_positions = {}
            for sym in self.symbols:
                ctx = contexts[sym]

                try:
                    sizing = strategy.calculate_size(ctx)
                    base_fraction = sizing.position_fraction
                except Exception as e:
                    base_fraction = 0.25

                # Apply leverage multiplier for intraday position
                target_fraction = base_fraction * leverage_multiplier

                # Calculate target quantity for intraday
                target_value = target_fraction * equity
                target_qty = target_value / ctx.current_price
                intraday_positions[sym] = target_qty

                # Execute trade to establish intraday position
                if target_qty != positions[sym]:
                    trade_value = (target_qty - positions[sym]) * ctx.current_price
                    cash -= trade_value
                    positions[sym] = target_qty
                    num_trades += 1
                    position_sizes.append(abs(target_fraction))

            # HOLD INTRADAY POSITIONS: Capture the day's price movement
            # Update positions to end-of-day prices
            eod_equity = cash
            for sym, qty in positions.items():
                eod_equity += qty * self.prices[sym][day+1] if day+1 < len(self.prices[sym]) else qty * self.prices[sym][day]

            equity_history.append(eod_equity)

            # END OF DAY: Enforce overnight leverage limit (2x max)
            # Use next day's opening price (or current EOD price as proxy)
            eod_price_idx = min(day + 1, len(self.prices[sym]) - 1)
            total_position_value = sum(
                abs(qty * self.prices[sym][eod_price_idx])
                for sym, qty in positions.items()
            )
            current_leverage = total_position_value / eod_equity if eod_equity > 0 else 0

            if current_leverage > self.max_overnight_leverage:
                # Scale down all positions proportionally to meet 2x limit
                scale_factor = self.max_overnight_leverage / current_leverage

                for sym in self.symbols:
                    if positions[sym] != 0:
                        # Reduce position at EOD
                        old_qty = positions[sym]
                        new_qty = old_qty * scale_factor
                        trade_value = (new_qty - old_qty) * self.prices[sym][eod_price_idx]
                        cash -= trade_value
                        positions[sym] = new_qty
                        num_trades += 1

        # Final equity
        final_equity = cash
        for sym, qty in positions.items():
            final_equity += qty * self.prices[sym][-1]

        equity_history.append(final_equity)

        # Calculate metrics
        total_return_pct = (final_equity - self.initial_cash) / self.initial_cash

        # Sharpe and Sortino ratios
        if len(equity_history) > 1:
            returns = np.diff(equity_history) / equity_history[:-1]

            # Sharpe ratio (uses total volatility)
            mean_return = np.mean(returns)
            volatility = np.std(returns) * np.sqrt(252)
            sharpe = mean_return / (np.std(returns) + 1e-10) * np.sqrt(252)

            # Sortino ratio (uses downside volatility only)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_volatility = np.std(downside_returns) * np.sqrt(252)
                sortino = mean_return / (np.std(downside_returns) + 1e-10) * np.sqrt(252)
            else:
                downside_volatility = 0.0
                sortino = sharpe  # No downside, Sortino = Sharpe
        else:
            sharpe = 0.0
            sortino = 0.0
            volatility = 0.0
            downside_volatility = 0.0

        # Max drawdown
        peak = self.initial_cash
        max_dd = 0.0
        for eq in equity_history:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd

        avg_position_size = np.mean(position_sizes) if position_sizes else 0.0
        avg_overnight_leverage = np.mean(overnight_leverages) if overnight_leverages else 0.0

        return LeverageStrategyResult(
            strategy_name=strategy_name,
            leverage_multiplier=leverage_multiplier,
            final_equity=final_equity,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_dd,
            num_trades=num_trades,
            avg_position_size=avg_position_size,
            volatility=volatility,
            downside_volatility=downside_volatility,
            total_interest_paid=total_interest_paid,
            avg_overnight_leverage=avg_overnight_leverage,
        )


def main():
    print("=" * 80)
    print("LEVERAGE + SIZING STRATEGY TESTING ON STOCKS")
    print("=" * 80)
    print()

    # Stock symbols only (have leverage access)
    symbols = ['NVDA', 'MSFT', 'GOOG', 'AAPL']
    print(f"Testing on {len(symbols)} stock symbols: {', '.join(symbols)}")
    print(f"Simulation period: 10 days")
    print(f"Initial capital: $100,000")
    print()
    print("Leverage constraints:")
    print("  • Max intraday: 4x")
    print("  • Max overnight: 2x")
    print("  • Interest rate: 6.5% annual (1.78 bps daily)")
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

    # Initialize backtester
    backtester = LeveragedBacktester(symbols, corr_data=corr_data)

    # Base strategies to test (focus on best return strategies)
    base_strategies = [
        (KellyStrategy(fraction=0.5, cap=1.0), "Kelly_50pct"),
        (KellyStrategy(fraction=0.75, cap=1.0), "Kelly_75pct"),
        (VolatilityAdjustedStrategy(corr_data=corr_data, target_vol_contribution=0.15), "VolAdjusted_15pct"),
        (FixedFractionStrategy(0.5), "Fixed_50pct"),
    ]

    # Leverage multipliers to test
    leverage_multipliers = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

    print(f"Testing {len(base_strategies)} strategies × {len(leverage_multipliers)} leverage levels...")
    print()

    # Run all combinations
    results = []
    for strategy, name in base_strategies:
        for lev in leverage_multipliers:
            result = backtester.run_strategy(strategy, name, leverage_multiplier=lev)
            results.append(result)
            print(f"  ✓ {name} @ {lev}x leverage")

    print()
    print("=" * 80)
    print("RESULTS (Sorted by Total Return)")
    print("=" * 80)
    print()

    # Sort by total return
    results.sort(key=lambda x: x.total_return_pct, reverse=True)

    # Display top 15
    df_data = []
    for r in results[:15]:
        df_data.append({
            'Strategy': r.strategy_name,
            'Leverage': f"{r.leverage_multiplier:.1f}x",
            'Return': f"{r.total_return_pct:.2%}",
            'Sharpe': f"{r.sharpe_ratio:.2f}",
            'Sortino': f"{r.sortino_ratio:.2f}",
            'Max DD': f"{r.max_drawdown_pct:.2%}",
            'Volatility': f"{r.volatility:.2%}",
            'Interest': f"${r.total_interest_paid:.0f}",
            'Avg Lev': f"{r.avg_overnight_leverage:.2f}x",
        })

    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))
    print()

    # Show top 5 by return
    print("=" * 80)
    print("TOP 5 BY TOTAL RETURN")
    print("=" * 80)
    for i, r in enumerate(results[:5], 1):
        print(f"{i}. {r.strategy_name} @ {r.leverage_multiplier}x")
        print(f"   Return: {r.total_return_pct:7.2%}  |  Sharpe: {r.sharpe_ratio:6.2f}  |  "
              f"Sortino: {r.sortino_ratio:6.2f}")
        print(f"   Max DD: {r.max_drawdown_pct:6.2%}  |  Interest: ${r.total_interest_paid:,.0f}  |  "
              f"Avg overnight leverage: {r.avg_overnight_leverage:.2f}x")
        print()

    # Show top 5 by Sortino ratio
    print("=" * 80)
    print("TOP 5 BY SORTINO RATIO (Better Risk-Adjusted)")
    print("=" * 80)
    sortino_sorted = sorted(results, key=lambda x: x.sortino_ratio, reverse=True)
    for i, r in enumerate(sortino_sorted[:5], 1):
        print(f"{i}. {r.strategy_name} @ {r.leverage_multiplier}x")
        print(f"   Sortino: {r.sortino_ratio:6.2f}  |  Sharpe: {r.sharpe_ratio:6.2f}  |  "
              f"Return: {r.total_return_pct:7.2%}")
        print(f"   Downside Vol: {r.downside_volatility:6.2%}  |  Total Vol: {r.volatility:6.2%}")
        print()

    # Show effect of leverage on best base strategy
    print("=" * 80)
    print("LEVERAGE IMPACT ON BEST BASE STRATEGY")
    print("=" * 80)

    # Find best base strategy at 1x
    baseline_results = [r for r in results if r.leverage_multiplier == 1.0]
    best_baseline = max(baseline_results, key=lambda x: x.total_return_pct)

    print(f"Best base strategy: {best_baseline.strategy_name}")
    print()

    # Show all leverage levels for this strategy
    strategy_results = [r for r in results if r.strategy_name == best_baseline.strategy_name]
    strategy_results.sort(key=lambda x: x.leverage_multiplier)

    print(f"{'Leverage':<10} {'Return':<12} {'Sharpe':<10} {'Sortino':<10} {'Interest':<12} {'Avg Lev':<10}")
    print("-" * 80)
    for r in strategy_results:
        print(f"{r.leverage_multiplier}x{' ':<8} "
              f"{r.total_return_pct:>10.2%}  "
              f"{r.sharpe_ratio:>8.2f}  "
              f"{r.sortino_ratio:>8.2f}  "
              f"${r.total_interest_paid:>9.0f}  "
              f"{r.avg_overnight_leverage:>8.2f}x")

    print()

    # Save results
    output_file = Path("experiments/leverage_sizing_stocks_results.json")
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'symbols': symbols,
        'initial_cash': backtester.initial_cash,
        'days': backtester.days,
        'max_intraday_leverage': backtester.max_intraday_leverage,
        'max_overnight_leverage': backtester.max_overnight_leverage,
        'annual_interest_rate': backtester.daily_interest_rate * 365,
        'results': [
            {
                'strategy': r.strategy_name,
                'leverage': r.leverage_multiplier,
                'final_equity': r.final_equity,
                'return_pct': r.total_return_pct,
                'sharpe': r.sharpe_ratio,
                'sortino': r.sortino_ratio,
                'max_dd_pct': r.max_drawdown_pct,
                'volatility': r.volatility,
                'downside_volatility': r.downside_volatility,
                'interest_paid': r.total_interest_paid,
                'avg_overnight_leverage': r.avg_overnight_leverage,
                'num_trades': r.num_trades,
            }
            for r in results
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    best = results[0]
    print(f"🏆 Highest Return: {best.strategy_name} @ {best.leverage_multiplier}x leverage")
    print(f"  • Total return: {best.total_return_pct:.2%}")
    print(f"  • Sharpe ratio: {best.sharpe_ratio:.2f}")
    print(f"  • Sortino ratio: {best.sortino_ratio:.2f}")
    print(f"  • Max drawdown: {best.max_drawdown_pct:.2%}")
    print(f"  • Interest cost: ${best.total_interest_paid:,.0f}")
    print(f"  • Avg overnight leverage: {best.avg_overnight_leverage:.2f}x")
    print()

    # Best Sortino recommendation
    print("⭐ Best Risk-Adjusted (Sortino Ratio):")
    best_sortino = max(results, key=lambda x: x.sortino_ratio)
    print(f"  {best_sortino.strategy_name} @ {best_sortino.leverage_multiplier}x leverage")
    print(f"  • Sortino ratio: {best_sortino.sortino_ratio:.2f}")
    print(f"  • Total return: {best_sortino.total_return_pct:.2%}")
    print(f"  • Sharpe ratio: {best_sortino.sharpe_ratio:.2f}")
    print(f"  • Downside volatility: {best_sortino.downside_volatility:.2%} (vs {best_sortino.volatility:.2%} total)")
    print()

    # Comparison
    if best_sortino.strategy_name != best.strategy_name or best_sortino.leverage_multiplier != best.leverage_multiplier:
        print("📊 Comparison:")
        print(f"  Highest Return strategy gives: {best.total_return_pct:.2%} return")
        print(f"  Best Sortino strategy gives: {best_sortino.total_return_pct:.2%} return")
        print(f"  Difference: {best.total_return_pct - best_sortino.total_return_pct:+.2%}")
        print()

    print("=" * 80)


if __name__ == "__main__":
    main()
