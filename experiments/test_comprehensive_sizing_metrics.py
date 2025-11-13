#!/usr/bin/env python3
"""
Comprehensive sizing strategy test with better metrics.

Uses Calmar Ratio and Squantified Sharpe to balance risk vs growth.

Asset types:
- Crypto (BTCUSD, ETHUSD): Long only, no leverage
- Stocks (NVDA, MSFT, GOOG, AAPL): Can short, up to 4x intraday leverage

Metrics:
- Calmar Ratio = CAGR / Max Drawdown
- Squantified Sharpe = 0.8 * Sharpe + 0.2 * CAGR
- Sortino Ratio = Return / Downside Volatility

Usage:
    python experiments/test_comprehensive_sizing_metrics.py
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
class ComprehensiveResult:
    """Results with better metrics."""
    strategy_name: str
    leverage_multiplier: float
    asset_mix: str  # "stocks_only", "crypto_only", "mixed"

    # Returns
    final_equity: float
    total_return_pct: float
    cagr: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    volatility: float
    downside_volatility: float

    # Combined metrics
    calmar_ratio: float
    squantified_sharpe: float

    # Other
    num_trades: int
    avg_position_size: float
    total_interest_paid: float
    avg_overnight_leverage: float


class ComprehensiveBacktester:
    """
    Backtester supporting both crypto (long only, no leverage)
    and stocks (can short, can leverage).
    """

    def __init__(
        self,
        crypto_symbols: List[str],
        stock_symbols: List[str],
        initial_cash: float = 100000,
        days: int = 10,
        corr_data: Dict = None,
        max_intraday_leverage: float = 4.0,
        max_overnight_leverage: float = 2.0,
        annual_interest_rate: float = 0.065,
    ):
        self.crypto_symbols = crypto_symbols
        self.stock_symbols = stock_symbols
        self.all_symbols = crypto_symbols + stock_symbols
        self.initial_cash = initial_cash
        self.days = days
        self.corr_data = corr_data
        self.max_intraday_leverage = max_intraday_leverage
        self.max_overnight_leverage = max_overnight_leverage
        self.daily_interest_rate = annual_interest_rate / 365

        # Generate synthetic data
        self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """Generate price data for both crypto and stocks."""
        np.random.seed(42)

        self.prices = {}
        self.forecasts = {}

        for symbol in self.all_symbols:
            is_crypto = symbol.endswith('USD')

            # Get volatility
            if self.corr_data and symbol in self.corr_data.get('volatility_metrics', {}):
                vol_metrics = self.corr_data['volatility_metrics'][symbol]
                annual_vol = vol_metrics['annualized_volatility']
            else:
                annual_vol = 0.50 if is_crypto else 0.25

            daily_vol = annual_vol / np.sqrt(252)

            # Generate price path
            drift = 0.0008 if is_crypto else 0.0005
            returns = np.random.normal(drift, daily_vol, self.days + 1)
            start_price = 2000.0 if is_crypto else 150.0
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
        asset_mix: str = "mixed",
    ) -> ComprehensiveResult:
        """
        Run strategy on specified asset mix.

        Args:
            strategy: Base sizing strategy
            strategy_name: Name for reporting
            leverage_multiplier: Multiplier for stocks only (1x-4x)
            asset_mix: "stocks_only", "crypto_only", or "mixed"

        Returns:
            ComprehensiveResult with all metrics
        """
        # Determine which symbols to trade
        if asset_mix == "stocks_only":
            active_symbols = self.stock_symbols
        elif asset_mix == "crypto_only":
            active_symbols = self.crypto_symbols
        else:  # mixed
            active_symbols = self.all_symbols

        cash = self.initial_cash
        positions = {sym: 0.0 for sym in active_symbols}
        equity_history = [cash]
        position_sizes = []
        num_trades = 0
        total_interest_paid = 0.0
        overnight_leverages = []

        for day in range(self.days):
            # Start of day equity
            equity = cash
            for sym, qty in positions.items():
                equity += qty * self.prices[sym][day]

            # Charge overnight interest (stocks only)
            if day > 0:
                stock_position_value = sum(
                    abs(qty * self.prices[sym][day])
                    for sym, qty in positions.items()
                    if sym in self.stock_symbols
                )
                overnight_leverage = stock_position_value / equity if equity > 0 else 0
                overnight_leverages.append(overnight_leverage)

                if overnight_leverage > 1.0:
                    borrowed_amount = (overnight_leverage - 1.0) * equity
                    daily_interest = borrowed_amount * self.daily_interest_rate
                    cash -= daily_interest
                    total_interest_paid += daily_interest
                    equity -= daily_interest

            # Build market contexts
            contexts = {}
            for sym in active_symbols:
                is_crypto = sym in self.crypto_symbols
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
            for sym in active_symbols:
                ctx = contexts[sym]
                is_crypto = sym in self.crypto_symbols

                try:
                    sizing = strategy.calculate_size(ctx)
                    base_fraction = sizing.position_fraction
                except Exception as e:
                    base_fraction = 0.25

                # Apply leverage (stocks only)
                if is_crypto:
                    target_fraction = base_fraction  # No leverage for crypto
                else:
                    target_fraction = base_fraction * leverage_multiplier

                # Crypto: Long only (no negative positions)
                if is_crypto and target_fraction < 0:
                    target_fraction = 0

                # Calculate target quantity
                target_value = target_fraction * equity
                target_qty = target_value / ctx.current_price

                # Execute trade
                if target_qty != positions[sym]:
                    trade_value = (target_qty - positions[sym]) * ctx.current_price
                    cash -= trade_value
                    positions[sym] = target_qty
                    num_trades += 1
                    position_sizes.append(abs(target_fraction))

            # Hold intraday, capture EOD prices
            eod_equity = cash
            for sym, qty in positions.items():
                eod_price_idx = min(day + 1, len(self.prices[sym]) - 1)
                eod_equity += qty * self.prices[sym][eod_price_idx]

            equity_history.append(eod_equity)

            # EOD: Enforce overnight leverage limit for stocks
            eod_price_idx = min(day + 1, len(self.prices[sym]) - 1)
            stock_position_value = sum(
                abs(qty * self.prices[sym][eod_price_idx])
                for sym, qty in positions.items()
                if sym in self.stock_symbols
            )
            current_leverage = stock_position_value / eod_equity if eod_equity > 0 else 0

            if current_leverage > self.max_overnight_leverage:
                scale_factor = self.max_overnight_leverage / current_leverage

                for sym in self.stock_symbols:
                    if sym in positions and positions[sym] != 0:
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

        # CAGR (annualized)
        years = self.days / 252
        cagr = (final_equity / self.initial_cash) ** (1 / years) - 1

        # Sharpe and Sortino
        if len(equity_history) > 1:
            returns = np.diff(equity_history) / equity_history[:-1]
            mean_return = np.mean(returns)
            volatility = np.std(returns) * np.sqrt(252)
            sharpe = mean_return / (np.std(returns) + 1e-10) * np.sqrt(252)

            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_volatility = np.std(downside_returns) * np.sqrt(252)
                sortino = mean_return / (np.std(downside_returns) + 1e-10) * np.sqrt(252)
            else:
                downside_volatility = 0.0
                sortino = sharpe
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

        # Calmar Ratio = CAGR / Max Drawdown
        if max_dd > 0:
            calmar = cagr / max_dd
        else:
            calmar = cagr * 1000  # High value if no drawdown

        # Squantified Sharpe = 0.8 * Sharpe + 0.2 * CAGR
        squantified_sharpe = 0.8 * sharpe + 0.2 * cagr

        avg_position_size = np.mean(position_sizes) if position_sizes else 0.0
        avg_overnight_leverage = np.mean(overnight_leverages) if overnight_leverages else 0.0

        return ComprehensiveResult(
            strategy_name=strategy_name,
            leverage_multiplier=leverage_multiplier,
            asset_mix=asset_mix,
            final_equity=final_equity,
            total_return_pct=total_return_pct,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_dd,
            volatility=volatility,
            downside_volatility=downside_volatility,
            calmar_ratio=calmar,
            squantified_sharpe=squantified_sharpe,
            num_trades=num_trades,
            avg_position_size=avg_position_size,
            total_interest_paid=total_interest_paid,
            avg_overnight_leverage=avg_overnight_leverage,
        )


def main():
    print("=" * 80)
    print("COMPREHENSIVE SIZING STRATEGY TEST")
    print("With Calmar Ratio & Squantified Sharpe")
    print("=" * 80)
    print()

    # Define symbols
    crypto_symbols = ['BTCUSD', 'ETHUSD']
    stock_symbols = ['NVDA', 'MSFT', 'GOOG', 'AAPL']

    print(f"Crypto (long only, no leverage): {', '.join(crypto_symbols)}")
    print(f"Stocks (can short, 4x leverage): {', '.join(stock_symbols)}")
    print(f"Simulation: 10 days, $100,000 initial")
    print()

    # Load correlation data
    print("Loading correlation data...")
    try:
        corr_data = load_correlation_matrix()
        print(f"✓ Loaded correlation matrix")
    except Exception as e:
        print(f"⚠️  Could not load: {e}")
        corr_data = None
    print()

    # Initialize backtester
    backtester = ComprehensiveBacktester(
        crypto_symbols=crypto_symbols,
        stock_symbols=stock_symbols,
        corr_data=corr_data,
    )

    # Strategies to test
    base_strategies = [
        (KellyStrategy(fraction=0.5, cap=1.0), "Kelly_50pct"),
        (KellyStrategy(fraction=0.75, cap=1.0), "Kelly_75pct"),
        (VolatilityAdjustedStrategy(corr_data=corr_data, target_vol_contribution=0.15), "VolAdjusted_15pct"),
        (FixedFractionStrategy(0.5), "Fixed_50pct"),
    ]

    # Asset mixes
    asset_mixes = ["stocks_only", "crypto_only", "mixed"]

    # Leverage levels (for stocks)
    leverage_levels = [1.0, 2.0, 3.0, 4.0]

    print("Running comprehensive tests...")
    print()

    results = []
    for strategy, name in base_strategies:
        for asset_mix in asset_mixes:
            if asset_mix == "crypto_only":
                # Crypto: no leverage
                result = backtester.run_strategy(strategy, name, leverage_multiplier=1.0, asset_mix=asset_mix)
                results.append(result)
                print(f"  ✓ {name} on {asset_mix}")
            else:
                # Stocks or mixed: test leverage levels
                for lev in leverage_levels:
                    result = backtester.run_strategy(strategy, name, leverage_multiplier=lev, asset_mix=asset_mix)
                    results.append(result)
                    print(f"  ✓ {name} @ {lev}x on {asset_mix}")

    print()
    print("=" * 80)
    print("TOP 10 BY SQUANTIFIED SHARPE (0.8*Sharpe + 0.2*CAGR)")
    print("=" * 80)
    print()

    # Sort by Squantified Sharpe
    results.sort(key=lambda x: x.squantified_sharpe, reverse=True)

    # Display top 10
    for i, r in enumerate(results[:10], 1):
        print(f"{i}. {r.strategy_name} @ {r.leverage_multiplier}x on {r.asset_mix}")
        print(f"   Squantified Sharpe: {r.squantified_sharpe:8.2f}  |  CAGR: {r.cagr:7.2%}  |  Sharpe: {r.sharpe_ratio:6.2f}")
        print(f"   Return: {r.total_return_pct:7.2%}  |  Max DD: {r.max_drawdown_pct:6.2%}  |  Calmar: {r.calmar_ratio:8.2f}")
        print()

    print("=" * 80)
    print("TOP 10 BY CALMAR RATIO (CAGR / Max Drawdown)")
    print("=" * 80)
    print()

    # Sort by Calmar
    calmar_sorted = sorted(results, key=lambda x: x.calmar_ratio, reverse=True)

    for i, r in enumerate(calmar_sorted[:10], 1):
        print(f"{i}. {r.strategy_name} @ {r.leverage_multiplier}x on {r.asset_mix}")
        print(f"   Calmar Ratio: {r.calmar_ratio:8.2f}  |  CAGR: {r.cagr:7.2%}  |  Max DD: {r.max_drawdown_pct:6.2%}")
        print(f"   Return: {r.total_return_pct:7.2%}  |  Sharpe: {r.sharpe_ratio:6.2f}  |  Sortino: {r.sortino_ratio:6.2f}")
        print()

    print("=" * 80)
    print("TOP 10 BY ABSOLUTE RETURN")
    print("=" * 80)
    print()

    # Sort by return
    return_sorted = sorted(results, key=lambda x: x.total_return_pct, reverse=True)

    for i, r in enumerate(return_sorted[:10], 1):
        print(f"{i}. {r.strategy_name} @ {r.leverage_multiplier}x on {r.asset_mix}")
        print(f"   Return: {r.total_return_pct:7.2%}  |  CAGR: {r.cagr:7.2%}")
        print(f"   Squantified Sharpe: {r.squantified_sharpe:8.2f}  |  Calmar: {r.calmar_ratio:8.2f}")
        print()

    # Save results
    output_file = Path("experiments/comprehensive_sizing_results.json")
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'crypto_symbols': crypto_symbols,
        'stock_symbols': stock_symbols,
        'days': backtester.days,
        'results': [
            {
                'strategy': r.strategy_name,
                'leverage': r.leverage_multiplier,
                'asset_mix': r.asset_mix,
                'return_pct': r.total_return_pct,
                'cagr': r.cagr,
                'sharpe': r.sharpe_ratio,
                'sortino': r.sortino_ratio,
                'calmar': r.calmar_ratio,
                'squantified_sharpe': r.squantified_sharpe,
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

    # Final recommendations
    print("=" * 80)
    print("🏆 FINAL RECOMMENDATIONS")
    print("=" * 80)
    print()

    best_squan = results[0]  # Already sorted by squantified sharpe
    print(f"Best Overall (Squantified Sharpe): {best_squan.strategy_name} @ {best_squan.leverage_multiplier}x on {best_squan.asset_mix}")
    print(f"  • Squantified Sharpe: {best_squan.squantified_sharpe:.2f}")
    print(f"  • CAGR: {best_squan.cagr:.2%}")
    print(f"  • Sharpe: {best_squan.sharpe_ratio:.2f}")
    print(f"  • Return: {best_squan.total_return_pct:.2%}")
    print(f"  • Max DD: {best_squan.max_drawdown_pct:.2%}")
    print(f"  • Calmar: {best_squan.calmar_ratio:.2f}")
    print()

    best_calmar = calmar_sorted[0]
    if best_calmar != best_squan:
        print(f"Best Risk-Adjusted (Calmar): {best_calmar.strategy_name} @ {best_calmar.leverage_multiplier}x on {best_calmar.asset_mix}")
        print(f"  • Calmar Ratio: {best_calmar.calmar_ratio:.2f}")
        print(f"  • CAGR: {best_calmar.cagr:.2%}")
        print(f"  • Max DD: {best_calmar.max_drawdown_pct:.2%}")
        print()

    best_return = return_sorted[0]
    print(f"Highest Return: {best_return.strategy_name} @ {best_return.leverage_multiplier}x on {best_return.asset_mix}")
    print(f"  • Return: {best_return.total_return_pct:.2%}")
    print(f"  • CAGR: {best_return.cagr:.2%}")
    print(f"  • Squantified Sharpe: {best_return.squantified_sharpe:.2f}")
    print()

    print("=" * 80)


if __name__ == "__main__":
    main()
