#!/usr/bin/env python3
"""
Example: Risk-adjusted position sizing using correlation and volatility.

This demonstrates how to combine correlation and volatility metrics
for smarter position sizing decisions.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from trainingdata.load_correlation_utils import (
    load_correlation_matrix,
    get_correlation,
    get_cluster_for_symbol,
    get_volatility_metrics,
    get_top_volatile_symbols,
    get_best_sharpe_symbols,
)


def calculate_risk_adjusted_size(
    symbol: str,
    base_notional: float,
    corr_data: dict,
    target_volatility: float = 0.15,  # 15% target vol
) -> dict:
    """
    Calculate risk-adjusted position size based on volatility.

    Args:
        symbol: Symbol to size
        base_notional: Base position size
        corr_data: Correlation matrix data
        target_volatility: Target portfolio contribution volatility

    Returns:
        Dict with sizing recommendation
    """
    vol_metrics = get_volatility_metrics(corr_data, symbol)

    # Volatility scaling
    symbol_vol = vol_metrics["annualized_volatility"]
    vol_scalar = target_volatility / symbol_vol if symbol_vol > 0 else 1.0

    # Cap scaling to reasonable range (0.5x to 2x)
    vol_scalar = max(0.5, min(2.0, vol_scalar))

    adjusted_notional = base_notional * vol_scalar

    return {
        "base_notional": base_notional,
        "adjusted_notional": adjusted_notional,
        "vol_scalar": vol_scalar,
        "symbol_volatility": symbol_vol,
        "sharpe_ratio": vol_metrics["sharpe_ratio"],
        "max_drawdown": vol_metrics["max_drawdown"],
    }


def main():
    print("=" * 80)
    print("RISK-ADJUSTED POSITION SIZING")
    print("=" * 80)
    print()

    # Load data
    print("Loading correlation and volatility data...")
    corr_data = load_correlation_matrix()
    print(f"✓ Loaded data for {len(corr_data['symbols'])} symbols")
    print()

    # Example 1: Compare volatility across asset classes
    print("1. VOLATILITY COMPARISON ACROSS ASSETS")
    print("-" * 80)

    test_symbols = ['BTCUSD', 'ETHUSD', 'AAPL', 'SPY', 'GLD', 'XOM']
    vol_comparison = []

    for symbol in test_symbols:
        try:
            metrics = get_volatility_metrics(corr_data, symbol)
            vol_comparison.append({
                'Symbol': symbol,
                'Ann. Vol': f"{metrics['annualized_volatility']:.1%}",
                'Sharpe': f"{metrics['sharpe_ratio']:.2f}",
                'Max DD': f"{metrics['max_drawdown']:.1%}",
                'VaR(95%)': f"{metrics['var_95']:.2%}",
            })
        except Exception as e:
            print(f"  Warning: {symbol} - {e}")

    if vol_comparison:
        df = pd.DataFrame(vol_comparison)
        print(df.to_string(index=False))
    print()

    # Example 2: Risk-adjusted position sizing
    print("2. RISK-ADJUSTED POSITION SIZING")
    print("-" * 80)
    print("Scenario: Want to allocate $50k across different assets")
    print("Target: Each position contributes 15% volatility to portfolio")
    print()

    base_size = 50000
    candidates = ['BTCUSD', 'AAPL', 'SPY', 'GLD']

    sizing_results = []
    for symbol in candidates:
        try:
            result = calculate_risk_adjusted_size(symbol, base_size, corr_data)
            sizing_results.append({
                'Symbol': symbol,
                'Base Size': f"${result['base_notional']:,.0f}",
                'Adjusted Size': f"${result['adjusted_notional']:,.0f}",
                'Vol Scalar': f"{result['vol_scalar']:.2f}x",
                'Symbol Vol': f"{result['symbol_volatility']:.1%}",
                'Sharpe': f"{result['sharpe_ratio']:.2f}",
            })
        except Exception as e:
            print(f"  Warning: {symbol} - {e}")

    if sizing_results:
        df = pd.DataFrame(sizing_results)
        print(df.to_string(index=False))
        print()
        print("Interpretation:")
        print("  - Higher volatility assets get smaller positions (vol scalar < 1)")
        print("  - Lower volatility assets get larger positions (vol scalar > 1)")
        print("  - This normalizes risk contribution across positions")
    print()

    # Example 3: Top opportunities by Sharpe ratio
    print("3. TOP RISK-ADJUSTED OPPORTUNITIES")
    print("-" * 80)
    print("Best Sharpe ratios (return per unit of risk):")
    print()

    best_sharpe = get_best_sharpe_symbols(corr_data, n=10)
    for i, (symbol, sharpe) in enumerate(best_sharpe.items(), 1):
        try:
            metrics = get_volatility_metrics(corr_data, symbol)
            cluster = get_cluster_for_symbol(corr_data, symbol)
            cluster_name = cluster['label'] if cluster else 'Uncorrelated'

            print(f"  {i:2d}. {symbol:12s} Sharpe: {sharpe:5.2f}  "
                  f"Vol: {metrics['annualized_volatility']:5.1%}  "
                  f"Return: {metrics['mean_return_annualized']:6.1%}  "
                  f"({cluster_name})")
        except Exception as e:
            print(f"  {i:2d}. {symbol:12s} Sharpe: {sharpe:5.2f}  (Error: {e})")
    print()

    # Example 4: Volatility-correlation matrix
    print("4. HIGH SHARPE + LOW CORRELATION COMBINATIONS")
    print("-" * 80)
    print("Finding pairs with good Sharpe AND low correlation for diversification:")
    print()

    # Get top Sharpe symbols
    top_sharpe_symbols = list(best_sharpe.head(20).index)

    # Find low-correlation pairs
    low_corr_pairs = []
    for i, sym1 in enumerate(top_sharpe_symbols):
        for sym2 in top_sharpe_symbols[i+1:]:
            try:
                corr = get_correlation(corr_data, sym1, sym2)
                if pd.notna(corr) and abs(corr) < 0.3:  # Low correlation
                    sharpe1 = best_sharpe[sym1]
                    sharpe2 = best_sharpe[sym2]
                    avg_sharpe = (sharpe1 + sharpe2) / 2
                    low_corr_pairs.append({
                        'Pair': f"{sym1}-{sym2}",
                        'Correlation': corr,
                        'Avg Sharpe': avg_sharpe,
                        'Sharpe1': sharpe1,
                        'Sharpe2': sharpe2,
                    })
            except Exception:
                continue

    if low_corr_pairs:
        # Sort by average Sharpe
        low_corr_pairs.sort(key=lambda x: x['Avg Sharpe'], reverse=True)

        print("Top 5 diversifying pairs (low correlation + high Sharpe):")
        for pair in low_corr_pairs[:5]:
            print(f"  {pair['Pair']:25s} Corr: {pair['Correlation']:+.3f}  "
                  f"Avg Sharpe: {pair['Avg Sharpe']:.2f}")
        print()
        print("These pairs offer good returns with natural diversification!")
    else:
        print("  No low-correlation pairs found among top Sharpe symbols")
    print()

    # Example 5: Volatility regimes
    print("5. CURRENT VOLATILITY REGIMES")
    print("-" * 80)
    print("Symbols with elevated volatility (>80th percentile vs historical):")
    print()

    elevated_vol = []
    for symbol in corr_data['symbols']:
        try:
            metrics = get_volatility_metrics(corr_data, symbol)
            if metrics.get('volatility_percentile') and metrics['volatility_percentile'] > 80:
                elevated_vol.append({
                    'Symbol': symbol,
                    'Percentile': f"{metrics['volatility_percentile']:.0f}%",
                    'Current Vol': f"{metrics['rolling_vol_30d']:.1%}" if metrics['rolling_vol_30d'] else 'N/A',
                    'Avg Vol': f"{metrics['annualized_volatility']:.1%}",
                })
        except Exception:
            continue

    if elevated_vol:
        # Sort by percentile
        elevated_vol.sort(key=lambda x: float(x['Percentile'].rstrip('%')), reverse=True)
        df = pd.DataFrame(elevated_vol[:10])
        print(df.to_string(index=False))
        print()
        print("⚠️  These assets are experiencing higher-than-normal volatility")
        print("   Consider reducing position sizes or tightening stops")
    else:
        print("  No symbols with elevated volatility detected")
    print()

    # Example 6: Risk budgeting
    print("6. RISK BUDGETING EXAMPLE")
    print("-" * 80)
    print("Allocating $500k across crypto, stocks, and commodities")
    print("Goal: Equal risk contribution from each asset class")
    print()

    portfolio = {
        'Crypto': ['BTCUSD', 'ETHUSD'],
        'Stocks': ['AAPL', 'MSFT', 'SPY'],
        'Commodities': ['GLD', 'USO'],
    }

    total_capital = 500000
    target_vol_per_class = 0.10  # 10% vol per asset class

    allocations = {}
    for asset_class, symbols in portfolio.items():
        class_allocations = []
        class_total = 0

        # Calculate average volatility for the class
        vols = []
        for symbol in symbols:
            try:
                metrics = get_volatility_metrics(corr_data, symbol)
                vols.append(metrics['annualized_volatility'])
            except Exception:
                continue

        if not vols:
            continue

        avg_vol = sum(vols) / len(vols)

        # Allocate capital to hit target volatility
        class_capital = (target_vol_per_class / avg_vol) * total_capital
        class_capital = min(class_capital, total_capital * 0.4)  # Cap at 40%

        # Distribute within class (equal weight for simplicity)
        per_symbol = class_capital / len(symbols)

        for symbol in symbols:
            try:
                metrics = get_volatility_metrics(corr_data, symbol)
                # Adjust individual positions by relative volatility
                vol_adjust = avg_vol / metrics['annualized_volatility']
                allocation = per_symbol * vol_adjust
                class_allocations.append({
                    'Symbol': symbol,
                    'Allocation': allocation,
                    'Volatility': metrics['annualized_volatility'],
                })
                class_total += allocation
            except Exception:
                continue

        allocations[asset_class] = {
            'symbols': class_allocations,
            'total': class_total,
            'target_vol': target_vol_per_class,
            'avg_vol': avg_vol,
        }

    for asset_class, data in allocations.items():
        print(f"{asset_class}:")
        print(f"  Target volatility: {data['target_vol']:.1%}")
        print(f"  Average asset vol: {data['avg_vol']:.1%}")
        print(f"  Total allocation: ${data['total']:,.0f} ({data['total']/total_capital:.1%})")
        print(f"  Individual positions:")
        for item in data['symbols']:
            print(f"    {item['Symbol']:10s} ${item['Allocation']:>10,.0f}  "
                  f"(vol: {item['Volatility']:.1%})")
        print()

    total_allocated = sum(d['total'] for d in allocations.values())
    print(f"Total allocated: ${total_allocated:,.0f} / ${total_capital:,.0f} "
          f"({total_allocated/total_capital:.1%})")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Key takeaways:")
    print("  1. High-volatility assets (crypto) need smaller positions")
    print("  2. Low-volatility assets (bonds/gold) can support larger positions")
    print("  3. Combine high Sharpe with low correlation for optimal portfolios")
    print("  4. Monitor volatility regimes and adjust sizing accordingly")
    print("  5. Risk budgeting ensures balanced exposure across asset classes")
    print("=" * 80)


if __name__ == "__main__":
    main()
