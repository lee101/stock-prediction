#!/usr/bin/env python3
"""
Example: Using the correlation matrix for risk management.

This demonstrates how to load and use the correlation matrix
to assess portfolio risk and make position sizing decisions.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trainingdata.load_correlation_utils import (
    load_correlation_matrix,
    get_correlation,
    get_symbol_correlations,
    get_cluster_for_symbol,
    get_cluster_exposure,
    get_portfolio_diversification_metrics,
    print_correlation_summary,
)


def main():
    print("=" * 70)
    print("CORRELATION MATRIX USAGE EXAMPLES")
    print("=" * 70)
    print()

    # Load the correlation matrix
    print("1. Loading correlation matrix...")
    corr_data = load_correlation_matrix("trainingdata/correlation_matrix.pkl")
    print(f"   ✓ Loaded {len(corr_data['symbols'])} symbols")
    print(f"   ✓ Identified {len(corr_data['clusters'])} correlation clusters")
    print()

    # Example 1: Check correlation between two symbols
    print("2. Get correlation between specific symbols")
    print("-" * 70)
    try:
        pairs = [
            ('BTCUSD', 'ETHUSD'),
            ('AAPL', 'MSFT'),
            ('SPY', 'QQQ'),
        ]
        for sym1, sym2 in pairs:
            corr = get_correlation(corr_data, sym1, sym2)
            print(f"   {sym1} ↔ {sym2}: {corr:.3f}")
    except Exception as e:
        print(f"   Note: Some symbols may not have overlapping data: {e}")
    print()

    # Example 2: Find most correlated symbols
    print("3. Find most correlated symbols with BTCUSD")
    print("-" * 70)
    try:
        corrs = get_symbol_correlations(corr_data, 'BTCUSD', top_n=10)
        for symbol, corr in corrs.items():
            if not pd.isna(corr):
                print(f"   {symbol}: {corr:+.3f}")
    except Exception as e:
        print(f"   Error: {e}")
    print()

    # Example 3: Get cluster information
    print("4. Find correlation cluster for symbols")
    print("-" * 70)
    test_symbols = ['BTCUSD', 'AAPL', 'XOM', 'SPY']
    for symbol in test_symbols:
        try:
            cluster = get_cluster_for_symbol(corr_data, symbol)
            if cluster:
                print(f"   {symbol}: {cluster['label']} ({cluster['size']} symbols, "
                      f"avg corr={cluster['avg_correlation']:.3f})")
            else:
                print(f"   {symbol}: Not in any cluster (uncorrelated)")
        except Exception as e:
            print(f"   {symbol}: Error - {e}")
    print()

    # Example 4: Calculate portfolio diversification
    print("5. Assess portfolio diversification")
    print("-" * 70)

    # Example portfolio
    portfolio = {
        'BTCUSD': 50000,  # $50k in Bitcoin
        'ETHUSD': 30000,  # $30k in Ethereum
        'AAPL': 40000,    # $40k in Apple
        'GLD': 20000,     # $20k in Gold
    }

    print("   Portfolio:")
    for symbol, value in portfolio.items():
        print(f"     {symbol}: ${value:,}")
    print()

    try:
        metrics = get_portfolio_diversification_metrics(corr_data, portfolio)
        print("   Diversification Metrics:")
        print(f"     Average pairwise correlation: {metrics['avg_pairwise_correlation']:.3f}")
        print(f"     Max correlation: {metrics['max_correlation']:.3f}")
        print(f"     Effective number of bets: {metrics['effective_num_bets']:.2f}")
        print(f"     Concentration score: {metrics['concentration_score']:.3f}")
        print()

        # Interpretation
        if metrics['effective_num_bets'] < 3:
            print("   ⚠️  Portfolio is highly concentrated (ENB < 3)")
        elif metrics['effective_num_bets'] < 5:
            print("   ⚠️  Portfolio has moderate diversification (ENB < 5)")
        else:
            print("   ✓ Portfolio is well diversified (ENB >= 5)")
    except Exception as e:
        print(f"   Error calculating metrics: {e}")
    print()

    # Example 5: Check cluster exposure
    print("6. Calculate exposure per correlation cluster")
    print("-" * 70)
    try:
        cluster_exp = get_cluster_exposure(corr_data, portfolio)
        print("   Cluster Exposures:")
        for cluster_id, exposure in sorted(cluster_exp.items(), key=lambda x: x[1], reverse=True):
            if exposure > 0:
                cluster_info = corr_data['clusters'][cluster_id]
                print(f"     {cluster_info['label']}: ${exposure:,.0f} "
                      f"({exposure / sum(portfolio.values()) * 100:.1f}%)")
    except Exception as e:
        print(f"   Error: {e}")
    print()

    # Example 6: Risk-aware position sizing
    print("7. Risk-aware position sizing example")
    print("-" * 70)
    print("   Scenario: Want to add SOLUSD to portfolio")
    print()

    new_symbol = 'SOLUSD'
    try:
        # Check which cluster it belongs to
        cluster = get_cluster_for_symbol(corr_data, new_symbol)
        if cluster:
            print(f"   {new_symbol} is in: {cluster['label']}")

            # Calculate current exposure to that cluster
            cluster_exp = get_cluster_exposure(corr_data, portfolio, cluster['cluster_id'])
            current_exposure = cluster_exp.get(cluster['cluster_id'], 0)

            print(f"   Current exposure to {cluster['label']}: ${current_exposure:,.0f}")

            # Risk limit: max 120% of portfolio in highly correlated group
            total_value = sum(portfolio.values())
            cluster_limit = total_value * 1.2  # 120%

            print(f"   Cluster limit (120% of portfolio): ${cluster_limit:,.0f}")
            print(f"   Remaining capacity: ${cluster_limit - current_exposure:,.0f}")

            if current_exposure < cluster_limit:
                print(f"   ✓ Safe to add {new_symbol} (within cluster limits)")
            else:
                print(f"   ⚠️  Adding {new_symbol} would exceed cluster limit")
        else:
            print(f"   ✓ {new_symbol} is uncorrelated - safe to add")
    except Exception as e:
        print(f"   Error: {e}")

    print()
    print("=" * 70)
    print("For more details, see:")
    print("  - trainingdata/correlation_matrix.json (human-readable)")
    print("  - trainingdata/correlation_matrix.csv (matrix in spreadsheet format)")
    print("  - docs/correlation_risk_management_plan.md (full documentation)")
    print("=" * 70)


if __name__ == "__main__":
    import pandas as pd
    main()
