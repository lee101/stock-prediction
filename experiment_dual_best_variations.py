#!/usr/bin/env python3
"""
Experiment: Dual Best Strategy Variations

Based on our findings that dual_best (2 positions) performed best with 27.03% return,
let's test variations to optimize it further:

1. Different position sizes around 47%
2. Different rebalancing frequencies  
3. Minimum return thresholds
4. Position sizing methods
"""

from portfolio_simulation_system import PortfolioSimulation, AllocationStrategy
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

def test_dual_best_variations():
    """Test systematic variations of the dual_best strategy"""
    
    simulation = PortfolioSimulation(initial_cash=100000.0)
    
    # Test variations of dual_best strategy
    strategies = []
    
    # 1. Position size variations around 47%
    position_sizes = [0.40, 0.44, 0.47, 0.50, 0.53]
    for size in position_sizes:
        strategies.append(AllocationStrategy(
            f"dual_pos{int(size*100)}", 
            max_positions=2, 
            max_position_size=size,
            rebalance_threshold=0.1
        ))
    
    # 2. Position count variations around 2
    position_counts = [(1, 0.95), (2, 0.47), (3, 0.32)]
    for count, size in position_counts:
        strategies.append(AllocationStrategy(
            f"positions_{count}_refined", 
            max_positions=count, 
            max_position_size=size,
            rebalance_threshold=0.05  # Tighter rebalancing
        ))
    
    # 3. Rebalancing threshold variations
    rebalance_thresholds = [0.05, 0.10, 0.15, 0.20]
    for threshold in rebalance_thresholds:
        strategies.append(AllocationStrategy(
            f"dual_rebal{int(threshold*100)}", 
            max_positions=2, 
            max_position_size=0.47,
            rebalance_threshold=threshold
        ))
    
    # 4. Conservative vs Aggressive variations
    strategies.extend([
        AllocationStrategy("dual_conservative", max_positions=2, max_position_size=0.40, rebalance_threshold=0.15),
        AllocationStrategy("dual_moderate", max_positions=2, max_position_size=0.47, rebalance_threshold=0.10),
        AllocationStrategy("dual_aggressive", max_positions=2, max_position_size=0.53, rebalance_threshold=0.05),
        AllocationStrategy("dual_ultra_aggressive", max_positions=2, max_position_size=0.60, rebalance_threshold=0.03),
    ])
    
    results = []
    
    print("Testing dual_best strategy variations...")
    print(f"Total strategies to test: {len(strategies)}")
    
    for i, strategy in enumerate(strategies):
        try:
            print(f"Testing {i+1}/{len(strategies)}: {strategy.name}")
            result = simulation.simulate_strategy(strategy, max_days=100)
            if result:
                results.append(result)
                print(f"  Result: {result['total_return']:.2%} return, {result['sharpe_ratio']:.3f} Sharpe")
            else:
                print(f"  No result for {strategy.name}")
        except Exception as e:
            print(f"  Strategy {strategy.name} failed: {e}")
    
    if not results:
        print("No results generated")
        return
    
    # Sort by total return
    results.sort(key=lambda x: x['total_return'], reverse=True)
    
    # Generate enhanced findings report
    report_content = f"""# Dual Best Strategy Variations - Experiment Results

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Strategies Tested:** {len(results)}  
**Focus:** Optimizing the dual_best strategy (2 positions)

## Executive Summary

The dual_best strategy showed the best performance in our initial tests with 27.03% return.
This experiment focuses on fine-tuning its parameters to maximize performance.

## Results Summary

### Top Performing Variations

"""
    
    for i, result in enumerate(results[:10]):  # Top 10
        report_content += f"""**#{i+1}: {result['strategy']}**
- **Total Return:** {result['total_return']:.2%}
- **Sharpe Ratio:** {result['sharpe_ratio']:.3f}
- **Max Drawdown:** {result['max_drawdown']:.2%}
- **Total Trades:** {result['total_trades']}
- **Win Rate:** {result.get('win_rate', 0):.1%}

"""
    
    # Analysis by parameter type
    best_result = results[0]
    
    # Position size analysis
    pos_size_results = [r for r in results if 'dual_pos' in r['strategy']]
    if pos_size_results:
        best_pos_size = max(pos_size_results, key=lambda x: x['total_return'])
        report_content += f"""## Position Size Analysis

**Best Position Size:** {best_pos_size['strategy']} with {best_pos_size['total_return']:.2%}

Position Size Performance:
"""
        for result in sorted(pos_size_results, key=lambda x: x['total_return'], reverse=True):
            size_pct = result['strategy'].replace('dual_pos', '')
            report_content += f"- {size_pct}%: {result['total_return']:.2%} return, {result['sharpe_ratio']:.3f} Sharpe\n"
    
    # Rebalancing analysis  
    rebal_results = [r for r in results if 'dual_rebal' in r['strategy']]
    if rebal_results:
        best_rebal = max(rebal_results, key=lambda x: x['total_return'])
        report_content += f"""
## Rebalancing Threshold Analysis

**Best Rebalancing:** {best_rebal['strategy']} with {best_rebal['total_return']:.2%}

Rebalancing Performance:
"""
        for result in sorted(rebal_results, key=lambda x: x['total_return'], reverse=True):
            threshold = result['strategy'].replace('dual_rebal', '')
            report_content += f"- {threshold}%: {result['total_return']:.2%} return, {result['sharpe_ratio']:.3f} Sharpe\n"
    
    # Risk profile analysis
    risk_results = [r for r in results if any(x in r['strategy'] for x in ['conservative', 'moderate', 'aggressive'])]
    if risk_results:
        report_content += f"""
## Risk Profile Analysis

"""
        for result in sorted(risk_results, key=lambda x: x['total_return'], reverse=True):
            report_content += f"**{result['strategy']}:** {result['total_return']:.2%} return, {result['max_drawdown']:.2%} drawdown\n"
    
    # Statistical analysis
    returns = [r['total_return'] for r in results]
    sharpe_ratios = [r['sharpe_ratio'] for r in results]
    
    report_content += f"""
## Statistical Summary

- **Mean Return:** {np.mean(returns):.2%}
- **Median Return:** {np.median(returns):.2%}
- **Return Std Dev:** {np.std(returns):.2%}
- **Best Return:** {max(returns):.2%}
- **Worst Return:** {min(returns):.2%}
- **Mean Sharpe:** {np.mean(sharpe_ratios):.3f}

## Key Insights

1. **Optimal Strategy:** {best_result['strategy']} achieved {best_result['total_return']:.2%}
2. **Performance Improvement:** {(best_result['total_return'] - 0.2703)*100:.2f}% vs original dual_best
3. **Consistency:** {len([r for r in results if r['total_return'] > 0.20])} strategies beat 20% return
4. **Risk Management:** Best max drawdown was {min(r['max_drawdown'] for r in results):.2%}

## Position Analysis

Top strategies are holding:
"""
    
    for result in results[:5]:
        positions = result.get('final_positions', {})
        active_positions = {k: v for k, v in positions.items() if v != 0}
        symbols = list(active_positions.keys())
        report_content += f"**{result['strategy']}:** {symbols}\n"
    
    # Recommendations for next experiment
    report_content += f"""

## Next Experiment Recommendations

Based on these results, the next experiment should focus on:

1. **Best Configuration:** Use {best_result['strategy']} as baseline for risk management tests
2. **Rebalancing Frequency:** Test different time-based rebalancing (daily, weekly, etc.)
3. **Risk Management:** Add stop-loss and take-profit to top 3 strategies
4. **Entry Filters:** Test minimum return thresholds and volatility filters
5. **Position Sizing:** Explore dynamic position sizing based on volatility or momentum

## Detailed Results

| Strategy | Return | Sharpe | Drawdown | Trades |
|----------|--------|--------|----------|---------|
"""
    
    for result in results:
        report_content += f"| {result['strategy']} | {result['total_return']:.2%} | {result['sharpe_ratio']:.3f} | {result['max_drawdown']:.2%} | {result['total_trades']} |\n"
    
    report_content += f"""
---
*Generated by experiment_dual_best_variations.py*
"""
    
    # Write report
    with open("findings.md", "w") as f:
        f.write(report_content)
    
    print(f"\nExperiment completed!")
    print(f"Strategies tested: {len(results)}")
    print(f"Best strategy: {best_result['strategy']} with {best_result['total_return']:.2%}")
    print(f"Results saved to findings.md")

if __name__ == "__main__":
    test_dual_best_variations()