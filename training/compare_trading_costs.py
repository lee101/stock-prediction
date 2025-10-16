#!/usr/bin/env python3
"""
Compare trading performance with realistic fees across different asset types
"""

import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from trading_config import get_trading_costs


def run_single_test(symbol, broker, episodes=30):
    """Run a single training test with specified parameters"""
    
    cmd = [
        'python', 'train_full_model.py',
        '--symbol', symbol,
        '--broker', broker,
        '--num_episodes', str(episodes),
        '--eval_interval', '10',
        '--update_interval', '5',
        '--initial_balance', '100000',
        '--patience', '20'
    ]
    
    print(f"\n🚀 Running: {symbol} on {broker}")
    print("-" * 40)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        # Parse output for key metrics
        output = result.stdout
        
        metrics = {}
        for line in output.split('\n'):
            if 'Final Balance:' in line:
                metrics['final_balance'] = float(line.split('$')[1].replace(',', ''))
            elif 'Total Profit/Loss:' in line:
                metrics['profit'] = float(line.split('$')[1].replace(',', ''))
            elif 'Total Fees Paid:' in line:
                metrics['fees'] = float(line.split('$')[1].replace(',', ''))
            elif 'ROI:' in line and 'roi_percent' not in metrics:
                metrics['roi'] = float(line.split(':')[1].strip().replace('%', ''))
            elif 'Total Return:' in line and '%' in line:
                metrics['return'] = float(line.split(':')[1].strip().replace('%', ''))
            elif 'Sharpe Ratio:' in line:
                metrics['sharpe'] = float(line.split(':')[1].strip())
            elif 'Max Drawdown:' in line:
                metrics['drawdown'] = float(line.split(':')[1].strip().replace('%', ''))
            elif 'Total Trades:' in line:
                metrics['trades'] = int(line.split(':')[1].strip())
            elif 'Trading Costs' in line:
                metrics['asset_type'] = 'CRYPTO' if 'CRYPTO' in line else 'STOCK'
        
        return metrics
        
    except subprocess.TimeoutExpired:
        print("  ⚠️ Training timeout")
        return None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def run_comparison_tests():
    """Run comprehensive comparison tests"""
    
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE TRADING COST COMPARISON")
    print("="*80)
    
    tests = [
        # Stock brokers (essentially free)
        {'symbol': 'STOCK', 'broker': 'alpaca', 'name': 'Alpaca (Stock)'},
        {'symbol': 'STOCK', 'broker': 'robinhood', 'name': 'Robinhood (Stock)'},
        {'symbol': 'STOCK', 'broker': 'td_ameritrade', 'name': 'TD Ameritrade (Stock)'},
        
        # Crypto exchanges
        {'symbol': 'CRYPTO', 'broker': 'binance', 'name': 'Binance (Crypto)'},
        {'symbol': 'CRYPTO', 'broker': 'default', 'name': 'Default Crypto (0.15%)'},
        {'symbol': 'CRYPTO', 'broker': 'coinbase', 'name': 'Coinbase (Crypto)'},
    ]
    
    results = []
    
    for test in tests:
        print(f"\n📊 Testing: {test['name']}")
        metrics = run_single_test(test['symbol'], test['broker'], episodes=30)
        
        if metrics:
            # Get cost structure
            asset_type = 'crypto' if 'Crypto' in test['name'] else 'stock'
            costs = get_trading_costs(asset_type, test['broker'])
            
            metrics['name'] = test['name']
            metrics['commission'] = costs.commission
            metrics['spread'] = costs.spread_pct
            metrics['slippage'] = costs.slippage_pct
            metrics['total_cost_pct'] = costs.commission + costs.spread_pct + costs.slippage_pct
            
            results.append(metrics)
            
            print(f"  ✅ ROI: {metrics.get('roi', 0):.2f}%")
            print(f"  💰 Fees: ${metrics.get('fees', 0):.2f}")
            print(f"  📈 Profit: ${metrics.get('profit', 0):.2f}")
    
    return results


def visualize_comparison(results):
    """Create comparison visualizations"""
    
    if not results:
        print("No results to visualize")
        return
    
    df = pd.DataFrame(results)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Trading Performance: Realistic Fee Comparison', fontsize=16, fontweight='bold')
    
    # 1. ROI Comparison
    ax1 = axes[0, 0]
    colors = ['green' if 'Stock' in name else 'orange' for name in df['name']]
    bars = ax1.bar(range(len(df)), df['roi'], color=colors, alpha=0.7)
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['name'], rotation=45, ha='right')
    ax1.set_ylabel('ROI (%)')
    ax1.set_title('Return on Investment')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, df['roi']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=8)
    
    # 2. Trading Fees
    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(df)), df['fees'], color=colors, alpha=0.7)
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df['name'], rotation=45, ha='right')
    ax2.set_ylabel('Total Fees ($)')
    ax2.set_title('Trading Fees Paid')
    ax2.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, df['fees']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${val:.0f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Net Profit
    ax3 = axes[0, 2]
    net_profit = df['profit']
    bars = ax3.bar(range(len(df)), net_profit, color=colors, alpha=0.7)
    ax3.set_xticks(range(len(df)))
    ax3.set_xticklabels(df['name'], rotation=45, ha='right')
    ax3.set_ylabel('Net Profit ($)')
    ax3.set_title('Net Profit After Fees')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    
    for bar, val in zip(bars, net_profit):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'${val:.0f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=8)
    
    # 4. Fee Structure Breakdown
    ax4 = axes[1, 0]
    width = 0.25
    x = np.arange(len(df))
    
    bars1 = ax4.bar(x - width, df['commission'] * 100, width, label='Commission', alpha=0.7)
    bars2 = ax4.bar(x, df['spread'] * 100, width, label='Spread', alpha=0.7)
    bars3 = ax4.bar(x + width, df['slippage'] * 100, width, label='Slippage', alpha=0.7)
    
    ax4.set_xlabel('Platform')
    ax4.set_ylabel('Cost (%)')
    ax4.set_title('Fee Structure Breakdown')
    ax4.set_xticks(x)
    ax4.set_xticklabels(df['name'], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Efficiency Ratio (Profit / Fees)
    ax5 = axes[1, 1]
    efficiency = df['profit'] / (df['fees'] + 1)  # Add 1 to avoid division by zero
    bars = ax5.bar(range(len(df)), efficiency, color=colors, alpha=0.7)
    ax5.set_xticks(range(len(df)))
    ax5.set_xticklabels(df['name'], rotation=45, ha='right')
    ax5.set_ylabel('Profit/Fee Ratio')
    ax5.set_title('Trading Efficiency')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=1, color='red', linestyle='--', alpha=0.3, label='Break-even')
    
    for bar, val in zip(bars, efficiency):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}x', ha='center', va='bottom' if val > 0 else 'top', fontsize=8)
    
    # 6. Summary Table
    ax6 = axes[1, 2]
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create summary statistics
    stock_results = df[df['name'].str.contains('Stock')]
    crypto_results = df[~df['name'].str.contains('Stock')]
    
    summary_data = [
        ['', 'Stocks', 'Crypto'],
        ['Avg ROI', f"{stock_results['roi'].mean():.2f}%", f"{crypto_results['roi'].mean():.2f}%"],
        ['Avg Fees', f"${stock_results['fees'].mean():.2f}", f"${crypto_results['fees'].mean():.2f}"],
        ['Avg Profit', f"${stock_results['profit'].mean():.2f}", f"${crypto_results['profit'].mean():.2f}"],
        ['Fee/Trade', f"{stock_results['total_cost_pct'].mean():.4%}", f"{crypto_results['total_cost_pct'].mean():.4%}"],
    ]
    
    table = ax6.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style the header row
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code the cells
    for i in range(1, 5):
        table[(i, 1)].set_facecolor('#e8f5e9')  # Light green for stocks
        table[(i, 2)].set_facecolor('#fff3e0')  # Light orange for crypto
    
    ax6.set_title('Summary Statistics', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'results/fee_comparison_{timestamp}.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\n📊 Comparison chart saved to: {save_path}")
    
    # Also save raw data
    csv_path = f'results/fee_comparison_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    print(f"📁 Raw data saved to: {csv_path}")
    
    plt.show()
    
    return df


def print_summary(results):
    """Print summary of results"""
    
    if not results:
        return
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("📊 TRADING COST IMPACT SUMMARY")
    print("="*80)
    
    # Stock vs Crypto comparison
    stock_df = df[df['name'].str.contains('Stock')]
    crypto_df = df[~df['name'].str.contains('Stock')]
    
    print("\n🏦 STOCK TRADING (Near-Zero Fees):")
    print("-" * 40)
    print(f"  Average ROI:      {stock_df['roi'].mean():.2f}%")
    print(f"  Average Fees:     ${stock_df['fees'].mean():.2f}")
    print(f"  Average Profit:   ${stock_df['profit'].mean():.2f}")
    print(f"  Fees per $100k:   ${stock_df['fees'].mean():.2f}")
    
    print("\n💰 CRYPTO TRADING (Higher Fees):")
    print("-" * 40)
    print(f"  Average ROI:      {crypto_df['roi'].mean():.2f}%")
    print(f"  Average Fees:     ${crypto_df['fees'].mean():.2f}")
    print(f"  Average Profit:   ${crypto_df['profit'].mean():.2f}")
    print(f"  Fees per $100k:   ${crypto_df['fees'].mean():.2f}")
    
    print("\n🎯 KEY FINDINGS:")
    print("-" * 40)
    
    fee_impact = (crypto_df['fees'].mean() - stock_df['fees'].mean())
    profit_diff = stock_df['profit'].mean() - crypto_df['profit'].mean()
    
    print(f"• Crypto fees are {crypto_df['fees'].mean() / (stock_df['fees'].mean() + 0.01):.1f}x higher than stocks")
    print(f"• Extra crypto fees cost: ${fee_impact:.2f} per $100k traded")
    print(f"• Profit difference: ${profit_diff:.2f} in favor of stocks")
    print(f"• Stock trading is {(stock_df['roi'].mean() / (crypto_df['roi'].mean() + 0.01) - 1) * 100:.0f}% more profitable due to lower fees")
    
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 40)
    print("• For HIGH FREQUENCY trading: Use stocks (near-zero fees)")
    print("• For CRYPTO trading: Minimize trade frequency")
    print("• Use limit orders to reduce spread costs")
    print("• Consider fee-reduction programs (BNB on Binance, etc.)")
    
    print("="*80)


if __name__ == '__main__':
    print("Starting comprehensive fee comparison...")
    
    # Ensure results directory exists
    Path('results').mkdir(exist_ok=True)
    
    # Run comparison tests
    results = run_comparison_tests()
    
    if results:
        # Visualize results
        df = visualize_comparison(results)
        
        # Print summary
        print_summary(results)
    else:
        print("\n❌ No successful test results to compare")