#!/usr/bin/env python3
"""
Quick comparison of trading with realistic fees
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append('..')

from trading_agent import TradingAgent
from trading_env import DailyTradingEnv
from ppo_trainer import PPOTrainer
from trading_config import get_trading_costs
from train_full_model import generate_synthetic_data, add_technical_indicators


def simulate_trading(asset_type='stock', broker='default', episodes=20):
    """Quick simulation with specific broker"""
    
    # Generate data - this returns capitalized columns already
    df = generate_synthetic_data(500)
    
    # Get costs
    costs = get_trading_costs(asset_type, broker)
    
    # Setup environment
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 
                'Rsi', 'Macd', 'Bb_Position', 'Volume_Ratio']
    available_features = [f for f in features if f in df.columns]
    
    env = DailyTradingEnv(
        df,
        window_size=30,
        initial_balance=100000,
        transaction_cost=costs.commission,
        spread_pct=costs.spread_pct,
        slippage_pct=costs.slippage_pct,
        min_commission=costs.min_commission,
        features=available_features
    )
    
    # Create simple agent
    input_dim = 30 * (len(available_features) + 3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    agent = TradingAgent(
        backbone_model=torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 768),
            torch.nn.ReLU()
        ),
        hidden_dim=768
    ).to(device)
    
    # Quick training
    trainer = PPOTrainer(agent, log_dir='./traininglogs_temp', device=device)
    
    for ep in range(episodes):
        trainer.train_episode(env)
        if (ep + 1) % 5 == 0:
            trainer.update()
    
    # Final evaluation
    env.reset()
    state = env.reset()
    done = False
    
    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, _, _ = agent.act(state_tensor, deterministic=True)
            action = action.cpu().numpy().flatten()
        state, _, done, _ = env.step(action)
    
    metrics = env.get_metrics()
    
    # Calculate total fees
    total_fees = sum([
        max(costs.commission * abs(t['new_position'] - t['old_position']) * t['balance'], 
            costs.min_commission) +
        costs.spread_pct * abs(t['new_position'] - t['old_position']) * t['balance'] +
        costs.slippage_pct * abs(t['new_position'] - t['old_position']) * t['balance']
        for t in env.trades
    ])
    
    trainer.close()
    
    return {
        'asset_type': asset_type,
        'broker': broker,
        'initial_balance': env.initial_balance,
        'final_balance': env.balance,
        'profit': env.balance - env.initial_balance,
        'fees': total_fees,
        'roi': (env.balance / env.initial_balance - 1) * 100,
        'trades': metrics['num_trades'],
        'sharpe': metrics['sharpe_ratio'],
        'commission': costs.commission,
        'spread': costs.spread_pct,
        'slippage': costs.slippage_pct,
        'total_cost_pct': costs.commission + costs.spread_pct + costs.slippage_pct
    }


if __name__ == '__main__':
    import torch
    
    print("\n" + "="*80)
    print("🎯 QUICK FEE COMPARISON - STOCKS vs CRYPTO")
    print("="*80)
    
    configs = [
        # Stocks (essentially free)
        {'asset_type': 'stock', 'broker': 'alpaca', 'name': 'Alpaca (Stock - $0 fees)'},
        {'asset_type': 'stock', 'broker': 'robinhood', 'name': 'Robinhood (Stock - $0 fees)'},
        
        # Crypto (higher fees)
        {'asset_type': 'crypto', 'broker': 'binance', 'name': 'Binance (Crypto - 0.1%)'},
        {'asset_type': 'crypto', 'broker': 'default', 'name': 'Crypto Default (0.15%)'},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n📊 Testing: {config['name']}")
        print("-" * 40)
        
        result = simulate_trading(
            asset_type=config['asset_type'],
            broker=config['broker'],
            episodes=20
        )
        
        result['name'] = config['name']
        results.append(result)
        
        print(f"  Initial: ${result['initial_balance']:,.2f}")
        print(f"  Final:   ${result['final_balance']:,.2f}")
        print(f"  Profit:  ${result['profit']:,.2f}")
        print(f"  Fees:    ${result['fees']:,.2f}")
        print(f"  ROI:     {result['roi']:.2f}%")
        print(f"  Trades:  {result['trades']}")
        print(f"  Cost/Trade: {result['total_cost_pct']:.4%}")
    
    # Summary comparison
    print("\n" + "="*80)
    print("📊 SUMMARY COMPARISON")
    print("="*80)
    
    df = pd.DataFrame(results)
    
    # Average by type
    stock_avg = df[df['asset_type'] == 'stock'].mean(numeric_only=True)
    crypto_avg = df[df['asset_type'] == 'crypto'].mean(numeric_only=True)
    
    print("\n🏦 STOCKS (Zero Commission):")
    print(f"  Avg Profit: ${stock_avg['profit']:,.2f}")
    print(f"  Avg Fees:   ${stock_avg['fees']:,.2f}")
    print(f"  Avg ROI:    {stock_avg['roi']:.2f}%")
    
    print("\n💰 CRYPTO (With Fees):")
    print(f"  Avg Profit: ${crypto_avg['profit']:,.2f}")
    print(f"  Avg Fees:   ${crypto_avg['fees']:,.2f}")
    print(f"  Avg ROI:    {crypto_avg['roi']:.2f}%")
    
    print("\n🎯 IMPACT OF FEES:")
    fee_difference = crypto_avg['fees'] - stock_avg['fees']
    profit_impact = stock_avg['profit'] - crypto_avg['profit']
    
    print(f"  Extra crypto fees: ${fee_difference:,.2f}")
    print(f"  Profit reduction:  ${profit_impact:,.2f}")
    print(f"  Fee multiplier:    {crypto_avg['fees'] / (stock_avg['fees'] + 0.01):.1f}x")
    
    # Create simple bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Profits
    ax1 = axes[0]
    colors = ['green' if 'Stock' in n else 'orange' for n in df['name']]
    ax1.bar(range(len(df)), df['profit'], color=colors, alpha=0.7)
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels([n.split('(')[0].strip() for n in df['name']], rotation=45)
    ax1.set_ylabel('Profit ($)')
    ax1.set_title('Net Profit Comparison')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    
    # Fees
    ax2 = axes[1]
    ax2.bar(range(len(df)), df['fees'], color=colors, alpha=0.7)
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels([n.split('(')[0].strip() for n in df['name']], rotation=45)
    ax2.set_ylabel('Total Fees ($)')
    ax2.set_title('Trading Fees Paid')
    ax2.grid(True, alpha=0.3)
    
    # Fee percentage
    ax3 = axes[2]
    ax3.bar(range(len(df)), df['total_cost_pct'] * 100, color=colors, alpha=0.7)
    ax3.set_xticks(range(len(df)))
    ax3.set_xticklabels([n.split('(')[0].strip() for n in df['name']], rotation=45)
    ax3.set_ylabel('Cost per Trade (%)')
    ax3.set_title('Trading Cost Structure')
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Impact of Realistic Trading Fees on Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    Path('results').mkdir(exist_ok=True)
    plt.savefig('results/quick_fee_comparison.png', dpi=100, bbox_inches='tight')
    print(f"\n📊 Chart saved to: results/quick_fee_comparison.png")
    
    print("\n✅ Comparison complete!")
    print("="*80)