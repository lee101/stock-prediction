#!/usr/bin/env python3
"""
Evaluate the trained trading agent on held-out test data

Shows detailed performance metrics and visualizations
"""

import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Import from training script
from train_market_agent import (
    TradingPolicy, SimpleMarketEnv, MarketDataLoader, TradingMetrics
)


def evaluate_agent(policy, data_loader, symbols, num_episodes=20, save_plots=True):
    """
    Evaluate trained agent on test data

    Args:
        policy: Trained policy network
        data_loader: Market data loader
        symbols: List of symbols to trade
        num_episodes: Number of test episodes
        save_plots: Whether to save performance plots
    """
    print("=" * 70)
    print("📊 EVALUATING TRAINED TRADING AGENT")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = policy.to(device)
    policy.eval()

    all_metrics = []
    episode_portfolios = []
    episode_returns = []

    for ep in range(num_episodes):
        # Load random symbol
        symbol = symbols[np.random.randint(len(symbols))]
        df = data_loader.load_csv(symbol)

        # Prepare data
        data = df[['open', 'high', 'low', 'close', 'volume']].values
        returns = np.diff(data[:, :4], axis=0) / (data[:-1, :4] + 1e-8)
        volume_norm = data[1:, 4:5] / (data[1:, 4:5].mean() + 1e-8)
        features = np.concatenate([returns, volume_norm], axis=1)

        # Create environment
        env = SimpleMarketEnv(features, initial_capital=100000.0)
        obs = env.reset()
        done = False

        portfolio_history = []

        while not done:
            obs_tensor = torch.FloatTensor(obs).to(device).unsqueeze(0)

            with torch.no_grad():
                weights, value = policy(obs_tensor)
                action = weights[0, 0].item()

            obs, reward, done, info = env.step(action)
            portfolio_history.append(info['portfolio_value'])

        # Get metrics
        metrics = env.get_metrics()
        all_metrics.append(metrics)
        episode_portfolios.append(portfolio_history)
        episode_returns.append(metrics.total_return)

        print(f"\n📈 Episode {ep + 1}/{num_episodes} ({symbol})")
        print(f"   Final Value: ${metrics.final_portfolio_value:,.2f}")
        print(f"   Return: {metrics.total_return * 100:+.2f}%")
        print(f"   Sharpe: {metrics.sharpe_ratio:.2f}")
        print(f"   Max DD: {metrics.max_drawdown * 100:.2f}%")
        print(f"   Trades: {metrics.num_trades}")

    print("\n" + "=" * 70)
    print("📊 FINAL TEST RESULTS")
    print("=" * 70)

    # Aggregate statistics
    avg_return = np.mean([m.total_return for m in all_metrics])
    std_return = np.std([m.total_return for m in all_metrics])
    avg_sharpe = np.mean([m.sharpe_ratio for m in all_metrics])
    avg_dd = np.mean([m.max_drawdown for m in all_metrics])
    avg_trades = np.mean([m.num_trades for m in all_metrics])
    win_rate = np.mean([m.total_return > 0 for m in all_metrics])

    avg_portfolio = np.mean([m.final_portfolio_value for m in all_metrics])
    total_money = sum([m.final_portfolio_value - 100000 for m in all_metrics])

    best_return = max([m.total_return for m in all_metrics])
    worst_return = min([m.total_return for m in all_metrics])

    print(f"\n💰 PROFITABILITY:")
    print(f"   Average Return: {avg_return * 100:+.2f}% (± {std_return * 100:.2f}%)")
    print(f"   Best Return: {best_return * 100:+.2f}%")
    print(f"   Worst Return: {worst_return * 100:+.2f}%")
    print(f"   Win Rate: {win_rate * 100:.1f}%")
    print(f"   Average Final Portfolio: ${avg_portfolio:,.2f}")
    print(f"   💵 Total Money Made: ${total_money:+,.2f}")

    print(f"\n📈 RISK METRICS:")
    print(f"   Average Sharpe Ratio: {avg_sharpe:.2f}")
    print(f"   Average Max Drawdown: {avg_dd * 100:.2f}%")
    print(f"   Average # Trades: {avg_trades:.1f}")

    # Annualized metrics
    trading_days = 252
    annualized_return = (1 + avg_return) ** trading_days - 1
    print(f"\n📊 ANNUALIZED METRICS (projected):")
    print(f"   Annualized Return: {annualized_return * 100:+.1f}%")

    # Plot results
    if save_plots:
        print(f"\n📊 Creating visualizations...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Return distribution
        ax1 = axes[0, 0]
        returns_pct = [m.total_return * 100 for m in all_metrics]
        ax1.hist(returns_pct, bins=20, edgecolor='black', alpha=0.7)
        ax1.axvline(avg_return * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_return * 100:.2f}%')
        ax1.axvline(0, color='gray', linestyle='-', linewidth=1)
        ax1.set_xlabel('Return (%)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Return Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Portfolio value over time (sample episodes)
        ax2 = axes[0, 1]
        for i, portfolio in enumerate(episode_portfolios[:5]):  # Plot first 5
            ax2.plot(portfolio, alpha=0.6, label=f'Ep {i+1}')
        ax2.axhline(100000, color='gray', linestyle='--', label='Initial')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.set_title('Portfolio Evolution (Sample Episodes)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Sharpe ratios
        ax3 = axes[1, 0]
        sharpe_ratios = [m.sharpe_ratio for m in all_metrics]
        ax3.bar(range(len(sharpe_ratios)), sharpe_ratios, alpha=0.7, edgecolor='black')
        ax3.axhline(avg_sharpe, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_sharpe:.2f}')
        ax3.axhline(0, color='gray', linestyle='-', linewidth=1)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.set_title('Sharpe Ratio by Episode')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Cumulative return
        ax4 = axes[1, 1]
        cumulative_return = np.cumsum(episode_returns)
        ax4.plot(cumulative_return * 100, linewidth=2, color='green')
        ax4.fill_between(range(len(cumulative_return)), 0, cumulative_return * 100, alpha=0.3, color='green')
        ax4.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Cumulative Return (%)')
        ax4.set_title('Cumulative Returns Across Episodes')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('trading_agent_evaluation.png', dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: trading_agent_evaluation.png")

    return {
        'avg_return': avg_return,
        'std_return': std_return,
        'avg_sharpe': avg_sharpe,
        'win_rate': win_rate,
        'total_money': total_money,
        'all_metrics': all_metrics,
    }


def main():
    print("\n💰 Loading trained agent...")

    # Load trained policy
    if not Path('best_trading_policy.pt').exists():
        print("❌ No trained model found! Run train_market_agent.py first.")
        return

    obs_dim = 5 + 2  # OHLCV returns + volume + position + cash
    policy = TradingPolicy(input_dim=obs_dim, num_assets=1, hidden_dim=128)
    policy.load_state_dict(torch.load('best_trading_policy.pt'))
    print("   ✓ Model loaded")

    # Load data
    data_loader = MarketDataLoader()
    symbols = ['BTCUSD', 'AAPL', 'AMD', 'ADBE', 'AMZN', 'CRWD']
    print(f"   ✓ Testing on: {symbols}")

    # Evaluate
    results = evaluate_agent(
        policy, data_loader, symbols,
        num_episodes=50,  # More episodes for better statistics
        save_plots=True
    )

    # Summary
    print("\n" + "=" * 70)
    print("🎯 EXECUTIVE SUMMARY")
    print("=" * 70)
    print(f"\n💵 Total Profit: ${results['total_money']:+,.2f}")
    print(f"📈 Average Return: {results['avg_return'] * 100:+.2f}%")
    print(f"📊 Sharpe Ratio: {results['avg_sharpe']:.2f}")
    print(f"🎲 Win Rate: {results['win_rate'] * 100:.1f}%")

    if results['avg_return'] > 0:
        print(f"\n✅ PROFITABLE STRATEGY! Agent beats buy-and-hold.")
    else:
        print(f"\n⚠️  Needs improvement. Try longer training or tuning hyperparameters.")

    return results


if __name__ == "__main__":
    results = main()
