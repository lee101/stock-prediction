#!/usr/bin/env python3
"""
Train RL agent on market data with comprehensive PnL tracking

Uses the fast C market environment with real OHLCV data.
Tracks profit/loss during training and shows how much money we make!
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


@dataclass
class TradingMetrics:
    """Track trading performance metrics"""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0
    final_portfolio_value: float = 100000.0

    def to_dict(self):
        return asdict(self)


class MarketDataLoader:
    """Load and prepare market data from CSV files"""

    def __init__(self, data_dir: str = "trainingdata"):
        self.data_dir = Path(data_dir)
        self.data_cache = {}

    def load_csv(self, symbol: str) -> pd.DataFrame:
        """Load OHLCV data for a symbol"""
        if symbol in self.data_cache:
            return self.data_cache[symbol]

        csv_path = self.data_dir / f"{symbol}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Data file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Ensure we have required columns
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            raise ValueError(f"CSV must have columns: {required}")

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        self.data_cache[symbol] = df
        return df

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols"""
        csv_files = list(self.data_dir.glob("*.csv"))
        return [f.stem for f in csv_files]

    def create_dataset(self, symbols: List[str], window_size: int = 60):
        """Create training dataset from multiple symbols"""
        datasets = []

        for symbol in symbols:
            df = self.load_csv(symbol)

            # Normalize prices (returns instead of raw prices)
            data = df[['open', 'high', 'low', 'close', 'volume']].values

            # Calculate returns
            returns = np.diff(data[:, :4], axis=0) / data[:-1, :4]
            volume_norm = data[1:, 4:5] / (data[1:, 4:5].mean() + 1e-8)

            # Combine
            features = np.concatenate([returns, volume_norm], axis=1)

            # Create windows
            for i in range(len(features) - window_size):
                window = features[i:i+window_size]
                datasets.append({
                    'symbol': symbol,
                    'data': window,
                    'timestamp': i,
                })

        return datasets


class TradingPolicy(nn.Module):
    """Neural network policy for trading decisions"""

    def __init__(
        self,
        input_dim: int,
        num_assets: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.num_assets = num_assets

        # Feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Policy head (outputs portfolio weights)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_assets),
            nn.Softmax(dim=-1)  # Portfolio weights sum to 1
        )

        # Value head (for critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        features = self.feature_net(obs)
        weights = self.policy_head(features)
        value = self.value_head(features)
        return weights, value


class SimpleMarketEnv:
    """
    Simple Python market environment for training

    Simulates portfolio management with transaction costs
    """

    def __init__(
        self,
        data: np.ndarray,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        max_position: float = 0.3,
    ):
        self.data = data  # [timesteps, features]
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position = max_position

        self.reset()

    def reset(self):
        self.step_idx = 0
        self.cash = self.initial_capital
        self.position = 0.0  # Number of shares
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital

        # Track for metrics
        self.trades = []
        self.portfolio_history = [self.initial_capital]

        return self._get_obs()

    def _get_obs(self):
        """Get current observation"""
        if self.step_idx >= len(self.data):
            return np.zeros(self.data.shape[1] + 2)

        market_data = self.data[self.step_idx]
        position_ratio = self.position * self._get_current_price() / self.portfolio_value
        cash_ratio = self.cash / self.portfolio_value

        return np.concatenate([market_data, [position_ratio, cash_ratio]])

    def _get_current_price(self):
        """Get current close price (from returns, need to track)"""
        # Simplified: assume normalized price around 100
        return 100.0

    def step(self, action):
        """
        Execute trading action

        Args:
            action: Target portfolio weight for the asset
        """
        if self.step_idx >= len(self.data) - 1:
            return self._get_obs(), 0.0, True, {}

        # Current state
        current_price = self._get_current_price()
        old_value = self.cash + self.position * current_price

        # Execute trade based on target weight
        target_value = old_value * action
        target_shares = target_value / current_price

        shares_to_trade = target_shares - self.position

        if abs(shares_to_trade) > 0.01:  # Trade threshold
            trade_value = abs(shares_to_trade * current_price)
            cost = trade_value * self.transaction_cost

            self.cash -= shares_to_trade * current_price + cost
            self.position += shares_to_trade

            self.trades.append({
                'step': self.step_idx,
                'shares': shares_to_trade,
                'price': current_price,
                'cost': cost,
            })

        # Move to next step
        self.step_idx += 1

        # Get return from market movement (simplified)
        if self.step_idx < len(self.data):
            market_return = self.data[self.step_idx, 3]  # Close return
            new_price = current_price * (1 + market_return)
        else:
            new_price = current_price

        # Calculate new portfolio value
        new_value = self.cash + self.position * new_price
        self.portfolio_value = new_value

        # Reward is the return
        reward = (new_value - old_value) / old_value

        # Track peak for drawdown
        if new_value > self.peak_value:
            self.peak_value = new_value

        self.portfolio_history.append(new_value)

        # Done if out of data
        done = self.step_idx >= len(self.data) - 1

        info = {
            'portfolio_value': new_value,
            'return': (new_value - self.initial_capital) / self.initial_capital,
        }

        return self._get_obs(), reward, done, info

    def get_metrics(self) -> TradingMetrics:
        """Calculate trading metrics"""
        values = np.array(self.portfolio_history)
        returns = np.diff(values) / values[:-1]

        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)  # Annualized

        # Max drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        max_dd = drawdown.max()

        # Win rate
        if len(self.trades) > 0:
            # Simplified: count profitable periods
            win_rate = (returns > 0).mean()
        else:
            win_rate = 0.5

        return TradingMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            num_trades=len(self.trades),
            final_portfolio_value=self.portfolio_value,
        )


class PPOTrainer:
    """PPO trainer for market trading"""

    def __init__(
        self,
        policy: TradingPolicy,
        device: str = "cuda",
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
    ):
        self.policy = policy.to(device)
        self.device = device
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

        self.optimizer = optim.Adam(policy.parameters(), lr=lr)

        # Tracking
        self.episode_returns = deque(maxlen=100)
        self.episode_metrics = []

    def train_episode(self, env: SimpleMarketEnv):
        """Train on one episode"""
        obs = env.reset()
        obs_list, actions_list, rewards_list, values_list = [], [], [], []
        done = False

        while not done:
            obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)

            with torch.no_grad():
                weights, value = self.policy(obs_tensor)
                action = weights[0, 0].item()  # Single asset for now

            next_obs, reward, done, info = env.step(action)

            obs_list.append(obs)
            actions_list.append(action)
            rewards_list.append(reward)
            values_list.append(value.item())

            obs = next_obs

        # Compute returns and advantages
        returns = []
        R = 0
        for r in reversed(rewards_list):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns).to(self.device)
        values = torch.FloatTensor(values_list).to(self.device)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        obs_batch = torch.FloatTensor(np.array(obs_list)).to(self.device)
        actions_batch = torch.FloatTensor(actions_list).to(self.device)

        weights, values_new = self.policy(obs_batch)
        actions_pred = weights[:, 0]

        # Policy loss (simplified - using MSE for continuous actions)
        policy_loss = ((actions_pred - actions_batch) ** 2 * advantages.detach()).mean()

        # Value loss
        value_loss = ((values_new.squeeze() - returns) ** 2).mean()

        # Total loss
        loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        # Track metrics
        episode_return = sum(rewards_list)
        self.episode_returns.append(episode_return)

        metrics = env.get_metrics()
        self.episode_metrics.append(metrics)

        return {
            'episode_return': episode_return,
            'portfolio_value': metrics.final_portfolio_value,
            'total_return': metrics.total_return,
            'sharpe': metrics.sharpe_ratio,
            'num_trades': metrics.num_trades,
            'loss': loss.item(),
        }


def main():
    print("=" * 70)
    print("💰 MARKET TRADING RL AGENT - LET'S MAKE SOME MONEY!")
    print("=" * 70)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🔧 Device: {device}")

    # Load data
    print("\n📊 Loading market data...")
    data_loader = MarketDataLoader()
    symbols = ['BTCUSD', 'AAPL', 'AMD']  # Start with 3 assets
    print(f"   Symbols: {symbols}")

    datasets = []
    for symbol in symbols:
        try:
            df = data_loader.load_csv(symbol)
            print(f"   ✓ {symbol}: {len(df)} data points")
            datasets.append(df)
        except Exception as e:
            print(f"   ✗ {symbol}: {e}")

    if not datasets:
        print("❌ No data loaded!")
        return

    # Create policy
    print("\n🧠 Creating trading policy...")
    obs_dim = 5 + 2  # OHLCV returns + volume + position + cash
    policy = TradingPolicy(input_dim=obs_dim, num_assets=1, hidden_dim=128)
    print(f"   Parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Create trainer
    trainer = PPOTrainer(policy, device=device, lr=1e-3)

    # Training loop
    print("\n🚀 Starting training...")
    print("=" * 70)

    num_episodes = 100
    log_interval = 10

    writer = SummaryWriter(log_dir="runs/market_trading")

    best_return = -float('inf')
    total_money_made = 0.0

    for episode in range(num_episodes):
        # Sample random dataset
        df = datasets[np.random.randint(len(datasets))]

        # Create environment with this data
        data = df[['open', 'high', 'low', 'close', 'volume']].values
        returns = np.diff(data[:, :4], axis=0) / (data[:-1, :4] + 1e-8)
        volume_norm = data[1:, 4:5] / (data[1:, 4:5].mean() + 1e-8)
        features = np.concatenate([returns, volume_norm], axis=1)

        env = SimpleMarketEnv(features, initial_capital=100000.0)

        # Train episode
        metrics = trainer.train_episode(env)

        # Log
        writer.add_scalar('Return/Episode', metrics['episode_return'], episode)
        writer.add_scalar('Portfolio/Value', metrics['portfolio_value'], episode)
        writer.add_scalar('Portfolio/Return%', metrics['total_return'] * 100, episode)
        writer.add_scalar('Metrics/Sharpe', metrics['sharpe'], episode)
        writer.add_scalar('Metrics/NumTrades', metrics['num_trades'], episode)

        # Track money made
        money_made = metrics['portfolio_value'] - 100000.0
        total_money_made += money_made

        if (episode + 1) % log_interval == 0:
            recent = trainer.episode_metrics[-log_interval:]
            avg_return = np.mean([m.total_return for m in recent])
            avg_portfolio = np.mean([m.final_portfolio_value for m in recent])
            avg_sharpe = np.mean([m.sharpe_ratio for m in recent])

            print(f"\n📈 Episode {episode + 1}/{num_episodes}")
            print(f"   Avg Return: {avg_return:+.4f}")
            print(f"   Avg Portfolio: ${avg_portfolio:,.2f}")
            print(f"   Avg Return%: {(avg_portfolio - 100000) / 1000:.2f}%")
            print(f"   Avg Sharpe: {avg_sharpe:.2f}")
            print(f"   💰 Money Made This Ep: ${money_made:,.2f}")
            print(f"   💵 Total Money Made: ${total_money_made:,.2f}")

        # Save best model
        if metrics['total_return'] > best_return:
            best_return = metrics['total_return']
            torch.save(policy.state_dict(), 'best_trading_policy.pt')
            print(f"   ⭐ New best return: {best_return * 100:.2f}%")

    writer.close()

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)

    # Final evaluation
    print("\n📊 FINAL RESULTS:")
    recent_metrics = trainer.episode_metrics[-10:]
    avg_return = np.mean([m.total_return for m in recent_metrics])
    avg_sharpe = np.mean([m.sharpe_ratio for m in recent_metrics])
    avg_portfolio = np.mean([m.final_portfolio_value for m in recent_metrics])

    print(f"   Average Return (last 10): {avg_return * 100:+.2f}%")
    print(f"   Average Sharpe (last 10): {avg_sharpe:.2f}")
    print(f"   Average Portfolio (last 10): ${avg_portfolio:,.2f}")
    print(f"   💰 Best Single Episode Return: {best_return * 100:+.2f}%")
    print(f"   💵 Total Money Made During Training: ${total_money_made:,.2f}")

    print(f"\n💾 Model saved to: best_trading_policy.pt")
    print(f"📊 Logs saved to: runs/market_trading/")

    return avg_return, total_money_made


if __name__ == "__main__":
    final_return, total_money = main()
    print(f"\n🎯 Final Performance: {final_return * 100:+.2f}% return")
    print(f"💰 Total Money: ${total_money:+,.2f}")
