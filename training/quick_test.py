#!/usr/bin/env python3
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append('..')

from trading_agent import TradingAgent
from trading_env import DailyTradingEnv
from ppo_trainer import PPOTrainer


def create_dummy_data(n_days=500):
    np.random.seed(42)
    
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    close_prices = [100.0]
    for _ in range(n_days - 1):
        change = np.random.normal(0.001, 0.02)
        close_prices.append(close_prices[-1] * (1 + change))
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': np.array(close_prices) * np.random.uniform(0.98, 1.02, n_days),
        'High': np.array(close_prices) * np.random.uniform(1.01, 1.05, n_days),
        'Low': np.array(close_prices) * np.random.uniform(0.95, 0.99, n_days),
        'Close': close_prices,
        'Volume': np.random.uniform(1e6, 1e7, n_days)
    })
    
    return df


def test_components():
    print("Testing RL Trading System Components...")
    print("=" * 50)
    
    print("\n1. Creating dummy data...")
    df = create_dummy_data(500)
    print(f"   Data shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    
    print("\n2. Creating environment...")
    env = DailyTradingEnv(
        df,
        window_size=20,
        initial_balance=10000,
        transaction_cost=0.001
    )
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")
    
    print("\n3. Testing environment reset and step...")
    obs = env.reset()
    print(f"   Initial observation shape: {obs.shape}")
    
    action = np.array([0.5])
    next_obs, reward, done, info = env.step(action)
    print(f"   Step executed successfully")
    print(f"   Reward: {reward:.4f}")
    print(f"   Info: {info}")
    
    print("\n4. Creating agent...")
    input_dim = 20 * 8
    
    backbone = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(input_dim, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 768),
        torch.nn.ReLU()
    )
    
    agent = TradingAgent(
        backbone_model=backbone,
        hidden_dim=768,
        action_std_init=0.5
    )
    print(f"   Agent created with {sum(p.numel() for p in agent.parameters())} parameters")
    
    print("\n5. Testing agent forward pass...")
    dummy_state = torch.randn(1, input_dim)
    action_mean, value = agent(dummy_state)
    print(f"   Action mean shape: {action_mean.shape}, Value shape: {value.shape}")
    
    action, logprob, value = agent.act(dummy_state)
    print(f"   Action: {action.item():.4f}, Value: {value.item():.4f}")
    
    print("\n6. Creating PPO trainer...")
    trainer = PPOTrainer(
        agent,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        eps_clip=0.2
    )
    print("   Trainer created successfully")
    
    print("\n7. Running short training episode...")
    env.reset()
    episode_reward, episode_length, info = trainer.train_episode(env, max_steps=50)
    print(f"   Episode reward: {episode_reward:.4f}")
    print(f"   Episode length: {episode_length}")
    print(f"   Final balance: ${info['balance']:.2f}")
    
    print("\n8. Testing PPO update...")
    for _ in range(3):
        env.reset()
        trainer.train_episode(env, max_steps=50)
    
    update_info = trainer.update()
    print(f"   Actor loss: {update_info['actor_loss']:.4f}")
    print(f"   Critic loss: {update_info['critic_loss']:.4f}")
    print(f"   Total loss: {update_info['total_loss']:.4f}")
    
    print("\n9. Getting environment metrics...")
    env.reset()
    done = False
    while not done:
        action = np.random.uniform(-1, 1, 1)
        _, _, done, _ = env.step(action)
    
    metrics = env.get_metrics()
    print(f"   Total return: {metrics['total_return']:.2%}")
    print(f"   Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Max drawdown: {metrics['max_drawdown']:.2%}")
    print(f"   Number of trades: {metrics['num_trades']}")
    
    print("\n" + "=" * 50)
    print("All tests passed successfully! ✓")
    print("\nYou can now run the full training with:")
    print("  python train_rl_agent.py --symbol AAPL --num_episodes 100")
    

if __name__ == '__main__':
    test_components()