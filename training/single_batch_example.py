#!/usr/bin/env python3
"""
Single Batch Training Example
This script demonstrates training on a single batch to verify the system works
and shows TensorBoard logging in action.
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.append('..')

from trading_agent import TradingAgent
from trading_env import DailyTradingEnv
from ppo_trainer import PPOTrainer


def create_sample_data(n_days=500, symbol='TEST'):
    """Create synthetic stock data for testing"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
    
    # Generate realistic price movement
    returns = np.random.normal(0.0005, 0.02, n_days)
    close_prices = 100 * np.exp(np.cumsum(returns))
    
    # Add some trend
    trend = np.linspace(0, 0.2, n_days)
    close_prices = close_prices * (1 + trend)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': close_prices * np.random.uniform(0.98, 1.02, n_days),
        'High': close_prices * np.random.uniform(1.01, 1.04, n_days),
        'Low': close_prices * np.random.uniform(0.96, 0.99, n_days),
        'Close': close_prices,
        'Volume': np.random.uniform(1e6, 5e6, n_days)
    })
    
    # Add technical indicators
    df['Returns'] = df['Close'].pct_change()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volume metrics
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1e-10)
    
    # Price ratios
    df['High_Low_Ratio'] = df['High'] / (df['Low'] + 1e-10)
    df['Close_Open_Ratio'] = df['Close'] / (df['Open'] + 1e-10)
    
    # Drop NaN rows
    df = df.dropna()
    
    print(f"Generated {len(df)} days of data for {symbol}")
    print(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    print(f"Average daily return: {df['Returns'].mean():.4%}")
    print(f"Volatility (std): {df['Returns'].std():.4%}")
    
    return df


def run_single_batch_training():
    print("=" * 80)
    print("SINGLE BATCH TRAINING EXAMPLE")
    print("=" * 80)
    
    # Configuration
    window_size = 30
    batch_episodes = 5  # Collect 5 episodes for one batch
    initial_balance = 10000
    
    print("\n1. GENERATING SAMPLE DATA")
    print("-" * 40)
    df = create_sample_data(n_days=500, symbol='SYNTHETIC')
    
    # Use available features
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                'Returns', 'RSI', 'Volume_Ratio', 
                'High_Low_Ratio', 'Close_Open_Ratio']
    
    available_features = [f for f in features if f in df.columns]
    print(f"\nUsing features: {available_features}")
    
    print("\n2. CREATING ENVIRONMENT")
    print("-" * 40)
    env = DailyTradingEnv(
        df,
        window_size=window_size,
        initial_balance=initial_balance,
        transaction_cost=0.001,
        features=available_features
    )
    print(f"Environment created:")
    print(f"  - Window size: {window_size}")
    print(f"  - Initial balance: ${initial_balance:,.2f}")
    print(f"  - Max episodes: {env.n_days}")
    
    print("\n3. INITIALIZING AGENT")
    print("-" * 40)
    input_dim = window_size * (len(available_features) + 3)
    
    # Create a simple backbone network
    backbone = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(input_dim, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(512, 768),
        torch.nn.ReLU()
    )
    
    agent = TradingAgent(
        backbone_model=backbone,
        hidden_dim=768,
        action_std_init=0.5
    )
    
    # Move agent to the correct device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = agent.to(device)
    
    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"Agent initialized:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    print("\n4. SETTING UP PPO TRAINER")
    print("-" * 40)
    trainer = PPOTrainer(
        agent,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4,
        entropy_coef=0.01,
        log_dir='./traininglogs'
    )
    print(f"PPO Trainer configured:")
    print(f"  - Learning rate (actor): 3e-4")
    print(f"  - Learning rate (critic): 1e-3")
    print(f"  - Gamma (discount): 0.99")
    print(f"  - PPO clip: 0.2")
    print(f"  - Update epochs: 4")
    
    print("\n5. COLLECTING BATCH DATA")
    print("-" * 40)
    print(f"Collecting {batch_episodes} episodes for the batch...")
    
    batch_rewards = []
    batch_lengths = []
    
    for episode in range(batch_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"\n  Episode {episode + 1}/{batch_episodes}:")
        
        while not done:
            # Get action from agent
            action, logprob, value = trainer.select_action(state, deterministic=False)
            
            # Step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition for training
            trainer.store_transition(
                state, action, logprob, reward,
                value[0], done
            )
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            # Print progress every 100 steps
            if episode_length % 100 == 0:
                print(f"    Step {episode_length}: Balance=${info['balance']:.2f}, Position={info['position']:.3f}")
        
        batch_rewards.append(episode_reward)
        batch_lengths.append(episode_length)
        
        metrics = env.get_metrics()
        print(f"    Completed: Reward={episode_reward:.4f}, Length={episode_length}")
        print(f"    Metrics: Return={metrics['total_return']:.2%}, Sharpe={metrics['sharpe_ratio']:.2f}, Trades={metrics['num_trades']}")
    
    print(f"\nBatch collection complete:")
    print(f"  - Average reward: {np.mean(batch_rewards):.4f}")
    print(f"  - Average length: {np.mean(batch_lengths):.1f}")
    print(f"  - Total transitions: {sum(batch_lengths)}")
    
    print("\n6. PERFORMING PPO UPDATE")
    print("-" * 40)
    print("Running PPO optimization on collected batch...")
    
    update_info = trainer.update()
    
    print(f"\nUpdate complete:")
    print(f"  - Actor loss: {update_info['actor_loss']:.6f}")
    print(f"  - Critic loss: {update_info['critic_loss']:.6f}")
    print(f"  - Total loss: {update_info['total_loss']:.6f}")
    
    print("\n7. EVALUATING UPDATED POLICY")
    print("-" * 40)
    print("Testing the updated policy (deterministic)...")
    
    state = env.reset()
    eval_reward = 0
    eval_length = 0
    done = False
    
    positions = []
    balances = []
    
    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, _, value = agent.act(state_tensor, deterministic=True)
            action = action.cpu().numpy().flatten()
        
        state, reward, done, info = env.step(action)
        eval_reward += reward
        eval_length += 1
        
        positions.append(info['position'])
        balances.append(info['balance'])
        
        if eval_length % 100 == 0:
            print(f"  Step {eval_length}: Balance=${info['balance']:.2f}, Position={info['position']:.3f}")
    
    final_metrics = env.get_metrics()
    
    print(f"\nEvaluation Results:")
    print(f"  - Total reward: {eval_reward:.4f}")
    print(f"  - Episode length: {eval_length}")
    print(f"  - Final balance: ${balances[-1]:.2f}")
    print(f"  - Total return: {final_metrics['total_return']:.2%}")
    print(f"  - Sharpe ratio: {final_metrics['sharpe_ratio']:.2f}")
    print(f"  - Max drawdown: {final_metrics['max_drawdown']:.2%}")
    print(f"  - Number of trades: {final_metrics['num_trades']}")
    print(f"  - Win rate: {final_metrics['win_rate']:.2%}")
    
    print("\n8. TENSORBOARD LOGGING")
    print("-" * 40)
    print("TensorBoard logs have been saved to: ./traininglogs/")
    print("To view the logs, run:")
    print("  tensorboard --logdir=./traininglogs")
    print("\nThen open your browser to: http://localhost:6006")
    
    # Close the writer
    trainer.close()
    
    print("\n" + "=" * 80)
    print("SINGLE BATCH TRAINING COMPLETE!")
    print("=" * 80)
    
    # Save a checkpoint
    checkpoint_path = Path('./models')
    checkpoint_path.mkdir(exist_ok=True)
    trainer.save_checkpoint(checkpoint_path / 'single_batch_model.pth')
    print(f"\nModel saved to: {checkpoint_path / 'single_batch_model.pth'}")
    
    return trainer, agent, env, final_metrics


if __name__ == '__main__':
    # Run the single batch example
    trainer, agent, env, metrics = run_single_batch_training()
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. View TensorBoard logs:")
    print("   tensorboard --logdir=./traininglogs")
    print("\n2. Run full training:")
    print("   python train_rl_agent.py --symbol AAPL --num_episodes 500")
    print("\n3. Load and continue training:")
    print("   trainer.load_checkpoint('./models/single_batch_model.pth')")
    print("=" * 80)