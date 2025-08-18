import torch
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from datetime import datetime
import argparse

from trading_agent import TradingAgent
from trading_env import DailyTradingEnv
from ppo_trainer import PPOTrainer


def load_data(symbol: str, data_dir: str = '../data') -> pd.DataFrame:
    data_path = Path(data_dir)
    
    csv_files = list(data_path.glob(f'*{symbol}*.csv'))
    if not csv_files:
        csv_files = list(data_path.glob('*.csv'))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        print(f"Using first available CSV: {csv_files[0]}")
    
    df = pd.read_csv(csv_files[0])
    
    columns_lower = [col.lower() for col in df.columns]
    df.columns = columns_lower
    
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        available_cols = list(df.columns)
        print(f"Warning: Missing columns {missing_cols}. Available: {available_cols}")
        
        if 'adj close' in df.columns and 'close' not in df.columns:
            df['close'] = df['adj close']
        if 'adj open' in df.columns and 'open' not in df.columns:
            df['open'] = df['adj open']
    
    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            if 'close' in df.columns:
                df[col] = df['close']
    
    if 'volume' not in df.columns:
        df['volume'] = 1000000
    
    df.columns = [col.title() for col in df.columns]
    
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['Returns'] = df['Close'].pct_change()
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    
    df = df.dropna()
    
    return df


def visualize_results(env: DailyTradingEnv, save_path: str = 'training_results.png'):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    axes[0].plot(env.balance_history)
    axes[0].set_title('Portfolio Balance Over Time')
    axes[0].set_xlabel('Days')
    axes[0].set_ylabel('Balance ($)')
    axes[0].grid(True)
    
    axes[1].plot(env.positions_history)
    axes[1].set_title('Position History')
    axes[1].set_xlabel('Days')
    axes[1].set_ylabel('Position Size')
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.3)
    axes[1].grid(True)
    
    if env.returns:
        cumulative_returns = np.cumprod(1 + np.array(env.returns))
        axes[2].plot(cumulative_returns)
        axes[2].set_title('Cumulative Returns')
        axes[2].set_xlabel('Days')
        axes[2].set_ylabel('Cumulative Return')
        axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Results visualization saved to {save_path}")


def evaluate_agent(agent, env, num_episodes: int = 5):
    agent.eval()
    
    all_metrics = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action, _, _ = agent.act(state_tensor, deterministic=True)
                action = action.cpu().numpy().flatten()
            
            state, reward, done, info = env.step(action)
            episode_reward += reward
        
        metrics = env.get_metrics()
        metrics['episode_reward'] = episode_reward
        all_metrics.append(metrics)
    
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = np.mean(values)
        avg_metrics[f'{key}_std'] = np.std(values)
    
    return avg_metrics


def main(args):
    print(f"Loading data for {args.symbol}...")
    df = load_data(args.symbol, args.data_dir)
    df = prepare_features(df)
    print(f"Data shape: {df.shape}")
    
    train_size = int(len(df) * args.train_ratio)
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                'Returns', 'RSI', 'Volume_Ratio', 
                'High_Low_Ratio', 'Close_Open_Ratio']
    
    available_features = [f for f in features if f in train_df.columns]
    
    train_env = DailyTradingEnv(
        train_df,
        window_size=args.window_size,
        initial_balance=args.initial_balance,
        transaction_cost=args.transaction_cost,
        features=available_features
    )
    
    test_env = DailyTradingEnv(
        test_df,
        window_size=args.window_size,
        initial_balance=args.initial_balance,
        transaction_cost=args.transaction_cost,
        features=available_features
    )
    
    input_dim = args.window_size * (len(available_features) + 3)
    
    agent = TradingAgent(
        backbone_model=torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 768),
            torch.nn.ReLU()
        ),
        hidden_dim=768,
        action_std_init=args.action_std
    )
    
    trainer = PPOTrainer(
        agent,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        eps_clip=args.eps_clip,
        k_epochs=args.k_epochs,
        entropy_coef=args.entropy_coef,
        log_dir='./traininglogs'
    )
    
    print("\nStarting training...")
    history = trainer.train(
        train_env,
        num_episodes=args.num_episodes,
        update_interval=args.update_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        save_dir=args.save_dir
    )
    
    print("\nEvaluating on test set...")
    test_metrics = evaluate_agent(agent, test_env, num_episodes=10)
    
    print("\nTest Set Performance:")
    print(f"  Average Return: {test_metrics['total_return']:.2%} ± {test_metrics['total_return_std']:.2%}")
    print(f"  Sharpe Ratio: {test_metrics['sharpe_ratio']:.2f} ± {test_metrics['sharpe_ratio_std']:.2f}")
    print(f"  Max Drawdown: {test_metrics['max_drawdown']:.2%} ± {test_metrics['max_drawdown_std']:.2%}")
    print(f"  Win Rate: {test_metrics['win_rate']:.2%} ± {test_metrics['win_rate_std']:.2%}")
    print(f"  Num Trades: {test_metrics['num_trades']:.1f} ± {test_metrics['num_trades_std']:.1f}")
    
    test_env.reset()
    state = test_env.reset()
    done = False
    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, _, _ = agent.act(state_tensor, deterministic=True)
            action = action.cpu().numpy().flatten()
        state, _, done, _ = test_env.step(action)
    
    visualize_results(test_env, f'{args.save_dir}/test_results.png')
    
    results = {
        'symbol': args.symbol,
        'timestamp': datetime.now().isoformat(),
        'test_metrics': test_metrics,
        'training_history': {
            'episode_rewards': history['episode_rewards'][-100:],
            'final_losses': {
                'actor': history['actor_losses'][-1] if history['actor_losses'] else None,
                'critic': history['critic_losses'][-1] if history['critic_losses'] else None
            }
        },
        'hyperparameters': vars(args)
    }
    
    with open(f'{args.save_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\nResults saved to {args.save_dir}/")
    
    # Close TensorBoard writer
    trainer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RL Trading Agent')
    
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol')
    parser.add_argument('--data_dir', type=str, default='../data', help='Data directory')
    parser.add_argument('--save_dir', type=str, default='./models', help='Save directory')
    
    parser.add_argument('--window_size', type=int, default=30, help='Observation window size')
    parser.add_argument('--initial_balance', type=float, default=10000, help='Initial balance')
    parser.add_argument('--transaction_cost', type=float, default=0.001, help='Transaction cost')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train/test split ratio')
    
    parser.add_argument('--num_episodes', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--update_interval', type=int, default=10, help='Policy update interval')
    parser.add_argument('--eval_interval', type=int, default=50, help='Evaluation interval')
    parser.add_argument('--save_interval', type=int, default=100, help='Model save interval')
    
    parser.add_argument('--lr_actor', type=float, default=3e-4, help='Actor learning rate')
    parser.add_argument('--lr_critic', type=float, default=1e-3, help='Critic learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='PPO clip parameter')
    parser.add_argument('--k_epochs', type=int, default=4, help='PPO update epochs')
    parser.add_argument('--action_std', type=float, default=0.5, help='Action std deviation')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient')
    
    args = parser.parse_args()
    
    Path(args.save_dir).mkdir(exist_ok=True)
    
    main(args)