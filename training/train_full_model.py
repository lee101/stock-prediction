#!/usr/bin/env python3
"""
Full Model Training with Realistic Fees and Comprehensive Visualization
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from tqdm import tqdm

sys.path.append('..')

from trading_agent import TradingAgent
from trading_env import DailyTradingEnv
from ppo_trainer import PPOTrainer
from trading_config import get_trading_costs, print_cost_comparison

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_and_prepare_data(symbol: str = 'AAPL', data_dir: str = '../data'):
    """Load and prepare real stock data with technical indicators"""
    
    print(f"\n📊 Loading data for {symbol}...")
    
    # Try to find the data file
    data_path = Path(data_dir)
    
    # Look for symbol-specific file first
    csv_files = list(data_path.glob(f'*{symbol}*.csv'))
    if not csv_files:
        # Use any available CSV for demo
        csv_files = list(data_path.glob('*.csv'))
        if csv_files:
            print(f"Symbol {symbol} not found, using: {csv_files[0].name}")
        else:
            print("No data files found, generating synthetic data...")
            return generate_synthetic_data()
    
    df = pd.read_csv(csv_files[0])
    
    # Standardize column names
    df.columns = [col.lower() for col in df.columns]
    
    # Ensure we have required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            if 'adj close' in df.columns and col == 'close':
                df[col] = df['adj close']
            elif 'adj open' in df.columns and col == 'open':
                df[col] = df['adj open']
            elif col in ['high', 'low']:
                df[col] = df['close'] if 'close' in df.columns else 100
            elif col == 'volume':
                df[col] = 1000000
    
    # Add date if not present
    if 'date' not in df.columns:
        df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
    
    # Calculate technical indicators
    df = add_technical_indicators(df)
    
    # Capitalize column names
    df.columns = [col.title() for col in df.columns]
    
    # Remove NaN values
    df = df.dropna()
    
    print(f"  ✅ Loaded {len(df)} days of data")
    print(f"  📈 Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    print(f"  📊 Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    
    return df


def generate_synthetic_data(n_days: int = 1000):
    """Generate realistic synthetic stock data for testing"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Generate realistic returns with volatility clustering
    returns = []
    volatility = 0.02
    for _ in range(n_days):
        # Volatility clustering
        volatility = 0.9 * volatility + 0.1 * np.random.uniform(0.01, 0.03)
        daily_return = np.random.normal(0.0005, volatility)
        returns.append(daily_return)
    
    # Generate prices
    close_prices = 100 * np.exp(np.cumsum(returns))
    
    # Add trend
    trend = np.linspace(0, 0.5, n_days)
    close_prices = close_prices * (1 + trend)
    
    df = pd.DataFrame({
        'date': dates,
        'open': close_prices * np.random.uniform(0.98, 1.02, n_days),
        'high': close_prices * np.random.uniform(1.01, 1.04, n_days),
        'low': close_prices * np.random.uniform(0.96, 0.99, n_days),
        'close': close_prices,
        'volume': np.random.uniform(1e6, 5e6, n_days) * (1 + np.random.normal(0, 0.3, n_days))
    })
    
    # Ensure lowercase for technical indicators
    df.columns = [col.lower() for col in df.columns]
    df = add_technical_indicators(df)
    df.columns = [col.title() for col in df.columns]
    df = df.dropna()
    
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicators"""
    df = df.copy()
    
    # Price-based indicators
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Moving averages
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_width'] + 1e-10)
    
    # Volume indicators
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Price ratios
    df['high_low_ratio'] = df['high'] / (df['low'] + 1e-10)
    df['close_open_ratio'] = df['close'] / (df['open'] + 1e-10)
    
    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()
    df['atr'] = calculate_atr(df)
    
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    return true_range.rolling(period).mean()


def create_advanced_model(input_dim: int, use_toto: bool = False):
    """Create an advanced trading model"""
    
    if use_toto:
        try:
            from toto.model.toto import Toto
            print("  🤖 Loading Toto backbone...")
            return TradingAgent(use_pretrained_toto=True)
        except ImportError:
            print("  ⚠️ Toto not available, using custom architecture")
    
    # Advanced custom architecture (without BatchNorm for single sample compatibility)
    backbone = torch.nn.Sequential(
        torch.nn.Flatten(),
        
        # Input layer
        torch.nn.Linear(input_dim, 1024),
        torch.nn.LayerNorm(1024),  # Use LayerNorm instead of BatchNorm
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        
        # Hidden layers
        torch.nn.Linear(1024, 512),
        torch.nn.LayerNorm(512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        
        torch.nn.Linear(512, 512),
        torch.nn.LayerNorm(512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        
        # Output projection
        torch.nn.Linear(512, 768),
        torch.nn.ReLU()
    )
    
    return TradingAgent(
        backbone_model=backbone,
        hidden_dim=768,
        action_std_init=0.5
    )


def visualize_results(env: DailyTradingEnv, history: dict, save_dir: str = './results'):
    """Create comprehensive visualization of results"""
    
    Path(save_dir).mkdir(exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Portfolio value over time
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(env.balance_history, label='Portfolio Value', linewidth=2)
    ax1.axhline(y=env.initial_balance, color='r', linestyle='--', alpha=0.5, label='Initial Balance')
    ax1.set_title('Portfolio Value Over Time', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative returns
    ax2 = plt.subplot(3, 3, 2)
    cumulative_returns = (np.array(env.balance_history) - env.initial_balance) / env.initial_balance * 100
    ax2.plot(cumulative_returns, label='Strategy Returns', linewidth=2, color='green')
    ax2.fill_between(range(len(cumulative_returns)), 0, cumulative_returns, alpha=0.3, color='green')
    ax2.set_title('Cumulative Returns (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Return (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Position history
    ax3 = plt.subplot(3, 3, 3)
    positions = np.array(env.positions_history)
    ax3.plot(positions, linewidth=1, alpha=0.8)
    ax3.fill_between(range(len(positions)), 0, positions, 
                     where=(positions > 0), color='green', alpha=0.3, label='Long')
    ax3.fill_between(range(len(positions)), 0, positions, 
                     where=(positions < 0), color='red', alpha=0.3, label='Short')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_title('Position History', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Days')
    ax3.set_ylabel('Position Size')
    ax3.set_ylim(-1.1, 1.1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Daily returns distribution
    ax4 = plt.subplot(3, 3, 4)
    daily_returns = np.array(env.returns) * 100
    ax4.hist(daily_returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax4.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Return (%)')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"Mean: {np.mean(daily_returns):.2f}%\nStd: {np.std(daily_returns):.2f}%"
    ax4.text(0.7, 0.9, stats_text, transform=ax4.transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 5. Drawdown
    ax5 = plt.subplot(3, 3, 5)
    cumulative = np.cumprod(1 + np.array(env.returns))
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max * 100
    ax5.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
    ax5.plot(drawdown, color='red', linewidth=1)
    ax5.set_title('Drawdown (%)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Days')
    ax5.set_ylabel('Drawdown (%)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Training loss curves
    ax6 = plt.subplot(3, 3, 6)
    if history and 'actor_losses' in history and len(history['actor_losses']) > 0:
        ax6.plot(history['actor_losses'], label='Actor Loss', alpha=0.7)
        ax6.plot(history['critic_losses'], label='Critic Loss', alpha=0.7)
        ax6.set_title('Training Losses', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Updates')
        ax6.set_ylabel('Loss')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # 7. Episode rewards
    ax7 = plt.subplot(3, 3, 7)
    if history and 'episode_rewards' in history and len(history['episode_rewards']) > 0:
        rewards = history['episode_rewards']
        ax7.plot(rewards, alpha=0.5, linewidth=1)
        
        # Add moving average
        window = min(20, len(rewards) // 4)
        if window > 1:
            ma = pd.Series(rewards).rolling(window=window).mean()
            ax7.plot(ma, label=f'MA({window})', linewidth=2, color='red')
        
        ax7.set_title('Episode Rewards', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Episode')
        ax7.set_ylabel('Reward')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    # 8. Trade analysis
    ax8 = plt.subplot(3, 3, 8)
    if env.trades:
        trade_balances = [t['balance'] for t in env.trades]
        ax8.plot(trade_balances, marker='o', markersize=2, linewidth=1, alpha=0.7)
        ax8.set_title(f'Balance After Each Trade ({len(env.trades)} trades)', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Trade Number')
        ax8.set_ylabel('Balance ($)')
        ax8.grid(True, alpha=0.3)
    
    # 9. Performance metrics table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('tight')
    ax9.axis('off')
    
    metrics = env.get_metrics()
    
    # Calculate additional metrics
    total_profit = env.balance - env.initial_balance
    roi = (env.balance / env.initial_balance - 1) * 100
    
    # Create metrics table
    table_data = [
        ['Metric', 'Value'],
        ['Initial Balance', f'${env.initial_balance:,.2f}'],
        ['Final Balance', f'${env.balance:,.2f}'],
        ['Total Profit/Loss', f'${total_profit:,.2f}'],
        ['ROI', f'{roi:.2f}%'],
        ['Total Return', f'{metrics["total_return"]:.2%}'],
        ['Sharpe Ratio', f'{metrics["sharpe_ratio"]:.3f}'],
        ['Max Drawdown', f'{metrics["max_drawdown"]:.2%}'],
        ['Number of Trades', f'{metrics["num_trades"]}'],
        ['Win Rate', f'{metrics["win_rate"]:.2%}'],
        ['Avg Daily Return', f'{np.mean(env.returns):.4%}'],
    ]
    
    table = ax9.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the header row
    for i in range(2):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.suptitle('Trading Strategy Performance Report', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save figure
    save_path = Path(save_dir) / f'performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\n📊 Performance report saved to: {save_path}")
    
    return fig


def run_full_training(args):
    """Run complete training pipeline"""
    
    print("\n" + "="*80)
    print("🚀 FULL MODEL TRAINING WITH REALISTIC FEES")
    print("="*80)
    
    # Load data
    df = load_and_prepare_data(args.symbol, args.data_dir)
    
    # Split data
    train_size = int(len(df) * args.train_ratio)
    val_size = int(len(df) * args.val_ratio)
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    print(f"\n📊 Data Split:")
    print(f"  Training: {len(train_df)} days")
    print(f"  Validation: {len(val_df)} days")
    print(f"  Testing: {len(test_df)} days")
    
    # Select features
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 
                    'Rsi', 'Macd', 'Bb_Position', 'Volume_Ratio', 
                    'Volatility', 'High_Low_Ratio', 'Close_Open_Ratio']
    
    available_features = [f for f in feature_cols if f in train_df.columns]
    print(f"\n🔧 Using {len(available_features)} features")
    
    # Get realistic trading costs based on asset type
    crypto_symbols = ['btc', 'eth', 'crypto', 'usdt', 'usdc', 'bnb', 'sol', 'ada', 'doge', 'matic']
    is_crypto = any(s in args.symbol.lower() for s in crypto_symbols)
    
    if args.broker == 'auto':
        if is_crypto:
            asset_type = 'crypto'
            broker = 'default'  # 0.15% fee as you specified
        else:
            asset_type = 'stock'
            broker = 'alpaca'  # Zero commission
    else:
        # User specified broker
        broker = args.broker
        if broker in ['binance', 'coinbase']:
            asset_type = 'crypto'
        else:
            asset_type = 'stock'
    
    costs = get_trading_costs(asset_type, broker)
    
    # Create environments with realistic fees
    print(f"\n💰 Trading Costs ({asset_type.upper()} - {broker}):")
    print(f"  Commission: {costs.commission:.4%} (min ${costs.min_commission})")
    print(f"  Spread: {costs.spread_pct:.5%}")
    print(f"  Slippage: {costs.slippage_pct:.5%}")
    print(f"  Total cost per trade: ~{(costs.commission + costs.spread_pct + costs.slippage_pct):.4%}")
    
    train_env = DailyTradingEnv(
        train_df,
        window_size=args.window_size,
        initial_balance=args.initial_balance,
        transaction_cost=costs.commission,
        spread_pct=costs.spread_pct,
        slippage_pct=costs.slippage_pct,
        min_commission=costs.min_commission,
        features=available_features
    )
    
    val_env = DailyTradingEnv(
        val_df,
        window_size=args.window_size,
        initial_balance=args.initial_balance,
        transaction_cost=costs.commission,
        spread_pct=costs.spread_pct,
        slippage_pct=costs.slippage_pct,
        min_commission=costs.min_commission,
        features=available_features
    )
    
    test_env = DailyTradingEnv(
        test_df,
        window_size=args.window_size,
        initial_balance=args.initial_balance,
        transaction_cost=costs.commission,
        spread_pct=costs.spread_pct,
        slippage_pct=costs.slippage_pct,
        min_commission=costs.min_commission,
        features=available_features
    )
    
    # Create model
    print(f"\n🤖 Initializing Model...")
    input_dim = args.window_size * (len(available_features) + 3)
    agent = create_advanced_model(input_dim, use_toto=args.use_toto)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = agent.to(device)
    
    total_params = sum(p.numel() for p in agent.parameters())
    print(f"  Model parameters: {total_params:,}")
    print(f"  Device: {device}")
    
    # Create trainer
    trainer = PPOTrainer(
        agent,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        eps_clip=args.eps_clip,
        k_epochs=args.k_epochs,
        entropy_coef=args.entropy_coef,
        device=device,
        log_dir='./traininglogs'
    )
    
    # Training loop with progress bar
    print(f"\n🏋️ Training for {args.num_episodes} episodes...")
    print("="*80)
    
    best_val_reward = -np.inf
    patience_counter = 0
    
    with tqdm(total=args.num_episodes, desc="Training Progress") as pbar:
        for episode in range(args.num_episodes):
            # Train episode
            train_reward, train_length, train_info = trainer.train_episode(train_env)
            
            # Update policy
            if (episode + 1) % args.update_interval == 0:
                update_info = trainer.update()
                pbar.set_postfix({
                    'reward': f'{train_reward:.3f}',
                    'actor_loss': f'{update_info["actor_loss"]:.4f}',
                    'critic_loss': f'{update_info["critic_loss"]:.4f}'
                })
            
            # Validation
            if (episode + 1) % args.eval_interval == 0:
                val_env.reset()
                val_reward, _, val_info = trainer.train_episode(val_env, deterministic=True)
                val_metrics = val_env.get_metrics()
                
                tqdm.write(f"\n📈 Episode {episode + 1} Validation:")
                tqdm.write(f"  Return: {val_metrics['total_return']:.2%}")
                tqdm.write(f"  Sharpe: {val_metrics['sharpe_ratio']:.3f}")
                tqdm.write(f"  Trades: {val_metrics['num_trades']}")
                
                # Early stopping
                if val_reward > best_val_reward:
                    best_val_reward = val_reward
                    trainer.save_checkpoint('./models/best_model.pth')
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        tqdm.write(f"\n⚠️ Early stopping at episode {episode + 1}")
                        break
            
            # Save checkpoint
            if (episode + 1) % args.save_interval == 0:
                trainer.save_checkpoint(f'./models/checkpoint_ep{episode + 1}.pth')
            
            pbar.update(1)
    
    print("\n" + "="*80)
    print("🎯 FINAL EVALUATION ON TEST SET")
    print("="*80)
    
    # Load best model
    trainer.load_checkpoint('./models/best_model.pth')
    
    # Test evaluation
    test_env.reset()
    state = test_env.reset()
    done = False
    
    print("\n📊 Running test evaluation...")
    
    with torch.no_grad():
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, _, _ = agent.act(state_tensor, deterministic=True)
            action = action.cpu().numpy().flatten()
            state, _, done, _ = test_env.step(action)
    
    # Calculate final metrics
    final_metrics = test_env.get_metrics()
    
    # Calculate profit with fees
    total_profit = test_env.balance - test_env.initial_balance
    total_fees = sum([
        max(costs.commission * abs(t['new_position'] - t['old_position']) * t['balance'], 
            costs.min_commission) +
        costs.spread_pct * abs(t['new_position'] - t['old_position']) * t['balance'] +
        costs.slippage_pct * abs(t['new_position'] - t['old_position']) * t['balance']
        for t in test_env.trades
    ])
    
    print("\n💰 FINAL RESULTS:")
    print("="*80)
    print(f"  Initial Balance:     ${test_env.initial_balance:,.2f}")
    print(f"  Final Balance:       ${test_env.balance:,.2f}")
    print(f"  Total Profit/Loss:   ${total_profit:,.2f}")
    print(f"  Total Fees Paid:     ${total_fees:,.2f}")
    print(f"  Net Profit:          ${total_profit:,.2f}")
    print(f"  ROI:                 {(test_env.balance/test_env.initial_balance - 1)*100:.2f}%")
    print(f"  Total Return:        {final_metrics['total_return']:.2%}")
    print(f"  Sharpe Ratio:        {final_metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown:        {final_metrics['max_drawdown']:.2%}")
    print(f"  Total Trades:        {final_metrics['num_trades']}")
    print(f"  Win Rate:            {final_metrics['win_rate']:.2%}")
    print(f"  Avg Trade Cost:      ${total_fees/max(final_metrics['num_trades'], 1):.2f}")
    print("="*80)
    
    # Visualize results
    print("\n📊 Generating performance visualizations...")
    fig = visualize_results(test_env, trainer.training_history, './results')
    
    # Save detailed results
    results = {
        'symbol': args.symbol,
        'timestamp': datetime.now().isoformat(),
        'final_metrics': final_metrics,
        'financial_summary': {
            'initial_balance': test_env.initial_balance,
            'final_balance': test_env.balance,
            'total_profit': total_profit,
            'total_fees': total_fees,
            'net_profit': total_profit,
            'roi_percent': (test_env.balance/test_env.initial_balance - 1)*100
        },
        'hyperparameters': vars(args),
        'test_period': {
            'start': str(test_df['Date'].iloc[0]) if 'Date' in test_df.columns else 'N/A',
            'end': str(test_df['Date'].iloc[-1]) if 'Date' in test_df.columns else 'N/A',
            'days': len(test_df)
        }
    }
    
    results_path = Path('./results') / f'results_{args.symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\n📁 Results saved to: {results_path}")
    
    # Close trainer
    trainer.close()
    
    print("\n✅ Training complete!")
    print("\n📊 To view TensorBoard logs:")
    print("   tensorboard --logdir=./traininglogs")
    
    return test_env, final_metrics, total_profit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Full RL Trading Model Training')
    
    # Data parameters
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock/crypto symbol')
    parser.add_argument('--data_dir', type=str, default='../data', help='Data directory')
    parser.add_argument('--broker', type=str, default='auto', help='Broker/exchange (auto, alpaca, robinhood, binance, coinbase)')
    
    # Environment parameters
    parser.add_argument('--window_size', type=int, default=30, help='Observation window')
    parser.add_argument('--initial_balance', type=float, default=100000, help='Starting capital')
    
    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=500, help='Number of episodes')
    parser.add_argument('--update_interval', type=int, default=10, help='Update frequency')
    parser.add_argument('--eval_interval', type=int, default=25, help='Validation frequency')
    parser.add_argument('--save_interval', type=int, default=100, help='Checkpoint frequency')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    
    # Model parameters
    parser.add_argument('--use_toto', action='store_true', help='Use Toto backbone')
    parser.add_argument('--lr_actor', type=float, default=1e-4, help='Actor learning rate')
    parser.add_argument('--lr_critic', type=float, default=5e-4, help='Critic learning rate')
    parser.add_argument('--gamma', type=float, default=0.995, help='Discount factor')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='PPO clip')
    parser.add_argument('--k_epochs', type=int, default=4, help='PPO epochs')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient')
    
    # Data split
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation data ratio')
    
    args = parser.parse_args()
    
    # Run training
    env, metrics, profit = run_full_training(args)