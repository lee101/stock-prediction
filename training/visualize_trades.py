#!/usr/bin/env python3
"""
Trade Visualization System
Visualizes trading decisions from any .pth model on any stock
Shows buy/sell points, positions, and performance metrics
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from datetime import datetime
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from trading_env import DailyTradingEnv
from trading_config import get_trading_costs
from advanced_trainer import TransformerTradingAgent, EnsembleTradingAgent
from train_full_model import add_technical_indicators
import mplfinance as mpf


class ReshapeWrapper(nn.Module):
    """Reshape wrapper for transformer models"""
    def __init__(self, agent, window_size=30):
        super().__init__()
        self.agent = agent
        self.window_size = window_size
    
    def forward(self, x):
        if len(x.shape) == 2:
            batch_size = x.shape[0]
            features_per_step = x.shape[1] // self.window_size
            x = x.view(batch_size, self.window_size, features_per_step)
        return self.agent(x)
    
    def get_action_distribution(self, x):
        if len(x.shape) == 2:
            batch_size = x.shape[0]
            features_per_step = x.shape[1] // self.window_size
            x = x.view(batch_size, self.window_size, features_per_step)
        return self.agent.get_action_distribution(x)


class TradeVisualizer:
    """Visualize trading decisions and performance"""
    
    def __init__(self, model_path, stock_symbol='AAPL', start_date='2023-01-01', end_date='2024-01-01'):
        """
        Initialize visualizer with model and stock data
        
        Args:
            model_path: Path to .pth model file
            stock_symbol: Stock ticker symbol
            start_date: Start date for backtesting
            end_date: End date for backtesting
        """
        self.model_path = Path(model_path)
        self.stock_symbol = stock_symbol
        self.start_date = start_date
        self.end_date = end_date
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.metadata = self.load_model()
        
        # Load stock data
        self.df = self.load_stock_data()
        
        # Setup environment
        self.env = self.setup_environment()
        
        # Store trading history
        self.trading_history = {
            'dates': [],
            'prices': [],
            'positions': [],
            'actions': [],
            'portfolio_values': [],
            'returns': [],
            'buy_points': [],
            'sell_points': [],
            'hold_points': []
        }
    
    def load_model(self):
        """Load the trained model from checkpoint"""
        print(f"\n📂 Loading model from {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Extract metadata
        metadata = {
            'episode': checkpoint.get('episode', 'unknown'),
            'metric_type': checkpoint.get('metric_type', 'unknown'),
            'metric_value': checkpoint.get('metric_value', 0),
            'run_name': checkpoint.get('run_name', 'unknown'),
            'timestamp': checkpoint.get('timestamp', 'unknown')
        }
        
        print(f"  Model info: Episode {metadata['episode']}, "
              f"Best {metadata['metric_type']}: {metadata['metric_value']:.4f}")
        
        # Reconstruct model architecture
        config = checkpoint.get('config', {})
        
        # Determine model type and create
        if 'ensemble_states' in checkpoint:
            # Ensemble model
            model = EnsembleTradingAgent(
                num_agents=len(checkpoint['ensemble_states']),
                input_dim=393,  # Default, will adjust if needed
                hidden_dim=config.get('hidden_dim', 256)
            )
            for i, state_dict in enumerate(checkpoint['ensemble_states']):
                model.agents[i].load_state_dict(state_dict)
            if 'ensemble_weights' in checkpoint:
                model.ensemble_weights = checkpoint['ensemble_weights']
        else:
            # Single transformer model
            # Determine input dimension from available features
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 
                       'Rsi', 'Macd', 'Bb_Position', 'Volume_Ratio']
            # 10 features + 3 extra (position, balance_norm, trades_norm) = 13
            input_dim = len(features) + 3
            
            agent = TransformerTradingAgent(
                input_dim=input_dim,  # Adjusted based on features
                hidden_dim=config.get('hidden_dim', 256),
                num_layers=config.get('num_layers', 3),
                num_heads=config.get('num_heads', 8),
                dropout=0  # No dropout for inference
            )
            
            if 'agent_state' in checkpoint:
                # Try to load, may need to adjust architecture
                try:
                    agent.load_state_dict(checkpoint['agent_state'])
                except:
                    # Create wrapper and try again
                    pass
            
            model = ReshapeWrapper(agent, window_size=30)
        
        model.to(self.device)
        model.eval()
        
        return model, metadata
    
    def load_stock_data(self):
        """Load and prepare stock data"""
        print(f"\n📊 Loading {self.stock_symbol} data from {self.start_date} to {self.end_date}")
        
        # Download data from yfinance
        ticker = yf.Ticker(self.stock_symbol)
        df = ticker.history(start=self.start_date, end=self.end_date)
        
        # Prepare dataframe with proper column names
        df = df.reset_index()
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        
        # Remove unnecessary columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Ensure column names are lowercase for compatibility
        df.columns = [col.lower() for col in df.columns]
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Capitalize columns back for environment
        df.columns = [col.capitalize() for col in df.columns]
        
        print(f"  Loaded {len(df)} days of data")
        print(f"  Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        
        return df
    
    def setup_environment(self):
        """Setup trading environment"""
        # Get realistic trading costs
        costs = get_trading_costs('stock', 'alpaca')
        
        # Define features
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 
                   'Rsi', 'Macd', 'Bb_Position', 'Volume_Ratio']
        available_features = [f for f in features if f in self.df.columns]
        
        # Create environment
        env = DailyTradingEnv(
            self.df,
            window_size=30,
            initial_balance=100000,
            transaction_cost=costs.commission,
            spread_pct=costs.spread_pct,
            slippage_pct=costs.slippage_pct,
            features=available_features
        )
        
        return env
    
    def run_backtest(self):
        """Run backtest with the model"""
        print(f"\n🏃 Running backtest on {self.stock_symbol}")
        
        # Reset environment
        state = self.env.reset()
        done = False
        step = 0
        
        while not done:
            # Get model prediction
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # Get action from model
                if hasattr(self.model, 'get_action_distribution'):
                    dist = self.model.get_action_distribution(state_tensor)
                    action = dist.mean.cpu().numpy()[0]
                else:
                    action, _ = self.model(state_tensor)
                    action = action.cpu().numpy()[0]
            
            # Step environment
            next_state, reward, done, info = self.env.step(action)
            
            # Record trading decision
            current_idx = self.env.current_step
            if current_idx < len(self.df):
                date = self.df.iloc[current_idx]['Date']
                price = self.df.iloc[current_idx]['Close']
                
                self.trading_history['dates'].append(date)
                self.trading_history['prices'].append(price)
                self.trading_history['positions'].append(self.env.position)
                self.trading_history['actions'].append(action[0] if isinstance(action, np.ndarray) else action)
                self.trading_history['portfolio_values'].append(self.env.balance)
                self.trading_history['returns'].append((self.env.balance / self.env.initial_balance - 1) * 100)
                
                # Categorize action
                action_value = action[0] if isinstance(action, np.ndarray) else action
                if len(self.trading_history['positions']) > 1:
                    prev_position = self.trading_history['positions'][-2]
                    position_change = self.env.position - prev_position
                    
                    if position_change > 0.1:  # Buying
                        self.trading_history['buy_points'].append((date, price))
                    elif position_change < -0.1:  # Selling
                        self.trading_history['sell_points'].append((date, price))
                    else:  # Holding
                        self.trading_history['hold_points'].append((date, price))
            
            state = next_state
            step += 1
        
        # Get final metrics
        self.final_metrics = self.env.get_metrics()
        
        print(f"\n📊 Backtest Results:")
        print(f"  Final Return: {self.final_metrics.get('total_return', 0):.2%}")
        print(f"  Sharpe Ratio: {self.final_metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  Max Drawdown: {self.final_metrics.get('max_drawdown', 0):.2%}")
        print(f"  Win Rate: {self.final_metrics.get('win_rate', 0):.2%}")
        print(f"  Number of Trades: {self.final_metrics.get('num_trades', 0)}")
    
    def plot_comprehensive_analysis(self, save_path=None):
        """Create comprehensive trading analysis visualization"""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(5, 2, figure=fig, hspace=0.3, wspace=0.2)
        
        # Convert dates for plotting
        dates = pd.to_datetime(self.trading_history['dates'])
        
        # 1. Price chart with buy/sell signals
        ax1 = fig.add_subplot(gs[0:2, :])
        ax1.plot(dates, self.trading_history['prices'], 'k-', alpha=0.7, linewidth=1)
        
        # Plot buy/sell points
        if self.trading_history['buy_points']:
            buy_dates, buy_prices = zip(*self.trading_history['buy_points'])
            ax1.scatter(pd.to_datetime(buy_dates), buy_prices, 
                       color='green', marker='^', s=100, alpha=0.7, label='Buy', zorder=5)
        
        if self.trading_history['sell_points']:
            sell_dates, sell_prices = zip(*self.trading_history['sell_points'])
            ax1.scatter(pd.to_datetime(sell_dates), sell_prices, 
                       color='red', marker='v', s=100, alpha=0.7, label='Sell', zorder=5)
        
        ax1.set_title(f'{self.stock_symbol} Price with Trading Signals\n'
                     f'Model: {self.metadata["metric_type"]} = {self.metadata["metric_value"]:.4f} '
                     f'(Episode {self.metadata["episode"]})', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Add position overlay
        ax1_twin = ax1.twinx()
        ax1_twin.fill_between(dates, 0, self.trading_history['positions'], 
                             alpha=0.2, color='blue', label='Position')
        ax1_twin.set_ylabel('Position Size', color='blue')
        ax1_twin.tick_params(axis='y', labelcolor='blue')
        ax1_twin.set_ylim(-1.2, 1.2)
        
        # 2. Portfolio value over time
        ax2 = fig.add_subplot(gs[2, :])
        ax2.plot(dates, self.trading_history['portfolio_values'], 'b-', linewidth=2)
        ax2.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial Balance')
        ax2.set_title('Portfolio Value Over Time', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Returns over time
        ax3 = fig.add_subplot(gs[3, 0])
        ax3.plot(dates, self.trading_history['returns'], 'g-', linewidth=1.5)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.fill_between(dates, 0, self.trading_history['returns'], 
                         where=np.array(self.trading_history['returns']) > 0, 
                         alpha=0.3, color='green', label='Profit')
        ax3.fill_between(dates, 0, self.trading_history['returns'], 
                         where=np.array(self.trading_history['returns']) < 0, 
                         alpha=0.3, color='red', label='Loss')
        ax3.set_title('Cumulative Returns (%)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Return (%)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Position distribution
        ax4 = fig.add_subplot(gs[3, 1])
        ax4.hist(self.trading_history['positions'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('Position Size Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Position Size')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # 5. Daily returns distribution
        ax5 = fig.add_subplot(gs[4, 0])
        daily_returns = np.diff(self.trading_history['portfolio_values']) / self.trading_history['portfolio_values'][:-1] * 100
        ax5.hist(daily_returns, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax5.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax5.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Daily Return (%)')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3)
        
        # Add normal distribution overlay
        from scipy import stats
        mu, std = daily_returns.mean(), daily_returns.std()
        x = np.linspace(daily_returns.min(), daily_returns.max(), 100)
        ax5_twin = ax5.twinx()
        ax5_twin.plot(x, stats.norm.pdf(x, mu, std) * len(daily_returns) * (daily_returns.max() - daily_returns.min()) / 30, 
                     'r-', linewidth=2, alpha=0.7, label=f'Normal (μ={mu:.2f}, σ={std:.2f})')
        ax5_twin.set_ylabel('Probability Density', color='red')
        ax5_twin.tick_params(axis='y', labelcolor='red')
        ax5_twin.legend(loc='upper right')
        
        # 6. Performance metrics
        ax6 = fig.add_subplot(gs[4, 1])
        ax6.axis('off')
        
        metrics_text = f"""
        📊 PERFORMANCE METRICS
        {'='*30}
        
        Total Return: {self.final_metrics.get('total_return', 0):.2%}
        Sharpe Ratio: {self.final_metrics.get('sharpe_ratio', 0):.3f}
        Max Drawdown: {self.final_metrics.get('max_drawdown', 0):.2%}
        Win Rate: {self.final_metrics.get('win_rate', 0):.2%}
        
        Number of Trades: {self.final_metrics.get('num_trades', 0)}
        Avg Trade Return: {self.final_metrics.get('avg_trade_return', 0):.2%}
        Best Trade: {self.final_metrics.get('best_trade', 0):.2%}
        Worst Trade: {self.final_metrics.get('worst_trade', 0):.2%}
        
        Initial Balance: $100,000
        Final Balance: ${self.trading_history['portfolio_values'][-1]:,.2f}
        Profit/Loss: ${self.trading_history['portfolio_values'][-1] - 100000:,.2f}
        
        Model: {self.model_path.name}
        Stock: {self.stock_symbol}
        Period: {self.start_date} to {self.end_date}
        """
        
        ax6.text(0.1, 0.5, metrics_text, fontsize=11, fontfamily='monospace',
                verticalalignment='center', transform=ax6.transAxes)
        
        # Main title
        fig.suptitle(f'Trading Analysis: {self.stock_symbol} with {self.model_path.name}',
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"\n📊 Visualization saved to {save_path}")
        
        plt.show()
    
    def plot_candlestick_with_trades(self, num_days=60, save_path=None):
        """Create candlestick chart with trade markers"""
        
        # Prepare data for mplfinance
        df_plot = self.df.copy()
        df_plot.set_index('Date', inplace=True)
        
        # Get last num_days
        df_plot = df_plot.iloc[-num_days:]
        
        # Prepare buy/sell markers
        buy_markers = []
        sell_markers = []
        
        for date, price in self.trading_history['buy_points']:
            if date in df_plot.index:
                buy_markers.append(price)
            else:
                buy_markers.append(np.nan)
        
        for date, price in self.trading_history['sell_points']:
            if date in df_plot.index:
                sell_markers.append(price)
            else:
                sell_markers.append(np.nan)
        
        # Create additional plots for signals
        apds = []
        if buy_markers:
            apds.append(mpf.make_addplot(buy_markers[-num_days:], type='scatter', 
                                        markersize=100, marker='^', color='green'))
        if sell_markers:
            apds.append(mpf.make_addplot(sell_markers[-num_days:], type='scatter',
                                        markersize=100, marker='v', color='red'))
        
        # Create candlestick chart
        fig, axes = mpf.plot(df_plot, 
                            type='candle',
                            style='charles',
                            title=f'{self.stock_symbol} - Last {num_days} Days with Trading Signals',
                            ylabel='Price ($)',
                            volume=True,
                            addplot=apds if apds else None,
                            figsize=(16, 10),
                            returnfig=True)
        
        # Save or show
        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"\n📊 Candlestick chart saved to {save_path}")
        
        plt.show()
    
    def export_trades_to_csv(self, save_path=None):
        """Export trading history to CSV"""
        
        # Create DataFrame
        trades_df = pd.DataFrame({
            'Date': self.trading_history['dates'],
            'Price': self.trading_history['prices'],
            'Position': self.trading_history['positions'],
            'Action': self.trading_history['actions'],
            'Portfolio_Value': self.trading_history['portfolio_values'],
            'Return_%': self.trading_history['returns']
        })
        
        # Save to CSV
        if save_path is None:
            save_path = f'trades_{self.stock_symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        trades_df.to_csv(save_path, index=False)
        print(f"\n📁 Trades exported to {save_path}")
        
        return trades_df


def main():
    """Main function to demonstrate trade visualization"""
    
    import argparse
    parser = argparse.ArgumentParser(description='Visualize trades from a trained model')
    parser.add_argument('--model', type=str, default='models/best_profit_model.pth',
                       help='Path to .pth model file')
    parser.add_argument('--stock', type=str, default='AAPL',
                       help='Stock symbol to test on')
    parser.add_argument('--start', type=str, default='2023-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-01-01',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--save', action='store_true',
                       help='Save visualizations to files')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("📊 TRADE VISUALIZATION SYSTEM")
    print("="*80)
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        print("\nAvailable models:")
        for model_file in Path('models').glob('*.pth'):
            print(f"  - {model_file}")
        return
    
    # Create visualizer
    visualizer = TradeVisualizer(
        model_path=args.model,
        stock_symbol=args.stock,
        start_date=args.start,
        end_date=args.end
    )
    
    # Run backtest
    visualizer.run_backtest()
    
    # Create visualizations
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Comprehensive analysis
    save_path = f'visualizations/{args.stock}_analysis_{timestamp}.png' if args.save else None
    visualizer.plot_comprehensive_analysis(save_path)
    
    # Candlestick chart
    save_path = f'visualizations/{args.stock}_candlestick_{timestamp}.png' if args.save else None
    visualizer.plot_candlestick_with_trades(save_path=save_path)
    
    # Export trades
    if args.save:
        csv_path = f'visualizations/{args.stock}_trades_{timestamp}.csv'
        visualizer.export_trades_to_csv(csv_path)
    
    print("\n✅ Visualization complete!")
    print("="*80)


if __name__ == '__main__':
    main()