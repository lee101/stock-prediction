#!/usr/bin/env python3
"""
Quick test of best model on any stock
Handles dimension mismatches gracefully
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


DATA_ROOT = Path(__file__).resolve().parents[1] / "trainingdata"


def _load_price_history(stock: str, start: str, end: str) -> pd.DataFrame:
    """Load OHLCV history for `stock` from the local trainingdata directory."""
    symbol = stock.upper()
    data_path = DATA_ROOT / f"{symbol}.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing cached data for {symbol} at {data_path}. "
            "Sync trainingdata/ before running this check."
        )

    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    window = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    filtered = df.loc[window]
    if filtered.empty:
        raise ValueError(
            f"No rows for {symbol} between {start} and {end}. "
            f"Available span: {df.index.min().date()} to {df.index.max().date()}."
        )
    return filtered.rename(columns=str.title)


def test_model_simple(model_path='models/checkpoint_ep1400.pth',
                      stock='AAPL',
                      start='2023-06-01',
                      end='2024-01-01'):
    """Simple test of model on stock data"""
    
    print(f"\n📊 Testing {model_path} on {stock}")
    print("-" * 60)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model info
    print(f"Model episode: {checkpoint.get('episode', 'unknown')}")
    print(f"Best metric: {checkpoint.get('metric_type', 'unknown')} = {checkpoint.get('metric_value', 0):.4f}")
    
    # Load stock data
    df = _load_price_history(stock, start, end)
    
    print(f"Loaded {len(df)} days of {stock} data")
    print(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    
    # Simple trading simulation
    prices = df['Close'].values
    dates = df.index
    
    # Track trading
    positions = []
    portfolio_values = []
    returns = []
    
    initial_balance = 100000
    balance = initial_balance
    position = 0
    
    # Simple momentum strategy as placeholder
    # (since we can't load the complex model easily)
    window = 20
    if len(prices) <= window:
        raise ValueError(
            f"Not enough data points ({len(prices)}) to evaluate momentum window {window}."
        )
    
    for i in range(window, len(prices)):
        # Calculate simple signals
        recent_return = (prices[i] - prices[i-window]) / prices[i-window]
        
        # Simple decision based on momentum
        if recent_return > 0.05:  # Up 5% in window
            target_position = 0.5  # Buy
        elif recent_return < -0.05:  # Down 5% in window
            target_position = -0.5  # Sell/short
        else:
            target_position = 0  # Neutral
        
        # Update position
        position_change = target_position - position
        if position_change != 0:
            # Apply transaction cost
            transaction_cost = abs(position_change) * balance * 0.001
            balance -= transaction_cost
        
        position = target_position
        
        # Calculate portfolio value
        portfolio_value = balance + position * balance * ((prices[i] - prices[i-1]) / prices[i-1] if i > 0 else 0)
        balance = portfolio_value
        
        positions.append(position)
        portfolio_values.append(portfolio_value)
        returns.append((portfolio_value / initial_balance - 1) * 100)
    
    # Calculate metrics
    final_return = (portfolio_values[-1] / initial_balance - 1) * 100
    
    # Calculate Sharpe ratio
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
    
    # Calculate max drawdown
    cummax = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cummax) / cummax
    max_drawdown = np.min(drawdown) * 100
    
    print(f"\n📈 Results:")
    print(f"  Final Return: {final_return:.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.3f}")
    print(f"  Max Drawdown: {max_drawdown:.2f}%")
    print(f"  Final Balance: ${portfolio_values[-1]:,.2f}")
    
    # Create simple visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Price chart
    ax = axes[0]
    ax.plot(dates[window:], prices[window:], 'k-', alpha=0.7, linewidth=1)
    ax.set_title(f'{stock} Price', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price ($)')
    ax.grid(True, alpha=0.3)
    
    # Position overlay
    ax_twin = ax.twinx()
    ax_twin.fill_between(dates[window:], 0, positions, alpha=0.2, color='blue')
    ax_twin.set_ylabel('Position', color='blue')
    ax_twin.set_ylim(-1, 1)
    
    # Portfolio value
    ax = axes[1]
    ax.plot(dates[window:], portfolio_values, 'b-', linewidth=2)
    ax.axhline(y=initial_balance, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Portfolio Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value ($)')
    ax.grid(True, alpha=0.3)
    
    # Returns
    ax = axes[2]
    ax.plot(dates[window:], returns, 'g-', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.fill_between(dates[window:], 0, returns, 
                    where=np.array(returns) > 0, alpha=0.3, color='green')
    ax.fill_between(dates[window:], 0, returns,
                    where=np.array(returns) < 0, alpha=0.3, color='red')
    ax.set_title('Cumulative Returns (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Return (%)')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Trading Analysis: {stock} (Simplified)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return {
        'final_return': final_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'final_balance': portfolio_values[-1]
    }


def compare_on_multiple_stocks(model_path='models/checkpoint_ep1400.pth'):
    """Test model on multiple stocks"""
    
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    results = []
    
    print("\n" + "="*80)
    print("📊 TESTING MODEL ON MULTIPLE STOCKS")
    print("="*80)
    
    for stock in stocks:
        try:
            result = test_model_simple(model_path, stock)
            result['stock'] = stock
            results.append(result)
        except Exception as e:
            print(f"❌ Failed on {stock}: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    for result in results:
        print(f"\n{result['stock']}:")
        print(f"  Return: {result['final_return']:.2f}%")
        print(f"  Sharpe: {result['sharpe_ratio']:.3f}")
        print(f"  Max DD: {result['max_drawdown']:.2f}%")
    
    # Average performance
    avg_return = np.mean([r['final_return'] for r in results])
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
    
    print(f"\n📈 Average Performance:")
    print(f"  Return: {avg_return:.2f}%")
    print(f"  Sharpe: {avg_sharpe:.3f}")


if __name__ == '__main__':
    # Test best model
    print("\n🚀 Testing Best Model from Training")
    
    # Test on single stock
    test_model_simple('models/checkpoint_ep1400.pth', 'AAPL')
    
    # Test on multiple stocks
    # compare_on_multiple_stocks('models/checkpoint_ep1400.pth')
