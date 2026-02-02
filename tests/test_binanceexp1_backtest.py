"""Simple backtest to verify trading strategy works."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.metrics_utils import compute_step_returns, annualized_sortino


class SimpleBacktester:
    """Simple backtester for binanceexp1 strategy."""
    
    def __init__(self, initial_cash=10000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0.0  # SOL holdings
        self.equity_history = []
        self.trades = []
        
    def get_equity(self, current_price):
        """Get current portfolio value."""
        return self.cash + (self.position * current_price)
    
    def execute_trade(self, timestamp, price, side, quantity):
        """Execute a trade."""
        if side == "buy" and quantity > 0:
            cost = quantity * price
            if cost <= self.cash:
                self.cash -= cost
                self.position += quantity
                self.trades.append({
                    "timestamp": timestamp,
                    "side": "buy",
                    "price": price,
                    "quantity": quantity,
                    "cost": cost
                })
                return True
        elif side == "sell" and quantity > 0:
            if quantity <= self.position:
                proceeds = quantity * price
                self.cash += proceeds
                self.position -= quantity
                self.trades.append({
                    "timestamp": timestamp,
                    "side": "sell",
                    "price": price,
                    "quantity": quantity,
                    "proceeds": proceeds
                })
                return True
        return False
    
    def run_simple_strategy(self, price_data, buy_threshold=-0.02, sell_threshold=0.02):
        """
        Run a simple momentum/mean-reversion strategy.
        
        Args:
            price_data: DataFrame with timestamp and close columns
            buy_threshold: Buy when price drops by this % from recent high
            sell_threshold: Sell when price rises by this % from entry
        """
        lookback = 24  # 24 hours
        entry_price = None
        
        for i in range(lookback, len(price_data)):
            row = price_data.iloc[i]
            timestamp = row['timestamp']
            current_price = row['close']
            
            # Track equity
            equity = self.get_equity(current_price)
            self.equity_history.append({
                "timestamp": timestamp,
                "equity": equity,
                "price": current_price
            })
            
            # Simple strategy: buy dips, sell rallies
            recent_prices = price_data['close'].iloc[i-lookback:i]
            recent_high = recent_prices.max()
            recent_low = recent_prices.min()
            
            # Calculate returns from recent high
            return_from_high = (current_price - recent_high) / recent_high
            
            # No position: look to buy
            if self.position == 0:
                if return_from_high < buy_threshold:
                    # Buy signal: price dropped from recent high
                    max_quantity = self.cash / current_price
                    quantity = max_quantity * 0.5  # Use 50% of cash
                    if self.execute_trade(timestamp, current_price, "buy", quantity):
                        entry_price = current_price
            
            # Have position: look to sell
            elif entry_price is not None:
                return_from_entry = (current_price - entry_price) / entry_price
                if return_from_entry > sell_threshold:
                    # Sell signal: profit target hit
                    self.execute_trade(timestamp, current_price, "sell", self.position)
                    entry_price = None
                elif return_from_entry < -0.05:  # Stop loss at -5%
                    self.execute_trade(timestamp, current_price, "sell", self.position)
                    entry_price = None
        
        # Close position at end
        if self.position > 0:
            final_price = price_data.iloc[-1]['close']
            self.execute_trade(
                price_data.iloc[-1]['timestamp'],
                final_price,
                "sell",
                self.position
            )
        
        return pd.DataFrame(self.equity_history)


def test_backtest_with_real_data():
    """Test backtest with real SOLUSD data."""
    data_file = Path("trainingdatahourly/crypto/SOLUSD.csv")
    
    if not data_file.exists():
        pytest.skip(f"Data file not found: {data_file}")
    
    # Load data
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Use recent data (last 1000 hours ~= 40 days)
    df_recent = df.tail(1000).reset_index(drop=True)
    
    # Run backtest
    backtester = SimpleBacktester(initial_cash=10000)
    equity_curve = backtester.run_simple_strategy(
        df_recent,
        buy_threshold=-0.03,  # Buy on 3% dip
        sell_threshold=0.02    # Sell on 2% gain
    )
    
    # Calculate metrics
    final_equity = equity_curve['equity'].iloc[-1]
    total_return = (final_equity - backtester.initial_cash) / backtester.initial_cash
    
    returns = compute_step_returns(equity_curve['equity'].values)
    sortino = annualized_sortino(returns, periods_per_year=24*365)
    
    num_trades = len(backtester.trades)
    
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Initial Cash:    ${backtester.initial_cash:,.2f}")
    print(f"Final Equity:    ${final_equity:,.2f}")
    print(f"Total Return:    {total_return*100:.2f}%")
    print(f"Sortino Ratio:   {sortino:.2f}")
    print(f"Num Trades:      {num_trades}")
    print(f"{'='*60}\n")
    
    # Assert basic sanity checks
    assert final_equity > 0, "Final equity should be positive"
    assert num_trades > 0, "Should have executed some trades"
    assert abs(total_return) < 10, "Return should be reasonable (< 1000%)"
    assert np.isfinite(sortino), "Sortino should be finite"
    
    # The strategy should at least not lose everything
    assert final_equity > backtester.initial_cash * 0.5, "Should not lose more than 50%"


def test_backtest_synthetic_uptrend():
    """Test with synthetic uptrending data."""
    # Create synthetic uptrend with noise
    np.random.seed(42)
    hours = 500
    timestamps = pd.date_range('2025-01-01', periods=hours, freq='h')
    
    # Generate uptrending prices with volatility
    base_price = 100
    trend = np.linspace(0, 20, hours)  # 20% uptrend
    noise = np.random.normal(0, 2, hours)  # 2% volatility
    prices = base_price + trend + noise
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices
    })
    
    # Run backtest
    backtester = SimpleBacktester(initial_cash=10000)
    equity_curve = backtester.run_simple_strategy(
        df,
        buy_threshold=-0.02,
        sell_threshold=0.015
    )
    
    # Calculate metrics
    final_equity = equity_curve['equity'].iloc[-1]
    total_return = (final_equity - backtester.initial_cash) / backtester.initial_cash
    
    print(f"\nSynthetic Uptrend Test:")
    print(f"  Initial: ${backtester.initial_cash:,.2f}")
    print(f"  Final:   ${final_equity:,.2f}")
    print(f"  Return:  {total_return*100:.2f}%")
    print(f"  Trades:  {len(backtester.trades)}")
    
    # In an uptrend, strategy should make money
    assert final_equity > backtester.initial_cash, "Should profit in uptrend"
    assert len(backtester.trades) > 0, "Should execute trades"


def test_backtest_synthetic_downtrend():
    """Test with synthetic downtrending data."""
    np.random.seed(43)
    hours = 500
    timestamps = pd.date_range('2025-01-01', periods=hours, freq='h')
    
    # Generate downtrending prices
    base_price = 100
    trend = np.linspace(0, -15, hours)  # -15% downtrend
    noise = np.random.normal(0, 1.5, hours)
    prices = base_price + trend + noise
    prices = np.maximum(prices, 50)  # Floor at 50
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices
    })
    
    # Run backtest
    backtester = SimpleBacktester(initial_cash=10000)
    equity_curve = backtester.run_simple_strategy(
        df,
        buy_threshold=-0.03,
        sell_threshold=0.02
    )
    
    final_equity = equity_curve['equity'].iloc[-1]
    total_return = (final_equity - backtester.initial_cash) / backtester.initial_cash
    
    print(f"\nSynthetic Downtrend Test:")
    print(f"  Initial: ${backtester.initial_cash:,.2f}")
    print(f"  Final:   ${final_equity:,.2f}")
    print(f"  Return:  {total_return*100:.2f}%")
    print(f"  Trades:  {len(backtester.trades)}")
    
    # In downtrend with stop losses, should limit losses
    assert final_equity > backtester.initial_cash * 0.7, "Should limit losses with stops"


def test_backtest_synthetic_sideways():
    """Test with synthetic sideways/ranging data."""
    np.random.seed(44)
    hours = 500
    timestamps = pd.date_range('2025-01-01', periods=hours, freq='h')
    
    # Generate mean-reverting sideways prices
    base_price = 100
    noise = np.random.normal(0, 3, hours)
    # Add sine wave for cyclical movement
    cycles = np.sin(np.linspace(0, 4*np.pi, hours)) * 5
    prices = base_price + noise + cycles
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices
    })
    
    # Run backtest
    backtester = SimpleBacktester(initial_cash=10000)
    equity_curve = backtester.run_simple_strategy(
        df,
        buy_threshold=-0.025,
        sell_threshold=0.015
    )
    
    final_equity = equity_curve['equity'].iloc[-1]
    total_return = (final_equity - backtester.initial_cash) / backtester.initial_cash
    
    print(f"\nSynthetic Sideways Test:")
    print(f"  Initial: ${backtester.initial_cash:,.2f}")
    print(f"  Final:   ${final_equity:,.2f}")
    print(f"  Return:  {total_return*100:.2f}%")
    print(f"  Trades:  {len(backtester.trades)}")
    
    # In sideways market, should have many trades
    # Note: Good mean-reversion strategies can make excellent returns in ranging markets!
    assert len(backtester.trades) > 5, "Should trade frequently in sideways market"
    assert abs(total_return) < 5.0, "Should have reasonable returns in sideways market"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
