#!/usr/bin/env python3
"""
Local Backtesting System for Trading Strategies
Simulates day-by-day trading using cached historical data and AI forecasts
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from loguru import logger
import sys

# Import existing modules
# Import with fallback for missing dependencies
try:
    from predict_stock_forecasting import make_predictions, load_stock_data_from_csv
except ImportError:
    logger.warning("Could not import predict_stock_forecasting - AI forecasts will not be available")
    make_predictions = None
    def load_stock_data_from_csv(path):
        return pd.read_csv(path)

try:
    from data_curate_daily import download_daily_stock_data
except ImportError:
    logger.warning("Could not import data_curate_daily")
    download_daily_stock_data = None
from src.fixtures import crypto_symbols
try:
    from src.sizing_utils import get_qty as _risk_get_qty
except Exception as exc:
    _risk_get_qty = None
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
logger.add("simulationresults/backtesting.log", rotation="10 MB")

if _risk_get_qty is None:
    logger.warning("Risk-weighted sizing offline: falling back to an equal-weight approximation.")

class LocalBacktester:
    """Main backtesting engine for simulating trading strategies locally"""
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 trading_fee: float = 0.001,  # 0.1% per trade
                 slippage: float = 0.0005,     # 0.05% slippage
                 max_positions: int = 5,
                 simulation_days: int = 25,
                 forecast_horizon: int = 7):    # 7-day forecast period
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trading_fee = trading_fee
        self.slippage = slippage
        self.max_positions = max_positions
        self.simulation_days = simulation_days
        self.forecast_horizon = forecast_horizon
        self._risk_sizer = _risk_get_qty
        
        # Portfolio state
        self.positions = {}  # symbol -> {'qty': float, 'entry_price': float, 'entry_date': datetime}
        self.cash = initial_capital
        self.portfolio_value_history = []
        self.trade_history = []
        self.daily_metrics = []
        
        # Setup directories
        self.results_dir = Path("simulationresults")
        self.results_dir.mkdir(exist_ok=True)
        self.cache_dir = Path("backtests/data_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_historical_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Load historical data from cache or download if needed"""
        logger.info(f"Loading historical data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        historical_data = {}
        
        for symbol in symbols:
            cache_file = self.cache_dir / f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            
            if cache_file.exists():
                logger.debug(f"Loading {symbol} from cache: {cache_file}")
                df = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
                historical_data[symbol] = df
            else:
                logger.info(f"Cache miss for {symbol}, downloading data...")
                # Download data using existing infrastructure
                try:
                    # Use the data directory structure from the existing system
                    data_path = Path("data") / symbol.replace("/", "-") 
                    csv_files = list(data_path.glob("*.csv"))
                    
                    if csv_files:
                        # Use most recent file
                        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
                        df = load_stock_data_from_csv(latest_file)
                        
                        # Filter to date range
                        if 'Date' in df.columns:
                            df['Date'] = pd.to_datetime(df['Date'])
                            df = df.set_index('Date')
                        
                        df = df.loc[start_date:end_date]
                        
                        # Cache the data
                        df.to_csv(cache_file)
                        historical_data[symbol] = df
                        logger.info(f"Cached {symbol} data to {cache_file}")
                    else:
                        logger.warning(f"No data files found for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error loading data for {symbol}: {e}")
                    
        return historical_data
    
    def generate_forecast_cache(self, symbols: List[str], forecast_date: datetime) -> Dict[str, Dict]:
        """Generate or load forecasts for a given date"""
        cache_file = self.cache_dir / f"forecasts_{forecast_date.strftime('%Y%m%d')}.json"
        
        if cache_file.exists():
            logger.debug(f"Loading forecasts from cache: {cache_file}")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        logger.info(f"Generating forecasts for {forecast_date}")
        
        # For simulation, we'll use historical price movements as "forecasts"
        # In production, this would call the AI model
        forecasts = {}
        
        for symbol in symbols:
            # Generate realistic forecast based on historical volatility
            # This is a placeholder - in production, use the real AI model
            forecasts[symbol] = {
                'close_total_predicted_change': np.random.normal(0.005, 0.02),  # 0.5% mean, 2% std
                'high_predicted_change': np.random.normal(0.01, 0.03),
                'low_predicted_change': np.random.normal(-0.01, 0.03),
                'confidence': np.random.uniform(0.4, 0.9),
                'forecast_date': forecast_date.isoformat(),
                'forecast_horizon_days': self.forecast_horizon
            }
        
        # Cache the forecasts
        with open(cache_file, 'w') as f:
            json.dump(forecasts, f, indent=2)
        
        return forecasts
    
    def calculate_position_size(self, symbol: str, price: float, strategy: str = 'equal_weight') -> float:
        """Calculate position size based on strategy"""
        if strategy == 'equal_weight':
            # Equal weight across max positions
            position_value = self.cash / self.max_positions
            return position_value / price
        
        elif strategy == 'risk_weighted':
            if self._risk_sizer is not None:
                # Use the existing risk-based sizing helper when available
                return self._risk_sizer(symbol, price, list(self.positions.values()))

            # Fall back to an equal-weight allocation when live sizing is unavailable
            logger.debug(
                "Risk-weighted sizing unavailable; defaulting to equal-weight sizing for {}",
                symbol,
            )
            fallback_positions = max(self.max_positions, 1)
            position_value = self.cash / fallback_positions
            return position_value / price
        
        elif strategy == 'single_position':
            # All capital in one position
            return (self.cash * 0.95) / price  # Keep 5% cash buffer
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def execute_trade(self, symbol: str, qty: float, price: float, trade_type: str, date: datetime):
        """Execute a trade and update portfolio"""
        # Calculate costs
        trade_value = qty * price
        fee = trade_value * self.trading_fee
        slippage_cost = trade_value * self.slippage
        total_cost = trade_value + fee + slippage_cost
        
        if trade_type == 'buy':
            if total_cost > self.cash:
                logger.warning(f"Insufficient cash for {symbol}: need ${total_cost:.2f}, have ${self.cash:.2f}")
                return False
            
            self.cash -= total_cost
            
            if symbol in self.positions:
                # Add to existing position
                old_qty = self.positions[symbol]['qty']
                old_price = self.positions[symbol]['entry_price']
                new_qty = old_qty + qty
                # Calculate weighted average price
                new_price = (old_qty * old_price + qty * price) / new_qty
                
                self.positions[symbol] = {
                    'qty': new_qty,
                    'entry_price': new_price,
                    'entry_date': date
                }
            else:
                self.positions[symbol] = {
                    'qty': qty,
                    'entry_price': price,
                    'entry_date': date
                }
            
            self.trade_history.append({
                'date': date,
                'symbol': symbol,
                'type': 'buy',
                'qty': qty,
                'price': price,
                'fee': fee,
                'slippage': slippage_cost,
                'total_cost': total_cost
            })
            
            logger.info(f"BUY {qty:.4f} {symbol} @ ${price:.2f} (total cost: ${total_cost:.2f})")
            return True
            
        elif trade_type == 'sell':
            if symbol not in self.positions:
                logger.error(f"No position to sell for {symbol}")
                return False
            
            position = self.positions[symbol]
            if qty > position['qty']:
                logger.warning(f"Trying to sell more than owned: {qty} > {position['qty']}")
                qty = position['qty']
            
            # Calculate proceeds
            proceeds = trade_value - fee - slippage_cost
            self.cash += proceeds
            
            # Calculate profit
            cost_basis = qty * position['entry_price']
            profit = proceeds - cost_basis
            
            # Update or remove position
            if qty >= position['qty']:
                del self.positions[symbol]
            else:
                position['qty'] -= qty
            
            self.trade_history.append({
                'date': date,
                'symbol': symbol,
                'type': 'sell',
                'qty': qty,
                'price': price,
                'fee': fee,
                'slippage': slippage_cost,
                'proceeds': proceeds,
                'profit': profit,
                'return_pct': (profit / cost_basis) * 100
            })
            
            logger.info(f"SELL {qty:.4f} {symbol} @ ${price:.2f} (profit: ${profit:.2f}, {profit/cost_basis*100:.1f}%)")
            return True
    
    def calculate_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        position_value = sum(
            self.positions[symbol]['qty'] * prices.get(symbol, self.positions[symbol]['entry_price'])
            for symbol in self.positions
        )
        return self.cash + position_value
    
    def run_backtest(self, symbols: List[str], strategy: str = 'equal_weight', 
                    start_date: Optional[datetime] = None) -> Dict:
        """Run the complete backtest simulation"""
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=self.simulation_days + 30)
        
        end_date = start_date + timedelta(days=self.simulation_days)
        
        logger.info(f"Starting backtest: {strategy} strategy, {start_date} to {end_date}")
        
        # Load all historical data
        historical_data = self.load_historical_data(symbols, 
                                                   start_date - timedelta(days=365), 
                                                   end_date + timedelta(days=30))
        
        # Simulation loop - day by day
        current_date = start_date
        day_count = 0
        
        while current_date <= end_date:
            logger.info(f"\n=== Day {day_count + 1}: {current_date.strftime('%Y-%m-%d')} ===")
            
            # Get current prices
            current_prices = {}
            for symbol, df in historical_data.items():
                if current_date in df.index:
                    current_prices[symbol] = df.loc[current_date, 'Close']
                else:
                    # Use last known price
                    valid_dates = df.index[df.index <= current_date]
                    if len(valid_dates) > 0:
                        current_prices[symbol] = df.loc[valid_dates[-1], 'Close']
            
            # Calculate current portfolio value
            portfolio_value = self.calculate_portfolio_value(current_prices)
            
            # Check for positions to close (after forecast horizon)
            positions_to_close = []
            for symbol, position in self.positions.items():
                holding_days = (current_date - position['entry_date']).days
                if holding_days >= self.forecast_horizon:
                    positions_to_close.append(symbol)
            
            # Close mature positions
            for symbol in positions_to_close:
                if symbol in current_prices:
                    self.execute_trade(symbol, self.positions[symbol]['qty'], 
                                     current_prices[symbol], 'sell', current_date)
            
            # Get forecasts for today
            forecasts = self.generate_forecast_cache(symbols, current_date)
            
            # Rank opportunities by expected return
            opportunities = []
            for symbol, forecast in forecasts.items():
                if symbol in current_prices and symbol not in self.positions:
                    expected_return = forecast['close_total_predicted_change']
                    confidence = forecast['confidence']
                    risk_score = expected_return * confidence
                    opportunities.append((symbol, risk_score, expected_return, confidence))
            
            opportunities.sort(key=lambda x: x[1], reverse=True)
            
            # Open new positions based on strategy
            if strategy == 'single_position' and len(self.positions) == 0:
                # Take the best opportunity
                if opportunities and opportunities[0][1] > 0:
                    symbol = opportunities[0][0]
                    qty = self.calculate_position_size(symbol, current_prices[symbol], strategy)
                    self.execute_trade(symbol, qty, current_prices[symbol], 'buy', current_date)
                    
            elif strategy in ['equal_weight', 'risk_weighted']:
                # Fill up to max positions
                positions_to_open = min(self.max_positions - len(self.positions), len(opportunities))
                
                for i in range(positions_to_open):
                    if opportunities[i][1] > 0:  # Positive expected return
                        symbol = opportunities[i][0]
                        qty = self.calculate_position_size(symbol, current_prices[symbol], strategy)
                        if qty > 0:
                            self.execute_trade(symbol, qty, current_prices[symbol], 'buy', current_date)
            
            # Record daily metrics
            self.daily_metrics.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'positions_value': portfolio_value - self.cash,
                'num_positions': len(self.positions),
                'return_pct': ((portfolio_value - self.initial_capital) / self.initial_capital) * 100
            })
            
            self.portfolio_value_history.append(portfolio_value)
            
            # Move to next day
            current_date += timedelta(days=1)
            day_count += 1
        
        # Close any remaining positions at end
        logger.info("\nClosing remaining positions at end of simulation...")
        for symbol in list(self.positions.keys()):
            if symbol in current_prices:
                self.execute_trade(symbol, self.positions[symbol]['qty'], 
                                 current_prices[symbol], 'sell', current_date)
        
        # Calculate final metrics
        final_value = self.calculate_portfolio_value(current_prices)
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Compile results
        results = {
            'strategy': strategy,
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'max_drawdown': self.calculate_max_drawdown(),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'num_trades': len(self.trade_history),
            'winning_trades': sum(1 for t in self.trade_history if t.get('profit', 0) > 0),
            'losing_trades': sum(1 for t in self.trade_history if t.get('profit', 0) < 0),
            'avg_win': np.mean([t['profit'] for t in self.trade_history if t.get('profit', 0) > 0]) if any(t.get('profit', 0) > 0 for t in self.trade_history) else 0,
            'avg_loss': np.mean([t['profit'] for t in self.trade_history if t.get('profit', 0) < 0]) if any(t.get('profit', 0) < 0 for t in self.trade_history) else 0,
            'portfolio_history': self.portfolio_value_history,
            'daily_metrics': self.daily_metrics,
            'trade_history': self.trade_history
        }
        
        return results
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage"""
        if not self.portfolio_value_history:
            return 0
        
        peak = self.portfolio_value_history[0]
        max_dd = 0
        
        for value in self.portfolio_value_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd * 100
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio (annualized)"""
        if len(self.daily_metrics) < 2:
            return 0
        
        daily_returns = []
        for i in range(1, len(self.daily_metrics)):
            prev_value = self.daily_metrics[i-1]['portfolio_value']
            curr_value = self.daily_metrics[i]['portfolio_value']
            daily_return = (curr_value - prev_value) / prev_value
            daily_returns.append(daily_return)
        
        if not daily_returns:
            return 0
        
        avg_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        
        if std_return == 0:
            return 0
        
        # Annualize
        annual_return = avg_return * 252
        annual_std = std_return * np.sqrt(252)
        
        return (annual_return - risk_free_rate) / annual_std
    
    def save_results(self, results: Dict, strategy_name: str):
        """Save backtest results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = self.results_dir / f"backtest_{strategy_name}_{timestamp}.json"
        with open(json_file, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            json_results = results.copy()
            json_results['daily_metrics'] = [
                {k: v.isoformat() if isinstance(v, datetime) else v for k, v in metric.items()}
                for metric in results['daily_metrics']
            ]
            json_results['trade_history'] = [
                {k: v.isoformat() if isinstance(v, datetime) else v for k, v in trade.items()}
                for trade in results['trade_history']
            ]
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {json_file}")
        
        # Save performance chart
        self.create_performance_chart(results, strategy_name)
        
        return json_file
    
    def create_performance_chart(self, results: Dict, strategy_name: str):
        """Create and save performance visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Backtest Results: {strategy_name} Strategy', fontsize=16)
        
        # Convert daily metrics to DataFrame for easier plotting
        df = pd.DataFrame(results['daily_metrics'])
        df['date'] = pd.to_datetime(df['date'])
        
        # 1. Portfolio value over time
        ax1.plot(df['date'], df['portfolio_value'], 'b-', linewidth=2)
        ax1.axhline(y=results['initial_capital'], color='r', linestyle='--', alpha=0.7, label='Initial Capital')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Portfolio Value Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _p: f'${x:,.0f}'))
        
        # 2. Daily returns distribution
        daily_returns = df['return_pct'].diff().dropna()
        ax2.hist(daily_returns, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Daily Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Daily Returns Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Number of positions over time
        ax3.plot(df['date'], df['num_positions'], 'g-', linewidth=2)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Number of Positions')
        ax3.set_title('Portfolio Positions Over Time')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, max(df['num_positions']) + 1)
        
        # 4. Summary statistics
        ax4.axis('off')
        summary_text = f"""
        === Performance Summary ===
        
        Initial Capital:     ${results['initial_capital']:,.2f}
        Final Value:        ${results['final_value']:,.2f}
        Total Return:       {results['total_return_pct']:.2f}%
        Max Drawdown:       {results['max_drawdown']:.2f}%
        Sharpe Ratio:       {results['sharpe_ratio']:.2f}
        
        Total Trades:       {results['num_trades']}
        Winning Trades:     {results['winning_trades']}
        Losing Trades:      {results['losing_trades']}
        Win Rate:           {(results['winning_trades'] / results['num_trades'] * 100) if results['num_trades'] > 0 else 0:.1f}%
        
        Avg Win:            ${results['avg_win']:,.2f}
        Avg Loss:           ${abs(results['avg_loss']):,.2f}
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_file = self.results_dir / f"backtest_chart_{strategy_name}_{timestamp}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance chart saved to {chart_file}")
        

def run_strategy_comparison(symbols: List[str], simulation_days: int = 25):
    """Run backtests for multiple strategies and compare results"""
    
    strategies = [
        'single_position',      # All capital in best opportunity
        'equal_weight',         # Equal weight across positions
        'risk_weighted',        # Risk-based position sizing
    ]
    
    all_results = {}
    
    for strategy in strategies:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running backtest for {strategy} strategy")
        logger.info(f"{'='*60}")
        
        # Reset backtester for each strategy
        backtester = LocalBacktester(
            initial_capital=100000,
            trading_fee=0.001,
            slippage=0.0005,
            max_positions=5 if strategy != 'single_position' else 1,
            simulation_days=simulation_days
        )
        
        results = backtester.run_backtest(symbols, strategy)
        backtester.save_results(results, strategy)
        all_results[strategy] = results
    
    # Create comparison chart
    create_strategy_comparison(all_results)
    
    return all_results


def create_strategy_comparison(all_results: Dict[str, Dict]):
    """Create comparison chart for all strategies"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Strategy Comparison', fontsize=16)
    
    # Extract data for comparison
    strategies = []
    returns = []
    sharpe_ratios = []
    max_drawdowns = []
    win_rates = []
    
    for strategy, results in all_results.items():
        strategies.append(strategy.replace('_', ' ').title())
        returns.append(results['total_return_pct'])
        sharpe_ratios.append(results['sharpe_ratio'])
        max_drawdowns.append(results['max_drawdown'])
        win_rate = (results['winning_trades'] / results['num_trades'] * 100) if results['num_trades'] > 0 else 0
        win_rates.append(win_rate)
    
    # 1. Returns comparison
    x = np.arange(len(strategies))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, returns, width, label='Total Return %', alpha=0.8)
    bars2 = ax1.bar(x + width/2, [-dd for dd in max_drawdowns], width, label='Max Drawdown %', alpha=0.8, color='red')
    
    ax1.set_xlabel('Strategy')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Returns vs Drawdown')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 2. Risk-adjusted returns
    ax2.scatter(win_rates, sharpe_ratios, s=200, alpha=0.7)
    
    for i, strategy in enumerate(strategies):
        ax2.annotate(strategy, (win_rates[i], sharpe_ratios[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    ax2.set_xlabel('Win Rate (%)')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Risk-Adjusted Performance')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = Path("simulationresults") / f"strategy_comparison_{timestamp}.png"
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Strategy comparison chart saved to {comparison_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Strategy':<20} {'Return %':>10} {'Sharpe':>10} {'Max DD %':>10} {'Win Rate %':>12}")
    print("-"*80)
    
    for strategy, ret, sharpe, dd, wr in zip(strategies, returns, sharpe_ratios, max_drawdowns, win_rates):
        print(f"{strategy:<20} {ret:>10.2f} {sharpe:>10.2f} {dd:>10.2f} {wr:>12.1f}")


if __name__ == "__main__":
    # Default symbols to test
    test_symbols = ['BTCUSD', 'ETHUSD', 'NVDA', 'TSLA', 'AAPL', 'GOOG', 'META', 'MSFT']
    
    logger.info("Starting Local Backtesting System")
    logger.info(f"Testing with symbols: {test_symbols}")
    
    # Run comparison
    results = run_strategy_comparison(test_symbols, simulation_days=25)
    
    logger.info("\nBacktesting complete! Check simulationresults/ directory for detailed results.")
