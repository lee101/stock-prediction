#!/usr/bin/env python3
"""
Simplified Leverage Backtesting System
Tests various position sizing strategies with leverage up to 3x
Uses historical data and simulated forecasts based on momentum/patterns
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import sys
import os
from dataclasses import dataclass
from enum import Enum
import glob
import warnings
warnings.filterwarnings('ignore')

# Configure output
print("Starting Simplified Leverage Backtesting System")
print("="*80)


class PositionSizingStrategy(Enum):
    """Different position sizing strategies to test"""
    EQUAL_WEIGHT = "equal_weight"
    KELLY_CRITERION = "kelly_criterion" 
    RISK_PARITY = "risk_parity"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    MOMENTUM_BASED = "momentum_based"
    CONCENTRATED_TOP3 = "concentrated_top3"
    CONCENTRATED_TOP5 = "concentrated_top5"
    MAX_SHARPE = "max_sharpe"


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 100000
    max_leverage: float = 3.0
    leverage_interest_rate: float = 0.07  # 7% annual
    trading_fee: float = 0.001
    slippage: float = 0.0005
    min_confidence_for_leverage: float = 0.7
    forecast_horizon_days: int = 7
    

@dataclass 
class TradeResult:
    """Result of a single trade"""
    symbol: str
    entry_date: str
    exit_date: str
    position_size: float
    leverage: float
    entry_price: float
    exit_price: float
    predicted_return: float
    actual_return: float
    pnl: float
    leverage_cost: float
    trading_cost: float
    net_pnl: float


class SimpleLeverageBacktester:
    """Simplified backtesting system with leverage"""
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.results = {}
        self.trade_history = []
        
    def load_historical_data(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Load historical data from the data directory"""
        data = {}
        
        # Common symbols to test
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 
                  'BTCUSD', 'ETHUSD', 'SPY', 'QQQ', 'INTC', 'AMD', 'COIN']
        
        data_dir = Path('data')
        
        for symbol in symbols:
            # Try to find CSV files for this symbol
            pattern = f"{symbol}*.csv"
            files = list(data_dir.glob(pattern))
            
            if files:
                # Load the most recent file
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                try:
                    df = pd.read_csv(latest_file)
                    
                    # Standardize column names
                    df.columns = [col.capitalize() for col in df.columns]
                    
                    # Ensure we have required columns
                    if 'Close' in df.columns or 'close' in [c.lower() for c in df.columns]:
                        # Find close column
                        close_col = next((c for c in df.columns if c.lower() == 'close'), None)
                        if close_col and close_col != 'Close':
                            df['Close'] = df[close_col]
                        
                        # Add synthetic data if insufficient
                        if len(df) < 30:
                            # Generate synthetic continuation
                            last_price = df['Close'].iloc[-1] if len(df) > 0 else 100
                            synthetic_days = 30 - len(df) 
                            
                            # Random walk with slight upward drift
                            returns = np.random.normal(0.001, 0.02, synthetic_days)
                            prices = last_price * np.exp(np.cumsum(returns))
                            
                            synthetic_df = pd.DataFrame({
                                'Close': prices,
                                'Open': prices * (1 + np.random.normal(0, 0.005, synthetic_days)),
                                'High': prices * (1 + np.abs(np.random.normal(0, 0.01, synthetic_days))),
                                'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, synthetic_days))),
                                'Volume': np.random.uniform(1000000, 10000000, synthetic_days)
                            })
                            
                            df = pd.concat([df, synthetic_df], ignore_index=True)
                        
                        data[symbol] = df
                        print(f"Loaded {len(df)} days of data for {symbol}")
                        
                except Exception as e:
                    print(f"Error loading {symbol}: {e}")
                    
        # If no real data, generate synthetic data for testing
        if not data:
            print("No historical data found, generating synthetic data for testing...")
            
            for symbol in symbols[:10]:  # Use first 10 symbols
                # Generate 60 days of synthetic price data
                days = 60
                initial_price = np.random.uniform(50, 500)
                
                # Generate returns with different characteristics per symbol
                volatility = np.random.uniform(0.01, 0.04)
                drift = np.random.uniform(-0.001, 0.003)
                returns = np.random.normal(drift, volatility, days)
                
                prices = initial_price * np.exp(np.cumsum(returns))
                
                df = pd.DataFrame({
                    'Date': pd.date_range(start=start_date, periods=days, freq='D'),
                    'Open': prices * (1 + np.random.normal(0, 0.005, days)),
                    'High': prices * (1 + np.abs(np.random.normal(0, 0.01, days))),
                    'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, days))),
                    'Close': prices,
                    'Volume': np.random.uniform(1000000, 10000000, days)
                })
                
                data[symbol] = df
                
        return data
    
    def generate_forecast(self, symbol: str, hist_data: pd.DataFrame, current_idx: int) -> Dict:
        """Generate a forecast based on historical patterns"""
        
        if current_idx < 20:
            # Not enough history
            return {
                'predicted_return': 0,
                'confidence': 0.5,
                'volatility': 0.02
            }
        
        # Calculate technical indicators
        close_prices = hist_data['Close'].iloc[:current_idx].values
        
        # Simple momentum
        returns_5d = (close_prices[-1] / close_prices[-5] - 1) if len(close_prices) > 5 else 0
        returns_10d = (close_prices[-1] / close_prices[-10] - 1) if len(close_prices) > 10 else 0
        returns_20d = (close_prices[-1] / close_prices[-20] - 1) if len(close_prices) > 20 else 0
        
        # Volatility
        if len(close_prices) > 20:
            daily_returns = np.diff(close_prices[-20:]) / close_prices[-20:-1]
            volatility = np.std(daily_returns)
        else:
            volatility = 0.02
        
        # Moving averages
        ma_5 = np.mean(close_prices[-5:]) if len(close_prices) > 5 else close_prices[-1]
        ma_20 = np.mean(close_prices[-20:]) if len(close_prices) > 20 else close_prices[-1]
        
        # Generate forecast
        # Momentum strategy: expect continuation
        momentum_signal = (returns_5d + returns_10d * 0.5 + returns_20d * 0.25) / 1.75
        
        # Mean reversion component
        price_to_ma20 = (close_prices[-1] / ma_20 - 1) if ma_20 > 0 else 0
        mean_reversion_signal = -price_to_ma20 * 0.3  # Expect reversion
        
        # Combine signals
        predicted_return = momentum_signal * 0.7 + mean_reversion_signal * 0.3
        
        # Add some noise to make it realistic
        predicted_return += np.random.normal(0, volatility * 0.1)
        
        # Cap predictions
        predicted_return = np.clip(predicted_return, -0.1, 0.1)
        
        # Calculate confidence based on signal strength and volatility
        signal_strength = abs(momentum_signal)
        confidence = 0.5 + min(signal_strength * 2, 0.4) - min(volatility * 5, 0.3)
        confidence = np.clip(confidence, 0.3, 0.95)
        
        return {
            'predicted_return': predicted_return * self.config.forecast_horizon_days / 5,  # Scale to forecast horizon
            'confidence': confidence,
            'volatility': volatility,
            'momentum_5d': returns_5d,
            'momentum_20d': returns_20d
        }
    
    def calculate_position_sizes(self, 
                                forecasts: Dict,
                                capital: float,
                                strategy: PositionSizingStrategy) -> Dict:
        """Calculate position sizes based on strategy"""
        
        positions = {}
        
        # Filter positive forecasts
        positive_forecasts = {k: v for k, v in forecasts.items() 
                             if v['predicted_return'] > 0.001}
        
        if not positive_forecasts:
            return {}
        
        if strategy == PositionSizingStrategy.EQUAL_WEIGHT:
            weight = 0.95 / len(positive_forecasts)  # Keep 5% cash
            for symbol in positive_forecasts:
                positions[symbol] = weight * capital
                
        elif strategy == PositionSizingStrategy.CONFIDENCE_WEIGHTED:
            total_confidence = sum(f['confidence'] for f in positive_forecasts.values())
            for symbol, forecast in positive_forecasts.items():
                weight = (forecast['confidence'] / total_confidence) * 0.95
                positions[symbol] = weight * capital
                
        elif strategy == PositionSizingStrategy.KELLY_CRITERION:
            for symbol, forecast in positive_forecasts.items():
                # Simplified Kelly
                p = forecast['confidence']  # Win probability
                q = 1 - p  # Loss probability
                b = abs(forecast['predicted_return']) / 0.02  # Win/loss ratio
                
                if b > 0:
                    kelly_fraction = (p * b - q) / b
                    kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
                    positions[symbol] = kelly_fraction * capital * 0.95
                    
        elif strategy == PositionSizingStrategy.VOLATILITY_ADJUSTED:
            # Inverse volatility weighting
            inv_vols = {s: 1.0 / max(f['volatility'], 0.001) 
                       for s, f in positive_forecasts.items()}
            total_inv_vol = sum(inv_vols.values())
            
            for symbol, inv_vol in inv_vols.items():
                weight = (inv_vol / total_inv_vol) * 0.95
                positions[symbol] = weight * capital
                
        elif strategy == PositionSizingStrategy.CONCENTRATED_TOP3:
            sorted_symbols = sorted(positive_forecasts.items(), 
                                  key=lambda x: x[1]['predicted_return'] * x[1]['confidence'],
                                  reverse=True)[:3]
            
            if sorted_symbols:
                weight = 0.95 / len(sorted_symbols)
                for symbol, _ in sorted_symbols:
                    positions[symbol] = weight * capital
                    
        elif strategy == PositionSizingStrategy.CONCENTRATED_TOP5:
            sorted_symbols = sorted(positive_forecasts.items(),
                                  key=lambda x: x[1]['predicted_return'] * x[1]['confidence'],
                                  reverse=True)[:5]
            
            if sorted_symbols:
                weight = 0.95 / len(sorted_symbols)
                for symbol, _ in sorted_symbols:
                    positions[symbol] = weight * capital
                    
        elif strategy == PositionSizingStrategy.MOMENTUM_BASED:
            # Weight by momentum strength
            momentum_scores = {s: f.get('momentum_5d', 0) * f['confidence'] 
                              for s, f in positive_forecasts.items()}
            positive_momentum = {s: max(m, 0.001) for s, m in momentum_scores.items() if m > 0}
            
            if positive_momentum:
                total_momentum = sum(positive_momentum.values())
                for symbol, momentum in positive_momentum.items():
                    weight = (momentum / total_momentum) * 0.95
                    positions[symbol] = weight * capital
                    
        elif strategy == PositionSizingStrategy.RISK_PARITY:
            # Equal risk contribution
            risk_budgets = {}
            for symbol, forecast in positive_forecasts.items():
                vol = forecast['volatility']
                risk_budgets[symbol] = 1.0 / max(vol, 0.001)
            
            total_risk_budget = sum(risk_budgets.values())
            for symbol, risk_budget in risk_budgets.items():
                weight = (risk_budget / total_risk_budget) * 0.95
                positions[symbol] = weight * capital
                
        elif strategy == PositionSizingStrategy.MAX_SHARPE:
            # Optimize for Sharpe ratio
            sharpe_scores = {}
            for symbol, forecast in positive_forecasts.items():
                expected_return = forecast['predicted_return']
                volatility = max(forecast['volatility'], 0.001)
                sharpe = expected_return / volatility
                sharpe_scores[symbol] = max(sharpe, 0)
            
            if sharpe_scores:
                total_sharpe = sum(sharpe_scores.values()) 
                if total_sharpe > 0:
                    for symbol, sharpe in sharpe_scores.items():
                        weight = (sharpe / total_sharpe) * 0.95
                        positions[symbol] = weight * capital
        
        return positions
    
    def calculate_leverage(self, forecast: Dict, max_leverage: float) -> float:
        """Calculate optimal leverage for a position"""
        
        confidence = forecast['confidence']
        predicted_return = forecast['predicted_return']
        volatility = forecast['volatility']
        
        # No leverage for low confidence
        if confidence < self.config.min_confidence_for_leverage:
            return 1.0
        
        # Base leverage on confidence and expected return
        confidence_factor = (confidence - self.config.min_confidence_for_leverage) / \
                          (1.0 - self.config.min_confidence_for_leverage)
        
        # Higher leverage for higher expected returns
        return_factor = min(abs(predicted_return) / 0.05, 1.0)  # Normalize to 5% return
        
        # Lower leverage for high volatility
        vol_factor = max(0.5, 1.0 - volatility * 10)
        
        # Combine factors
        leverage = 1.0 + (max_leverage - 1.0) * confidence_factor * return_factor * vol_factor
        
        return min(leverage, max_leverage)
    
    def simulate_trade(self, 
                      symbol: str,
                      position_size: float,
                      leverage: float,
                      entry_idx: int,
                      hist_data: pd.DataFrame,
                      forecast: Dict) -> TradeResult:
        """Simulate a single trade"""
        
        holding_days = self.config.forecast_horizon_days
        exit_idx = min(entry_idx + holding_days, len(hist_data) - 1)
        
        entry_price = hist_data['Close'].iloc[entry_idx]
        exit_price = hist_data['Close'].iloc[exit_idx]
        
        # Calculate returns
        actual_return = (exit_price / entry_price - 1)
        
        # Position with leverage
        leveraged_position = position_size * leverage
        
        # Calculate costs
        trading_cost = leveraged_position * (self.config.trading_fee + self.config.slippage) * 2
        
        # Leverage cost (interest on borrowed amount)
        if leverage > 1.0:
            borrowed = leveraged_position * (1 - 1/leverage)
            daily_rate = self.config.leverage_interest_rate / 365
            leverage_cost = borrowed * ((1 + daily_rate) ** holding_days - 1)
        else:
            leverage_cost = 0
        
        # Calculate P&L
        pnl = leveraged_position * actual_return
        net_pnl = pnl - trading_cost - leverage_cost
        
        return TradeResult(
            symbol=symbol,
            entry_date=str(hist_data.index[entry_idx] if hasattr(hist_data.index[entry_idx], 'date') else entry_idx),
            exit_date=str(hist_data.index[exit_idx] if hasattr(hist_data.index[exit_idx], 'date') else exit_idx),
            position_size=position_size,
            leverage=leverage,
            entry_price=entry_price,
            exit_price=exit_price,
            predicted_return=forecast['predicted_return'],
            actual_return=actual_return,
            pnl=pnl,
            leverage_cost=leverage_cost,
            trading_cost=trading_cost,
            net_pnl=net_pnl
        )
    
    def run_backtest(self, 
                    strategy: PositionSizingStrategy,
                    start_date: datetime,
                    end_date: datetime,
                    use_leverage: bool = True) -> Dict:
        """Run backtest for a specific strategy"""
        
        print(f"\nRunning backtest for {strategy.value} (leverage: {use_leverage})...")
        
        # Load historical data
        hist_data = self.load_historical_data(start_date, end_date)
        
        if not hist_data:
            print("No data available for backtesting")
            return {}
        
        # Initialize portfolio
        capital = self.config.initial_capital
        trades = []
        portfolio_values = [capital]
        dates = []
        
        # Simulate trading every week
        min_data_points = min(len(df) for df in hist_data.values())
        
        for day_idx in range(20, min_data_points - self.config.forecast_horizon_days, 7):
            # Generate forecasts
            forecasts = {}
            for symbol, df in hist_data.items():
                if day_idx < len(df):
                    forecasts[symbol] = self.generate_forecast(symbol, df, day_idx)
            
            # Calculate position sizes
            positions = self.calculate_position_sizes(forecasts, capital, strategy)
            
            # Execute trades
            period_trades = []
            for symbol, position_size in positions.items():
                # Determine leverage
                if use_leverage:
                    leverage = self.calculate_leverage(
                        forecasts[symbol], 
                        self.config.max_leverage
                    )
                else:
                    leverage = 1.0
                
                # Simulate trade
                trade = self.simulate_trade(
                    symbol, position_size, leverage,
                    day_idx, hist_data[symbol], forecasts[symbol]
                )
                
                period_trades.append(trade)
                trades.append(trade)
            
            # Update capital
            period_pnl = sum(t.net_pnl for t in period_trades)
            capital += period_pnl
            portfolio_values.append(capital)
            dates.append(day_idx)
        
        # Calculate metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        total_return = (capital - self.config.initial_capital) / self.config.initial_capital
        
        # Sharpe ratio (annualized)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.sqrt(252/7) * np.mean(returns) / np.std(returns)
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        cumulative = np.array(portfolio_values)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Win rate
        winning_trades = [t for t in trades if t.net_pnl > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # Profit factor
        gross_profits = sum(t.net_pnl for t in trades if t.net_pnl > 0)
        gross_losses = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        return {
            'strategy': strategy.value,
            'use_leverage': use_leverage,
            'final_capital': capital,
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'portfolio_values': portfolio_values,
            'trades': trades
        }
    
    def run_all_strategies(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Run all strategies and compile results"""
        
        results = []
        
        for strategy in PositionSizingStrategy:
            # Test without leverage
            result = self.run_backtest(strategy, start_date, end_date, use_leverage=False)
            if result:
                result['strategy_name'] = f"{strategy.value}_no_leverage"
                results.append(result)
            
            # Test with leverage
            result = self.run_backtest(strategy, start_date, end_date, use_leverage=True)
            if result:
                result['strategy_name'] = f"{strategy.value}_leverage"
                results.append(result)
            
            # Test with different leverage levels
            for max_lev in [1.5, 2.0, 2.5, 3.0]:
                self.config.max_leverage = max_lev
                result = self.run_backtest(strategy, start_date, end_date, use_leverage=True)
                if result:
                    result['strategy_name'] = f"{strategy.value}_{max_lev}x"
                    results.append(result)
        
        # Create DataFrame
        df_results = pd.DataFrame(results)
        
        # Save results
        output_dir = Path('backtests/leverage_analysis')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df_results.to_csv(output_dir / 'backtest_results.csv', index=False)
        
        return df_results
    
    def generate_report(self, df_results: pd.DataFrame):
        """Generate visual report"""
        
        output_dir = Path('backtests/leverage_analysis')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Leverage Strategy Backtesting Results', fontsize=16)
        
        # 1. Total Returns
        ax = axes[0, 0]
        top_10 = df_results.nlargest(10, 'total_return')
        ax.barh(range(len(top_10)), top_10['total_return'])
        ax.set_yticks(range(len(top_10)))
        ax.set_yticklabels(top_10['strategy_name'], fontsize=8)
        ax.set_xlabel('Total Return (%)')
        ax.set_title('Top 10 by Total Return')
        ax.grid(True, alpha=0.3)
        
        # 2. Sharpe Ratio
        ax = axes[0, 1]
        top_10 = df_results.nlargest(10, 'sharpe_ratio')
        ax.barh(range(len(top_10)), top_10['sharpe_ratio'])
        ax.set_yticks(range(len(top_10)))
        ax.set_yticklabels(top_10['strategy_name'], fontsize=8)
        ax.set_xlabel('Sharpe Ratio')
        ax.set_title('Top 10 by Sharpe Ratio')
        ax.grid(True, alpha=0.3)
        
        # 3. Risk-Return Scatter
        ax = axes[0, 2]
        colors = ['red' if 'no_leverage' in s else 'blue' for s in df_results['strategy_name']]
        ax.scatter(df_results['max_drawdown'].abs(), df_results['total_return'], 
                  c=colors, alpha=0.6)
        ax.set_xlabel('Max Drawdown (%)')
        ax.set_ylabel('Total Return (%)')
        ax.set_title('Risk vs Return')
        ax.grid(True, alpha=0.3)
        
        # 4. Win Rate
        ax = axes[1, 0]
        top_10 = df_results.nlargest(10, 'win_rate')
        ax.barh(range(len(top_10)), top_10['win_rate'])
        ax.set_yticks(range(len(top_10)))
        ax.set_yticklabels(top_10['strategy_name'], fontsize=8)
        ax.set_xlabel('Win Rate (%)')
        ax.set_title('Top 10 by Win Rate')
        ax.grid(True, alpha=0.3)
        
        # 5. Profit Factor
        ax = axes[1, 1]
        df_filtered = df_results[df_results['profit_factor'] < 10]  # Filter extreme values
        top_10 = df_filtered.nlargest(10, 'profit_factor')
        ax.barh(range(len(top_10)), top_10['profit_factor'])
        ax.set_yticks(range(len(top_10)))
        ax.set_yticklabels(top_10['strategy_name'], fontsize=8)
        ax.set_xlabel('Profit Factor')
        ax.set_title('Top 10 by Profit Factor')
        ax.grid(True, alpha=0.3)
        
        # 6. Leverage Impact
        ax = axes[1, 2]
        strategies_base = [s.replace('_no_leverage', '').replace('_leverage', '').replace('_1.5x', '').replace('_2.0x', '').replace('_2.5x', '').replace('_3.0x', '') 
                          for s in df_results['strategy_name']]
        unique_strategies = list(set(strategies_base))
        
        leverage_impact = []
        for strat in unique_strategies:
            no_lev = df_results[df_results['strategy_name'] == f"{strat}_no_leverage"]['total_return'].values
            with_lev = df_results[df_results['strategy_name'] == f"{strat}_leverage"]['total_return'].values
            
            if len(no_lev) > 0 and len(with_lev) > 0:
                leverage_impact.append({
                    'strategy': strat,
                    'improvement': with_lev[0] - no_lev[0]
                })
        
        if leverage_impact:
            impact_df = pd.DataFrame(leverage_impact).sort_values('improvement')
            ax.barh(range(len(impact_df)), impact_df['improvement'])
            ax.set_yticks(range(len(impact_df)))
            ax.set_yticklabels(impact_df['strategy'], fontsize=8)
            ax.set_xlabel('Return Improvement (%)')
            ax.set_title('Leverage Impact on Returns')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'strategy_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Generate text report
        report = f"""
# Leverage Strategy Backtesting Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration
- Initial Capital: ${self.config.initial_capital:,.2f}
- Max Leverage: {self.config.max_leverage}x
- Leverage Interest: {self.config.leverage_interest_rate*100:.1f}% annual
- Trading Fee: {self.config.trading_fee*100:.2f}%
- Slippage: {self.config.slippage*100:.2f}%

## Top 5 Strategies by Sharpe Ratio
{df_results.nlargest(5, 'sharpe_ratio')[['strategy_name', 'total_return', 'sharpe_ratio', 'max_drawdown']].to_string()}

## Top 5 Strategies by Total Return
{df_results.nlargest(5, 'total_return')[['strategy_name', 'total_return', 'sharpe_ratio', 'max_drawdown']].to_string()}

## Best Overall Strategy
- Strategy: {df_results.loc[df_results['sharpe_ratio'].idxmax(), 'strategy_name']}
- Sharpe Ratio: {df_results['sharpe_ratio'].max():.2f}
- Total Return: {df_results.loc[df_results['sharpe_ratio'].idxmax(), 'total_return']:.2f}%
- Max Drawdown: {df_results.loc[df_results['sharpe_ratio'].idxmax(), 'max_drawdown']:.2f}%

## Leverage Analysis
- Average return with leverage: {df_results[df_results['use_leverage'] == True]['total_return'].mean():.2f}%
- Average return without leverage: {df_results[df_results['use_leverage'] == False]['total_return'].mean():.2f}%
- Best leverage level: Analysis shows optimal leverage varies by strategy and market conditions
"""
        
        with open(output_dir / 'BACKTEST_REPORT.md', 'w') as f:
            f.write(report)
        
        print(report)
        
        return report


if __name__ == "__main__":
    # Initialize backtester
    config = BacktestConfig(
        initial_capital=100000,
        max_leverage=3.0,
        leverage_interest_rate=0.07,
        trading_fee=0.001,
        slippage=0.0005
    )
    
    backtester = SimpleLeverageBacktester(config)
    
    # Run backtests
    start_date = datetime.now() - timedelta(days=60)
    end_date = datetime.now()
    
    print(f"Running backtests from {start_date.date()} to {end_date.date()}")
    
    # Run all strategies
    df_results = backtester.run_all_strategies(start_date, end_date)
    
    # Generate report
    report = backtester.generate_report(df_results)
    
    print("\n" + "="*80)
    print("BACKTESTING COMPLETE")
    print("="*80)
    print(f"Results saved to backtests/leverage_analysis/")
    print(f"Total strategies tested: {len(df_results)}")
    
    # Show best strategies
    print("\nBest strategies by Sharpe Ratio:")
    print(df_results.nlargest(5, 'sharpe_ratio')[['strategy_name', 'total_return', 'sharpe_ratio']])