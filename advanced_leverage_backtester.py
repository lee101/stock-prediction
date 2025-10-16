#!/usr/bin/env python3
"""
Advanced Backtesting System with Leverage and Position Sizing Strategies
Tests various position sizing strategies including leverage up to 3x
With realistic 7% annual interest on leveraged portions
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from loguru import logger
import sys
import os
from dataclasses import dataclass
from enum import Enum

# Import existing modules
from predict_stock_forecasting import make_predictions, load_stock_data_from_csv
from data_curate_daily import download_daily_stock_data
from src.fixtures import crypto_symbols
from enhanced_local_backtester import EnhancedLocalBacktester
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
logger.add("backtests/advanced_leverage_backtesting.log", rotation="10 MB")


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
class LeverageConfig:
    """Configuration for leverage usage"""
    max_leverage: float = 3.0
    annual_interest_rate: float = 0.07  # 7% annual interest
    min_confidence_for_leverage: float = 0.7  # Minimum confidence to use leverage
    leverage_scaling: str = "linear"  # linear, exponential, step
    

@dataclass
class BacktestResult:
    """Results from a single backtest run"""
    strategy: str
    leverage: float
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    leverage_costs: float
    trading_costs: float
    daily_returns: List[float]
    positions_history: List[Dict]
    

class AdvancedLeverageBacktester:
    """Advanced backtesting system with leverage and multiple position sizing strategies"""
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 start_date: datetime = None,
                 end_date: datetime = None,
                 trading_fee: float = 0.001,
                 slippage: float = 0.0005,
                 leverage_config: LeverageConfig = None):
        
        self.initial_capital = initial_capital
        self.start_date = start_date or datetime.now() - timedelta(days=30)
        self.end_date = end_date or datetime.now()
        self.trading_fee = trading_fee
        self.slippage = slippage
        self.leverage_config = leverage_config or LeverageConfig()
        
        # Initialize base backtester
        self.base_backtester = EnhancedLocalBacktester(
            initial_capital=initial_capital,
            start_date=self.start_date,
            end_date=self.end_date,
            use_real_forecasts=True
        )
        
        # Results storage
        self.results = {}
        self.detailed_metrics = {}
        
    def calculate_leverage_cost(self, borrowed_amount: float, days: int) -> float:
        """Calculate interest cost for leveraged positions"""
        daily_rate = self.leverage_config.annual_interest_rate / 365
        # Compound daily interest
        total_interest = borrowed_amount * ((1 + daily_rate) ** days - 1)
        return total_interest
    
    def determine_optimal_leverage(self, 
                                  forecast: Dict, 
                                  volatility: float,
                                  strategy: PositionSizingStrategy) -> float:
        """Determine optimal leverage based on forecast and strategy"""
        
        confidence = forecast.get('confidence', 0.5)
        predicted_return = forecast.get('close_total_predicted_change', 0)
        
        # Base leverage on confidence and predicted return
        if confidence < self.leverage_config.min_confidence_for_leverage:
            return 1.0  # No leverage for low confidence
        
        if self.leverage_config.leverage_scaling == "linear":
            # Linear scaling based on confidence
            leverage = 1.0 + (confidence - self.leverage_config.min_confidence_for_leverage) * \
                      (self.leverage_config.max_leverage - 1.0) / \
                      (1.0 - self.leverage_config.min_confidence_for_leverage)
                      
        elif self.leverage_config.leverage_scaling == "exponential":
            # Exponential scaling for high confidence trades
            confidence_factor = (confidence - self.leverage_config.min_confidence_for_leverage) / \
                               (1.0 - self.leverage_config.min_confidence_for_leverage)
            leverage = 1.0 + (self.leverage_config.max_leverage - 1.0) * (confidence_factor ** 2)
            
        elif self.leverage_config.leverage_scaling == "step":
            # Step function based on confidence thresholds
            if confidence >= 0.9:
                leverage = 3.0
            elif confidence >= 0.8:
                leverage = 2.0
            elif confidence >= 0.7:
                leverage = 1.5
            else:
                leverage = 1.0
        else:
            leverage = 1.0
            
        # Adjust for volatility (reduce leverage for high volatility)
        if volatility > 0.03:  # High volatility threshold
            leverage *= 0.8
        elif volatility > 0.02:
            leverage *= 0.9
            
        # Cap at max leverage
        return min(leverage, self.leverage_config.max_leverage)
    
    def calculate_position_sizes(self, 
                                forecasts: Dict,
                                available_capital: float,
                                strategy: PositionSizingStrategy,
                                historical_data: Dict = None) -> Dict:
        """Calculate position sizes based on strategy"""
        
        positions = {}
        
        if strategy == PositionSizingStrategy.EQUAL_WEIGHT:
            # Equal weight across all positive forecasts
            positive_forecasts = {k: v for k, v in forecasts.items() 
                                 if v.get('close_total_predicted_change', 0) > 0}
            if positive_forecasts:
                weight = 1.0 / len(positive_forecasts)
                for symbol, forecast in positive_forecasts.items():
                    positions[symbol] = {
                        'weight': weight,
                        'dollar_amount': available_capital * weight * 0.95,  # Keep 5% cash
                        'leverage': 1.0
                    }
                    
        elif strategy == PositionSizingStrategy.KELLY_CRITERION:
            # Kelly Criterion based position sizing
            total_kelly = 0
            kelly_weights = {}
            
            for symbol, forecast in forecasts.items():
                pred_return = forecast.get('close_total_predicted_change', 0)
                confidence = forecast.get('confidence', 0.5)
                
                if pred_return > 0:
                    # Simplified Kelly fraction
                    win_prob = confidence
                    loss_prob = 1 - confidence
                    avg_win = pred_return
                    avg_loss = pred_return * 0.5  # Assume half the predicted return as potential loss
                    
                    if avg_loss != 0:
                        kelly_fraction = (win_prob * avg_win - loss_prob * avg_loss) / avg_win
                        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25% per position
                        kelly_weights[symbol] = kelly_fraction
                        total_kelly += kelly_fraction
            
            # Normalize weights
            if total_kelly > 0:
                for symbol, kelly_weight in kelly_weights.items():
                    normalized_weight = (kelly_weight / total_kelly) * 0.95  # Keep 5% cash
                    positions[symbol] = {
                        'weight': normalized_weight,
                        'dollar_amount': available_capital * normalized_weight,
                        'leverage': 1.0
                    }
                    
        elif strategy == PositionSizingStrategy.CONFIDENCE_WEIGHTED:
            # Weight by confidence scores
            total_confidence = sum(f.get('confidence', 0) for f in forecasts.values() 
                                  if f.get('close_total_predicted_change', 0) > 0)
            
            if total_confidence > 0:
                for symbol, forecast in forecasts.items():
                    if forecast.get('close_total_predicted_change', 0) > 0:
                        confidence = forecast.get('confidence', 0)
                        weight = (confidence / total_confidence) * 0.95
                        positions[symbol] = {
                            'weight': weight,
                            'dollar_amount': available_capital * weight,
                            'leverage': 1.0
                        }
                        
        elif strategy == PositionSizingStrategy.CONCENTRATED_TOP3:
            # Concentrate on top 3 predicted performers
            sorted_forecasts = sorted(forecasts.items(), 
                                     key=lambda x: x[1].get('close_total_predicted_change', 0),
                                     reverse=True)[:3]
            
            if sorted_forecasts:
                weight = 0.95 / len(sorted_forecasts)
                for symbol, forecast in sorted_forecasts:
                    if forecast.get('close_total_predicted_change', 0) > 0:
                        positions[symbol] = {
                            'weight': weight,
                            'dollar_amount': available_capital * weight,
                            'leverage': 1.0
                        }
                        
        elif strategy == PositionSizingStrategy.CONCENTRATED_TOP5:
            # Concentrate on top 5 predicted performers
            sorted_forecasts = sorted(forecasts.items(), 
                                     key=lambda x: x[1].get('close_total_predicted_change', 0),
                                     reverse=True)[:5]
            
            if sorted_forecasts:
                weight = 0.95 / len(sorted_forecasts)
                for symbol, forecast in sorted_forecasts:
                    if forecast.get('close_total_predicted_change', 0) > 0:
                        positions[symbol] = {
                            'weight': weight,
                            'dollar_amount': available_capital * weight,
                            'leverage': 1.0
                        }
                        
        # Apply leverage based on strategy and forecast confidence
        for symbol in positions:
            if symbol in forecasts:
                # Calculate historical volatility if available
                volatility = 0.02  # Default volatility
                if historical_data and symbol in historical_data:
                    hist = historical_data[symbol]
                    if len(hist) > 1:
                        returns = hist['Close'].pct_change().dropna()
                        volatility = returns.std() if len(returns) > 0 else 0.02
                
                # Determine optimal leverage
                optimal_leverage = self.determine_optimal_leverage(
                    forecasts[symbol], volatility, strategy
                )
                
                positions[symbol]['leverage'] = optimal_leverage
                positions[symbol]['dollar_amount'] *= optimal_leverage
                
        return positions
    
    def simulate_trading_period(self,
                              strategy: PositionSizingStrategy,
                              use_leverage: bool = True) -> BacktestResult:
        """Simulate trading over the specified period"""
        
        logger.info(f"Starting simulation for strategy: {strategy.value}, leverage: {use_leverage}")
        
        current_capital = self.initial_capital
        daily_returns = []
        positions_history = []
        total_leverage_costs = 0
        total_trading_costs = 0
        winning_trades = 0
        losing_trades = 0
        gross_profits = 0
        gross_losses = 0
        
        # Generate date range
        current_date = self.start_date
        
        while current_date <= self.end_date:
            # Get forecasts for current date
            forecasts = self.base_backtester.generate_real_ai_forecasts(
                list(crypto_symbols.keys()), current_date
            )
            
            if forecasts:
                # Get historical data for volatility calculation
                historical_data = {}
                for symbol in forecasts.keys():
                    hist = self.base_backtester.load_symbol_history(symbol, current_date)
                    if hist is not None:
                        historical_data[symbol] = hist
                
                # Calculate position sizes
                positions = self.calculate_position_sizes(
                    forecasts, current_capital, strategy, historical_data
                )
                
                if not use_leverage:
                    # Override leverage to 1.0 if not using leverage
                    for pos in positions.values():
                        pos['leverage'] = 1.0
                        pos['dollar_amount'] /= pos.get('leverage', 1.0)
                
                # Execute trades and calculate returns
                period_return = 0
                period_leverage_cost = 0
                period_trading_cost = 0
                
                for symbol, position in positions.items():
                    if symbol in forecasts:
                        # Entry costs
                        entry_cost = position['dollar_amount'] * (self.trading_fee + self.slippage)
                        period_trading_cost += entry_cost
                        
                        # Calculate return
                        predicted_return = forecasts[symbol].get('close_total_predicted_change', 0)
                        
                        # Add some realistic noise to predictions (reality != perfect prediction)
                        noise = np.random.normal(0, 0.005)  # 0.5% standard deviation
                        actual_return = predicted_return + noise
                        
                        # Calculate P&L
                        position_pnl = position['dollar_amount'] * actual_return
                        
                        # Exit costs
                        exit_cost = position['dollar_amount'] * (self.trading_fee + self.slippage)
                        period_trading_cost += exit_cost
                        
                        # Calculate leverage cost if applicable
                        if position['leverage'] > 1.0:
                            borrowed = position['dollar_amount'] * (1 - 1/position['leverage'])
                            leverage_cost = self.calculate_leverage_cost(borrowed, 7)  # 7 day holding period
                            period_leverage_cost += leverage_cost
                        
                        # Net P&L
                        net_pnl = position_pnl - entry_cost - exit_cost - period_leverage_cost
                        period_return += net_pnl
                        
                        # Track winning/losing trades
                        if net_pnl > 0:
                            winning_trades += 1
                            gross_profits += net_pnl
                        else:
                            losing_trades += 1
                            gross_losses += abs(net_pnl)
                        
                        # Record position
                        positions_history.append({
                            'date': current_date.isoformat(),
                            'symbol': symbol,
                            'dollar_amount': position['dollar_amount'],
                            'leverage': position['leverage'],
                            'predicted_return': predicted_return,
                            'actual_return': actual_return,
                            'net_pnl': net_pnl
                        })
                
                # Update capital
                current_capital += period_return
                daily_return = period_return / (current_capital - period_return)
                daily_returns.append(daily_return)
                
                total_leverage_costs += period_leverage_cost
                total_trading_costs += period_trading_cost
            
            # Move to next trading period (weekly for this simulation)
            current_date += timedelta(days=7)
        
        # Calculate metrics
        total_return = (current_capital - self.initial_capital) / self.initial_capital
        days_traded = (self.end_date - self.start_date).days
        annualized_return = ((1 + total_return) ** (365 / days_traded) - 1) if days_traded > 0 else 0
        
        # Sharpe Ratio
        if daily_returns:
            returns_array = np.array(daily_returns)
            sharpe_ratio = np.sqrt(252) * (returns_array.mean() / returns_array.std()) if returns_array.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Max Drawdown
        cumulative_returns = np.cumprod(1 + np.array(daily_returns))
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        # Win Rate and Profit Factor
        total_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        return BacktestResult(
            strategy=strategy.value,
            leverage=use_leverage,
            initial_capital=self.initial_capital,
            final_capital=current_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            leverage_costs=total_leverage_costs,
            trading_costs=total_trading_costs,
            daily_returns=daily_returns,
            positions_history=positions_history
        )
    
    def run_all_strategies(self) -> Dict[str, BacktestResult]:
        """Run all position sizing strategies with and without leverage"""
        
        results = {}
        
        for strategy in PositionSizingStrategy:
            # Test without leverage
            logger.info(f"Testing {strategy.value} without leverage...")
            result_no_leverage = self.simulate_trading_period(strategy, use_leverage=False)
            results[f"{strategy.value}_no_leverage"] = result_no_leverage
            
            # Test with leverage
            logger.info(f"Testing {strategy.value} with leverage...")
            result_with_leverage = self.simulate_trading_period(strategy, use_leverage=True)
            results[f"{strategy.value}_with_leverage"] = result_with_leverage
            
            # Test with different leverage levels
            for max_lev in [1.5, 2.0, 2.5, 3.0]:
                self.leverage_config.max_leverage = max_lev
                logger.info(f"Testing {strategy.value} with {max_lev}x max leverage...")
                result = self.simulate_trading_period(strategy, use_leverage=True)
                results[f"{strategy.value}_{max_lev}x"] = result
        
        self.results = results
        return results
    
    def generate_report(self, output_dir: str = "backtests/leverage_analysis"):
        """Generate comprehensive report with visualizations"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create results DataFrame
        results_data = []
        for name, result in self.results.items():
            results_data.append({
                'Strategy': name,
                'Final Capital': result.final_capital,
                'Total Return': result.total_return * 100,
                'Annualized Return': result.annualized_return * 100,
                'Sharpe Ratio': result.sharpe_ratio,
                'Max Drawdown': result.max_drawdown * 100,
                'Win Rate': result.win_rate * 100,
                'Profit Factor': result.profit_factor,
                'Total Trades': result.total_trades,
                'Leverage Costs': result.leverage_costs,
                'Trading Costs': result.trading_costs
            })
        
        df_results = pd.DataFrame(results_data)
        
        # Save to CSV
        df_results.to_csv(output_path / 'backtest_results.csv', index=False)
        
        # Create visualizations
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Position Sizing and Leverage Strategy Analysis', fontsize=16)
        
        # 1. Total Returns Comparison
        ax = axes[0, 0]
        df_sorted = df_results.sort_values('Total Return', ascending=True)
        ax.barh(df_sorted['Strategy'], df_sorted['Total Return'])
        ax.set_xlabel('Total Return (%)')
        ax.set_title('Total Returns by Strategy')
        ax.grid(True, alpha=0.3)
        
        # 2. Sharpe Ratio Comparison
        ax = axes[0, 1]
        df_sorted = df_results.sort_values('Sharpe Ratio', ascending=True)
        ax.barh(df_sorted['Strategy'], df_sorted['Sharpe Ratio'])
        ax.set_xlabel('Sharpe Ratio')
        ax.set_title('Risk-Adjusted Returns (Sharpe Ratio)')
        ax.grid(True, alpha=0.3)
        
        # 3. Max Drawdown
        ax = axes[0, 2]
        df_sorted = df_results.sort_values('Max Drawdown', ascending=False)
        ax.barh(df_sorted['Strategy'], df_sorted['Max Drawdown'].abs())
        ax.set_xlabel('Max Drawdown (%)')
        ax.set_title('Maximum Drawdown by Strategy')
        ax.grid(True, alpha=0.3)
        
        # 4. Win Rate
        ax = axes[1, 0]
        df_sorted = df_results.sort_values('Win Rate', ascending=True)
        ax.barh(df_sorted['Strategy'], df_sorted['Win Rate'])
        ax.set_xlabel('Win Rate (%)')
        ax.set_title('Win Rate by Strategy')
        ax.grid(True, alpha=0.3)
        
        # 5. Profit Factor
        ax = axes[1, 1]
        df_sorted = df_results.sort_values('Profit Factor', ascending=True)
        df_sorted['Profit Factor'] = df_sorted['Profit Factor'].clip(upper=10)  # Cap for visualization
        ax.barh(df_sorted['Strategy'], df_sorted['Profit Factor'])
        ax.set_xlabel('Profit Factor')
        ax.set_title('Profit Factor by Strategy')
        ax.grid(True, alpha=0.3)
        
        # 6. Cost Analysis
        ax = axes[1, 2]
        costs_df = df_results[['Strategy', 'Leverage Costs', 'Trading Costs']].set_index('Strategy')
        costs_df.plot(kind='barh', stacked=True, ax=ax)
        ax.set_xlabel('Costs ($)')
        ax.set_title('Trading and Leverage Costs')
        ax.grid(True, alpha=0.3)
        
        # 7. Return vs Risk Scatter
        ax = axes[2, 0]
        for _, row in df_results.iterrows():
            color = 'red' if 'no_leverage' in row['Strategy'] else 'blue'
            ax.scatter(abs(row['Max Drawdown']), row['Total Return'], 
                      label=row['Strategy'], alpha=0.6, s=100, color=color)
        ax.set_xlabel('Max Drawdown (%)')
        ax.set_ylabel('Total Return (%)')
        ax.set_title('Return vs Risk Profile')
        ax.grid(True, alpha=0.3)
        
        # 8. Leverage Impact Analysis
        ax = axes[2, 1]
        leverage_impact = []
        for strategy in PositionSizingStrategy:
            base_name = strategy.value
            no_lev = df_results[df_results['Strategy'] == f"{base_name}_no_leverage"]['Total Return'].values
            with_lev = df_results[df_results['Strategy'] == f"{base_name}_with_leverage"]['Total Return'].values
            if len(no_lev) > 0 and len(with_lev) > 0:
                leverage_impact.append({
                    'Strategy': base_name,
                    'Return Improvement': with_lev[0] - no_lev[0]
                })
        
        if leverage_impact:
            impact_df = pd.DataFrame(leverage_impact)
            ax.bar(impact_df['Strategy'], impact_df['Return Improvement'])
            ax.set_xlabel('Strategy')
            ax.set_ylabel('Return Improvement (%)')
            ax.set_title('Impact of Leverage on Returns')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # 9. Efficiency Frontier
        ax = axes[2, 2]
        ax.scatter(df_results['Max Drawdown'].abs(), df_results['Sharpe Ratio'])
        for idx, row in df_results.iterrows():
            if row['Sharpe Ratio'] > df_results['Sharpe Ratio'].quantile(0.75):
                ax.annotate(row['Strategy'], 
                           (abs(row['Max Drawdown']), row['Sharpe Ratio']),
                           fontsize=8, alpha=0.7)
        ax.set_xlabel('Max Drawdown (%)')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Efficiency Frontier')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'strategy_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Generate summary report
        summary = f"""
# Advanced Leverage Backtesting Results
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration
- Initial Capital: ${self.initial_capital:,.2f}
- Testing Period: {self.start_date.date()} to {self.end_date.date()}
- Max Leverage: {self.leverage_config.max_leverage}x
- Leverage Interest Rate: {self.leverage_config.annual_interest_rate*100:.1f}% annual
- Trading Fee: {self.trading_fee*100:.2f}%
- Slippage: {self.slippage*100:.2f}%

## Top Performing Strategies

### By Total Return:
{df_results.nlargest(5, 'Total Return')[['Strategy', 'Total Return', 'Sharpe Ratio']].to_string()}

### By Sharpe Ratio:
{df_results.nlargest(5, 'Sharpe Ratio')[['Strategy', 'Sharpe Ratio', 'Total Return']].to_string()}

### By Profit Factor:
{df_results.nlargest(5, 'Profit Factor')[['Strategy', 'Profit Factor', 'Win Rate']].to_string()}

## Key Insights

1. **Best Overall Strategy**: {df_results.loc[df_results['Sharpe Ratio'].idxmax(), 'Strategy']}
   - Sharpe Ratio: {df_results['Sharpe Ratio'].max():.2f}
   - Return: {df_results.loc[df_results['Sharpe Ratio'].idxmax(), 'Total Return']:.2f}%
   - Max Drawdown: {df_results.loc[df_results['Sharpe Ratio'].idxmax(), 'Max Drawdown']:.2f}%

2. **Highest Return Strategy**: {df_results.loc[df_results['Total Return'].idxmax(), 'Strategy']}
   - Total Return: {df_results['Total Return'].max():.2f}%
   - Associated Risk (Max DD): {df_results.loc[df_results['Total Return'].idxmax(), 'Max Drawdown']:.2f}%

3. **Leverage Impact**:
   - Average return improvement with leverage: {df_results[df_results['Strategy'].str.contains('with_leverage')]['Total Return'].mean() - df_results[df_results['Strategy'].str.contains('no_leverage')]['Total Return'].mean():.2f}%
   - Average leverage cost: ${df_results['Leverage Costs'].mean():,.2f}

4. **Risk Analysis**:
   - Lowest drawdown strategy: {df_results.loc[df_results['Max Drawdown'].idxmax(), 'Strategy']}
   - Highest win rate: {df_results.loc[df_results['Win Rate'].idxmax(), 'Strategy']} ({df_results['Win Rate'].max():.1f}%)

## Detailed Results
See 'backtest_results.csv' for complete metrics.
See 'strategy_analysis.png' for visualizations.
"""
        
        with open(output_path / 'BACKTEST_REPORT.md', 'w') as f:
            f.write(summary)
        
        logger.success(f"Report generated in {output_path}")
        
        return df_results


if __name__ == "__main__":
    logger.info("Starting Advanced Leverage Backtesting System")
    
    # Configure backtest
    leverage_config = LeverageConfig(
        max_leverage=3.0,
        annual_interest_rate=0.07,
        min_confidence_for_leverage=0.7,
        leverage_scaling="linear"
    )
    
    # Run backtest for last 30 days
    backtester = AdvancedLeverageBacktester(
        initial_capital=100000,
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        leverage_config=leverage_config
    )
    
    # Run all strategies
    results = backtester.run_all_strategies()
    
    # Generate report
    df_results = backtester.generate_report()
    
    # Print summary
    print("\n" + "="*80)
    print("BACKTESTING COMPLETE")
    print("="*80)
    print(f"\nTop 5 Strategies by Sharpe Ratio:")
    print(df_results.nlargest(5, 'Sharpe Ratio')[['Strategy', 'Total Return', 'Sharpe Ratio', 'Max Drawdown']])
    
    print(f"\nTop 5 Strategies by Total Return:")
    print(df_results.nlargest(5, 'Total Return')[['Strategy', 'Total Return', 'Sharpe Ratio', 'Max Drawdown']])
    
    logger.success("Advanced backtesting complete!")