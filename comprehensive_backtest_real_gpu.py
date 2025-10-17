#!/usr/bin/env python3
"""
Comprehensive backtesting system using real GPU forecasts and multiple position sizing strategies.
This system integrates with the actual trade_stock_e2e trading logic to test various strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

# Import actual trading modules
from trade_stock_e2e import analyze_symbols, backtest_forecasts
from src.position_sizing_optimizer import (
    constant_sizing,
    expected_return_sizing,
    volatility_scaled_sizing,
    top_n_expected_return_sizing,
    backtest_position_sizing_series,
    sharpe_ratio
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveBacktester:
    """
    Comprehensive backtesting system that uses real GPU forecasts and multiple position sizing strategies.
    """
    
    def __init__(self, symbols: List[str], start_date: str = None, end_date: str = None):
        self.symbols = symbols
        self.start_date = start_date or "2021-01-01"
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.results = {}
        
    def get_real_gpu_forecasts(self, symbol: str, num_simulations: int = 100) -> pd.DataFrame:
        """
        Get real GPU forecasts for a symbol using the actual trading system.
        This uses the same analyze_symbols function as the live trading system.
        """
        try:
            logger.info(f"Getting real GPU forecasts for {symbol}")
            
            # Use the actual backtest_forecasts function from trade_stock_e2e
            backtest_df = backtest_forecasts(symbol, num_simulations)
            
            # Calculate actual returns for the backtesting period
            actual_returns = []
            predicted_returns = []
            
            for idx, row in backtest_df.iterrows():
                # Calculate actual return (next day's close / current close - 1)
                actual_return = (row.get('next_close', row['close']) / row['close'] - 1) if row['close'] > 0 else 0
                
                # Calculate predicted return based on the model's prediction
                predicted_return = (row['predicted_close'] / row['close'] - 1) if row['close'] > 0 else 0
                
                actual_returns.append(actual_return)
                predicted_returns.append(predicted_return)
            
            # Create DataFrame with actual and predicted returns
            df = pd.DataFrame({
                'actual_return': actual_returns,
                'predicted_return': predicted_returns,
                'timestamp': pd.date_range(start=self.start_date, periods=len(actual_returns), freq='D')
            })
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting GPU forecasts for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_all_forecasts(self) -> Dict[str, pd.DataFrame]:
        """
        Get GPU forecasts for all symbols.
        """
        all_forecasts = {}
        
        for symbol in self.symbols:
            forecasts = self.get_real_gpu_forecasts(symbol)
            if not forecasts.empty:
                all_forecasts[symbol] = forecasts
                logger.info(f"Got {len(forecasts)} forecasts for {symbol}")
        
        return all_forecasts
    
    def create_multi_asset_data(self, forecasts: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create multi-asset actual and predicted returns DataFrames.
        """
        actual_data = {}
        predicted_data = {}
        
        for symbol, df in forecasts.items():
            if not df.empty:
                actual_data[symbol] = df.set_index('timestamp')['actual_return']
                predicted_data[symbol] = df.set_index('timestamp')['predicted_return']
        
        actual_df = pd.DataFrame(actual_data)
        predicted_df = pd.DataFrame(predicted_data)
        
        # Align indices and forward fill missing values
        common_index = actual_df.index.intersection(predicted_df.index)
        actual_df = actual_df.loc[common_index].fillna(0)
        predicted_df = predicted_df.loc[common_index].fillna(0)
        
        return actual_df, predicted_df
    
    def test_position_sizing_strategies(self, actual_df: pd.DataFrame, predicted_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Test multiple position sizing strategies and return performance results.
        """
        strategies = {
            'constant_1x': lambda p: constant_sizing(p, factor=1.0),
            'constant_0.5x': lambda p: constant_sizing(p, factor=0.5),
            'constant_2x': lambda p: constant_sizing(p, factor=2.0),
            'expected_return_1x': lambda p: expected_return_sizing(p, risk_factor=1.0),
            'expected_return_0.5x': lambda p: expected_return_sizing(p, risk_factor=0.5),
            'expected_return_2x': lambda p: expected_return_sizing(p, risk_factor=2.0),
            'volatility_scaled': lambda p: volatility_scaled_sizing(p, window=10),
            'top_1_best': lambda p: top_n_expected_return_sizing(p, n=1, leverage=1.0),
            'top_2_best': lambda p: top_n_expected_return_sizing(p, n=2, leverage=1.0),
            'top_3_best': lambda p: top_n_expected_return_sizing(p, n=3, leverage=1.0),
            'top_1_high_lev': lambda p: top_n_expected_return_sizing(p, n=1, leverage=2.0),
            'balanced_k2': lambda p: predicted_df / 2,  # K-divisor approach
            'balanced_k3': lambda p: predicted_df / 3,  # K-divisor approach
            'balanced_k5': lambda p: predicted_df / 5,  # K-divisor approach
        }
        
        results = {}
        
        for name, strategy_func in strategies.items():
            logger.info(f"Testing strategy: {name}")
            
            try:
                # Get position sizes
                sizes = strategy_func(predicted_df)
                
                # Ensure sizes are properly clipped to reasonable bounds
                sizes = sizes.clip(-5, 5)  # Reasonable leverage bounds
                
                # Calculate PnL series
                pnl_series = backtest_position_sizing_series(
                    actual_df, 
                    predicted_df, 
                    lambda _: sizes,
                    trading_fee=0.001  # 0.1% trading fee
                )
                
                # Calculate performance metrics
                total_return = pnl_series.sum()
                sharpe = sharpe_ratio(pnl_series, risk_free_rate=0.02)  # 2% risk-free rate
                max_drawdown = self.calculate_max_drawdown(pnl_series.cumsum())
                volatility = pnl_series.std() * np.sqrt(252)  # Annualized volatility
                
                results[name] = {
                    'pnl_series': pnl_series,
                    'cumulative_pnl': pnl_series.cumsum(),
                    'total_return': total_return,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_drawdown,
                    'volatility': volatility,
                    'num_trades': len(pnl_series),
                    'win_rate': (pnl_series > 0).mean()
                }
                
                logger.info(f"{name}: Total Return={total_return:.4f}, Sharpe={sharpe:.3f}, Max DD={max_drawdown:.4f}")
                
            except Exception as e:
                logger.error(f"Error testing strategy {name}: {e}")
                continue
        
        return results
    
    def calculate_max_drawdown(self, cumulative_pnl: pd.Series) -> float:
        """Calculate maximum drawdown from cumulative PnL series."""
        peak = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - peak) / peak.abs()
        return drawdown.min()
    
    def generate_performance_plots(self, results: Dict[str, Dict], output_dir: str = "backtest_results"):
        """
        Generate comprehensive performance plots and save them.
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Cumulative PnL Plot
        ax1 = plt.subplot(4, 2, 1)
        for name, metrics in results.items():
            if 'cumulative_pnl' in metrics:
                plt.plot(metrics['cumulative_pnl'], label=name, alpha=0.8)
        plt.title('Cumulative PnL by Strategy', fontsize=14, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Cumulative PnL')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 2. Risk-Return Scatter Plot
        ax2 = plt.subplot(4, 2, 2)
        returns = [metrics['total_return'] for metrics in results.values()]
        risks = [metrics['volatility'] for metrics in results.values()]
        names = list(results.keys())
        
        scatter = plt.scatter(risks, returns, c=range(len(names)), cmap='viridis', s=100, alpha=0.7)
        for i, name in enumerate(names):
            plt.annotate(name, (risks[i], returns[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)
        plt.title('Risk-Return Profile', fontsize=14, fontweight='bold')
        plt.xlabel('Volatility (Risk)')
        plt.ylabel('Total Return')
        plt.grid(True, alpha=0.3)
        
        # 3. Sharpe Ratio Bar Chart
        ax3 = plt.subplot(4, 2, 3)
        sharpe_ratios = [metrics['sharpe_ratio'] for metrics in results.values()]
        bars = plt.bar(names, sharpe_ratios, color='skyblue', alpha=0.8)
        plt.title('Sharpe Ratio by Strategy', fontsize=14, fontweight='bold')
        plt.ylabel('Sharpe Ratio')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, sharpe_ratios):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 4. Maximum Drawdown Bar Chart
        ax4 = plt.subplot(4, 2, 4)
        drawdowns = [metrics['max_drawdown'] for metrics in results.values()]
        bars = plt.bar(names, drawdowns, color='lightcoral', alpha=0.8)
        plt.title('Maximum Drawdown by Strategy', fontsize=14, fontweight='bold')
        plt.ylabel('Max Drawdown')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, drawdowns):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.01, 
                    f'{value:.3f}', ha='center', va='top', fontsize=8)
        
        # 5. Win Rate Bar Chart
        ax5 = plt.subplot(4, 2, 5)
        win_rates = [metrics['win_rate'] for metrics in results.values()]
        bars = plt.bar(names, win_rates, color='lightgreen', alpha=0.8)
        plt.title('Win Rate by Strategy', fontsize=14, fontweight='bold')
        plt.ylabel('Win Rate')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, win_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.1%}', ha='center', va='bottom', fontsize=8)
        
        # 6. Rolling Sharpe Ratio
        ax6 = plt.subplot(4, 2, 6)
        for name, metrics in results.items():
            if 'pnl_series' in metrics:
                rolling_sharpe = metrics['pnl_series'].rolling(window=30).apply(lambda x: sharpe_ratio(x, risk_free_rate=0.02))
                plt.plot(rolling_sharpe, label=name, alpha=0.7)
        plt.title('30-Day Rolling Sharpe Ratio', fontsize=14, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Rolling Sharpe Ratio')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 7. Performance Summary Table
        ax7 = plt.subplot(4, 2, 7)
        ax7.axis('tight')
        ax7.axis('off')
        
        # Create performance summary table
        table_data = []
        for name, metrics in results.items():
            table_data.append([
                name,
                f"{metrics['total_return']:.4f}",
                f"{metrics['sharpe_ratio']:.3f}",
                f"{metrics['max_drawdown']:.4f}",
                f"{metrics['volatility']:.4f}",
                f"{metrics['win_rate']:.1%}"
            ])
        
        table = ax7.table(cellText=table_data,
                         colLabels=['Strategy', 'Total Return', 'Sharpe', 'Max DD', 'Volatility', 'Win Rate'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        plt.title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
        
        # 8. Distribution of Daily Returns
        ax8 = plt.subplot(4, 2, 8)
        for name, metrics in results.items():
            if 'pnl_series' in metrics:
                plt.hist(metrics['pnl_series'], bins=50, alpha=0.5, label=name, density=True)
        plt.title('Distribution of Daily Returns', fontsize=14, fontweight='bold')
        plt.xlabel('Daily Return')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the comprehensive plot
        output_file = output_path / f"comprehensive_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Comprehensive results saved to {output_file}")
        
        # Save results to CSV
        csv_data = []
        for name, metrics in results.items():
            csv_data.append({
                'Strategy': name,
                'Total_Return': metrics['total_return'],
                'Sharpe_Ratio': metrics['sharpe_ratio'],
                'Max_Drawdown': metrics['max_drawdown'],
                'Volatility': metrics['volatility'],
                'Win_Rate': metrics['win_rate'],
                'Num_Trades': metrics['num_trades']
            })
        
        results_df = pd.DataFrame(csv_data)
        csv_file = output_path / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(csv_file, index=False)
        logger.info(f"Results CSV saved to {csv_file}")
        
        return output_file, csv_file
    
    def run_comprehensive_backtest(self, output_dir: str = "backtest_results"):
        """
        Run the comprehensive backtest with real GPU forecasts.
        """
        logger.info("Starting comprehensive backtest with real GPU forecasts...")
        
        # Get real GPU forecasts for all symbols
        logger.info("Getting real GPU forecasts...")
        forecasts = self.get_all_forecasts()
        
        if not forecasts:
            logger.error("No forecasts available. Cannot run backtest.")
            return
        
        logger.info(f"Got forecasts for {len(forecasts)} symbols")
        
        # Create multi-asset data
        logger.info("Creating multi-asset data...")
        actual_df, predicted_df = self.create_multi_asset_data(forecasts)
        
        if actual_df.empty or predicted_df.empty:
            logger.error("No data available for backtesting.")
            return
        
        logger.info(f"Created data with {len(actual_df)} time periods and {len(actual_df.columns)} assets")
        
        # Test position sizing strategies
        logger.info("Testing position sizing strategies...")
        results = self.test_position_sizing_strategies(actual_df, predicted_df)
        
        if not results:
            logger.error("No strategy results available.")
            return
        
        # Generate performance plots
        logger.info("Generating performance plots...")
        plot_file, csv_file = self.generate_performance_plots(results, output_dir)
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE BACKTEST RESULTS SUMMARY")
        logger.info("="*80)
        
        # Sort by Sharpe ratio
        sorted_results = sorted(results.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True)
        
        for name, metrics in sorted_results[:5]:  # Top 5 strategies
            logger.info(f"{name:20} | Return: {metrics['total_return']:8.4f} | Sharpe: {metrics['sharpe_ratio']:6.3f} | Max DD: {metrics['max_drawdown']:8.4f} | Win Rate: {metrics['win_rate']:6.1%}")
        
        logger.info("="*80)
        logger.info(f"Results saved to: {plot_file}")
        logger.info(f"CSV data saved to: {csv_file}")
        
        return results, plot_file, csv_file


def main():
    """
    Main function to run the comprehensive backtest.
    """
    # Define symbols to test (same as in trade_stock_e2e.py)
    symbols = [
        "COUR", "GOOG", "TSLA", "NVDA", "AAPL", "U", "ADSK", 
        "ADBE", "COIN", "MSFT", "NFLX", "UNIUSD", "ETHUSD", "BTCUSD"
    ]
    
    # Create backtester
    backtester = ComprehensiveBacktester(
        symbols=symbols,
        start_date="2023-01-01",
        end_date="2024-12-31"
    )
    
    # Run comprehensive backtest
    results, plot_file, csv_file = backtester.run_comprehensive_backtest()
    
    return results, plot_file, csv_file


if __name__ == "__main__":
    main()
