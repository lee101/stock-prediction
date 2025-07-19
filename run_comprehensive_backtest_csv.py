#!/usr/bin/env python3
"""
Comprehensive backtesting system using real CSV prediction files and multiple position sizing strategies.
This system works with the actual generated prediction CSV files to test various trading strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import warnings
import ast
import re

warnings.filterwarnings('ignore')

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

# Import actual trading modules
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

class CSVBacktester:
    """
    Comprehensive backtesting system that uses real CSV prediction files and multiple position sizing strategies.
    """
    
    def __init__(self, results_dir: str = "results", output_dir: str = "backtest_results"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def load_prediction_csvs(self, limit_files: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Load prediction CSV files from the results directory.
        """
        csv_files = list(self.results_dir.glob("predictions-*.csv"))
        csv_files = sorted(csv_files, key=lambda x: x.name)
        
        if limit_files:
            csv_files = csv_files[-limit_files:]  # Take the most recent files
            
        logger.info(f"Loading {len(csv_files)} prediction CSV files...")
        
        all_predictions = {}
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                timestamp = self.extract_timestamp_from_filename(csv_file.name)
                
                # Process each row (instrument) in the CSV
                for _, row in df.iterrows():
                    instrument = row['instrument']
                    
                    if instrument not in all_predictions:
                        all_predictions[instrument] = []
                    
                    # Extract the actual movements and predictions
                    prediction_data = self.extract_prediction_data(row, timestamp)
                    if prediction_data:
                        all_predictions[instrument].append(prediction_data)
                        
            except Exception as e:
                logger.warning(f"Error loading {csv_file}: {e}")
                continue
        
        # Convert to DataFrames
        processed_predictions = {}
        for instrument, data_list in all_predictions.items():
            if len(data_list) > 10:  # Only include instruments with sufficient data
                df = pd.DataFrame(data_list)
                df = df.sort_values('timestamp').reset_index(drop=True)
                processed_predictions[instrument] = df
                logger.info(f"Loaded {len(df)} predictions for {instrument}")
        
        return processed_predictions
    
    def extract_timestamp_from_filename(self, filename: str) -> datetime:
        """Extract timestamp from prediction CSV filename."""
        # Format: predictions-2023-06-23_12-13-09.csv
        match = re.search(r'predictions-(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv', filename)
        if match:
            timestamp_str = match.group(1)
            return datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
        return datetime.now()
    
    def extract_prediction_data(self, row, timestamp: datetime) -> Optional[Dict]:
        """Extract prediction and actual movement data from CSV row."""
        try:
            # Parse the tensor strings to get actual values
            close_actuals = self.parse_tensor_string(row['close_actual_movement_values'])
            close_predictions = self.parse_tensor_string(row['close_predictions'])
            
            if close_actuals is None or close_predictions is None:
                return None
            
            # Calculate additional metrics
            data = {
                'timestamp': timestamp,
                'instrument': row['instrument'],
                'close_last_price': row['close_last_price'],
                'close_predicted_price': row['close_predicted_price'],
                'close_actual_movements': close_actuals,
                'close_predictions': close_predictions,
                'close_val_loss': row['close_val_loss'],
                'takeprofit_profit': row.get('takeprofit_profit', 0),
                'entry_takeprofit_profit': row.get('entry_takeprofit_profit', 0),
                'maxdiffprofit_profit': row.get('maxdiffprofit_profit', 0),
            }
            
            # Add high/low data if available
            if 'high_actual_movement_values' in row and pd.notna(row['high_actual_movement_values']):
                data['high_actual_movements'] = self.parse_tensor_string(row['high_actual_movement_values'])
                data['high_predictions'] = self.parse_tensor_string(row['high_predictions'])
            
            if 'low_actual_movement_values' in row and pd.notna(row['low_actual_movement_values']):
                data['low_actual_movements'] = self.parse_tensor_string(row['low_actual_movement_values'])
                data['low_predictions'] = self.parse_tensor_string(row['low_predictions'])
            
            return data
            
        except Exception as e:
            logger.debug(f"Error extracting data for {row.get('instrument', 'unknown')}: {e}")
            return None
    
    def parse_tensor_string(self, tensor_str) -> Optional[List[float]]:
        """Parse tensor string like 'tensor([0.1, -0.2, 0.3])' to list of floats."""
        if pd.isna(tensor_str) or not isinstance(tensor_str, str):
            return None
        
        try:
            # Remove 'tensor(' and ')' and parse the list
            tensor_str = tensor_str.strip()
            if tensor_str.startswith('tensor('):
                tensor_str = tensor_str[7:-1]  # Remove 'tensor(' and ')'
            
            # Handle the list format
            if tensor_str.startswith('[') and tensor_str.endswith(']'):
                # Use ast.literal_eval for safe evaluation
                values = ast.literal_eval(tensor_str)
                return [float(v) for v in values]
            
        except Exception as e:
            logger.debug(f"Error parsing tensor string '{tensor_str}': {e}")
            
        return None
    
    def create_multi_asset_data(self, predictions: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create multi-asset actual and predicted returns DataFrames using the prediction sequence data.
        """
        actual_data = {}
        predicted_data = {}
        
        for instrument, df in predictions.items():
            if len(df) < 5:  # Need minimum data
                continue
                
            # Use the prediction sequence data (each row has a sequence of 6-7 predictions/actuals)
            instrument_actuals = []
            instrument_predictions = []
            timestamps = []
            
            for _, row in df.iterrows():
                # Each row contains a sequence, we'll use the last prediction/actual from each sequence
                if row['close_actual_movements'] and row['close_predictions']:
                    # Take the last value from each sequence as the "next day" prediction
                    actual_return = row['close_actual_movements'][-1] if row['close_actual_movements'] else 0
                    predicted_return = row['close_predictions'][-1] if row['close_predictions'] else 0
                    
                    instrument_actuals.append(actual_return)
                    instrument_predictions.append(predicted_return)
                    timestamps.append(row['timestamp'])
            
            if len(instrument_actuals) > 5:  # Minimum threshold
                # Create index from timestamps
                actual_series = pd.Series(instrument_actuals, index=timestamps, name=instrument)
                predicted_series = pd.Series(instrument_predictions, index=timestamps, name=instrument)
                
                actual_data[instrument] = actual_series
                predicted_data[instrument] = predicted_series
        
        if not actual_data:
            logger.error("No valid data found for backtesting")
            return pd.DataFrame(), pd.DataFrame()
        
        # Create DataFrames and align
        actual_df = pd.DataFrame(actual_data)
        predicted_df = pd.DataFrame(predicted_data)
        
        # Align indices and forward fill missing values
        common_index = actual_df.index.intersection(predicted_df.index)
        if len(common_index) == 0:
            logger.error("No common timestamps found")
            return pd.DataFrame(), pd.DataFrame()
            
        actual_df = actual_df.loc[common_index].fillna(0)
        predicted_df = predicted_df.loc[common_index].fillna(0)
        
        logger.info(f"Created multi-asset data with {len(actual_df)} time periods and {len(actual_df.columns)} assets")
        
        return actual_df, predicted_df
    
    def test_position_sizing_strategies(self, actual_df: pd.DataFrame, predicted_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Test multiple position sizing strategies and return performance results.
        """
        strategies = {
            # Basic strategies
            'constant_1x': lambda p: constant_sizing(p, factor=1.0),
            'constant_0.5x': lambda p: constant_sizing(p, factor=0.5),
            'constant_2x': lambda p: constant_sizing(p, factor=2.0),
            
            # Expected return strategies
            'expected_return_1x': lambda p: expected_return_sizing(p, risk_factor=1.0),
            'expected_return_0.5x': lambda p: expected_return_sizing(p, risk_factor=0.5),
            'expected_return_2x': lambda p: expected_return_sizing(p, risk_factor=2.0),
            'expected_return_5x': lambda p: expected_return_sizing(p, risk_factor=5.0),
            
            # Volatility-based strategies
            'volatility_scaled': lambda p: volatility_scaled_sizing(p, window=10),
            'volatility_scaled_5d': lambda p: volatility_scaled_sizing(p, window=5),
            'volatility_scaled_20d': lambda p: volatility_scaled_sizing(p, window=20),
            
            # Top-N strategies  
            'top_1_best': lambda p: top_n_expected_return_sizing(p, n=1, leverage=1.0),
            'top_2_best': lambda p: top_n_expected_return_sizing(p, n=2, leverage=1.0),
            'top_3_best': lambda p: top_n_expected_return_sizing(p, n=3, leverage=1.0),
            'top_1_high_lev': lambda p: top_n_expected_return_sizing(p, n=1, leverage=2.0),
            'top_2_high_lev': lambda p: top_n_expected_return_sizing(p, n=2, leverage=2.0),
            
            # K-divisor approaches (balanced exposure)
            'balanced_k2': lambda p: predicted_df / 2,
            'balanced_k3': lambda p: predicted_df / 3,
            'balanced_k5': lambda p: predicted_df / 5,
            'balanced_k10': lambda p: predicted_df / 10,
            
            # Sign-only strategies (direction only)
            'sign_only': lambda p: np.sign(predicted_df),
            'sign_scaled_0.5': lambda p: np.sign(predicted_df) * 0.5,
            
            # Threshold strategies
            'threshold_01': lambda p: predicted_df * (predicted_df.abs() > 0.01),
            'threshold_02': lambda p: predicted_df * (predicted_df.abs() > 0.02),
            'threshold_05': lambda p: predicted_df * (predicted_df.abs() > 0.05),
            
            # Best absolute predictions
            'abs_best_1': lambda p: self.get_absolute_best_n(predicted_df, n=1),
            'abs_best_2': lambda p: self.get_absolute_best_n(predicted_df, n=2),
            'abs_best_3': lambda p: self.get_absolute_best_n(predicted_df, n=3),
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
                    'win_rate': (pnl_series > 0).mean(),
                    'avg_trade': pnl_series.mean(),
                    'hit_rate': (pnl_series > 0).sum() / len(pnl_series) if len(pnl_series) > 0 else 0
                }
                
                logger.info(f"{name}: Total Return={total_return:.4f}, Sharpe={sharpe:.3f}, Max DD={max_drawdown:.4f}")
                
            except Exception as e:
                logger.error(f"Error testing strategy {name}: {e}")
                continue
        
        return results
    
    def get_absolute_best_n(self, predicted_df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Get the n assets with highest absolute predicted returns."""
        abs_pred = predicted_df.abs()
        ranks = abs_pred.rank(axis=1, ascending=False, method='first')
        selected = ranks <= n
        
        # Apply original signs
        sizes = selected * np.sign(predicted_df)
        
        # Normalize by number of selected assets
        counts = selected.sum(axis=1).replace(0, np.nan)
        sizes = sizes.div(counts, axis=0).fillna(0.0)
        
        return sizes
    
    def calculate_max_drawdown(self, cumulative_pnl: pd.Series) -> float:
        """Calculate maximum drawdown from cumulative PnL series."""
        if len(cumulative_pnl) == 0:
            return 0.0
        peak = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - peak) / np.maximum(peak.abs(), 1e-9)  # Avoid division by zero
        return drawdown.min()
    
    def generate_performance_plots(self, results: Dict[str, Dict], output_dir: str = "backtest_results"):
        """
        Generate comprehensive performance plots and save them.
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 24))
        
        # Filter out strategies with no valid results
        valid_results = {k: v for k, v in results.items() if v['total_return'] != 0}
        
        if not valid_results:
            logger.warning("No valid results to plot")
            return None, None
        
        # 1. Cumulative PnL Plot
        ax1 = plt.subplot(4, 2, 1)
        for name, metrics in valid_results.items():
            if 'cumulative_pnl' in metrics and len(metrics['cumulative_pnl']) > 0:
                plt.plot(metrics['cumulative_pnl'], label=name, alpha=0.8)
        plt.title('Cumulative PnL by Strategy', fontsize=14, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Cumulative PnL')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 2. Risk-Return Scatter Plot
        ax2 = plt.subplot(4, 2, 2)
        returns = [metrics['total_return'] for metrics in valid_results.values()]
        risks = [metrics['volatility'] for metrics in valid_results.values()]
        names = list(valid_results.keys())
        
        scatter = plt.scatter(risks, returns, c=range(len(names)), cmap='viridis', s=100, alpha=0.7)
        for i, name in enumerate(names):
            if i < 10:  # Only annotate first 10 to avoid clutter
                plt.annotate(name, (risks[i], returns[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)
        plt.title('Risk-Return Profile', fontsize=14, fontweight='bold')
        plt.xlabel('Volatility (Risk)')
        plt.ylabel('Total Return')
        plt.grid(True, alpha=0.3)
        
        # 3. Sharpe Ratio Bar Chart
        ax3 = plt.subplot(4, 2, 3)
        sharpe_ratios = [metrics['sharpe_ratio'] for metrics in valid_results.values()]
        names_short = [name[:10] for name in names]  # Truncate names for display
        
        bars = plt.bar(range(len(names_short)), sharpe_ratios, color='skyblue', alpha=0.8)
        plt.title('Sharpe Ratio by Strategy', fontsize=14, fontweight='bold')
        plt.ylabel('Sharpe Ratio')
        plt.xticks(range(len(names_short)), names_short, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # 4. Maximum Drawdown Bar Chart
        ax4 = plt.subplot(4, 2, 4)
        drawdowns = [metrics['max_drawdown'] for metrics in valid_results.values()]
        bars = plt.bar(range(len(names_short)), drawdowns, color='lightcoral', alpha=0.8)
        plt.title('Maximum Drawdown by Strategy', fontsize=14, fontweight='bold')
        plt.ylabel('Max Drawdown')
        plt.xticks(range(len(names_short)), names_short, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # 5. Win Rate Bar Chart
        ax5 = plt.subplot(4, 2, 5)
        win_rates = [metrics['win_rate'] for metrics in valid_results.values()]
        bars = plt.bar(range(len(names_short)), win_rates, color='lightgreen', alpha=0.8)
        plt.title('Win Rate by Strategy', fontsize=14, fontweight='bold')
        plt.ylabel('Win Rate')
        plt.xticks(range(len(names_short)), names_short, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # 6. Rolling Sharpe Ratio (for top 5 strategies)
        ax6 = plt.subplot(4, 2, 6)
        top_5_strategies = sorted(valid_results.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True)[:5]
        
        for name, metrics in top_5_strategies:
            if 'pnl_series' in metrics and len(metrics['pnl_series']) > 30:
                rolling_sharpe = metrics['pnl_series'].rolling(window=30).apply(lambda x: sharpe_ratio(x, risk_free_rate=0.02))
                plt.plot(rolling_sharpe, label=name, alpha=0.7)
        plt.title('30-Day Rolling Sharpe Ratio (Top 5)', fontsize=14, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Rolling Sharpe Ratio')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 7. Performance Summary Table
        ax7 = plt.subplot(4, 2, 7)
        ax7.axis('tight')
        ax7.axis('off')
        
        # Create performance summary table for top 10 strategies
        top_10_strategies = sorted(valid_results.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True)[:10]
        table_data = []
        for name, metrics in top_10_strategies:
            table_data.append([
                name[:15],  # Truncate name
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
        plt.title('Performance Summary (Top 10)', fontsize=14, fontweight='bold', pad=20)
        
        # 8. Distribution of Daily Returns (top 3 strategies)
        ax8 = plt.subplot(4, 2, 8)
        top_3_strategies = sorted(valid_results.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True)[:3]
        
        for name, metrics in top_3_strategies:
            if 'pnl_series' in metrics and len(metrics['pnl_series']) > 0:
                plt.hist(metrics['pnl_series'], bins=30, alpha=0.5, label=name, density=True)
        plt.title('Distribution of Daily Returns (Top 3)', fontsize=14, fontweight='bold')
        plt.xlabel('Daily Return')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the comprehensive plot
        output_file = output_path / f"csv_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Comprehensive results saved to {output_file}")
        
        # Save results to CSV
        csv_data = []
        for name, metrics in valid_results.items():
            csv_data.append({
                'Strategy': name,
                'Total_Return': metrics['total_return'],
                'Sharpe_Ratio': metrics['sharpe_ratio'],
                'Max_Drawdown': metrics['max_drawdown'],
                'Volatility': metrics['volatility'],
                'Win_Rate': metrics['win_rate'],
                'Num_Trades': metrics['num_trades'],
                'Avg_Trade': metrics['avg_trade'],
                'Hit_Rate': metrics['hit_rate']
            })
        
        results_df = pd.DataFrame(csv_data)
        results_df = results_df.sort_values('Sharpe_Ratio', ascending=False)
        csv_file = output_path / f"csv_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(csv_file, index=False)
        logger.info(f"Results CSV saved to {csv_file}")
        
        plt.close()
        
        return output_file, csv_file
    
    def run_comprehensive_backtest(self, limit_files: Optional[int] = 50):
        """
        Run the comprehensive backtest with real CSV forecasts.
        """
        logger.info("Starting comprehensive backtest with CSV prediction files...")
        
        # Load prediction CSV files
        logger.info("Loading prediction CSV files...")
        predictions = self.load_prediction_csvs(limit_files=limit_files)
        
        if not predictions:
            logger.error("No prediction data loaded. Cannot run backtest.")
            return None, None, None
        
        logger.info(f"Loaded predictions for {len(predictions)} instruments")
        
        # Create multi-asset data
        logger.info("Creating multi-asset data...")
        actual_df, predicted_df = self.create_multi_asset_data(predictions)
        
        if actual_df.empty or predicted_df.empty:
            logger.error("No data available for backtesting.")
            return None, None, None
        
        logger.info(f"Created data with {len(actual_df)} time periods and {len(actual_df.columns)} assets")
        
        # Test position sizing strategies
        logger.info("Testing position sizing strategies...")
        results = self.test_position_sizing_strategies(actual_df, predicted_df)
        
        if not results:
            logger.error("No strategy results available.")
            return None, None, None
        
        # Generate performance plots
        logger.info("Generating performance plots...")
        plot_file, csv_file = self.generate_performance_plots(results, self.output_dir)
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("CSV BACKTEST RESULTS SUMMARY")
        logger.info("="*80)
        
        # Sort by Sharpe ratio
        sorted_results = sorted(results.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True)
        
        logger.info(f"{'Strategy':<20} | {'Return':<8} | {'Sharpe':<6} | {'Max DD':<8} | {'Win Rate':<8} | {'# Trades':<8}")
        logger.info("-" * 80)
        
        for name, metrics in sorted_results[:10]:  # Top 10 strategies
            logger.info(f"{name:<20} | {metrics['total_return']:8.4f} | {metrics['sharpe_ratio']:6.3f} | {metrics['max_drawdown']:8.4f} | {metrics['win_rate']:8.1%} | {metrics['num_trades']:8d}")
        
        logger.info("="*80)
        if plot_file:
            logger.info(f"Plots saved to: {plot_file}")
        if csv_file:
            logger.info(f"CSV data saved to: {csv_file}")
        
        return results, plot_file, csv_file


def main():
    """
    Main function to run the CSV-based comprehensive backtest.
    """
    logger.info("Starting CSV-based comprehensive backtest...")
    
    # Create backtester
    backtester = CSVBacktester()
    
    # Run comprehensive backtest
    results, plot_file, csv_file = backtester.run_comprehensive_backtest(limit_files=100)
    
    if results:
        logger.info("CSV backtest completed successfully!")
        return results, plot_file, csv_file
    else:
        logger.error("CSV backtest failed!")
        return None, None, None


if __name__ == "__main__":
    main()
