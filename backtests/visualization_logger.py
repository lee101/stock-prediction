#!/usr/bin/env python3
"""
Comprehensive visualization and logging system for trading strategy simulation.
Creates detailed graphs and TensorBoard logs for analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logger = logging.getLogger(__name__)

class VisualizationLogger:
    """Handles all visualization and logging for trading strategies."""
    
    def __init__(self, output_dir: str = "trading_results", tb_log_dir: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # TensorBoard setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if tb_log_dir is None:
            tb_log_dir = f"./logs/trading_simulation_{timestamp}"
        self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        logger.info(f"Visualization logger initialized - Output: {self.output_dir}, TensorBoard: {tb_log_dir}")
    
    def log_forecasts_to_tensorboard(self, forecasts: Dict, step: int = 0):
        """Log forecast data to TensorBoard."""
        logger.info("Logging forecasts to TensorBoard...")
        
        # Aggregate forecast metrics
        predicted_returns = []
        symbols = []
        
        for symbol, data in forecasts.items():
            if 'close_total_predicted_change' in data:
                predicted_returns.append(data['close_total_predicted_change'])
                symbols.append(symbol)
        
        if predicted_returns:
            # Log distribution of predicted returns
            self.tb_writer.add_histogram('forecasts/predicted_returns_distribution', 
                                        np.array(predicted_returns), step)
            
            # Log individual predictions
            for i, (symbol, pred_return) in enumerate(zip(symbols, predicted_returns)):
                self.tb_writer.add_scalar(f'forecasts/individual/{symbol}', pred_return, step)
            
            # Log summary statistics
            self.tb_writer.add_scalar('forecasts/mean_predicted_return', np.mean(predicted_returns), step)
            self.tb_writer.add_scalar('forecasts/std_predicted_return', np.std(predicted_returns), step)
            self.tb_writer.add_scalar('forecasts/max_predicted_return', np.max(predicted_returns), step)
            self.tb_writer.add_scalar('forecasts/min_predicted_return', np.min(predicted_returns), step)
            
            # Log positive vs negative predictions
            positive_preds = sum(1 for x in predicted_returns if x > 0)
            negative_preds = sum(1 for x in predicted_returns if x <= 0)
            self.tb_writer.add_scalar('forecasts/positive_predictions_count', positive_preds, step)
            self.tb_writer.add_scalar('forecasts/negative_predictions_count', negative_preds, step)
    
    def log_strategies_to_tensorboard(self, strategies: Dict, step: int = 0):
        """Log strategy performance to TensorBoard."""
        logger.info("Logging strategies to TensorBoard...")
        
        for strategy_name, strategy_data in strategies.items():
            if 'error' in strategy_data:
                continue
            
            # Log basic metrics
            expected_return = strategy_data.get('expected_return', 0)
            self.tb_writer.add_scalar(f'strategies/{strategy_name}/expected_return', 
                                    expected_return, step)
            
            # Log performance if available
            perf = strategy_data.get('performance', {})
            if perf:
                self.tb_writer.add_scalar(f'strategies/{strategy_name}/simulated_return', 
                                        perf.get('simulated_actual_return', 0), step)
                self.tb_writer.add_scalar(f'strategies/{strategy_name}/profit_loss', 
                                        perf.get('profit_loss', 0), step)
                self.tb_writer.add_scalar(f'strategies/{strategy_name}/outperformance', 
                                        perf.get('outperformance', 0), step)
            
            # Log allocation diversity
            allocation = strategy_data.get('allocation', {})
            if allocation:
                num_positions = len(allocation)
                max_allocation = max(allocation.values())
                allocation_entropy = -sum(w * np.log(w + 1e-10) for w in allocation.values())
                
                self.tb_writer.add_scalar(f'strategies/{strategy_name}/num_positions', 
                                        num_positions, step)
                self.tb_writer.add_scalar(f'strategies/{strategy_name}/max_allocation', 
                                        max_allocation, step)
                self.tb_writer.add_scalar(f'strategies/{strategy_name}/allocation_entropy', 
                                        allocation_entropy, step)
    
    def create_forecast_visualization(self, forecasts: Dict, filename: str = None) -> str:
        """Create comprehensive forecast visualization."""
        logger.info("Creating forecast visualization...")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"forecasts_{timestamp}.png"
        
        # Prepare data
        symbols = []
        predicted_returns = []
        predicted_prices = []
        last_prices = []
        
        for symbol, data in forecasts.items():
            if 'close_total_predicted_change' in data:
                symbols.append(symbol)
                predicted_returns.append(data['close_total_predicted_change'])
                predicted_prices.append(data.get('close_predicted_price_value', 0))
                last_prices.append(data.get('close_last_price', 0))
        
        if not symbols:
            logger.warning("No forecast data to visualize")
            return None
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Stock Forecasts Analysis', fontsize=16, fontweight='bold')
        
        # 1. Predicted Returns Bar Chart
        colors = ['green' if x > 0 else 'red' for x in predicted_returns]
        bars1 = ax1.bar(symbols, predicted_returns, color=colors, alpha=0.7)
        ax1.set_title('Predicted Returns by Symbol', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Predicted Return (%)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels on bars
        for bar, value in zip(bars1, predicted_returns):
            height = bar.get_height()
            ax1.annotate(f'{value:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=8)
        
        # 2. Price Comparison
        x_pos = np.arange(len(symbols))
        width = 0.35
        
        bars2a = ax2.bar(x_pos - width/2, last_prices, width, label='Current Price', alpha=0.7)
        bars2b = ax2.bar(x_pos + width/2, predicted_prices, width, label='Predicted Price', alpha=0.7)
        
        ax2.set_title('Current vs Predicted Prices', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Price ($)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(symbols, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Return Distribution
        ax3.hist(predicted_returns, bins=min(20, len(predicted_returns)), alpha=0.7, edgecolor='black')
        ax3.set_title('Distribution of Predicted Returns', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Predicted Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero Return')
        ax3.axvline(x=np.mean(predicted_returns), color='green', linestyle='--', alpha=0.7, 
                   label=f'Mean: {np.mean(predicted_returns):.3f}')
        ax3.legend()
        
        # 4. Top/Bottom Performers
        sorted_data = sorted(zip(symbols, predicted_returns), key=lambda x: x[1])
        top_5 = sorted_data[-5:]
        bottom_5 = sorted_data[:5]
        
        # Combine and create horizontal bar chart
        combined_symbols = [x[0] for x in bottom_5 + top_5]
        combined_returns = [x[1] for x in bottom_5 + top_5]
        colors_combined = ['red' if x < 0 else 'green' for x in combined_returns]
        
        y_pos = np.arange(len(combined_symbols))
        bars4 = ax4.barh(y_pos, combined_returns, color=colors_combined, alpha=0.7)
        ax4.set_title('Top & Bottom Predicted Performers', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Predicted Return (%)')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(combined_symbols)
        ax4.grid(True, alpha=0.3)
        ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for bar, value in zip(bars4, combined_returns):
            width_bar = bar.get_width()
            ax4.annotate(f'{value:.3f}',
                        xy=(width_bar, bar.get_y() + bar.get_height() / 2),
                        xytext=(3 if width_bar >= 0 else -3, 0),
                        textcoords="offset points",
                        ha='left' if width_bar >= 0 else 'right', va='center',
                        fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Forecast visualization saved to {output_path}")
        
        plt.close()
        return str(output_path)
    
    def create_strategy_comparison(self, strategies: Dict, filename: str = None) -> str:
        """Create strategy comparison visualization."""
        logger.info("Creating strategy comparison visualization...")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"strategy_comparison_{timestamp}.png"
        
        # Filter out error strategies
        valid_strategies = {k: v for k, v in strategies.items() if 'error' not in v}
        
        if not valid_strategies:
            logger.warning("No valid strategies to compare")
            return None
        
        # Prepare data
        strategy_names = list(valid_strategies.keys())
        expected_returns = [s.get('expected_return', 0) for s in valid_strategies.values()]
        simulated_returns = [s.get('performance', {}).get('simulated_actual_return', 0) for s in valid_strategies.values()]
        profit_losses = [s.get('performance', {}).get('profit_loss', 0) for s in valid_strategies.values()]
        num_positions = [s.get('num_positions', len(s.get('allocation', {}))) for s in valid_strategies.values()]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Trading Strategy Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Expected vs Simulated Returns
        x_pos = np.arange(len(strategy_names))
        width = 0.35
        
        bars1a = ax1.bar(x_pos - width/2, expected_returns, width, label='Expected', alpha=0.7)
        bars1b = ax1.bar(x_pos + width/2, simulated_returns, width, label='Simulated', alpha=0.7)
        
        ax1.set_title('Expected vs Simulated Returns', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Return (%)')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(strategy_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for bars in [bars1a, bars1b]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3 if height >= 0 else -15),
                            textcoords="offset points",
                            ha='center', va='bottom' if height >= 0 else 'top',
                            fontsize=8)
        
        # 2. Profit/Loss
        colors = ['green' if x > 0 else 'red' for x in profit_losses]
        bars2 = ax2.bar(strategy_names, profit_losses, color=colors, alpha=0.7)
        ax2.set_title('Profit/Loss by Strategy', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Profit/Loss ($)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for bar, value in zip(bars2, profit_losses):
            height = bar.get_height()
            ax2.annotate(f'${value:,.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=8)
        
        # 3. Risk vs Return Scatter Plot
        risks = []  # We'll use number of positions as a proxy for risk (inverse relationship)
        for s in valid_strategies.values():
            num_pos = s.get('num_positions', len(s.get('allocation', {})))
            risk_proxy = 1.0 / max(num_pos, 1)  # Higher positions = lower risk
            risks.append(risk_proxy)
        
        scatter = ax3.scatter(risks, simulated_returns, c=profit_losses, s=100, alpha=0.7, cmap='RdYlGn')
        ax3.set_title('Risk vs Return Profile', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Risk Level (1/num_positions)')
        ax3.set_ylabel('Simulated Return (%)')
        ax3.grid(True, alpha=0.3)
        
        # Add strategy labels
        for i, name in enumerate(strategy_names):
            ax3.annotate(name, (risks[i], simulated_returns[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Profit/Loss ($)')
        
        # 4. Allocation Diversity
        diversification_scores = []
        for s in valid_strategies.values():
            allocation = s.get('allocation', {})
            if allocation:
                # Calculate entropy as measure of diversification
                weights = list(allocation.values())
                entropy = -sum(w * np.log(w + 1e-10) for w in weights if w > 0)
                diversification_scores.append(entropy)
            else:
                diversification_scores.append(0)
        
        bars4 = ax4.bar(strategy_names, diversification_scores, alpha=0.7)
        ax4.set_title('Portfolio Diversification (Higher = More Diverse)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Diversification Score (Entropy)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars4, diversification_scores):
            height = bar.get_height()
            ax4.annotate(f'{value:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Strategy comparison saved to {output_path}")
        
        plt.close()
        return str(output_path)
    
    def create_portfolio_allocation_plots(self, strategies: Dict, filename: str = None) -> str:
        """Create detailed portfolio allocation visualizations."""
        logger.info("Creating portfolio allocation visualizations...")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"portfolio_allocations_{timestamp}.png"
        
        # Filter valid strategies with allocations
        strategies_with_allocations = {k: v for k, v in strategies.items() 
                                     if 'error' not in v and v.get('allocation')}
        
        if not strategies_with_allocations:
            logger.warning("No strategies with allocation data")
            return None
        
        # Calculate subplot layout
        num_strategies = len(strategies_with_allocations)
        cols = min(3, num_strategies)
        rows = (num_strategies + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
        fig.suptitle('Portfolio Allocations by Strategy', fontsize=16, fontweight='bold')
        
        # Handle single subplot case
        if num_strategies == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, list) else [axes]
        else:
            axes = axes.flatten()
        
        # Create pie charts for each strategy
        for i, (strategy_name, strategy_data) in enumerate(strategies_with_allocations.items()):
            allocation = strategy_data.get('allocation', {})
            
            if not allocation:
                continue
            
            # Prepare data for pie chart
            labels = []
            sizes = []
            colors = plt.cm.Set3(np.linspace(0, 1, len(allocation)))
            
            for symbol, weight in sorted(allocation.items(), key=lambda x: x[1], reverse=True):
                labels.append(f'{symbol}\n({weight:.1%})')
                sizes.append(weight)
            
            # Create pie chart
            wedges, texts, autotexts = axes[i].pie(sizes, labels=labels, autopct='%1.1f%%',
                                                  colors=colors, startangle=90)
            
            axes[i].set_title(f'{strategy_name.replace("_", " ").title()}\n'
                            f'Return: {strategy_data.get("expected_return", 0):.3f}',
                            fontsize=12, fontweight='bold')
            
            # Enhance text visibility
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(8)
        
        # Hide empty subplots
        for j in range(num_strategies, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Portfolio allocation plots saved to {output_path}")
        
        plt.close()
        return str(output_path)
    
    def create_performance_timeline(self, strategies: Dict, days: int = 30, filename: str = None) -> str:
        """Create simulated performance timeline."""
        logger.info("Creating performance timeline simulation...")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_timeline_{timestamp}.png"
        
        # Filter valid strategies
        valid_strategies = {k: v for k, v in strategies.items() if 'error' not in v}
        
        if not valid_strategies:
            logger.warning("No valid strategies for timeline")
            return None
        
        # Generate timeline data (simulated)
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                             end=datetime.now(), freq='D')
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        fig.suptitle('Strategy Performance Timeline (Simulated)', fontsize=16, fontweight='bold')
        
        # Generate simulated daily returns for each strategy
        np.random.seed(42)  # For reproducible results
        
        cumulative_returns = {}
        daily_pnl = {}
        
        for strategy_name, strategy_data in valid_strategies.items():
            expected_return = strategy_data.get('expected_return', 0)
            
            # Generate realistic daily returns around expected performance
            daily_volatility = abs(expected_return) * 0.1  # 10% of expected return as daily vol
            daily_returns = np.random.normal(expected_return / days, daily_volatility, len(dates))
            
            # Apply some mean reversion and trend
            for i in range(1, len(daily_returns)):
                daily_returns[i] += 0.1 * (expected_return / days - daily_returns[i-1])
            
            cumulative_returns[strategy_name] = np.cumsum(daily_returns)
            daily_pnl[strategy_name] = daily_returns * 100000  # Assuming $100k initial capital
        
        # Plot 1: Cumulative Returns
        for strategy_name, cum_returns in cumulative_returns.items():
            ax1.plot(dates, cum_returns * 100, label=strategy_name.replace('_', ' ').title(), 
                    linewidth=2, alpha=0.8)
        
        ax1.set_title('Cumulative Returns Over Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Add horizontal line at 0
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 2: Daily P&L
        for strategy_name, pnl in daily_pnl.items():
            ax2.bar(dates, pnl, alpha=0.6, label=strategy_name.replace('_', ' ').title(), width=0.8)
        
        ax2.set_title('Daily P&L', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Daily P&L ($)')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax2.xaxis.set_major_locator(mdates.WeekdayLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Add horizontal line at 0
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Performance timeline saved to {output_path}")
        
        plt.close()
        return str(output_path)
    
    def log_comprehensive_analysis(self, results: Dict, step: int = 0):
        """Log comprehensive analysis to TensorBoard."""
        logger.info("Logging comprehensive analysis to TensorBoard...")
        
        # Log forecast analysis
        if 'forecasts' in results:
            self.log_forecasts_to_tensorboard(results['forecasts'], step)
        
        # Log strategy analysis
        if 'strategies' in results:
            self.log_strategies_to_tensorboard(results['strategies'], step)
        
        # Log additional metrics
        if 'simulation_params' in results:
            params = results['simulation_params']
            self.tb_writer.add_scalar('simulation/initial_capital', params.get('initial_capital', 0), step)
            self.tb_writer.add_scalar('simulation/forecast_days', params.get('forecast_days', 0), step)
            self.tb_writer.add_scalar('simulation/symbols_count', len(params.get('symbols_available', [])), step)
        
        # Create strategy comparison table for TensorBoard
        if 'strategies' in results:
            strategy_table = []
            headers = ['Strategy', 'Expected Return', 'Simulated Return', 'Profit/Loss', 'Positions']
            
            for strategy_name, strategy_data in results['strategies'].items():
                if 'error' not in strategy_data:
                    row = [
                        strategy_name,
                        f"{strategy_data.get('expected_return', 0):.4f}",
                        f"{strategy_data.get('performance', {}).get('simulated_actual_return', 0):.4f}",
                        f"${strategy_data.get('performance', {}).get('profit_loss', 0):,.0f}",
                        str(strategy_data.get('num_positions', 'N/A'))
                    ]
                    strategy_table.append(row)
            
            # Log as text
            table_text = "Strategy Comparison:\n"
            table_text += " | ".join(headers) + "\n"
            table_text += "-" * 80 + "\n"
            for row in strategy_table:
                table_text += " | ".join(row) + "\n"
            
            self.tb_writer.add_text('analysis/strategy_comparison', table_text, step)
        
        self.tb_writer.flush()
    
    def create_all_visualizations(self, results: Dict) -> List[str]:
        """Create all visualization plots and return list of file paths."""
        logger.info("Creating all visualizations...")
        
        created_files = []
        
        try:
            # Create forecast visualization
            if 'forecasts' in results:
                forecast_plot = self.create_forecast_visualization(results['forecasts'])
                if forecast_plot:
                    created_files.append(forecast_plot)
            
            # Create strategy comparison
            if 'strategies' in results:
                strategy_plot = self.create_strategy_comparison(results['strategies'])
                if strategy_plot:
                    created_files.append(strategy_plot)
            
            # Create portfolio allocation plots
            if 'strategies' in results:
                allocation_plot = self.create_portfolio_allocation_plots(results['strategies'])
                if allocation_plot:
                    created_files.append(allocation_plot)
            
            # Create performance timeline
            if 'strategies' in results:
                timeline_plot = self.create_performance_timeline(results['strategies'])
                if timeline_plot:
                    created_files.append(timeline_plot)
            
            # Log to TensorBoard
            self.log_comprehensive_analysis(results)
            
            logger.info(f"Created {len(created_files)} visualization files")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
        
        return created_files
    
    def close(self):
        """Close TensorBoard writer."""
        if hasattr(self, 'tb_writer'):
            self.tb_writer.close()
            logger.info("TensorBoard writer closed")


if __name__ == "__main__":
    # Example usage
    print("Visualization Logger module loaded successfully!")