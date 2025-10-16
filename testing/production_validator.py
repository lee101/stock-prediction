#!/usr/bin/env python3
"""
Production Model Validation Framework
Comprehensive testing for production-ready models
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# Import production systems
import sys
sys.path.append('hfinference')
from production_engine import ProductionTradingEngine, PredictionResult


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    start_date: str = '2023-01-01'
    end_date: str = '2024-01-01'
    initial_capital: float = 100000
    transaction_cost: float = 0.001  # 0.1%
    symbols: List[str] = None
    rebalance_frequency: str = 'weekly'  # 'daily', 'weekly', 'monthly'
    max_position_size: float = 0.2  # 20% max per stock
    stop_loss: float = 0.05  # 5% stop loss
    take_profit: float = 0.15  # 15% take profit


@dataclass
class PerformanceMetrics:
    """Performance metrics for backtesting"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    total_trades: int
    profit_factor: float
    calmar_ratio: float
    
    def to_dict(self) -> Dict:
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'total_trades': self.total_trades,
            'profit_factor': self.profit_factor,
            'calmar_ratio': self.calmar_ratio
        }


class ProductionValidator:
    """Comprehensive validation for production models"""
    
    def __init__(self, engine: ProductionTradingEngine):
        self.engine = engine
        self.setup_logging()
        
        # Create output directories
        self.output_dir = Path('testing/results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_logging(self):
        """Setup validation logging"""
        log_dir = Path('testing/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_historical_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Download historical data for backtesting"""
        self.logger.info(f"Downloading historical data for {len(symbols)} symbols")
        
        data = {}
        
        def download_symbol(symbol):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                
                if len(df) < 100:
                    self.logger.warning(f"Insufficient data for {symbol}")
                    return symbol, None
                
                df.columns = df.columns.str.lower()
                df = df.reset_index()
                return symbol, df
                
            except Exception as e:
                self.logger.error(f"Failed to download {symbol}: {e}")
                return symbol, None
        
        # Download in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_symbol = {
                executor.submit(download_symbol, symbol): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol, df = future.result()
                if df is not None:
                    data[symbol] = df
        
        self.logger.info(f"Downloaded data for {len(data)} symbols")
        return data
    
    def simulate_historical_predictions(self, symbol: str, df: pd.DataFrame, 
                                      lookback_days: int = 100) -> List[Dict]:
        """Simulate predictions on historical data"""
        
        predictions = []
        sequence_length = self.engine.config.sequence_length
        
        # Start from where we have enough data
        start_idx = max(lookback_days, sequence_length + 10)
        
        for i in range(start_idx, len(df) - 5, 5):  # Every 5 days
            try:
                # Get data up to current point
                historical_data = df.iloc[:i+1].copy()
                
                # Prepare sequence
                sequence = self.engine.prepare_sequence(historical_data)
                
                # Generate prediction
                with torch.no_grad():
                    base_outputs = self.engine.base_model(sequence)
                    
                    specialist_outputs = None
                    if symbol in self.engine.specialists:
                        specialist_outputs = self.engine.specialists[symbol](sequence)
                    
                    # Get ensemble weights
                    base_weight, specialist_weight = self.engine.calculate_ensemble_weights(symbol)
                    
                    # Process prediction for 1-day horizon
                    if specialist_outputs and 'horizon_1' in base_outputs:
                        base_pred = base_outputs['horizon_1']['action_probs']
                        specialist_pred = specialist_outputs['horizon_1']['action_probs']
                        ensemble_probs = base_weight * base_pred + specialist_weight * specialist_pred
                    elif 'horizon_1' in base_outputs:
                        ensemble_probs = base_outputs['horizon_1']['action_probs']
                    else:
                        ensemble_probs = base_outputs.get('action_probs', torch.tensor([[0.33, 0.34, 0.33]]))
                    
                    action_idx = torch.argmax(ensemble_probs).item()
                    confidence = torch.max(ensemble_probs).item()
                    
                    # Get actual future prices (if available)
                    current_price = df.iloc[i]['close']
                    future_prices = []
                    
                    for j in range(1, 6):  # Next 5 days
                        if i + j < len(df):
                            future_prices.append(df.iloc[i + j]['close'])
                    
                    predictions.append({
                        'date': df.iloc[i]['date'],
                        'current_price': current_price,
                        'predicted_action': action_idx,
                        'confidence': confidence,
                        'future_prices': future_prices,
                        'base_weight': base_weight,
                        'specialist_weight': specialist_weight
                    })
                    
            except Exception as e:
                self.logger.error(f"Prediction error at index {i}: {e}")
                continue
        
        return predictions
    
    def calculate_prediction_accuracy(self, predictions: List[Dict]) -> Dict[str, float]:
        """Calculate prediction accuracy metrics"""
        
        correct_predictions = 0
        total_predictions = 0
        
        directional_correct = 0
        price_mae = []
        confidence_scores = []
        
        for pred in predictions:
            if len(pred['future_prices']) == 0:
                continue
            
            current_price = pred['current_price']
            next_price = pred['future_prices'][0]
            predicted_action = pred['predicted_action']
            
            # Actual price movement
            price_change = (next_price - current_price) / current_price
            
            # Determine actual action
            if price_change > 0.01:  # >1% up
                actual_action = 0  # Buy
            elif price_change < -0.01:  # >1% down
                actual_action = 2  # Sell
            else:
                actual_action = 1  # Hold
            
            # Check if prediction was correct
            if predicted_action == actual_action:
                correct_predictions += 1
            
            # Directional accuracy (up vs down)
            predicted_direction = 1 if predicted_action == 0 else -1 if predicted_action == 2 else 0
            actual_direction = 1 if price_change > 0 else -1 if price_change < 0 else 0
            
            if predicted_direction * actual_direction > 0 or (predicted_direction == 0 and abs(price_change) < 0.01):
                directional_correct += 1
            
            total_predictions += 1
            price_mae.append(abs(price_change))
            confidence_scores.append(pred['confidence'])
        
        return {
            'accuracy': correct_predictions / max(total_predictions, 1),
            'directional_accuracy': directional_correct / max(total_predictions, 1),
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'price_mae': np.mean(price_mae) if price_mae else 0,
            'total_predictions': total_predictions
        }
    
    def run_backtest(self, config: BacktestConfig) -> Tuple[PerformanceMetrics, pd.DataFrame]:
        """Run comprehensive backtest"""
        
        self.logger.info(f"Running backtest from {config.start_date} to {config.end_date}")
        
        # Get historical data
        if config.symbols is None:
            config.symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META']
        
        historical_data = self.get_historical_data(config.symbols, config.start_date, config.end_date)
        
        # Initialize portfolio
        portfolio_value = config.initial_capital
        cash = config.initial_capital
        positions = {}  # symbol -> {shares, entry_price, entry_date}
        
        # Track performance
        portfolio_history = []
        trade_log = []
        
        # Get trading dates
        sample_df = list(historical_data.values())[0]
        trading_dates = sample_df['date'].tolist()
        
        rebalance_interval = {'daily': 1, 'weekly': 5, 'monthly': 20}[config.rebalance_frequency]
        
        for i, date in enumerate(trading_dates[100::rebalance_interval]):  # Start after enough history
            current_date = pd.to_datetime(date)
            
            try:
                # Get predictions for each symbol
                symbol_predictions = {}
                
                for symbol in config.symbols:
                    if symbol not in historical_data:
                        continue
                    
                    df = historical_data[symbol]
                    date_idx = df[df['date'] <= date].index.max()
                    
                    if date_idx < 100:  # Need enough history
                        continue
                    
                    # Get historical data up to current date
                    hist_data = df.iloc[:date_idx + 1]
                    
                    try:
                        # Simulate prediction
                        sequence = self.engine.prepare_sequence(hist_data)
                        
                        with torch.no_grad():
                            base_outputs = self.engine.base_model(sequence)
                            
                            specialist_outputs = None
                            if symbol in self.engine.specialists:
                                specialist_outputs = self.engine.specialists[symbol](sequence)
                            
                            # Get ensemble prediction
                            base_weight, specialist_weight = self.engine.calculate_ensemble_weights(symbol)
                            
                            if specialist_outputs and 'horizon_1' in base_outputs:
                                base_pred = base_outputs['horizon_1']['action_probs']
                                specialist_pred = specialist_outputs['horizon_1']['action_probs']
                                ensemble_probs = base_weight * base_pred + specialist_weight * specialist_pred
                            else:
                                ensemble_probs = base_outputs.get('action_probs', torch.tensor([[0.33, 0.34, 0.33]]))
                            
                            action_idx = torch.argmax(ensemble_probs).item()
                            confidence = torch.max(ensemble_probs).item()
                            
                            symbol_predictions[symbol] = {
                                'action': action_idx,
                                'confidence': confidence,
                                'current_price': hist_data['close'].iloc[-1]
                            }
                            
                    except Exception as e:
                        self.logger.error(f"Prediction error for {symbol} on {date}: {e}")
                        continue
                
                # Execute trades based on predictions
                current_portfolio_value = cash
                
                # Calculate current position values
                for symbol, position in positions.items():
                    if symbol in historical_data:
                        df = historical_data[symbol]
                        date_idx = df[df['date'] <= date].index.max()
                        if date_idx >= 0:
                            current_price = df.iloc[date_idx]['close']
                            position_value = position['shares'] * current_price
                            current_portfolio_value += position_value
                
                # Trading logic
                for symbol, pred in symbol_predictions.items():
                    action = pred['action']
                    confidence = pred['confidence']
                    current_price = pred['current_price']
                    
                    # Only trade with sufficient confidence
                    if confidence < 0.4:
                        continue
                    
                    # Buy signal
                    if action == 0 and symbol not in positions:
                        max_position_value = current_portfolio_value * config.max_position_size
                        shares_to_buy = int(max_position_value / current_price)
                        cost = shares_to_buy * current_price * (1 + config.transaction_cost)
                        
                        if cost <= cash and shares_to_buy > 0:
                            cash -= cost
                            positions[symbol] = {
                                'shares': shares_to_buy,
                                'entry_price': current_price,
                                'entry_date': current_date
                            }
                            
                            trade_log.append({
                                'date': current_date,
                                'symbol': symbol,
                                'action': 'BUY',
                                'shares': shares_to_buy,
                                'price': current_price,
                                'confidence': confidence
                            })
                    
                    # Sell signal or stop loss/take profit
                    elif symbol in positions:
                        position = positions[symbol]
                        entry_price = position['entry_price']
                        shares = position['shares']
                        
                        # Calculate return
                        price_return = (current_price - entry_price) / entry_price
                        
                        should_sell = (
                            action == 2 or  # Sell signal
                            price_return <= -config.stop_loss or  # Stop loss
                            price_return >= config.take_profit  # Take profit
                        )
                        
                        if should_sell:
                            sell_value = shares * current_price * (1 - config.transaction_cost)
                            cash += sell_value
                            
                            trade_log.append({
                                'date': current_date,
                                'symbol': symbol,
                                'action': 'SELL',
                                'shares': shares,
                                'price': current_price,
                                'confidence': confidence,
                                'return': price_return
                            })
                            
                            del positions[symbol]
                
                # Record portfolio value
                total_value = cash
                for symbol, position in positions.items():
                    if symbol in historical_data:
                        df = historical_data[symbol]
                        date_idx = df[df['date'] <= date].index.max()
                        if date_idx >= 0:
                            current_price = df.iloc[date_idx]['close']
                            total_value += position['shares'] * current_price
                
                portfolio_history.append({
                    'date': current_date,
                    'portfolio_value': total_value,
                    'cash': cash,
                    'positions_value': total_value - cash
                })
                
            except Exception as e:
                self.logger.error(f"Backtest error on {date}: {e}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(portfolio_history)
        
        # Calculate performance metrics
        if len(results_df) > 1:
            returns = results_df['portfolio_value'].pct_change().dropna()
            
            total_return = (results_df['portfolio_value'].iloc[-1] / config.initial_capital) - 1
            
            # Calculate other metrics
            trading_days = len(returns)
            annualized_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
            volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
            sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0  # Assume 2% risk-free rate
            
            # Max drawdown
            peak = results_df['portfolio_value'].expanding(min_periods=1).max()
            drawdown = (results_df['portfolio_value'] - peak) / peak
            max_drawdown = abs(drawdown.min())
            
            # Trading metrics
            trades_df = pd.DataFrame(trade_log)
            win_trades = trades_df[trades_df['return'] > 0] if 'return' in trades_df.columns else pd.DataFrame()
            loss_trades = trades_df[trades_df['return'] <= 0] if 'return' in trades_df.columns else pd.DataFrame()
            
            win_rate = len(win_trades) / max(len(trades_df[trades_df['action'] == 'SELL']), 1)
            avg_win = win_trades['return'].mean() if len(win_trades) > 0 else 0
            avg_loss = abs(loss_trades['return'].mean()) if len(loss_trades) > 0 else 0
            
            profit_factor = (avg_win * len(win_trades)) / max(avg_loss * len(loss_trades), 1e-6) if avg_loss > 0 else float('inf')
            calmar_ratio = annualized_return / max(max_drawdown, 1e-6)
            
            metrics = PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                total_trades=len(trades_df),
                profit_factor=profit_factor,
                calmar_ratio=calmar_ratio
            )
        else:
            # Default metrics if no data
            metrics = PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        return metrics, results_df
    
    def validate_model_accuracy(self, symbols: List[str], test_period_months: int = 6) -> Dict[str, Dict]:
        """Validate model accuracy on historical data"""
        
        self.logger.info(f"Validating model accuracy for {len(symbols)} symbols")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=test_period_months * 30 + 200)  # Extra for model history
        
        historical_data = self.get_historical_data(
            symbols, 
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        accuracy_results = {}
        
        for symbol, df in historical_data.items():
            self.logger.info(f"Validating {symbol}")
            
            # Generate historical predictions
            predictions = self.simulate_historical_predictions(symbol, df)
            
            if not predictions:
                self.logger.warning(f"No predictions generated for {symbol}")
                continue
            
            # Calculate accuracy metrics
            accuracy_metrics = self.calculate_prediction_accuracy(predictions)
            
            accuracy_results[symbol] = accuracy_metrics
            
            self.logger.info(f"{symbol}: Accuracy={accuracy_metrics['accuracy']:.3f}, "
                           f"Directional={accuracy_metrics['directional_accuracy']:.3f}")
        
        return accuracy_results
    
    def generate_report(self, backtest_metrics: PerformanceMetrics, 
                       accuracy_results: Dict[str, Dict], 
                       results_df: pd.DataFrame) -> str:
        """Generate comprehensive validation report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f'validation_report_{timestamp}.json'
        
        report = {
            'timestamp': timestamp,
            'backtest_performance': backtest_metrics.to_dict(),
            'model_accuracy': accuracy_results,
            'summary': {
                'avg_accuracy': np.mean([r['accuracy'] for r in accuracy_results.values()]) if accuracy_results else 0,
                'avg_directional_accuracy': np.mean([r['directional_accuracy'] for r in accuracy_results.values()]) if accuracy_results else 0,
                'total_symbols_tested': len(accuracy_results),
                'backtest_sharpe_ratio': backtest_metrics.sharpe_ratio,
                'backtest_max_drawdown': backtest_metrics.max_drawdown,
                'backtest_win_rate': backtest_metrics.win_rate
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Validation report saved to {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("PRODUCTION MODEL VALIDATION REPORT")
        print("="*60)
        print(f"Total Return: {backtest_metrics.total_return:.2%}")
        print(f"Annualized Return: {backtest_metrics.annualized_return:.2%}")
        print(f"Sharpe Ratio: {backtest_metrics.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {backtest_metrics.max_drawdown:.2%}")
        print(f"Win Rate: {backtest_metrics.win_rate:.2%}")
        print(f"Total Trades: {backtest_metrics.total_trades}")
        print()
        print(f"Average Accuracy: {report['summary']['avg_accuracy']:.2%}")
        print(f"Average Directional Accuracy: {report['summary']['avg_directional_accuracy']:.2%}")
        print(f"Symbols Tested: {report['summary']['total_symbols_tested']}")
        print("="*60)
        
        return str(report_path)
    
    def run_full_validation(self, test_symbols: List[str] = None) -> str:
        """Run complete validation suite"""
        
        if test_symbols is None:
            test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'JPM', 'BAC']
        
        self.logger.info("Starting full production validation")
        
        # 1. Model accuracy validation
        accuracy_results = self.validate_model_accuracy(test_symbols, test_period_months=6)
        
        # 2. Backtest validation
        backtest_config = BacktestConfig(
            start_date='2023-06-01',
            end_date='2024-01-01',
            symbols=test_symbols,
            initial_capital=100000
        )
        
        backtest_metrics, results_df = self.run_backtest(backtest_config)
        
        # 3. Generate comprehensive report
        report_path = self.generate_report(backtest_metrics, accuracy_results, results_df)
        
        return report_path


def main():
    """Run production validation"""
    print("Production Model Validation")
    print("="*50)
    
    try:
        # Load production engine
        engine = ProductionTradingEngine()
        
        # Create validator
        validator = ProductionValidator(engine)
        
        # Run validation
        report_path = validator.run_full_validation()
        
        print(f"\nValidation complete! Report: {report_path}")
        
    except FileNotFoundError as e:
        print(f"Models not found: {e}")
        print("Please run train_production_v2.py first to train production models")
    except Exception as e:
        print(f"Validation failed: {e}")


if __name__ == "__main__":
    main()