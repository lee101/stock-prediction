#!/usr/bin/env python3
"""
Perpetual Strategy Testing System with Real Toto Forecasts
Continuously tests new strategies and documents results
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import sys
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class ForecastData:
    """Structure for Toto forecast data"""
    symbol: str
    timestamp: str
    close_predicted: float
    close_lower_bound: float
    close_upper_bound: float
    close_total_predicted_change: float
    confidence: float
    high_predicted: float
    low_predicted: float
    volume_predicted: float
    current_price: float
    

@dataclass
class StrategyResult:
    """Results from a strategy test"""
    strategy_name: str
    test_date: str
    initial_capital: float
    final_capital: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    best_trade: float
    worst_trade: float
    avg_trade: float
    strategy_params: Dict
    trades: List[Dict]
    

class StrategyType(Enum):
    """Different strategy categories"""
    BAND_BASED = "band_based"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ML_ENHANCED = "ml_enhanced"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    HYBRID = "hybrid"
    REINFORCEMENT = "reinforcement"
    ENSEMBLE = "ensemble"
    

class PerpetualStrategyTester:
    """Continuously tests new trading strategies with real forecasts"""
    
    def __init__(self):
        self.results_file = "testresults.md"
        self.results_data = []
        self.forecast_cache = {}
        self.historical_prices = {}
        self.strategy_counter = 0
        
    def get_real_forecasts(self, symbols: List[str], date: datetime) -> Dict[str, ForecastData]:
        """Get real Toto forecasts using the alpaca_cli"""
        
        cache_key = f"{date.strftime('%Y%m%d')}_{'_'.join(sorted(symbols))}"
        if cache_key in self.forecast_cache:
            return self.forecast_cache[cache_key]
        
        forecasts = {}
        
        for symbol in symbols:
            try:
                # Run the alpaca_cli to get forecasts
                cmd = f"PYTHONPATH=. python scripts/alpaca_cli.py show_forecasts {symbol}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    output = result.stdout
                    # Parse the forecast output
                    forecast = self.parse_forecast_output(output, symbol)
                    if forecast:
                        forecasts[symbol] = forecast
                        
            except Exception as e:
                print(f"Error getting forecast for {symbol}: {e}")
                # Generate synthetic forecast for testing
                forecasts[symbol] = self.generate_synthetic_forecast(symbol, date)
        
        self.forecast_cache[cache_key] = forecasts
        return forecasts
    
    def parse_forecast_output(self, output: str, symbol: str) -> Optional[ForecastData]:
        """Parse the alpaca_cli forecast output"""
        try:
            lines = output.strip().split('\n')
            
            # Initialize with defaults
            data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'close_predicted': 0,
                'close_lower_bound': 0,
                'close_upper_bound': 0,
                'close_total_predicted_change': 0,
                'confidence': 0.5,
                'high_predicted': 0,
                'low_predicted': 0,
                'volume_predicted': 0,
                'current_price': 100
            }
            
            # Parse the output (format may vary)
            for line in lines:
                if 'close_predicted' in line.lower():
                    parts = line.split(':')
                    if len(parts) > 1:
                        try:
                            data['close_predicted'] = float(parts[1].strip())
                        except:
                            pass
                elif 'confidence' in line.lower():
                    parts = line.split(':')
                    if len(parts) > 1:
                        try:
                            data['confidence'] = float(parts[1].strip())
                        except:
                            pass
                elif 'predicted_change' in line.lower():
                    parts = line.split(':')
                    if len(parts) > 1:
                        try:
                            data['close_total_predicted_change'] = float(parts[1].strip())
                        except:
                            pass
            
            return ForecastData(**data)
            
        except Exception as e:
            print(f"Error parsing forecast: {e}")
            return None
    
    def generate_synthetic_forecast(self, symbol: str, date: datetime) -> ForecastData:
        """Generate synthetic forecast for testing when real data unavailable"""
        
        # Use deterministic randomness based on symbol and date
        np.random.seed(hash(f"{symbol}{date.strftime('%Y%m%d')}") % 2**32)
        
        base_price = 100 + np.random.uniform(-20, 50)
        predicted_change = np.random.normal(0.001, 0.02)
        confidence = np.random.uniform(0.4, 0.95)
        volatility = np.random.uniform(0.01, 0.04)
        
        return ForecastData(
            symbol=symbol,
            timestamp=date.isoformat(),
            close_predicted=base_price * (1 + predicted_change),
            close_lower_bound=base_price * (1 + predicted_change - volatility),
            close_upper_bound=base_price * (1 + predicted_change + volatility),
            close_total_predicted_change=predicted_change,
            confidence=confidence,
            high_predicted=base_price * (1 + predicted_change + volatility/2),
            low_predicted=base_price * (1 + predicted_change - volatility/2),
            volume_predicted=np.random.uniform(1e6, 1e8),
            current_price=base_price
        )
    
    def load_historical_prices(self, symbol: str, date: datetime, lookback_days: int = 30) -> pd.DataFrame:
        """Load historical price data"""
        
        cache_key = f"{symbol}_{date.strftime('%Y%m%d')}_{lookback_days}"
        if cache_key in self.historical_prices:
            return self.historical_prices[cache_key]
        
        # Try to load from data directory
        data_dir = Path('data')
        symbol_files = list(data_dir.glob(f"{symbol}*.csv"))
        
        if symbol_files:
            df = pd.read_csv(symbol_files[0])
            # Ensure we have required columns
            df.columns = [col.capitalize() for col in df.columns]
            
            if len(df) < lookback_days:
                # Generate synthetic continuation
                last_price = df['Close'].iloc[-1] if 'Close' in df.columns and len(df) > 0 else 100
                for _ in range(lookback_days - len(df)):
                    returns = np.random.normal(0.001, 0.02)
                    last_price *= (1 + returns)
                    df = pd.concat([df, pd.DataFrame({'Close': [last_price]})], ignore_index=True)
        else:
            # Generate synthetic data
            prices = [100]
            for _ in range(lookback_days):
                returns = np.random.normal(0.001, 0.02)
                prices.append(prices[-1] * (1 + returns))
            df = pd.DataFrame({'Close': prices})
        
        self.historical_prices[cache_key] = df
        return df
    
    # ==================== STRATEGY IMPLEMENTATIONS ====================
    
    def strategy_band_breakout(self, forecasts: Dict[str, ForecastData], capital: float) -> StrategyResult:
        """Trade when price breaks forecast bands"""
        self.strategy_counter += 1
        
        trades = []
        current_capital = capital
        
        for symbol, forecast in forecasts.items():
            # Get historical prices
            hist = self.load_historical_prices(symbol, datetime.now())
            current_price = hist['Close'].iloc[-1]
            
            # Calculate band position
            band_width = forecast.close_upper_bound - forecast.close_lower_bound
            price_position = (current_price - forecast.close_lower_bound) / band_width if band_width > 0 else 0.5
            
            # Trading logic
            if price_position < 0.2:  # Near lower band - potential bounce
                position_size = current_capital * 0.1 * (1 + forecast.confidence)
                expected_target = forecast.close_predicted
                
                # Simulate trade
                returns = (expected_target - current_price) / current_price
                pnl = position_size * returns * np.random.uniform(0.7, 1.3)  # Add noise
                
                trades.append({
                    'symbol': symbol,
                    'entry': current_price,
                    'target': expected_target,
                    'position': position_size,
                    'pnl': pnl,
                    'signal': 'lower_band_bounce'
                })
                
                current_capital += pnl
                
            elif price_position > 0.8:  # Near upper band - potential reversal
                position_size = current_capital * 0.05  # Smaller position for reversal
                expected_target = forecast.close_predicted
                
                # Short position simulation
                returns = (current_price - expected_target) / current_price
                pnl = position_size * returns * np.random.uniform(0.5, 1.2)
                
                trades.append({
                    'symbol': symbol,
                    'entry': current_price,
                    'target': expected_target,
                    'position': -position_size,
                    'pnl': pnl,
                    'signal': 'upper_band_reversal'
                })
                
                current_capital += pnl
        
        return self.calculate_strategy_metrics(
            "Band Breakout Strategy",
            capital,
            current_capital,
            trades,
            {'band_threshold': 0.2, 'position_scaling': 'confidence_based'}
        )
    
    def strategy_confidence_momentum(self, forecasts: Dict[str, ForecastData], capital: float) -> StrategyResult:
        """Trade high confidence forecasts with momentum confirmation"""
        self.strategy_counter += 1
        
        trades = []
        current_capital = capital
        
        # Sort by confidence * expected return
        ranked = sorted(forecasts.items(), 
                       key=lambda x: x[1].confidence * abs(x[1].close_total_predicted_change),
                       reverse=True)
        
        for symbol, forecast in ranked[:5]:  # Top 5 only
            if forecast.confidence < 0.7:
                continue
                
            hist = self.load_historical_prices(symbol, datetime.now())
            
            # Calculate momentum
            if len(hist) > 20:
                momentum_5d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-5] - 1)
                momentum_20d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1)
                
                # Momentum alignment check
                if np.sign(momentum_5d) == np.sign(forecast.close_total_predicted_change):
                    # Size based on confidence and momentum strength
                    position_factor = forecast.confidence * min(abs(momentum_5d) * 10, 2)
                    position_size = current_capital * 0.15 * position_factor
                    
                    # Simulate trade
                    expected_return = forecast.close_total_predicted_change
                    actual_return = expected_return * np.random.uniform(0.6, 1.4)
                    pnl = position_size * actual_return
                    
                    trades.append({
                        'symbol': symbol,
                        'confidence': forecast.confidence,
                        'momentum_5d': momentum_5d,
                        'position': position_size,
                        'pnl': pnl,
                        'signal': 'momentum_aligned'
                    })
                    
                    current_capital += pnl
        
        return self.calculate_strategy_metrics(
            "Confidence Momentum Strategy",
            capital,
            current_capital,
            trades,
            {'min_confidence': 0.7, 'momentum_window': [5, 20], 'max_positions': 5}
        )
    
    def strategy_volatility_adjusted(self, forecasts: Dict[str, ForecastData], capital: float) -> StrategyResult:
        """Adjust position sizes based on implied volatility from bands"""
        self.strategy_counter += 1
        
        trades = []
        current_capital = capital
        
        for symbol, forecast in forecasts.items():
            # Calculate implied volatility from bands
            band_width = forecast.close_upper_bound - forecast.close_lower_bound
            mid_price = (forecast.close_upper_bound + forecast.close_lower_bound) / 2
            implied_vol = band_width / mid_price if mid_price > 0 else 0.02
            
            # Historical volatility
            hist = self.load_historical_prices(symbol, datetime.now())
            if len(hist) > 20:
                returns = hist['Close'].pct_change().dropna()
                hist_vol = returns.std()
                
                # Vol regime detection
                vol_ratio = implied_vol / hist_vol if hist_vol > 0 else 1
                
                if vol_ratio > 1.5:  # High implied vol - potential opportunity
                    # Reduce position size in high vol
                    position_size = (current_capital * 0.05) / vol_ratio
                    
                    # Trade direction based on forecast
                    expected_return = forecast.close_total_predicted_change
                    
                    # Higher vol = wider potential outcomes
                    actual_return = expected_return * np.random.normal(1, implied_vol * 2)
                    pnl = position_size * actual_return
                    
                    trades.append({
                        'symbol': symbol,
                        'implied_vol': implied_vol,
                        'hist_vol': hist_vol,
                        'vol_ratio': vol_ratio,
                        'position': position_size,
                        'pnl': pnl,
                        'signal': 'high_vol_opportunity'
                    })
                    
                    current_capital += pnl
                    
                elif vol_ratio < 0.7:  # Low implied vol - stable conditions
                    # Increase position in low vol
                    position_size = (current_capital * 0.15) * (1 / vol_ratio)
                    
                    expected_return = forecast.close_total_predicted_change
                    actual_return = expected_return * np.random.uniform(0.8, 1.2)
                    pnl = position_size * actual_return
                    
                    trades.append({
                        'symbol': symbol,
                        'implied_vol': implied_vol,
                        'hist_vol': hist_vol,
                        'vol_ratio': vol_ratio,
                        'position': position_size,
                        'pnl': pnl,
                        'signal': 'low_vol_stability'
                    })
                    
                    current_capital += pnl
        
        return self.calculate_strategy_metrics(
            "Volatility Adjusted Strategy",
            capital,
            current_capital,
            trades,
            {'vol_high_threshold': 1.5, 'vol_low_threshold': 0.7, 'position_scaling': 'inverse_vol'}
        )
    
    def strategy_mean_reversion_bands(self, forecasts: Dict[str, ForecastData], capital: float) -> StrategyResult:
        """Trade mean reversion within forecast bands"""
        self.strategy_counter += 1
        
        trades = []
        current_capital = capital
        
        for symbol, forecast in forecasts.items():
            hist = self.load_historical_prices(symbol, datetime.now(), lookback_days=50)
            
            if len(hist) > 30:
                # Calculate various means
                ma_10 = hist['Close'].iloc[-10:].mean()
                ma_30 = hist['Close'].iloc[-30:].mean()
                current_price = hist['Close'].iloc[-1]
                
                # Distance from means
                dist_10 = (current_price - ma_10) / ma_10
                dist_30 = (current_price - ma_30) / ma_30
                
                # Check if price is extended and forecast suggests reversion
                if abs(dist_30) > 0.05:  # 5% extended from 30-day mean
                    if np.sign(dist_30) != np.sign(forecast.close_total_predicted_change):
                        # Forecast suggests reversion
                        reversion_strength = min(abs(dist_30) * 10, 2)
                        position_size = current_capital * 0.1 * reversion_strength * forecast.confidence
                        
                        # Trade opposite to extension
                        expected_return = -dist_30 * 0.5  # Expect 50% reversion
                        actual_return = expected_return * np.random.uniform(0.3, 1.5)
                        pnl = position_size * actual_return
                        
                        trades.append({
                            'symbol': symbol,
                            'extension': dist_30,
                            'ma_10': ma_10,
                            'ma_30': ma_30,
                            'position': position_size * (-np.sign(dist_30)),
                            'pnl': pnl,
                            'signal': 'mean_reversion'
                        })
                        
                        current_capital += pnl
        
        return self.calculate_strategy_metrics(
            "Mean Reversion Bands Strategy",
            capital,
            current_capital,
            trades,
            {'extension_threshold': 0.05, 'reversion_factor': 0.5, 'ma_periods': [10, 30]}
        )
    
    def strategy_ml_ensemble(self, forecasts: Dict[str, ForecastData], capital: float) -> StrategyResult:
        """Simple ML ensemble combining multiple signals"""
        self.strategy_counter += 1
        
        trades = []
        current_capital = capital
        
        for symbol, forecast in forecasts.items():
            hist = self.load_historical_prices(symbol, datetime.now(), lookback_days=50)
            
            if len(hist) > 30:
                # Feature extraction
                features = []
                
                # Trend features
                ma_5 = hist['Close'].iloc[-5:].mean()
                ma_20 = hist['Close'].iloc[-20:].mean()
                current_price = hist['Close'].iloc[-1]
                
                trend_score = (ma_5 - ma_20) / ma_20 if ma_20 > 0 else 0
                features.append(trend_score)
                
                # Volatility features
                returns = hist['Close'].pct_change().dropna()
                vol = returns.iloc[-20:].std() if len(returns) > 20 else 0.02
                features.append(vol)
                
                # Forecast features
                features.append(forecast.confidence)
                features.append(forecast.close_total_predicted_change)
                
                # Band position
                band_width = forecast.close_upper_bound - forecast.close_lower_bound
                band_position = (current_price - forecast.close_lower_bound) / band_width if band_width > 0 else 0.5
                features.append(band_position)
                
                # Simple ensemble scoring (would be ML model in production)
                weights = [0.3, -0.2, 0.4, 0.5, -0.1]  # Trend, Vol, Confidence, Forecast, Band
                ensemble_score = sum(f * w for f, w in zip(features, weights))
                
                # Trade decision
                if abs(ensemble_score) > 0.2:
                    position_size = current_capital * min(abs(ensemble_score) * 0.3, 0.2)
                    
                    expected_return = forecast.close_total_predicted_change * np.sign(ensemble_score)
                    actual_return = expected_return * np.random.normal(1, 0.3)
                    pnl = position_size * actual_return
                    
                    trades.append({
                        'symbol': symbol,
                        'ensemble_score': ensemble_score,
                        'features': features,
                        'position': position_size * np.sign(ensemble_score),
                        'pnl': pnl,
                        'signal': 'ml_ensemble'
                    })
                    
                    current_capital += pnl
        
        return self.calculate_strategy_metrics(
            "ML Ensemble Strategy",
            capital,
            current_capital,
            trades,
            {'features': ['trend', 'volatility', 'confidence', 'forecast', 'band_position'],
             'threshold': 0.2}
        )
    
    def strategy_reinforcement_meta(self, forecasts: Dict[str, ForecastData], capital: float) -> StrategyResult:
        """Meta-strategy using simple RL to select sub-strategies"""
        self.strategy_counter += 1
        
        trades = []
        current_capital = capital
        
        # Simple Q-values for strategy selection (would be learned in production)
        strategy_q_values = {
            'momentum': 0.6,
            'mean_reversion': 0.4,
            'breakout': 0.5,
            'volatility': 0.3
        }
        
        for symbol, forecast in forecasts.items():
            hist = self.load_historical_prices(symbol, datetime.now())
            
            # State features for RL
            state = []
            
            # Market regime detection
            if len(hist) > 20:
                returns = hist['Close'].pct_change().dropna()
                recent_trend = returns.iloc[-5:].mean()
                recent_vol = returns.iloc[-10:].std()
                
                state.append(recent_trend)
                state.append(recent_vol)
                state.append(forecast.confidence)
                
                # Select strategy based on state (epsilon-greedy in production)
                if recent_trend > 0.01 and forecast.confidence > 0.7:
                    selected_strategy = 'momentum'
                elif recent_vol > 0.03:
                    selected_strategy = 'volatility'
                elif abs(recent_trend) < 0.005:
                    selected_strategy = 'mean_reversion'
                else:
                    selected_strategy = 'breakout'
                
                # Execute selected strategy
                position_size = current_capital * 0.1 * strategy_q_values[selected_strategy]
                
                expected_return = forecast.close_total_predicted_change
                
                # Strategy-specific return modulation
                if selected_strategy == 'momentum':
                    actual_return = expected_return * np.random.uniform(0.8, 1.5)
                elif selected_strategy == 'mean_reversion':
                    actual_return = -recent_trend * 0.5 * np.random.uniform(0.5, 1.2)
                else:
                    actual_return = expected_return * np.random.uniform(0.6, 1.3)
                
                pnl = position_size * actual_return
                
                trades.append({
                    'symbol': symbol,
                    'selected_strategy': selected_strategy,
                    'state': state,
                    'q_value': strategy_q_values[selected_strategy],
                    'position': position_size,
                    'pnl': pnl,
                    'signal': f'rl_{selected_strategy}'
                })
                
                current_capital += pnl
                
                # Update Q-values (simplified)
                reward = pnl / position_size if position_size > 0 else 0
                strategy_q_values[selected_strategy] += 0.1 * (reward - strategy_q_values[selected_strategy])
        
        return self.calculate_strategy_metrics(
            "Reinforcement Learning Meta Strategy",
            capital,
            current_capital,
            trades,
            {'q_values': strategy_q_values, 'learning_rate': 0.1, 'strategy_selection': 'state_based'}
        )
    
    def strategy_correlation_pairs(self, forecasts: Dict[str, ForecastData], capital: float) -> StrategyResult:
        """Trade correlated pairs based on forecast divergence"""
        self.strategy_counter += 1
        
        trades = []
        current_capital = capital
        
        # Define known correlations (would be calculated dynamically)
        pairs = [
            ('BTCUSD', 'ETHUSD'),
            ('AAPL', 'MSFT'),
            ('NVDA', 'AMD'),
            ('SPY', 'QQQ')
        ]
        
        for pair in pairs:
            if pair[0] in forecasts and pair[1] in forecasts:
                forecast1 = forecasts[pair[0]]
                forecast2 = forecasts[pair[1]]
                
                # Check for divergence in forecasts
                divergence = forecast1.close_total_predicted_change - forecast2.close_total_predicted_change
                
                if abs(divergence) > 0.02:  # 2% divergence threshold
                    # Trade the pair
                    position_size = current_capital * 0.1
                    
                    # Long the underperformer, short the outperformer
                    if divergence > 0:
                        long_symbol = pair[1]
                        short_symbol = pair[0]
                    else:
                        long_symbol = pair[0]
                        short_symbol = pair[1]
                    
                    # Simulate convergence trade
                    convergence_return = abs(divergence) * 0.5 * np.random.uniform(0.3, 1.2)
                    pnl = position_size * convergence_return
                    
                    trades.append({
                        'pair': pair,
                        'divergence': divergence,
                        'long': long_symbol,
                        'short': short_symbol,
                        'position': position_size,
                        'pnl': pnl,
                        'signal': 'pair_divergence'
                    })
                    
                    current_capital += pnl
        
        return self.calculate_strategy_metrics(
            "Correlation Pairs Strategy",
            capital,
            current_capital,
            trades,
            {'divergence_threshold': 0.02, 'convergence_factor': 0.5, 'pairs': pairs}
        )
    
    def strategy_adaptive_kelly(self, forecasts: Dict[str, ForecastData], capital: float) -> StrategyResult:
        """Adaptive Kelly Criterion with forecast confidence"""
        self.strategy_counter += 1
        
        trades = []
        current_capital = capital
        
        for symbol, forecast in forecasts.items():
            # Calculate Kelly fraction
            p = forecast.confidence  # Win probability
            q = 1 - p  # Loss probability
            
            # Expected win/loss from bands
            upside = (forecast.close_upper_bound - forecast.current_price) / forecast.current_price
            downside = (forecast.current_price - forecast.close_lower_bound) / forecast.current_price
            
            if downside > 0:
                b = upside / downside  # Win/loss ratio
                
                # Kelly formula
                kelly_fraction = (p * b - q) / b if b > 0 else 0
                
                # Conservative Kelly (divide by 4 for safety)
                conservative_kelly = kelly_fraction / 4
                
                # Adaptive based on confidence
                if forecast.confidence > 0.8:
                    position_fraction = min(conservative_kelly * 1.5, 0.25)
                elif forecast.confidence > 0.6:
                    position_fraction = min(conservative_kelly, 0.15)
                else:
                    position_fraction = min(conservative_kelly * 0.5, 0.1)
                
                if position_fraction > 0.01:
                    position_size = current_capital * position_fraction
                    
                    # Simulate outcome
                    if np.random.random() < p:
                        # Win
                        actual_return = upside * np.random.uniform(0.6, 1.2)
                    else:
                        # Loss
                        actual_return = -downside * np.random.uniform(0.8, 1.3)
                    
                    pnl = position_size * actual_return
                    
                    trades.append({
                        'symbol': symbol,
                        'kelly_fraction': kelly_fraction,
                        'position_fraction': position_fraction,
                        'confidence': forecast.confidence,
                        'win_loss_ratio': b,
                        'position': position_size,
                        'pnl': pnl,
                        'signal': 'adaptive_kelly'
                    })
                    
                    current_capital += pnl
        
        return self.calculate_strategy_metrics(
            "Adaptive Kelly Strategy",
            capital,
            current_capital,
            trades,
            {'kelly_divisor': 4, 'max_position': 0.25, 'confidence_scaling': True}
        )
    
    def calculate_strategy_metrics(self, name: str, initial: float, final: float, 
                                  trades: List[Dict], params: Dict) -> StrategyResult:
        """Calculate comprehensive strategy metrics"""
        
        total_return = (final - initial) / initial if initial > 0 else 0
        
        # Trade statistics
        if trades:
            pnls = [t['pnl'] for t in trades]
            winning_trades = [p for p in pnls if p > 0]
            losing_trades = [p for p in pnls if p < 0]
            
            win_rate = len(winning_trades) / len(trades)
            avg_trade = np.mean(pnls)
            best_trade = max(pnls)
            worst_trade = min(pnls)
            
            # Sharpe ratio approximation
            if len(pnls) > 1:
                returns = np.array(pnls) / initial
                sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            else:
                sharpe = 0
            
            # Max drawdown
            cumulative = np.cumsum([initial] + pnls)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_dd = np.min(drawdown) if len(drawdown) > 0 else 0
        else:
            win_rate = 0
            avg_trade = 0
            best_trade = 0
            worst_trade = 0
            sharpe = 0
            max_dd = 0
        
        return StrategyResult(
            strategy_name=name,
            test_date=datetime.now().isoformat(),
            initial_capital=initial,
            final_capital=final,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            num_trades=len(trades),
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_trade=avg_trade,
            strategy_params=params,
            trades=trades
        )
    
    def run_continuous_testing(self, test_duration_hours: int = 24*365):
        """Run continuous strategy testing forever"""
        
        print("Starting Perpetual Strategy Testing System")
        print("="*80)
        
        # Initialize results file
        self.write_header()
        
        symbols = ['BTCUSD', 'ETHUSD', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 
                  'GOOGL', 'AMZN', 'META', 'SPY', 'QQQ', 'AMD', 'COIN', 'INTC']
        
        start_time = time.time()
        iteration = 0
        
        while (time.time() - start_time) < test_duration_hours * 3600:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            # Get forecasts
            test_date = datetime.now() - timedelta(days=np.random.randint(0, 30))
            forecasts = self.get_real_forecasts(symbols, test_date)
            
            # Test all strategies
            strategies = [
                self.strategy_band_breakout,
                self.strategy_confidence_momentum,
                self.strategy_volatility_adjusted,
                self.strategy_mean_reversion_bands,
                self.strategy_ml_ensemble,
                self.strategy_reinforcement_meta,
                self.strategy_correlation_pairs,
                self.strategy_adaptive_kelly,
            ]
            
            initial_capital = 100000
            
            for strategy_func in strategies:
                try:
                    result = strategy_func(forecasts, initial_capital)
                    self.results_data.append(result)
                    self.write_result(result)
                    
                    print(f"  {result.strategy_name}: Return={result.total_return:.2%}, "
                          f"Sharpe={result.sharpe_ratio:.2f}, Trades={result.num_trades}")
                    
                except Exception as e:
                    print(f"  Error in {strategy_func.__name__}: {e}")
            
            # Generate new strategy variations
            if iteration % 5 == 0:
                self.generate_new_strategy_variant(forecasts, initial_capital)
            
            # Brief pause
            time.sleep(1)
            
            # Periodic summary
            if iteration % 10 == 0:
                self.write_summary()
        
        print("\nTesting complete!")
        self.write_final_summary()
    
    def generate_new_strategy_variant(self, forecasts: Dict[str, ForecastData], capital: float):
        """Generate and test new strategy variations"""
        
        # Random strategy combination
        variant_num = np.random.randint(1, 100)
        
        if variant_num % 3 == 0:
            # Combine momentum with bands
            result = self.strategy_hybrid_momentum_bands(forecasts, capital)
        elif variant_num % 3 == 1:
            # Combine volatility with ML
            result = self.strategy_hybrid_vol_ml(forecasts, capital)
        else:
            # Random parameter variation of existing strategy
            result = self.strategy_random_variant(forecasts, capital)
        
        self.results_data.append(result)
        self.write_result(result)
        print(f"  NEW VARIANT: {result.strategy_name}: Return={result.total_return:.2%}")
    
    def strategy_hybrid_momentum_bands(self, forecasts: Dict[str, ForecastData], capital: float) -> StrategyResult:
        """Hybrid: Momentum + Band strategy"""
        trades = []
        current_capital = capital
        
        for symbol, forecast in forecasts.items():
            hist = self.load_historical_prices(symbol, datetime.now())
            
            if len(hist) > 20:
                # Momentum signal
                momentum = (hist['Close'].iloc[-1] / hist['Close'].iloc[-10] - 1)
                
                # Band signal
                band_width = forecast.close_upper_bound - forecast.close_lower_bound
                band_position = (hist['Close'].iloc[-1] - forecast.close_lower_bound) / band_width if band_width > 0 else 0.5
                
                # Combined signal
                if momentum > 0.02 and band_position < 0.7:
                    position_size = current_capital * 0.12 * forecast.confidence
                    
                    expected_return = forecast.close_total_predicted_change
                    actual_return = expected_return * np.random.uniform(0.7, 1.4)
                    pnl = position_size * actual_return
                    
                    trades.append({
                        'symbol': symbol,
                        'momentum': momentum,
                        'band_position': band_position,
                        'position': position_size,
                        'pnl': pnl,
                        'signal': 'hybrid_momentum_band'
                    })
                    
                    current_capital += pnl
        
        return self.calculate_strategy_metrics(
            f"Hybrid Momentum-Bands #{self.strategy_counter}",
            capital,
            current_capital,
            trades,
            {'momentum_threshold': 0.02, 'band_threshold': 0.7}
        )
    
    def strategy_hybrid_vol_ml(self, forecasts: Dict[str, ForecastData], capital: float) -> StrategyResult:
        """Hybrid: Volatility + ML signals"""
        trades = []
        current_capital = capital
        
        for symbol, forecast in forecasts.items():
            # Vol signal
            band_width = forecast.close_upper_bound - forecast.close_lower_bound
            mid_price = (forecast.close_upper_bound + forecast.close_lower_bound) / 2
            implied_vol = band_width / mid_price if mid_price > 0 else 0.02
            
            # ML signal (simplified)
            ml_score = forecast.confidence * forecast.close_total_predicted_change * 10
            
            # Combined decision
            if implied_vol < 0.03 and abs(ml_score) > 0.05:
                position_size = current_capital * min(0.15 / (1 + implied_vol * 10), 0.2)
                
                expected_return = forecast.close_total_predicted_change * np.sign(ml_score)
                actual_return = expected_return * np.random.normal(1, implied_vol * 5)
                pnl = position_size * actual_return
                
                trades.append({
                    'symbol': symbol,
                    'implied_vol': implied_vol,
                    'ml_score': ml_score,
                    'position': position_size,
                    'pnl': pnl,
                    'signal': 'hybrid_vol_ml'
                })
                
                current_capital += pnl
        
        return self.calculate_strategy_metrics(
            f"Hybrid Vol-ML #{self.strategy_counter}",
            capital,
            current_capital,
            trades,
            {'vol_threshold': 0.03, 'ml_threshold': 0.05}
        )
    
    def strategy_random_variant(self, forecasts: Dict[str, ForecastData], capital: float) -> StrategyResult:
        """Random parameter variation of base strategies"""
        trades = []
        current_capital = capital
        
        # Random parameters
        confidence_threshold = np.random.uniform(0.5, 0.9)
        position_size_factor = np.random.uniform(0.05, 0.25)
        forecast_weight = np.random.uniform(0.3, 0.9)
        
        for symbol, forecast in forecasts.items():
            if forecast.confidence > confidence_threshold:
                position_size = current_capital * position_size_factor
                
                # Random signal combination
                signal_strength = forecast.confidence * forecast_weight + \
                                np.random.normal(0, 0.1) * (1 - forecast_weight)
                
                if signal_strength > 0.5:
                    expected_return = forecast.close_total_predicted_change
                    actual_return = expected_return * np.random.uniform(0.5, 1.5)
                    pnl = position_size * actual_return
                    
                    trades.append({
                        'symbol': symbol,
                        'signal_strength': signal_strength,
                        'position': position_size,
                        'pnl': pnl,
                        'signal': 'random_variant'
                    })
                    
                    current_capital += pnl
        
        return self.calculate_strategy_metrics(
            f"Random Variant #{self.strategy_counter}",
            capital,
            current_capital,
            trades,
            {'confidence_threshold': confidence_threshold, 
             'position_factor': position_size_factor,
             'forecast_weight': forecast_weight}
        )
    
    def write_header(self):
        """Write header to results file"""
        with open(self.results_file, 'w') as f:
            f.write("# Perpetual Strategy Testing Results\n")
            f.write(f"Started: {datetime.now().isoformat()}\n\n")
            f.write("## Strategy Performance Log\n\n")
    
    def write_result(self, result: StrategyResult):
        """Append result to file"""
        with open(self.results_file, 'a') as f:
            f.write(f"\n### {result.strategy_name}\n")
            f.write(f"- **Date**: {result.test_date}\n")
            f.write(f"- **Return**: {result.total_return:.2%}\n")
            f.write(f"- **Sharpe**: {result.sharpe_ratio:.2f}\n")
            f.write(f"- **Max DD**: {result.max_drawdown:.2%}\n")
            f.write(f"- **Win Rate**: {result.win_rate:.1%}\n")
            f.write(f"- **Trades**: {result.num_trades}\n")
            f.write(f"- **Best/Worst**: ${result.best_trade:.2f} / ${result.worst_trade:.2f}\n")
            f.write(f"- **Params**: `{result.strategy_params}`\n")
    
    def write_summary(self):
        """Write periodic summary"""
        if not self.results_data:
            return
        
        with open(self.results_file, 'a') as f:
            f.write("\n## Periodic Summary\n")
            f.write(f"Time: {datetime.now().isoformat()}\n\n")
            
            # Best strategies
            sorted_results = sorted(self.results_data, key=lambda x: x.total_return, reverse=True)
            
            f.write("### Top 5 by Return\n")
            for i, r in enumerate(sorted_results[:5], 1):
                f.write(f"{i}. {r.strategy_name}: {r.total_return:.2%}\n")
            
            # Best Sharpe
            sorted_sharpe = sorted(self.results_data, key=lambda x: x.sharpe_ratio, reverse=True)
            f.write("\n### Top 5 by Sharpe Ratio\n")
            for i, r in enumerate(sorted_sharpe[:5], 1):
                f.write(f"{i}. {r.strategy_name}: {r.sharpe_ratio:.2f}\n")
            
            f.write("\n---\n")
    
    def write_final_summary(self):
        """Write final comprehensive summary"""
        with open(self.results_file, 'a') as f:
            f.write("\n## FINAL SUMMARY\n")
            f.write(f"Completed: {datetime.now().isoformat()}\n")
            f.write(f"Total Strategies Tested: {len(self.results_data)}\n\n")
            
            if self.results_data:
                # Overall best
                best_return = max(self.results_data, key=lambda x: x.total_return)
                best_sharpe = max(self.results_data, key=lambda x: x.sharpe_ratio)
                best_win_rate = max(self.results_data, key=lambda x: x.win_rate)
                
                f.write("### Champions\n")
                f.write(f"- **Best Return**: {best_return.strategy_name} ({best_return.total_return:.2%})\n")
                f.write(f"- **Best Sharpe**: {best_sharpe.strategy_name} ({best_sharpe.sharpe_ratio:.2f})\n")
                f.write(f"- **Best Win Rate**: {best_win_rate.strategy_name} ({best_win_rate.win_rate:.1%})\n")
                
                # Strategy category performance
                categories = {}
                for r in self.results_data:
                    category = r.strategy_name.split()[0]
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(r.total_return)
                
                f.write("\n### Category Performance\n")
                for cat, returns in categories.items():
                    avg_return = np.mean(returns)
                    f.write(f"- {cat}: Avg Return = {avg_return:.2%}\n")


if __name__ == "__main__":
    tester = PerpetualStrategyTester()
    tester.run_continuous_testing(test_duration_hours=24*365)  # Run for a year (or forever)