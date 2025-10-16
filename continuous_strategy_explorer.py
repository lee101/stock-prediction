#!/usr/bin/env python3
"""
Continuous Strategy Explorer - Tests endless strategy variations
Uses realistic synthetic forecasts and explores novel combinations
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import sys
import os
import time
from dataclasses import dataclass, asdict
import itertools
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Trade:
    symbol: str
    entry_time: datetime
    exit_time: datetime  
    entry_price: float
    exit_price: float
    position_size: float
    leverage: float
    pnl: float
    return_pct: float
    strategy: str
    signals: Dict

class ContinuousStrategyExplorer:
    """Explores endless strategy combinations and optimizations"""
    
    def __init__(self):
        self.results_file = "testresults.md"
        self.iteration = 0
        self.all_results = []
        self.best_strategies = []
        self.strategy_dna = {}  # Store successful strategy "genes"
        
        # Strategy components that can be mixed
        self.signal_generators = [
            'momentum', 'mean_reversion', 'breakout', 'volatility',
            'volume', 'correlation', 'ml_ensemble', 'pattern'
        ]
        
        self.position_sizers = [
            'fixed', 'kelly', 'volatility_scaled', 'confidence_weighted',
            'risk_parity', 'optimal_f', 'martingale', 'anti_martingale'
        ]
        
        self.risk_managers = [
            'stop_loss', 'trailing_stop', 'time_stop', 'volatility_stop',
            'correlation_hedge', 'portfolio_heat', 'drawdown_control'
        ]
        
        self.entry_filters = [
            'trend_filter', 'volatility_filter', 'volume_filter',
            'time_of_day', 'correlation_filter', 'regime_filter'
        ]
        
    def generate_realistic_forecast(self, symbol: str, lookback_data: pd.DataFrame = None) -> Dict:
        """Generate realistic Toto-style forecast with bounds"""
        
        # Base parameters for different symbols
        symbol_characteristics = {
            'BTCUSD': {'volatility': 0.04, 'trend': 0.001, 'mean_reversion': 0.3},
            'ETHUSD': {'volatility': 0.05, 'trend': 0.0015, 'mean_reversion': 0.35},
            'AAPL': {'volatility': 0.02, 'trend': 0.0008, 'mean_reversion': 0.5},
            'TSLA': {'volatility': 0.06, 'trend': 0.002, 'mean_reversion': 0.2},
            'NVDA': {'volatility': 0.045, 'trend': 0.0025, 'mean_reversion': 0.25},
        }
        
        chars = symbol_characteristics.get(symbol, 
                {'volatility': 0.03, 'trend': 0.001, 'mean_reversion': 0.4})
        
        # Current market regime (changes over time)
        regime = np.random.choice(['trending', 'ranging', 'volatile'], p=[0.3, 0.5, 0.2])
        
        # Generate forecast based on regime
        if regime == 'trending':
            predicted_change = np.random.normal(chars['trend'] * 2, chars['volatility'] * 0.5)
            confidence = np.random.uniform(0.65, 0.85)
        elif regime == 'ranging':
            predicted_change = np.random.normal(0, chars['volatility'] * 0.3)
            confidence = np.random.uniform(0.5, 0.7)
        else:  # volatile
            predicted_change = np.random.normal(chars['trend'], chars['volatility'] * 1.5)
            confidence = np.random.uniform(0.4, 0.6)
        
        # Add mean reversion component
        if lookback_data is not None and len(lookback_data) > 20:
            current = lookback_data['Close'].iloc[-1]
            ma20 = lookback_data['Close'].iloc[-20:].mean()
            extension = (current - ma20) / ma20
            
            if abs(extension) > 0.05:  # Extended from mean
                reversion_component = -extension * chars['mean_reversion'] * confidence
                predicted_change += reversion_component
        
        # Calculate bounds (Toto-style)
        volatility = chars['volatility']
        upper_bound = predicted_change + volatility * (2 - confidence)  # Tighter bands for higher confidence
        lower_bound = predicted_change - volatility * (2 - confidence)
        
        return {
            'predicted_change': predicted_change,
            'upper_bound': upper_bound,
            'lower_bound': lower_bound,
            'confidence': confidence,
            'volatility': volatility,
            'regime': regime
        }
    
    def load_or_generate_price_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """Load real data or generate realistic synthetic prices"""
        
        # Try to load real data first
        data_dir = Path('data')
        symbol_files = list(data_dir.glob(f"{symbol}*.csv"))
        
        if symbol_files:
            try:
                df = pd.read_csv(symbol_files[0])
                if 'Close' in df.columns or 'close' in df.columns:
                    df.columns = [col.capitalize() for col in df.columns]
                    if len(df) >= days:
                        return df.iloc[-days:]
            except:
                pass
        
        # Generate realistic synthetic data
        prices = []
        current_price = {
            'BTCUSD': 45000, 'ETHUSD': 3000, 'AAPL': 180,
            'TSLA': 250, 'NVDA': 500, 'MSFT': 400
        }.get(symbol, 100)
        
        # Generate with realistic patterns
        trend = np.random.choice([1.0002, 1.0, 0.9998])  # Slight trend
        
        for i in range(days):
            # Daily return with volatility clustering
            if i == 0:
                volatility = 0.02
            else:
                # GARCH-like volatility
                volatility = 0.02 * (0.94 + 0.06 * abs(prices[-1]['return']) / 0.02)
            
            daily_return = np.random.normal(0, volatility) * trend
            current_price *= (1 + daily_return)
            
            prices.append({
                'Date': datetime.now() - timedelta(days=days-i),
                'Open': current_price * np.random.uniform(0.99, 1.01),
                'High': current_price * np.random.uniform(1.0, 1.02),
                'Low': current_price * np.random.uniform(0.98, 1.0),
                'Close': current_price,
                'Volume': np.random.uniform(1e6, 1e8),
                'return': daily_return
            })
        
        df = pd.DataFrame(prices)
        return df
    
    def test_strategy_variant(self, strategy_config: Dict) -> Dict:
        """Test a specific strategy configuration"""
        
        symbols = ['BTCUSD', 'ETHUSD', 'AAPL', 'TSLA', 'NVDA']
        initial_capital = 100000
        capital = initial_capital
        trades = []
        
        for symbol in symbols:
            # Load price data
            price_data = self.load_or_generate_price_data(symbol, 100)
            
            # Generate forecast
            forecast = self.generate_realistic_forecast(symbol, price_data)
            
            # Generate signals based on strategy config
            signals = self.generate_signals(
                price_data, forecast, strategy_config['signal_generator']
            )
            
            # Apply entry filters
            if self.apply_entry_filters(
                price_data, forecast, signals, strategy_config['entry_filter']
            ):
                # Calculate position size
                position_size = self.calculate_position_size(
                    capital, forecast, signals, strategy_config['position_sizer']
                )
                
                # Determine leverage
                leverage = self.calculate_leverage(forecast, strategy_config)
                
                # Simulate trade
                trade = self.simulate_trade(
                    symbol, price_data, forecast, position_size, leverage, strategy_config
                )
                
                if trade:
                    trades.append(trade)
                    capital += trade.pnl
        
        # Calculate metrics
        total_return = (capital - initial_capital) / initial_capital
        
        if trades:
            returns = [t.return_pct for t in trades]
            winning = [t for t in trades if t.pnl > 0]
            
            metrics = {
                'total_return': total_return,
                'num_trades': len(trades),
                'win_rate': len(winning) / len(trades),
                'avg_return': np.mean(returns),
                'sharpe': np.sqrt(252) * np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
                'max_drawdown': self.calculate_max_drawdown([t.pnl for t in trades], initial_capital)
            }
        else:
            metrics = {
                'total_return': 0,
                'num_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'sharpe': 0,
                'max_drawdown': 0
            }
        
        return {
            'config': strategy_config,
            'metrics': metrics,
            'trades': trades
        }
    
    def generate_signals(self, price_data: pd.DataFrame, forecast: Dict, signal_type: str) -> Dict:
        """Generate trading signals based on signal type"""
        
        signals = {}
        
        if signal_type == 'momentum':
            # Momentum signals
            returns_5d = (price_data['Close'].iloc[-1] / price_data['Close'].iloc[-6] - 1) if len(price_data) > 5 else 0
            returns_20d = (price_data['Close'].iloc[-1] / price_data['Close'].iloc[-21] - 1) if len(price_data) > 20 else 0
            
            signals['momentum_5d'] = returns_5d
            signals['momentum_20d'] = returns_20d
            signals['signal_strength'] = (returns_5d + returns_20d * 0.5) / 1.5
            
        elif signal_type == 'mean_reversion':
            # Mean reversion signals
            if len(price_data) > 20:
                ma20 = price_data['Close'].iloc[-20:].mean()
                current = price_data['Close'].iloc[-1]
                extension = (current - ma20) / ma20
                
                signals['extension'] = extension
                signals['signal_strength'] = -extension if abs(extension) > 0.03 else 0
            else:
                signals['signal_strength'] = 0
                
        elif signal_type == 'breakout':
            # Breakout signals
            if len(price_data) > 20:
                high_20d = price_data['High'].iloc[-20:].max()
                low_20d = price_data['Low'].iloc[-20:].min()
                current = price_data['Close'].iloc[-1]
                
                if current > high_20d * 0.99:
                    signals['signal_strength'] = 1
                elif current < low_20d * 1.01:
                    signals['signal_strength'] = -1
                else:
                    signals['signal_strength'] = 0
            else:
                signals['signal_strength'] = 0
                
        elif signal_type == 'volatility':
            # Volatility-based signals
            if len(price_data) > 20:
                returns = price_data['Close'].pct_change().dropna()
                current_vol = returns.iloc[-5:].std() if len(returns) > 5 else 0.02
                hist_vol = returns.iloc[-20:].std() if len(returns) > 20 else 0.02
                
                vol_ratio = current_vol / hist_vol if hist_vol > 0 else 1
                
                # Trade when volatility is extreme
                if vol_ratio > 1.5:
                    signals['signal_strength'] = -0.5  # Expect reversion
                elif vol_ratio < 0.7:
                    signals['signal_strength'] = 0.5  # Expect expansion
                else:
                    signals['signal_strength'] = 0
                    
                signals['vol_ratio'] = vol_ratio
            else:
                signals['signal_strength'] = 0
                
        elif signal_type == 'ml_ensemble':
            # Combine multiple signals
            mom_signal = self.generate_signals(price_data, forecast, 'momentum')
            rev_signal = self.generate_signals(price_data, forecast, 'mean_reversion')
            vol_signal = self.generate_signals(price_data, forecast, 'volatility')
            
            # Weight combination
            ensemble_strength = (
                mom_signal.get('signal_strength', 0) * 0.3 +
                rev_signal.get('signal_strength', 0) * 0.3 +
                vol_signal.get('signal_strength', 0) * 0.2 +
                forecast['predicted_change'] * 10 * 0.2
            )
            
            signals['signal_strength'] = ensemble_strength
            signals['components'] = {
                'momentum': mom_signal.get('signal_strength', 0),
                'reversion': rev_signal.get('signal_strength', 0),
                'volatility': vol_signal.get('signal_strength', 0),
                'forecast': forecast['predicted_change']
            }
        else:
            # Default or pattern recognition
            signals['signal_strength'] = forecast['predicted_change'] * 10 * forecast['confidence']
        
        signals['forecast_aligned'] = np.sign(signals.get('signal_strength', 0)) == np.sign(forecast['predicted_change'])
        
        return signals
    
    def apply_entry_filters(self, price_data: pd.DataFrame, forecast: Dict, 
                          signals: Dict, filter_type: str) -> bool:
        """Apply entry filters to validate trade entry"""
        
        if filter_type == 'trend_filter':
            # Only trade in trending markets
            if len(price_data) > 20:
                ma20 = price_data['Close'].iloc[-20:].mean()
                ma50 = price_data['Close'].iloc[-50:].mean() if len(price_data) > 50 else ma20
                return ma20 > ma50 or signals.get('signal_strength', 0) > 0.5
            return True
            
        elif filter_type == 'volatility_filter':
            # Avoid extremely high volatility
            return forecast['volatility'] < 0.06
            
        elif filter_type == 'volume_filter':
            # Ensure adequate volume
            if 'Volume' in price_data.columns:
                avg_volume = price_data['Volume'].iloc[-20:].mean()
                recent_volume = price_data['Volume'].iloc[-1]
                return recent_volume > avg_volume * 0.7
            return True
            
        elif filter_type == 'correlation_filter':
            # Check correlation with market (simplified)
            return forecast['confidence'] > 0.5
            
        elif filter_type == 'regime_filter':
            # Trade based on market regime
            return forecast.get('regime') in ['trending', 'ranging']
            
        else:  # No filter or time_of_day (always true for backtesting)
            return True
    
    def calculate_position_size(self, capital: float, forecast: Dict, 
                               signals: Dict, sizing_method: str) -> float:
        """Calculate position size based on method"""
        
        base_size = capital * 0.1  # 10% base position
        
        if sizing_method == 'fixed':
            return base_size
            
        elif sizing_method == 'kelly':
            # Simplified Kelly Criterion
            p = forecast['confidence']
            q = 1 - p
            b = abs(forecast['predicted_change']) / forecast['volatility'] if forecast['volatility'] > 0 else 1
            
            kelly_fraction = (p * b - q) / b if b > 0 else 0
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            return capital * kelly_fraction
            
        elif sizing_method == 'volatility_scaled':
            # Inverse volatility scaling
            target_vol = 0.02
            position_size = base_size * (target_vol / forecast['volatility'])
            return min(position_size, capital * 0.2)
            
        elif sizing_method == 'confidence_weighted':
            return base_size * (0.5 + forecast['confidence'])
            
        elif sizing_method == 'risk_parity':
            # Equal risk contribution (simplified)
            return base_size / (1 + forecast['volatility'] * 10)
            
        elif sizing_method == 'optimal_f':
            # Simplified optimal f
            signal_strength = abs(signals.get('signal_strength', 0))
            return base_size * min(signal_strength * 2, 2)
            
        elif sizing_method == 'martingale':
            # Increase after losses (dangerous but included for testing)
            # In real implementation, would track recent losses
            return base_size * np.random.uniform(1, 1.5)
            
        elif sizing_method == 'anti_martingale':
            # Increase after wins
            return base_size * np.random.uniform(0.8, 1.2)
            
        else:
            return base_size
    
    def calculate_leverage(self, forecast: Dict, strategy_config: Dict) -> float:
        """Calculate appropriate leverage"""
        
        max_leverage = strategy_config.get('max_leverage', 2.0)
        
        # Base leverage on confidence and volatility
        if forecast['confidence'] < 0.6:
            return 1.0
        
        confidence_factor = (forecast['confidence'] - 0.6) / 0.4
        volatility_factor = max(0.5, 1 - forecast['volatility'] * 10)
        
        leverage = 1 + (max_leverage - 1) * confidence_factor * volatility_factor
        
        return min(leverage, max_leverage)
    
    def simulate_trade(self, symbol: str, price_data: pd.DataFrame, forecast: Dict,
                      position_size: float, leverage: float, strategy_config: Dict) -> Optional[Trade]:
        """Simulate a trade execution"""
        
        if len(price_data) < 2:
            return None
        
        entry_price = price_data['Close'].iloc[-1]
        
        # Simulate future price (would use next day's actual price in real backtest)
        predicted_return = forecast['predicted_change']
        
        # Add realistic noise
        actual_return = predicted_return + np.random.normal(0, forecast['volatility'] * 0.5)
        
        # Apply leverage
        leveraged_return = actual_return * leverage
        
        # Calculate exit price
        exit_price = entry_price * (1 + actual_return)
        
        # Calculate P&L
        leveraged_position = position_size * leverage
        pnl = leveraged_position * actual_return
        
        # Apply costs
        trading_cost = leveraged_position * 0.001  # 0.1% trading cost
        
        if leverage > 1:
            # Leverage cost (7% annual for borrowed amount)
            borrowed = leveraged_position * (1 - 1/leverage)
            leverage_cost = borrowed * 0.07 / 365 * 7  # 7 day holding
            pnl -= leverage_cost
        
        pnl -= trading_cost
        
        return Trade(
            symbol=symbol,
            entry_time=datetime.now(),
            exit_time=datetime.now() + timedelta(days=7),
            entry_price=entry_price,
            exit_price=exit_price,
            position_size=position_size,
            leverage=leverage,
            pnl=pnl,
            return_pct=pnl / position_size if position_size > 0 else 0,
            strategy=strategy_config['name'],
            signals={'forecast': forecast}
        )
    
    def calculate_max_drawdown(self, pnls: List[float], initial_capital: float) -> float:
        """Calculate maximum drawdown"""
        
        if not pnls:
            return 0
        
        cumulative = [initial_capital]
        for pnl in pnls:
            cumulative.append(cumulative[-1] + pnl)
        
        cumulative = np.array(cumulative)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(np.min(drawdown))
    
    def generate_strategy_variant(self) -> Dict:
        """Generate a new strategy variant to test"""
        
        self.iteration += 1
        
        # Mix and match components
        config = {
            'name': f'Strategy_{self.iteration}',
            'signal_generator': np.random.choice(self.signal_generators),
            'position_sizer': np.random.choice(self.position_sizers),
            'risk_manager': np.random.choice(self.risk_managers),
            'entry_filter': np.random.choice(self.entry_filters),
            'max_leverage': np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0]),
            'stop_loss': np.random.uniform(0.02, 0.1),
            'take_profit': np.random.uniform(0.02, 0.2),
            'max_positions': np.random.randint(3, 10)
        }
        
        # Sometimes create hybrid strategies
        if self.iteration % 5 == 0:
            # Combine successful elements
            if self.best_strategies:
                parent = np.random.choice(self.best_strategies)
                config['signal_generator'] = parent['config']['signal_generator']
                config['name'] = f"Evolved_{self.iteration}"
        
        return config
    
    def run_forever(self):
        """Run continuous strategy exploration"""
        
        print("Starting Continuous Strategy Explorer")
        print("="*80)
        
        # Initialize results file
        with open(self.results_file, 'w') as f:
            f.write("# Continuous Strategy Testing Results\n")
            f.write(f"Started: {datetime.now()}\n\n")
        
        while True:
            # Generate new strategy variant
            strategy_config = self.generate_strategy_variant()
            
            # Test it
            result = self.test_strategy_variant(strategy_config)
            
            # Store results
            self.all_results.append(result)
            
            # Update best strategies
            if result['metrics']['sharpe'] > 1.0 or result['metrics']['total_return'] > 0.1:
                self.best_strategies.append(result)
                # Keep only top 20
                self.best_strategies = sorted(
                    self.best_strategies, 
                    key=lambda x: x['metrics']['sharpe'], 
                    reverse=True
                )[:20]
            
            # Write to file
            self.write_result(result)
            
            # Print progress
            print(f"Iteration {self.iteration}: {strategy_config['name']}")
            print(f"  Return: {result['metrics']['total_return']:.2%}")
            print(f"  Sharpe: {result['metrics']['sharpe']:.2f}")
            print(f"  Trades: {result['metrics']['num_trades']}")
            
            # Periodic summary
            if self.iteration % 100 == 0:
                self.write_summary()
            
            # Generate variations of successful strategies
            if self.iteration % 10 == 0 and self.best_strategies:
                self.explore_successful_variants()
            
            # Brief pause
            time.sleep(0.1)
    
    def explore_successful_variants(self):
        """Create variations of successful strategies"""
        
        if not self.best_strategies:
            return
        
        # Pick a successful strategy
        parent = np.random.choice(self.best_strategies)
        
        # Create mutations
        for _ in range(5):
            mutant_config = parent['config'].copy()
            
            # Mutate random parameter
            mutation = np.random.choice([
                'signal_generator', 'position_sizer', 
                'risk_manager', 'entry_filter'
            ])
            
            if mutation == 'signal_generator':
                mutant_config['signal_generator'] = np.random.choice(self.signal_generators)
            elif mutation == 'position_sizer':
                mutant_config['position_sizer'] = np.random.choice(self.position_sizers)
            elif mutation == 'risk_manager':
                mutant_config['risk_manager'] = np.random.choice(self.risk_managers)
            else:
                mutant_config['entry_filter'] = np.random.choice(self.entry_filters)
            
            mutant_config['name'] = f"Mutant_{self.iteration}_{mutation}"
            
            # Test mutant
            result = self.test_strategy_variant(mutant_config)
            self.all_results.append(result)
            
            print(f"  Mutant: {mutant_config['name']} -> Return: {result['metrics']['total_return']:.2%}")
    
    def write_result(self, result: Dict):
        """Write result to file"""
        
        with open(self.results_file, 'a') as f:
            f.write(f"\n## {result['config']['name']}\n")
            f.write(f"- Time: {datetime.now()}\n")
            f.write(f"- Return: {result['metrics']['total_return']:.2%}\n")
            f.write(f"- Sharpe: {result['metrics']['sharpe']:.2f}\n")
            f.write(f"- Win Rate: {result['metrics']['win_rate']:.1%}\n")
            f.write(f"- Max DD: {result['metrics']['max_drawdown']:.2%}\n")
            f.write(f"- Config: `{result['config']}`\n")
    
    def write_summary(self):
        """Write periodic summary"""
        
        with open(self.results_file, 'a') as f:
            f.write(f"\n# Summary at Iteration {self.iteration}\n")
            f.write(f"Time: {datetime.now()}\n\n")
            
            if self.best_strategies:
                f.write("## Top 5 Strategies by Sharpe\n")
                for i, s in enumerate(self.best_strategies[:5], 1):
                    f.write(f"{i}. {s['config']['name']}: Sharpe={s['metrics']['sharpe']:.2f}, Return={s['metrics']['total_return']:.2%}\n")
                
                # Analyze winning components
                signal_counts = {}
                sizer_counts = {}
                
                for s in self.best_strategies:
                    sig = s['config']['signal_generator']
                    siz = s['config']['position_sizer']
                    
                    signal_counts[sig] = signal_counts.get(sig, 0) + 1
                    sizer_counts[siz] = sizer_counts.get(siz, 0) + 1
                
                f.write("\n## Winning Components\n")
                f.write("### Best Signal Generators\n")
                for sig, count in sorted(signal_counts.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"- {sig}: {count} appearances\n")
                
                f.write("\n### Best Position Sizers\n")
                for siz, count in sorted(sizer_counts.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"- {siz}: {count} appearances\n")
            
            f.write("\n---\n")


if __name__ == "__main__":
    explorer = ContinuousStrategyExplorer()
    explorer.run_forever()