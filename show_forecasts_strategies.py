#!/usr/bin/env python3
"""
Enhanced Forecasting Strategies

This module implements sophisticated forecasting strategies that exploit:
1. Prediction magnitude (larger moves get more allocation)
2. Directional confidence (multiple signals alignment)
3. Risk-adjusted position sizing
4. Dynamic strategy selection based on market conditions
"""

import sys
from pathlib import Path
import pandas as pd
from loguru import logger
from datetime import datetime, timedelta
import numpy as np
import json

import pytz
import alpaca_wrapper
from predict_stock_forecasting import make_predictions, load_stock_data_from_csv
from data_curate_daily import download_daily_stock_data
from show_forecasts import get_cached_predictions


class ForecastingStrategy:
    """Base class for forecasting strategies"""
    
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.results = {}
    
    def calculate_signal_strength(self, predictions):
        """Calculate signal strength from predictions (0-1 scale)"""
        raise NotImplementedError
    
    def calculate_position_size(self, signal_strength, base_capital=10000):
        """Calculate position size based on signal strength"""
        raise NotImplementedError
    
    def get_recommendation(self, predictions, current_price=None):
        """Get trading recommendation"""
        signal_strength = self.calculate_signal_strength(predictions)
        position_size = self.calculate_position_size(signal_strength)
        
        return {
            'strategy': self.name,
            'signal_strength': signal_strength,
            'position_size': position_size,
            'recommendation': self._get_action(signal_strength),
            'confidence': self._get_confidence_level(signal_strength)
        }
    
    def _get_action(self, signal_strength):
        """Convert signal strength to action"""
        if signal_strength > 0.7:
            return "STRONG_BUY"
        elif signal_strength > 0.5:
            return "BUY"
        elif signal_strength > 0.3:
            return "WEAK_BUY"
        elif signal_strength > -0.3:
            return "HOLD"
        elif signal_strength > -0.5:
            return "WEAK_SELL"
        elif signal_strength > -0.7:
            return "SELL"
        else:
            return "STRONG_SELL"
    
    def _get_confidence_level(self, signal_strength):
        """Get confidence level"""
        confidence = abs(signal_strength)
        if confidence > 0.8:
            return "VERY_HIGH"
        elif confidence > 0.6:
            return "HIGH"
        elif confidence > 0.4:
            return "MEDIUM"
        elif confidence > 0.2:
            return "LOW"
        else:
            return "VERY_LOW"


class MagnitudeBasedStrategy(ForecastingStrategy):
    """Strategy that allocates based on predicted price movement magnitude"""
    
    def __init__(self):
        super().__init__(
            "magnitude_based",
            "Allocates more capital to positions with larger predicted price movements"
        )
    
    def calculate_signal_strength(self, predictions):
        """Calculate signal based on prediction magnitude"""
        try:
            # Get current and predicted prices
            current_close = float(predictions['close_last_price'].iloc[0])
            predicted_close = self._extract_numeric_value(predictions['close_predicted_price_value'].iloc[0])
            
            # Calculate percentage change
            pct_change = (predicted_close - current_close) / current_close
            
            # Scale by magnitude - larger moves get stronger signals
            # Use tanh to bound between -1 and 1, scaled by 10 to make it responsive
            signal_strength = np.tanh(pct_change * 10)
            
            return signal_strength
            
        except Exception as e:
            logger.warning(f"Error calculating magnitude signal: {e}")
            return 0.0
    
    def calculate_position_size(self, signal_strength, base_capital=10000):
        """Position size based on signal strength magnitude"""
        # Use square root to moderate extreme positions
        size_multiplier = np.sqrt(abs(signal_strength))
        
        # Base position is 20% of capital, can scale up to 80% for very strong signals
        base_size = 0.2
        max_additional = 0.6
        
        position_fraction = base_size + (size_multiplier * max_additional)
        return int(base_capital * position_fraction)
    
    def _extract_numeric_value(self, value):
        """Extract numeric value from various formats"""
        if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
            return float(value.strip('()').rstrip(','))
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return float(str(value))


class ConsensusStrategy(ForecastingStrategy):
    """Strategy that uses consensus across multiple prediction metrics"""
    
    def __init__(self):
        super().__init__(
            "consensus_based", 
            "Uses consensus across multiple prediction signals for higher confidence"
        )
    
    def calculate_signal_strength(self, predictions):
        """Calculate consensus signal from multiple metrics"""
        try:
            signals = []
            row = predictions.iloc[0]
            
            # Price direction signals
            current_close = float(row['close_last_price'])
            predicted_close = self._extract_numeric_value(row['close_predicted_price_value'])
            close_signal = 1 if predicted_close > current_close else -1
            signals.append(close_signal)
            
            # Trading strategy signals
            strategy_cols = ['entry_takeprofit_profit', 'maxdiffprofit_profit', 'takeprofit_profit']
            for col in strategy_cols:
                if col in predictions.columns:
                    try:
                        value = self._extract_numeric_value(row[col])
                        signals.append(1 if value > 0.02 else (-1 if value < -0.02 else 0))  # 2% threshold
                    except:
                        continue
            
            # High/low range signals
            if 'high_predicted_price_value' in predictions.columns and 'low_predicted_price_value' in predictions.columns:
                try:
                    predicted_high = self._extract_numeric_value(row['high_predicted_price_value'])
                    predicted_low = self._extract_numeric_value(row['low_predicted_price_value'])
                    range_midpoint = (predicted_high + predicted_low) / 2
                    range_signal = 1 if range_midpoint > current_close else -1
                    signals.append(range_signal)
                except:
                    pass
            
            if not signals:
                return 0.0
            
            # Calculate consensus strength
            consensus_ratio = sum(signals) / len(signals)
            agreement_strength = abs(consensus_ratio)  # How much do signals agree
            
            # Boost signal if there's strong agreement
            signal_strength = consensus_ratio * (0.5 + 0.5 * agreement_strength)
            
            return signal_strength
            
        except Exception as e:
            logger.warning(f"Error calculating consensus signal: {e}")
            return 0.0
    
    def calculate_position_size(self, signal_strength, base_capital=10000):
        """Position size based on consensus strength"""
        # Higher consensus gets more allocation
        confidence = abs(signal_strength)
        
        if confidence > 0.8:
            position_fraction = 0.75  # Very strong consensus
        elif confidence > 0.6:
            position_fraction = 0.55  # Strong consensus  
        elif confidence > 0.4:
            position_fraction = 0.35  # Moderate consensus
        elif confidence > 0.2:
            position_fraction = 0.20  # Weak consensus
        else:
            position_fraction = 0.10  # Very weak consensus
            
        return int(base_capital * position_fraction)
    
    def _extract_numeric_value(self, value):
        """Extract numeric value from various formats"""
        if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
            return float(value.strip('()').rstrip(','))
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return float(str(value))


class VolatilityAdjustedStrategy(ForecastingStrategy):
    """Strategy that adjusts position size based on predicted volatility"""
    
    def __init__(self):
        super().__init__(
            "volatility_adjusted",
            "Adjusts position sizes based on predicted price volatility (range)"
        )
    
    def calculate_signal_strength(self, predictions):
        """Calculate signal strength considering volatility"""
        try:
            row = predictions.iloc[0]
            current_close = float(row['close_last_price'])
            predicted_close = self._extract_numeric_value(row['close_predicted_price_value'])
            
            # Basic direction signal
            direction = 1 if predicted_close > current_close else -1
            magnitude = abs(predicted_close - current_close) / current_close
            
            # Calculate predicted volatility from high/low range
            if 'high_predicted_price_value' in predictions.columns and 'low_predicted_price_value' in predictions.columns:
                predicted_high = self._extract_numeric_value(row['high_predicted_price_value'])
                predicted_low = self._extract_numeric_value(row['low_predicted_price_value'])
                
                # Volatility as percentage of current price
                volatility = (predicted_high - predicted_low) / current_close
                
                # Higher volatility = higher potential but needs smaller position
                # Moderate the signal based on risk-adjusted return
                risk_adjusted_magnitude = magnitude / max(volatility, 0.01)  # Avoid division by zero
                
                # Cap the signal to reasonable bounds
                signal_strength = direction * np.tanh(risk_adjusted_magnitude * 5)
            else:
                # Fallback to simple magnitude if no range data
                signal_strength = direction * np.tanh(magnitude * 10)
            
            return signal_strength
            
        except Exception as e:
            logger.warning(f"Error calculating volatility-adjusted signal: {e}")
            return 0.0
    
    def calculate_position_size(self, signal_strength, base_capital=10000):
        """Position size inversely related to volatility"""
        signal_magnitude = abs(signal_strength)
        
        # Conservative approach - strong signals get moderate positions
        # Weak signals get small positions
        if signal_magnitude > 0.7:
            position_fraction = 0.6  # Strong signal but volatility-adjusted
        elif signal_magnitude > 0.5:
            position_fraction = 0.45
        elif signal_magnitude > 0.3:
            position_fraction = 0.3
        else:
            position_fraction = 0.15  # Small position for weak signals
            
        return int(base_capital * position_fraction)
    
    def _extract_numeric_value(self, value):
        """Extract numeric value from various formats"""
        if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
            return float(value.strip('()').rstrip(','))
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return float(str(value))


class MomentumVolatilityStrategy(ForecastingStrategy):
    """Strategy that combines momentum and volatility signals with enhanced position sizing"""
    
    def __init__(self):
        super().__init__(
            "momentum_volatility",
            "Combines momentum trends with volatility-adjusted risk management"
        )
    
    def calculate_signal_strength(self, predictions):
        """Calculate signal considering both momentum and volatility"""
        try:
            row = predictions.iloc[0]
            current_close = float(row['close_last_price'])
            predicted_close = self._extract_numeric_value(row['close_predicted_price_value'])
            
            # Basic momentum signal
            momentum = (predicted_close - current_close) / current_close
            momentum_signal = np.tanh(momentum * 15)  # Stronger momentum response
            
            # Volatility component
            if 'high_predicted_price_value' in predictions.columns and 'low_predicted_price_value' in predictions.columns:
                predicted_high = self._extract_numeric_value(row['high_predicted_price_value'])
                predicted_low = self._extract_numeric_value(row['low_predicted_price_value'])
                
                volatility = (predicted_high - predicted_low) / current_close
                
                # Higher volatility = higher potential reward but needs careful sizing
                # Use volatility as a multiplier for momentum signal
                volatility_multiplier = 1 + (volatility * 2)  # Scale with volatility
                enhanced_signal = momentum_signal * volatility_multiplier
                
                # Cap the signal to prevent extreme positions
                signal_strength = np.tanh(enhanced_signal)
            else:
                signal_strength = momentum_signal
            
            return signal_strength
            
        except Exception as e:
            logger.warning(f"Error calculating momentum-volatility signal: {e}")
            return 0.0
    
    def calculate_position_size(self, signal_strength, base_capital=10000):
        """Aggressive position sizing for strong momentum-volatility signals"""
        signal_magnitude = abs(signal_strength)
        
        if signal_magnitude > 0.8:
            position_fraction = 0.85  # Very aggressive for strong signals
        elif signal_magnitude > 0.6:
            position_fraction = 0.65
        elif signal_magnitude > 0.4:
            position_fraction = 0.45
        elif signal_magnitude > 0.2:
            position_fraction = 0.25
        else:
            position_fraction = 0.10
            
        return int(base_capital * position_fraction)
    
    def _extract_numeric_value(self, value):
        """Extract numeric value from various formats"""
        if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
            return float(value.strip('()').rstrip(','))
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return float(str(value))


class ProfitTargetStrategy(ForecastingStrategy):
    """Strategy that focuses on trading profit metrics from predictions"""
    
    def __init__(self):
        super().__init__(
            "profit_target",
            "Uses predicted trading profits to determine position sizing"
        )
    
    def calculate_signal_strength(self, predictions):
        """Calculate signal based on predicted trading profits"""
        try:
            row = predictions.iloc[0]
            
            # Look for profit metrics in the predictions
            profit_signals = []
            profit_cols = ['entry_takeprofit_profit', 'maxdiffprofit_profit', 'takeprofit_profit']
            
            for col in profit_cols:
                if col in predictions.columns:
                    try:
                        profit_value = self._extract_numeric_value(row[col])
                        # Convert profit to signal strength
                        profit_signals.append(np.tanh(profit_value * 100))  # Scale profit values
                    except:
                        continue
            
            # If we have profit signals, use them
            if profit_signals:
                avg_profit_signal = np.mean(profit_signals)
                
                # Enhance with directional price signal
                current_close = float(row['close_last_price'])
                predicted_close = self._extract_numeric_value(row['close_predicted_price_value'])
                direction_signal = 1 if predicted_close > current_close else -1
                
                # Combine profit expectation with direction
                signal_strength = avg_profit_signal * direction_signal
                
                return signal_strength
            else:
                # Fallback to basic price direction
                current_close = float(row['close_last_price'])
                predicted_close = self._extract_numeric_value(row['close_predicted_price_value'])
                pct_change = (predicted_close - current_close) / current_close
                return np.tanh(pct_change * 10)
            
        except Exception as e:
            logger.warning(f"Error calculating profit target signal: {e}")
            return 0.0
    
    def calculate_position_size(self, signal_strength, base_capital=10000):
        """Position sizing based on profit potential"""
        signal_magnitude = abs(signal_strength)
        
        # More aggressive sizing for profit-based signals
        if signal_magnitude > 0.7:
            position_fraction = 0.75
        elif signal_magnitude > 0.5:
            position_fraction = 0.60
        elif signal_magnitude > 0.3:
            position_fraction = 0.40
        elif signal_magnitude > 0.1:
            position_fraction = 0.20
        else:
            position_fraction = 0.05
            
        return int(base_capital * position_fraction)
    
    def _extract_numeric_value(self, value):
        """Extract numeric value from various formats"""
        if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
            return float(value.strip('()').rstrip(','))
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return float(str(value))


class HybridProfitVolatilityStrategy(ForecastingStrategy):
    """Ultra-optimized strategy combining profit targeting with volatility adjustment"""
    
    def __init__(self):
        super().__init__(
            "hybrid_profit_volatility",
            "Combines profit targeting with volatility-adjusted risk management for optimal returns"
        )
    
    def calculate_signal_strength(self, predictions):
        """Calculate signal combining profit targets and volatility adjustment"""
        try:
            row = predictions.iloc[0]
            
            # Component 1: Profit-based signal (strongest performer)
            profit_signal = self._calculate_profit_signal(row)
            
            # Component 2: Volatility-adjusted signal (consistently strong)
            volatility_signal = self._calculate_volatility_signal(row, predictions)
            
            # Component 3: Momentum confirmation
            momentum_signal = self._calculate_momentum_signal(row)
            
            # Weight the signals based on performance insights
            # Profit signal gets highest weight (50%), volatility (35%), momentum (15%)
            combined_signal = (0.50 * profit_signal + 
                             0.35 * volatility_signal + 
                             0.15 * momentum_signal)
            
            # Apply enhancement multiplier for strong consensus
            if abs(profit_signal) > 0.7 and abs(volatility_signal) > 0.7:
                combined_signal *= 1.2  # Boost when both strong signals agree
            
            return np.tanh(combined_signal)  # Bound between -1 and 1
            
        except Exception as e:
            logger.warning(f"Error calculating hybrid signal: {e}")
            return 0.0
    
    def _calculate_profit_signal(self, row):
        """Calculate profit-based signal component"""
        profit_signals = []
        profit_cols = ['entry_takeprofit_profit', 'maxdiffprofit_profit', 'takeprofit_profit']
        
        for col in profit_cols:
            if col in row.index:
                try:
                    profit_value = self._extract_numeric_value(row[col])
                    profit_signals.append(np.tanh(profit_value * 150))  # Higher scaling for profit
                except:
                    continue
        
        if profit_signals:
            return np.mean(profit_signals)
        else:
            return 0.0
    
    def _calculate_volatility_signal(self, row, predictions):
        """Calculate volatility-adjusted signal component"""
        try:
            current_close = float(row['close_last_price'])
            predicted_close = self._extract_numeric_value(row['close_predicted_price_value'])
            
            direction = 1 if predicted_close > current_close else -1
            magnitude = abs(predicted_close - current_close) / current_close
            
            if 'high_predicted_price_value' in predictions.columns and 'low_predicted_price_value' in predictions.columns:
                predicted_high = self._extract_numeric_value(row['high_predicted_price_value'])
                predicted_low = self._extract_numeric_value(row['low_predicted_price_value'])
                
                volatility = (predicted_high - predicted_low) / current_close
                risk_adjusted_magnitude = magnitude / max(volatility, 0.01)
                return direction * np.tanh(risk_adjusted_magnitude * 8)
            else:
                return direction * np.tanh(magnitude * 12)
                
        except:
            return 0.0
    
    def _calculate_momentum_signal(self, row):
        """Calculate momentum confirmation signal"""
        try:
            current_close = float(row['close_last_price'])
            predicted_close = self._extract_numeric_value(row['close_predicted_price_value'])
            
            momentum = (predicted_close - current_close) / current_close
            return np.tanh(momentum * 20)  # Strong momentum scaling
        except:
            return 0.0
    
    def calculate_position_size(self, signal_strength, base_capital=10000):
        """Ultra-aggressive position sizing for hybrid strategy"""
        signal_magnitude = abs(signal_strength)
        
        if signal_magnitude > 0.9:
            position_fraction = 0.95  # Maximum confidence
        elif signal_magnitude > 0.8:
            position_fraction = 0.85
        elif signal_magnitude > 0.7:
            position_fraction = 0.75
        elif signal_magnitude > 0.6:
            position_fraction = 0.60
        elif signal_magnitude > 0.4:
            position_fraction = 0.45
        elif signal_magnitude > 0.2:
            position_fraction = 0.25
        else:
            position_fraction = 0.10
            
        return int(base_capital * position_fraction)
    
    def _extract_numeric_value(self, value):
        """Extract numeric value from various formats"""
        if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
            return float(value.strip('()').rstrip(','))
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return float(str(value))


class AdaptiveStrategy(ForecastingStrategy):
    """Strategy that adapts approach based on recent prediction accuracy"""
    
    def __init__(self):
        super().__init__(
            "adaptive",
            "Adapts strategy selection based on recent prediction performance"
        )
        self.sub_strategies = [
            MagnitudeBasedStrategy(),
            ConsensusStrategy(), 
            VolatilityAdjustedStrategy(),
            MomentumVolatilityStrategy(),
            ProfitTargetStrategy(),
            HybridProfitVolatilityStrategy()
        ]
        self.performance_history = {}
    
    def calculate_signal_strength(self, predictions):
        """Use the best performing sub-strategy"""
        # For now, use a weighted ensemble of all strategies
        signals = []
        weights = []
        
        for strategy in self.sub_strategies:
            try:
                signal = strategy.calculate_signal_strength(predictions)
                signals.append(signal)
                # Weight based on recent performance (equal weights for now)
                weights.append(1.0)
            except Exception as e:
                logger.warning(f"Error in {strategy.name}: {e}")
                continue
        
        if not signals:
            return 0.0
        
        # Weighted average of signals
        total_weight = sum(weights)
        weighted_signal = sum(s * w for s, w in zip(signals, weights)) / total_weight
        
        return weighted_signal
    
    def calculate_position_size(self, signal_strength, base_capital=10000):
        """Conservative position sizing for ensemble"""
        signal_magnitude = abs(signal_strength)
        
        # More conservative than individual strategies
        if signal_magnitude > 0.8:
            position_fraction = 0.5
        elif signal_magnitude > 0.6:
            position_fraction = 0.4
        elif signal_magnitude > 0.4:
            position_fraction = 0.25
        elif signal_magnitude > 0.2:
            position_fraction = 0.15
        else:
            position_fraction = 0.05
            
        return int(base_capital * position_fraction)


def run_forecasting_strategies(symbol, base_capital=10000):
    """Run all forecasting strategies on a symbol"""
    logger.info(f"\n=== Enhanced Forecasting Strategies for {symbol} ===")
    
    # Get predictions
    try:
        # Try to get fresh predictions first
        is_crypto = symbol in ['BTCUSD', 'ETHUSD', 'LTCUSD', 'ADAUSD', 'DOTUSD']
        market_clock = alpaca_wrapper.get_clock()
        is_market_open = market_clock.is_open
        
        if is_crypto or is_market_open:
            try:
                current_time_formatted = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
                data_df = download_daily_stock_data(current_time_formatted)
                predictions = make_predictions(current_time_formatted, alpaca_wrapper=alpaca_wrapper)
                symbol_predictions = predictions[predictions['instrument'] == symbol]
                
                if symbol_predictions.empty:
                    raise Exception("No fresh predictions found")
                    
                logger.info("Using fresh predictions")
            except Exception as e:
                logger.warning(f"Error getting fresh data: {e}")
                symbol_predictions = get_cached_predictions(symbol)
                if symbol_predictions is None:
                    logger.error("No cached predictions available")
                    return
                logger.info("Using cached predictions")
        else:
            symbol_predictions = get_cached_predictions(symbol)
            if symbol_predictions is None:
                logger.error("No cached predictions available")
                return
            logger.info("Using cached predictions")
            
    except Exception as e:
        logger.error(f"Error loading predictions: {e}")
        return
    
    # Initialize strategies
    strategies = [
        MagnitudeBasedStrategy(),
        ConsensusStrategy(),
        VolatilityAdjustedStrategy(),
        MomentumVolatilityStrategy(),
        ProfitTargetStrategy(),
        HybridProfitVolatilityStrategy(),
        AdaptiveStrategy()
    ]
    
    # Get current price for reference
    current_price = float(symbol_predictions['close_last_price'].iloc[0])
    predicted_price = None
    try:
        predicted_price = float(symbol_predictions['close_predicted_price_value'].iloc[0])
    except:
        try:
            pred_val = symbol_predictions['close_predicted_price_value'].iloc[0]
            if isinstance(pred_val, str) and pred_val.startswith('(') and pred_val.endswith(')'):
                predicted_price = float(pred_val.strip('()').rstrip(','))
        except:
            pass
    
    logger.info(f"Current price: ${current_price:.2f}")
    if predicted_price:
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        logger.info(f"Predicted price: ${predicted_price:.2f} ({price_change_pct:+.2f}%)")
    
    # Run all strategies
    results = []
    logger.info(f"\n=== Strategy Recommendations (Base Capital: ${base_capital:,}) ===")
    
    for strategy in strategies:
        try:
            recommendation = strategy.get_recommendation(symbol_predictions, current_price)
            recommendation['symbol'] = symbol
            recommendation['current_price'] = current_price
            recommendation['predicted_price'] = predicted_price
            recommendation['timestamp'] = datetime.now().isoformat()
            
            results.append(recommendation)
            
            # Display recommendation
            logger.info(f"\n{strategy.name.upper()}:")
            logger.info(f"  Signal Strength: {recommendation['signal_strength']:.3f}")
            logger.info(f"  Recommendation: {recommendation['recommendation']}")
            logger.info(f"  Position Size: ${recommendation['position_size']:,}")
            logger.info(f"  Confidence: {recommendation['confidence']}")
            
        except Exception as e:
            logger.error(f"Error running {strategy.name}: {e}")
            continue
    
    # Save results to file
    save_strategy_results(symbol, results)
    
    # Generate summary
    generate_strategy_report(symbol, results, current_price, predicted_price)
    
    return results


def save_strategy_results(symbol, results):
    """Save strategy results to JSON file"""
    results_dir = Path(__file__).parent / "strategy_results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = results_dir / f"{symbol}_strategies_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {filename}")


def generate_strategy_report(symbol, results, current_price, predicted_price):
    """Generate markdown report of strategy results"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate consensus
    num_strategies = len(results)
    buy_signals = sum(1 for r in results if r['recommendation'] in ['STRONG_BUY', 'BUY', 'WEAK_BUY'])
    sell_signals = sum(1 for r in results if r['recommendation'] in ['STRONG_SELL', 'SELL', 'WEAK_SELL'])
    hold_signals = sum(1 for r in results if r['recommendation'] == 'HOLD')
    
    avg_position_size = np.mean([r['position_size'] for r in results])
    avg_signal_strength = np.mean([abs(r['signal_strength']) for r in results])
    
    # Price movement info
    price_change_info = ""
    if predicted_price:
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        price_change_info = f"**Predicted Move:** ${price_change:+.2f} ({price_change_pct:+.2f}%)"
    
    report_content = f"""# Enhanced Forecasting Strategies Report

**Symbol:** {symbol}  
**Generated:** {timestamp}  
**Current Price:** ${current_price:.2f}  
{price_change_info}

## Strategy Consensus

- **Buy Signals:** {buy_signals}/{num_strategies} strategies
- **Sell Signals:** {sell_signals}/{num_strategies} strategies  
- **Hold Signals:** {hold_signals}/{num_strategies} strategies
- **Average Signal Strength:** {avg_signal_strength:.3f}
- **Average Position Size:** ${avg_position_size:,.0f}

## Individual Strategy Results

"""
    
    # Sort results by signal strength (absolute value)
    sorted_results = sorted(results, key=lambda x: abs(x['signal_strength']), reverse=True)
    
    for i, result in enumerate(sorted_results, 1):
        direction = "↗️" if result['signal_strength'] > 0 else "↘️" if result['signal_strength'] < 0 else "➡️"
        
        report_content += f"""### #{i}: {result['strategy'].replace('_', ' ').title()} {direction}

- **Recommendation:** {result['recommendation']}
- **Signal Strength:** {result['signal_strength']:.3f}
- **Position Size:** ${result['position_size']:,}
- **Confidence:** {result['confidence']}

"""
    
    # Analysis and insights
    strongest_signal = max(results, key=lambda x: abs(x['signal_strength']))
    largest_position = max(results, key=lambda x: x['position_size'])
    
    report_content += f"""## Key Insights

1. **Strongest Signal:** {strongest_signal['strategy'].replace('_', ' ').title()} with {strongest_signal['signal_strength']:.3f} strength
2. **Largest Position:** {largest_position['strategy'].replace('_', ' ').title()} suggests ${largest_position['position_size']:,}
3. **Market Sentiment:** {"Bullish" if buy_signals > sell_signals else "Bearish" if sell_signals > buy_signals else "Neutral"}
4. **Strategy Agreement:** {max(buy_signals, sell_signals, hold_signals)}/{num_strategies} strategies agree

## Recommended Action

"""
    
    majority_threshold = max(2, num_strategies // 2)
    strong_threshold = max(3, (num_strategies * 2) // 3)
    
    if buy_signals >= strong_threshold:
        report_content += "**STRONG BUY** - Most strategies are bullish\n"
    elif buy_signals >= majority_threshold:
        report_content += "**BUY** - Majority of strategies are bullish\n"
    elif sell_signals >= strong_threshold:
        report_content += "**STRONG SELL** - Most strategies are bearish\n"
    elif sell_signals >= majority_threshold:
        report_content += "**SELL** - Majority of strategies are bearish\n"
    else:
        report_content += "**HOLD** - Mixed signals, wait for clearer opportunity\n"
    
    report_content += f"""
**Suggested Position Size:** ${avg_position_size:,.0f} (average across strategies)

---
*Generated by Enhanced Forecasting Strategies v1.0*
"""
    
    # Write report
    with open("strategy_findings.md", "w") as f:
        f.write(report_content)
    
    logger.info("Strategy report saved to strategy_findings.md")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python show_forecasts_strategies.py <symbol>")
        sys.exit(1)

    symbol = sys.argv[1].upper()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, format="{time} | {level} | {message}")
    
    # Run enhanced strategies
    results = run_forecasting_strategies(symbol, base_capital=10000)
    
    if results:
        print(f"\n✅ Analysis complete! Check strategy_findings.md for detailed report.")
    else:
        print("❌ Failed to run analysis - check logs for errors.")