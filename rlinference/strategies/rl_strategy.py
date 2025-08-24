from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from loguru import logger


class RLTradingStrategy:
    def __init__(self, config, model_manager, data_preprocessor):
        self.config = config
        self.model_manager = model_manager
        self.data_preprocessor = data_preprocessor
        
        # Strategy parameters
        self.min_confidence = 0.3
        self.position_scaling = True
        self.use_ensemble = config.ensemble_predictions
        
    def generate_signals(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        current_position: Optional[Dict] = None
    ) -> Dict:
        """Generate trading signals for a symbol."""
        
        # Prepare observation
        current_pos_size = 0.0
        entry_price = 0.0
        
        if current_position:
            current_pos_size = current_position['qty']
            if current_position['side'] == 'short':
                current_pos_size = -current_pos_size
            entry_price = current_position.get('entry_price', 0.0)
        
        observation = self.data_preprocessor.prepare_observation(
            market_data,
            current_position=current_pos_size,
            current_balance=self.config.initial_balance,  # Simplified
            initial_balance=self.config.initial_balance,
            entry_price=entry_price
        )
        
        # Get model prediction
        action, value = self.model_manager.predict(
            symbol,
            observation,
            deterministic=True,
            use_ensemble=self.use_ensemble
        )
        
        # Generate signal
        signal = {
            'symbol': symbol,
            'action': action,
            'value': value,
            'confidence': abs(action),
            'timestamp': pd.Timestamp.now()
        }
        
        # Determine position recommendation
        if abs(action) < 0.1:  # Near zero = no position
            signal['recommendation'] = 'close'
            signal['side'] = 'neutral'
            signal['position_size'] = 0.0
        elif action > 0:
            signal['recommendation'] = 'long'
            signal['side'] = 'buy'
            signal['position_size'] = abs(action) * self.config.max_position_size
        else:
            signal['recommendation'] = 'short'
            signal['side'] = 'sell'
            signal['position_size'] = abs(action) * self.config.max_position_size
        
        # Add technical context
        latest = market_data.iloc[-1]
        signal['rsi'] = latest.get('RSI', 50)
        signal['volume_ratio'] = latest.get('Volume_Ratio', 1.0)
        signal['price'] = latest['Close']
        
        # Add trend information
        if 'SMA_20' in latest and 'SMA_50' in latest:
            signal['trend'] = 'bullish' if latest['SMA_20'] > latest['SMA_50'] else 'bearish'
        
        return signal
    
    def filter_signals(
        self,
        signals: List[Dict],
        current_positions: Dict[str, Dict]
    ) -> List[Dict]:
        """Filter and prioritize signals."""
        
        filtered = []
        
        for signal in signals:
            symbol = signal['symbol']
            
            # Check minimum confidence
            if signal['confidence'] < self.min_confidence:
                logger.debug(f"Skipping {symbol}: confidence {signal['confidence']:.2%} below threshold")
                continue
            
            # Check if we should act on this signal
            if symbol in current_positions:
                current_pos = current_positions[symbol]
                
                # Check if signal suggests closing
                if signal['recommendation'] == 'close':
                    filtered.append(signal)
                    continue
                
                # Check if signal suggests opposite side (flip position)
                if (current_pos['side'] == 'long' and signal['side'] == 'sell') or \
                   (current_pos['side'] == 'short' and signal['side'] == 'buy'):
                    # Only flip if confidence is high
                    if signal['confidence'] > 0.6:
                        filtered.append(signal)
                    continue
                
                # Check if we should increase position
                if current_pos['side'] == signal['recommendation']:
                    # Only increase if high confidence and position not too large
                    if signal['confidence'] > 0.7:
                        current_size = current_pos.get('qty', 0) * current_pos.get('current_price', 1)
                        if current_size < self.config.max_position_value * 0.8:
                            filtered.append(signal)
            else:
                # New position
                filtered.append(signal)
        
        # Sort by confidence
        filtered.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Limit to max positions
        max_new_positions = self.config.max_positions - len(current_positions)
        if max_new_positions > 0:
            filtered = filtered[:max_new_positions]
        
        return filtered
    
    def calculate_position_sizes(
        self,
        signals: List[Dict],
        account_equity: float,
        current_positions: Dict[str, Dict]
    ) -> List[Dict]:
        """Calculate actual position sizes based on account and risk."""
        
        # Calculate available capital
        positions_value = sum(
            pos.get('market_value', 0) for pos in current_positions.values()
        )
        available_capital = account_equity - positions_value
        
        # Allocate capital to signals
        total_confidence = sum(s['confidence'] for s in signals)
        
        for signal in signals:
            # Base allocation proportional to confidence
            if total_confidence > 0:
                allocation_pct = signal['confidence'] / total_confidence
            else:
                allocation_pct = 1.0 / len(signals)
            
            # Calculate position value
            position_value = available_capital * allocation_pct * signal['position_size']
            
            # Apply limits
            position_value = min(position_value, self.config.max_position_value)
            
            # Calculate shares
            if signal.get('price'):
                signal['target_shares'] = int(position_value / signal['price'])
                signal['target_value'] = position_value
            else:
                signal['target_shares'] = 0
                signal['target_value'] = 0
        
        return signals
    
    def generate_orders(
        self,
        signals: List[Dict],
        current_positions: Dict[str, Dict]
    ) -> List[Dict]:
        """Convert signals to actual orders."""
        
        orders = []
        
        for signal in signals:
            symbol = signal['symbol']
            
            # Determine order type and quantity
            if symbol in current_positions:
                current_pos = current_positions[symbol]
                
                if signal['recommendation'] == 'close':
                    # Close position
                    orders.append({
                        'symbol': symbol,
                        'action': 'close',
                        'order_type': 'market',
                        'reason': 'Model signal to close'
                    })
                elif signal['side'] != current_pos['side']:
                    # Flip position - close then open
                    orders.append({
                        'symbol': symbol,
                        'action': 'close',
                        'order_type': 'market',
                        'reason': 'Closing to flip position'
                    })
                    if signal['target_shares'] > 0:
                        orders.append({
                            'symbol': symbol,
                            'action': 'open',
                            'side': signal['side'],
                            'qty': signal['target_shares'],
                            'order_type': 'market',
                            'reason': f"Opening {signal['side']} position"
                        })
                else:
                    # Adjust position size
                    current_qty = current_pos.get('qty', 0)
                    target_qty = signal['target_shares']
                    
                    if target_qty > current_qty * 1.1:  # Increase by more than 10%
                        additional_qty = target_qty - current_qty
                        orders.append({
                            'symbol': symbol,
                            'action': 'add',
                            'side': signal['side'],
                            'qty': additional_qty,
                            'order_type': 'market',
                            'reason': 'Increasing position size'
                        })
            else:
                # Open new position
                if signal['target_shares'] > 0:
                    orders.append({
                        'symbol': symbol,
                        'action': 'open',
                        'side': signal['side'],
                        'qty': signal['target_shares'],
                        'order_type': 'market',
                        'reason': f"Opening {signal['side']} position",
                        'confidence': signal['confidence']
                    })
        
        return orders