from typing import List, Dict, Optional
from loguru import logger
import numpy as np


class RiskManager:
    def __init__(self, config):
        self.config = config
        self.stop_loss = config.stop_loss
        self.take_profit = config.take_profit
        self.max_drawdown_stop = config.max_drawdown_stop
        self.max_position_value = config.max_position_value
        self.max_positions = config.max_positions
        self.circuit_breaker_loss = config.circuit_breaker_loss
    
    def filter_recommendations(
        self, 
        recommendations: List[Dict],
        current_positions: Dict[str, dict],
        portfolio_tracker
    ) -> List[Dict]:
        """Filter recommendations based on risk criteria."""
        
        filtered = []
        
        # Get portfolio metrics
        metrics = portfolio_tracker.get_metrics()
        daily_return = metrics.get('daily_return', 0)
        max_drawdown = metrics.get('max_drawdown', 0)
        
        # Check circuit breaker
        if daily_return < -self.circuit_breaker_loss:
            logger.warning(f"Circuit breaker: Daily loss {daily_return:.2%} exceeds limit")
            return []
        
        # Check max drawdown
        if abs(max_drawdown) > self.max_drawdown_stop:
            logger.warning(f"Max drawdown {max_drawdown:.2%} exceeds limit")
            # Only allow risk-reducing trades
            for rec in recommendations:
                if rec['symbol'] in current_positions:
                    # Allow closing or reducing positions
                    current_pos = current_positions[rec['symbol']]
                    if (current_pos['side'] == 'long' and rec['side'] == 'sell') or \
                       (current_pos['side'] == 'short' and rec['side'] == 'buy'):
                        filtered.append(rec)
            return filtered
        
        # Normal filtering
        position_count = len(current_positions)
        
        for rec in recommendations:
            symbol = rec['symbol']
            
            # Check position limits
            if symbol not in current_positions and position_count >= self.max_positions:
                logger.debug(f"Skipping {symbol}: max positions reached")
                continue
            
            # Check position value limit
            if rec.get('last_price'):
                position_value = rec['position_size'] * rec['last_price'] * 10000  # Rough estimate
                if position_value > self.max_position_value:
                    logger.debug(f"Reducing position size for {symbol}: exceeds max value")
                    rec['position_size'] = self.max_position_value / (rec['last_price'] * 10000)
            
            # Check confidence threshold
            if rec.get('confidence', 0) < 0.3:
                logger.debug(f"Skipping {symbol}: low confidence {rec.get('confidence', 0):.2%}")
                continue
            
            # Avoid flipping positions too frequently
            if symbol in current_positions:
                current_pos = current_positions[symbol]
                if rec['side'] != current_pos['side']:
                    # Only flip if confidence is high
                    if rec.get('confidence', 0) < 0.6:
                        logger.debug(f"Skipping {symbol}: not confident enough to flip position")
                        continue
            
            filtered.append(rec)
        
        return filtered
    
    def check_circuit_breaker(self, portfolio_tracker) -> bool:
        """Check if circuit breaker should be triggered."""
        
        metrics = portfolio_tracker.get_metrics()
        daily_return = metrics.get('daily_return', 0)
        
        if daily_return < -self.circuit_breaker_loss:
            logger.error(f"CIRCUIT BREAKER TRIGGERED: Daily loss {daily_return:.2%}")
            return True
        
        return False
    
    def calculate_position_size(
        self,
        symbol: str,
        account_equity: float,
        current_positions: Dict[str, dict],
        volatility: Optional[float] = None
    ) -> float:
        """Calculate appropriate position size based on risk."""
        
        # Base position size
        base_size = self.config.max_position_size
        
        # Adjust for number of positions (diversification)
        position_count = len(current_positions)
        if position_count > 0:
            # Reduce size as we have more positions
            diversification_factor = 1.0 / (1 + position_count * 0.2)
            base_size *= diversification_factor
        
        # Adjust for volatility if available
        if volatility is not None:
            # Higher volatility = smaller position
            if volatility > 0.03:  # 3% daily volatility
                volatility_factor = 0.03 / volatility
                base_size *= min(volatility_factor, 1.0)
        
        # Ensure minimum and maximum bounds
        base_size = max(0.05, min(base_size, self.config.max_position_size))
        
        return base_size
    
    def should_close_position(
        self,
        position: dict,
        current_price: float,
        recommendation: Optional[Dict] = None
    ) -> bool:
        """Determine if a position should be closed."""
        
        entry_price = position['entry_price']
        position_return = (current_price - entry_price) / entry_price
        
        if position['side'] == 'short':
            position_return = -position_return
        
        # Check stop loss
        if self.stop_loss and position_return < -self.stop_loss:
            logger.info(f"Stop loss triggered: {position_return:.2%}")
            return True
        
        # Check take profit
        if self.take_profit and position_return > self.take_profit:
            logger.info(f"Take profit triggered: {position_return:.2%}")
            return True
        
        # Check if recommendation suggests opposite side
        if recommendation and recommendation['side'] != position['side']:
            if recommendation.get('confidence', 0) > 0.6:
                logger.info(f"Closing due to side change recommendation")
                return True
        
        return False