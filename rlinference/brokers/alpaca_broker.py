from typing import List, Optional, Dict
from datetime import datetime
import pytz
from loguru import logger
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data import StockHistoricalDataClient

from src.date_utils import is_nyse_open_on_date


class AlpacaBroker:
    def __init__(self, config, paper: bool = True):
        self.config = config
        self.paper = paper
        
        # Initialize trading client
        self.trading_client = TradingClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper=paper
        )
        
        # Initialize data client
        self.data_client = StockHistoricalDataClient(
            api_key=config.api_key,
            secret_key=config.secret_key
        )
        
        logger.info(f"AlpacaBroker initialized ({'PAPER' if paper else 'LIVE'} mode)")
    
    def get_account(self):
        """Get account information."""
        try:
            return self.trading_client.get_account()
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def get_positions(self) -> List:
        """Get all open positions."""
        try:
            return self.trading_client.get_all_positions()
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_position(self, symbol: str):
        """Get position for specific symbol."""
        try:
            return self.trading_client.get_open_position(symbol)
        except Exception as e:
            logger.debug(f"No position for {symbol}: {e}")
            return None
    
    def close_position(self, symbol: str):
        """Close position for symbol."""
        try:
            return self.trading_client.close_position(symbol)
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return None
    
    def place_order(
        self,
        symbol: str,
        qty: Optional[int],
        side: str,
        order_type: str = 'market',
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = 'day'
    ):
        """Place an order."""
        try:
            # Convert side string to OrderSide enum
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            
            # Convert time_in_force to enum
            tif = TimeInForce.DAY
            if time_in_force.lower() == 'gtc':
                tif = TimeInForce.GTC
            elif time_in_force.lower() == 'ioc':
                tif = TimeInForce.IOC
            
            # Get position quantity if not specified (for exit orders)
            if qty is None:
                position = self.get_position(symbol)
                if position:
                    qty = abs(int(position.qty))
                else:
                    logger.warning(f"No position found for {symbol}, cannot place exit order")
                    return None
            
            # Create appropriate order request
            if order_type == 'market':
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif
                )
            elif order_type == 'limit':
                if limit_price is None:
                    raise ValueError("Limit price required for limit order")
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif,
                    limit_price=limit_price
                )
            elif order_type == 'stop':
                if stop_price is None:
                    raise ValueError("Stop price required for stop order")
                order_request = StopOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif,
                    stop_price=stop_price
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            # Submit order
            order = self.trading_client.submit_order(order_request)
            logger.info(f"Order placed: {order_type} {side} {qty} {symbol}")
            return order
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def cancel_order(self, order_id: str):
        """Cancel an order."""
        try:
            return self.trading_client.cancel_order_by_id(order_id)
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return None
    
    def get_orders(self, status: str = 'open'):
        """Get orders by status."""
        try:
            return self.trading_client.get_orders(filter={'status': status})
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []
    
    def is_market_open(self) -> bool:
        """Check if market is open."""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            # Fallback to time-based check
            return self._is_market_hours()
    
    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours.

        Uses exchange_calendars for accurate holiday detection.
        """
        est = pytz.timezone('US/Eastern')
        now = datetime.now(est)

        # Use calendar-based check for holidays and weekends
        if not is_nyse_open_on_date(now):
            return False

        # Market hours: 9:30 AM - 4:00 PM EST
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= now <= market_close
    
    def is_new_trading_day(self) -> bool:
        """Check if it's a new trading day."""
        # Simple implementation - could be enhanced with persistent state
        est = pytz.timezone('US/Eastern')
        now = datetime.now(est)
        return now.hour == 9 and now.minute < 35  # Just after market open