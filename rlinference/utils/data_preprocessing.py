import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from pathlib import Path
import yfinance as yf
from datetime import datetime, timedelta
from loguru import logger


class DataPreprocessor:
    def __init__(self, window_size: int = 30, features: List[str] = None):
        self.window_size = window_size
        self.features = features or [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Returns', 'RSI', 'Volume_Ratio',
            'High_Low_Ratio', 'Close_Open_Ratio'
        ]
        self.price_features = ['Open', 'High', 'Low', 'Close']
        
    def fetch_realtime_data(self, symbol: str, period: str = "2mo") -> pd.DataFrame:
        """Fetch real-time data using yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                logger.error(f"No data fetched for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            df.columns = [col.replace(' ', '') for col in df.columns]
            
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators and features."""
        df = df.copy()
        
        # Basic features
        df['Returns'] = df['Close'].pct_change()
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Volume features
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Price ratios
        df['High_Low_Ratio'] = df['High'] / (df['Low'] + 1e-10)
        df['Close_Open_Ratio'] = df['Close'] / (df['Open'] + 1e-10)
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-10)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def prepare_observation(
        self, 
        df: pd.DataFrame, 
        current_position: float = 0.0,
        current_balance: float = 10000.0,
        initial_balance: float = 10000.0,
        entry_price: float = 0.0
    ) -> np.ndarray:
        """Prepare observation for RL model."""
        
        # Get last window_size rows
        if len(df) < self.window_size:
            logger.warning(f"Not enough data: {len(df)} < {self.window_size}")
            # Pad with zeros if not enough data
            padding = self.window_size - len(df)
            df = pd.concat([pd.DataFrame(0, index=range(padding), columns=df.columns), df])
        
        window_data = df.tail(self.window_size)
        
        # Extract available features
        available_features = [f for f in self.features if f in window_data.columns]
        feature_data = window_data[available_features].values
        
        # Normalize features
        normalized_data = (feature_data - np.mean(feature_data, axis=0)) / (np.std(feature_data, axis=0) + 1e-8)
        
        # Add position info
        position_info = np.full((self.window_size, 1), current_position)
        
        # Add balance ratio
        balance_ratio = current_balance / initial_balance
        balance_info = np.full((self.window_size, 1), balance_ratio)
        
        # Add P&L info
        if current_position != 0 and entry_price > 0:
            current_price = window_data['Close'].iloc[-1]
            pnl = (current_price - entry_price) / entry_price * current_position
        else:
            pnl = 0.0
        pnl_info = np.full((self.window_size, 1), pnl)
        
        # Combine all features
        observation = np.concatenate([
            normalized_data,
            position_info,
            balance_info,
            pnl_info
        ], axis=1)
        
        return observation.astype(np.float32)
    
    def get_latest_prices(self, symbol: str) -> Tuple[float, float, float]:
        """Get latest bid, ask, and last prices."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Try to get bid/ask from info
            bid = info.get('bid', None)
            ask = info.get('ask', None)
            last = info.get('regularMarketPrice', info.get('price', None))
            
            # Fallback to last price if bid/ask not available
            if bid is None or ask is None:
                if last is not None:
                    spread = last * 0.001  # Assume 0.1% spread
                    bid = last - spread/2
                    ask = last + spread/2
                else:
                    # Get from recent history
                    hist = ticker.history(period="1d", interval="1m")
                    if not hist.empty:
                        last = hist['Close'].iloc[-1]
                        bid = last * 0.999
                        ask = last * 1.001
                    else:
                        return None, None, None
            
            return float(bid), float(ask), float(last)
            
        except Exception as e:
            logger.error(f"Error getting prices for {symbol}: {e}")
            return None, None, None