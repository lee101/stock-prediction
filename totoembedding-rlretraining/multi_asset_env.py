#!/usr/bin/env python3
"""
Multi-Asset Trading Environment for RL Training with Toto Embeddings
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import torch
from collections import defaultdict, deque
import random

# Import toto embedding system
import sys
sys.path.append('../totoembedding')
from embedding_model import TotoEmbeddingModel


class MultiAssetTradingEnv(gym.Env):
    """
    Multi-asset trading environment that uses toto embeddings
    for cross-asset relationship modeling
    """
    
    def __init__(
        self,
        data_dir: str = "trainingdata/train",
        symbols: List[str] = None,
        embedding_model_path: str = None,
        window_size: int = 30,
        initial_balance: float = 100000.0,
        max_positions: int = 5,
        transaction_cost: float = 0.001,
        spread_pct: float = 0.0001,
        slippage_pct: float = 0.0001,
        min_commission: float = 1.0,
        correlation_lookback: int = 252,  # Days for correlation calculation
        rebalance_frequency: int = 1,  # Steps between rebalancing (1 = every step, 1440 = daily)
        max_position_size: float = 0.6,  # Maximum position size per asset
        confidence_threshold: float = 0.4,  # Minimum confidence for trades
        diversification_bonus: float = 0.001,  # Reward for diversification
        **kwargs
    ):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        
        # Default symbols from your trainingdata
        if symbols is None:
            symbols = [
                'AAPL', 'ADBE', 'ADSK', 'BTCUSD', 'COIN', 'COUR',
                'ETHUSD', 'GOOG', 'LTCUSD', 'MSFT', 'NFLX', 'NVDA',
                'PYPL', 'SAP', 'SONY', 'TSLA', 'U', 'UNIUSD'
            ]
        
        self.symbols = symbols
        self.num_assets = len(symbols)
        self.symbol_to_id = {sym: i for i, sym in enumerate(symbols)}
        
        # Classify assets by type (crypto trades 24/7, stocks only during market hours)
        self.crypto_symbols = {s for s in symbols if any(crypto in s.upper() for crypto in ['USD', 'BTC', 'ETH', 'LTC', 'UNI', 'PAXG', 'DOGE', 'DOT', 'ADA', 'ALGO', 'ATOM', 'AVAX', 'LINK', 'MATIC', 'SHIB', 'SOL', 'XLM', 'XRP'])}
        self.stock_symbols = set(symbols) - self.crypto_symbols
        
        # Environment parameters
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.max_positions = max_positions
        self.max_position_size = max_position_size
        self.transaction_cost = transaction_cost
        self.confidence_threshold = confidence_threshold
        self.diversification_bonus = diversification_bonus
        self.spread_pct = spread_pct
        self.slippage_pct = slippage_pct
        self.min_commission = min_commission
        self.correlation_lookback = correlation_lookback
        self.rebalance_frequency = rebalance_frequency
        self.steps_since_rebalance = 0  # Track steps since last rebalance
        
        # Load toto embedding model
        self.embedding_model = None
        if embedding_model_path:
            self.embedding_model = self._load_embedding_model(embedding_model_path)
        
        # Load market data
        self.market_data = self._load_market_data()
        self.prepare_features()
        
        # Calculate data length (minimum across all symbols)
        self.data_length = min(len(df) for df in self.market_data.values()) - window_size - 1
        
        # Action space: continuous allocation weights for each asset [-1, 1]
        # -1 = max short, 0 = no position, 1 = max long
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self.num_assets,), 
            dtype=np.float32
        )
        
        # Observation space: embeddings + portfolio state + market features
        embedding_dim = 128  # From toto embedding model
        portfolio_dim = self.num_assets * 3  # positions, values, pnl per asset
        market_dim = self.num_assets * 10  # price features per asset
        correlation_dim = self.num_assets * (self.num_assets - 1) // 2  # Pairwise correlations
        
        obs_dim = embedding_dim + portfolio_dim + market_dim + correlation_dim + 10  # +10 for global features
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.reset()
    
    def _load_embedding_model(self, model_path: str) -> TotoEmbeddingModel:
        """Load the toto embedding model"""
        try:
            # You'll need to specify the pretrained model path
            pretrained_path = "training/models/modern_best_sharpe.pth"  # Adjust as needed
            model = TotoEmbeddingModel(
                pretrained_model_path=pretrained_path,
                num_symbols=len(self.symbols)
            )
            
            # Load embedding model weights if they exist
            if Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
            
            model.eval()
            return model
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
            return None
    
    def _load_market_data(self) -> Dict[str, pd.DataFrame]:
        """Load market data for all symbols"""
        market_data = {}
        
        for symbol in self.symbols:
            filepath = self.data_dir / f"{symbol}.csv"
            if filepath.exists():
                df = pd.read_csv(filepath, parse_dates=['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                market_data[symbol] = df
            else:
                print(f"Warning: Data file not found for {symbol}")
        
        return market_data
    
    def prepare_features(self):
        """Prepare technical features for all symbols"""
        for symbol, df in self.market_data.items():
            # Price features
            df['Returns'] = df['Close'].pct_change()
            df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['HL_Ratio'] = (df['High'] - df['Low']) / df['Close']
            df['OC_Ratio'] = (df['Open'] - df['Close']) / df['Close']
            
            # Moving averages and ratios
            for window in [5, 10, 20, 50]:
                df[f'MA_{window}'] = df['Close'].rolling(window).mean()
                df[f'MA_Ratio_{window}'] = df['Close'] / df[f'MA_{window}']
            
            # Volatility features
            df['Volatility_5'] = df['Returns'].rolling(5).std()
            df['Volatility_20'] = df['Returns'].rolling(20).std()
            
            # Volume features (if available)
            if 'Volume' in df.columns:
                df['Volume_MA'] = df['Volume'].rolling(20).mean()
                df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            else:
                df['Volume_Ratio'] = 1.0
            
            # RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta).where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Time features
            df['Hour'] = df['timestamp'].dt.hour
            df['DayOfWeek'] = df['timestamp'].dt.dayofweek
            df['Month'] = df['timestamp'].dt.month
            
            # Fill NaN values
            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)
            
            self.market_data[symbol] = df
    
    def reset(self) -> np.ndarray:
        """Reset the environment"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}  # Position sizes (-1 to 1)
        self.position_values = {symbol: 0.0 for symbol in self.symbols}  # Dollar values
        self.entry_prices = {symbol: 0.0 for symbol in self.symbols}
        
        # Portfolio tracking
        self.portfolio_history = []
        self.trades_history = []
        self.returns_history = []
        self.correlation_matrix = np.eye(self.num_assets)
        
        # Performance metrics
        self.total_trades = 0
        self.total_fees = 0.0
        self.steps_since_rebalance = 0  # Reset rebalance counter
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        action = np.clip(action, -1.0, 1.0)
        
        # Get current prices
        current_prices = self._get_current_prices()
        
        # Calculate current portfolio value
        portfolio_value = self._calculate_portfolio_value(current_prices)
        
        # Only execute trades if it's time to rebalance
        can_rebalance = self.steps_since_rebalance >= self.rebalance_frequency
        if can_rebalance:
            # Update positions based on action
            reward, fees = self._execute_trades(action, current_prices, portfolio_value)
            self.steps_since_rebalance = 0
        else:
            # No trading allowed yet
            reward, fees = 0.0, 0.0
            self.steps_since_rebalance += 1
        
        # Update portfolio tracking
        new_portfolio_value = self._calculate_portfolio_value(current_prices)
        self.balance = new_portfolio_value
        
        # Calculate returns
        if len(self.portfolio_history) > 0:
            portfolio_return = (new_portfolio_value - self.portfolio_history[-1]) / self.portfolio_history[-1]
            self.returns_history.append(portfolio_return)
        else:
            portfolio_return = 0.0
        
        self.portfolio_history.append(new_portfolio_value)
        
        # Update correlation matrix periodically
        if self.current_step % 20 == 0:
            self._update_correlation_matrix()
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.data_length
        
        # Calculate reward (risk-adjusted returns)
        reward = self._calculate_reward(portfolio_return, fees)
        
        # Get next observation
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        
        info = {
            'portfolio_value': new_portfolio_value,
            'portfolio_return': portfolio_return,
            'total_fees': self.total_fees,
            'num_trades': self.total_trades,
            'positions': self.positions.copy(),
            'balance': self.balance
        }
        
        return obs, reward, done, info
    
    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols"""
        prices = {}
        idx = self.current_step + self.window_size
        
        for symbol in self.symbols:
            if idx < len(self.market_data[symbol]):
                prices[symbol] = self.market_data[symbol].iloc[idx]['Close']
            else:
                # Use last available price
                prices[symbol] = self.market_data[symbol].iloc[-1]['Close']
        
        return prices
    
    def _calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        total_value = 0.0
        
        for symbol in self.symbols:
            if abs(self.positions[symbol]) > 1e-6:  # Has position
                position_value = abs(self.positions[symbol]) * self.balance * current_prices[symbol] / current_prices[symbol]  # Simplified
                if self.positions[symbol] > 0:  # Long position
                    if self.entry_prices[symbol] > 0:
                        pnl = (current_prices[symbol] - self.entry_prices[symbol]) / self.entry_prices[symbol]
                        position_value = self.position_values[symbol] * (1 + pnl)
                else:  # Short position
                    if self.entry_prices[symbol] > 0:
                        pnl = (self.entry_prices[symbol] - current_prices[symbol]) / self.entry_prices[symbol]
                        position_value = abs(self.position_values[symbol]) * (1 + pnl)
                
                total_value += position_value
            else:
                total_value += self.position_values[symbol]  # Cash portion
        
        # Add remaining cash
        used_balance = sum(abs(self.position_values[symbol]) for symbol in self.symbols)
        total_value += max(0, self.initial_balance - used_balance)
        
        return total_value
    
    def _execute_trades(self, target_positions: np.ndarray, prices: Dict[str, float], portfolio_value: float) -> Tuple[float, float]:
        """Execute trades to reach target positions"""
        total_fees = 0.0
        total_reward = 0.0
        
        for i, symbol in enumerate(self.symbols):
            target_pos = target_positions[i]
            current_pos = self.positions[symbol]
            
            # Check if we need to trade
            position_change = abs(target_pos - current_pos)
            if position_change > 0.01:  # Minimum change threshold
                
                # Calculate trade size - use optimized max position size
                max_trade_pct = self.max_position_size / self.max_positions  # Distribute across positions
                trade_value = position_change * portfolio_value * max_trade_pct
                
                # Calculate fees
                commission = max(self.transaction_cost * trade_value, self.min_commission)
                spread_cost = self.spread_pct * trade_value
                slippage_cost = self.slippage_pct * trade_value
                
                total_fees += commission + spread_cost + slippage_cost
                
                # Update position
                self.positions[symbol] = target_pos
                self.position_values[symbol] = target_pos * portfolio_value * 0.2
                self.entry_prices[symbol] = prices[symbol]
                
                self.total_trades += 1
                self.total_fees += total_fees
                
                # Record trade
                self.trades_history.append({
                    'step': self.current_step,
                    'symbol': symbol,
                    'action': target_pos,
                    'price': prices[symbol],
                    'fees': commission + spread_cost + slippage_cost
                })
        
        return total_reward, total_fees
    
    def _calculate_reward(self, portfolio_return: float, fees: float) -> float:
        """Calculate reward for the step"""
        # Base reward from returns
        reward = portfolio_return
        
        # Penalize fees
        fee_penalty = fees / self.initial_balance
        reward -= fee_penalty
        
        # Risk adjustment
        if len(self.returns_history) > 20:
            volatility = np.std(self.returns_history[-20:])
            if volatility > 0:
                reward = reward / (volatility + 1e-8)
        
        # Diversification bonus - reward having multiple positions up to max_positions
        active_positions = sum(1 for pos in self.positions.values() if abs(pos) > 0.1)
        diversification_bonus = min(active_positions / self.max_positions, 1.0) * self.diversification_bonus
        reward += diversification_bonus
        
        # Concentration penalty - penalize over-concentration in few assets
        position_values = [abs(pos) for pos in self.positions.values()]
        if position_values:
            concentration = max(position_values) / sum(position_values) if sum(position_values) > 0 else 0
            concentration_penalty = max(0, concentration - (1.0 / self.max_positions)) * 0.01
            reward -= concentration_penalty
        
        return reward
    
    def _update_correlation_matrix(self):
        """Update correlation matrix between assets"""
        if self.current_step < self.correlation_lookback:
            return
        
        # Get recent returns for all symbols
        returns_data = []
        for symbol in self.symbols:
            start_idx = max(0, self.current_step + self.window_size - self.correlation_lookback)
            end_idx = self.current_step + self.window_size
            
            symbol_returns = self.market_data[symbol].iloc[start_idx:end_idx]['Returns'].values
            returns_data.append(symbol_returns)
        
        # Calculate correlation matrix
        returns_array = np.array(returns_data)
        self.correlation_matrix = np.corrcoef(returns_array)
        
        # Handle NaN values
        self.correlation_matrix = np.nan_to_num(self.correlation_matrix, nan=0.0)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        features = []
        
        # Get toto embeddings if model is available
        if self.embedding_model is not None:
            embedding_features = self._get_embedding_features()
            features.extend(embedding_features)
        else:
            # Fallback to zeros if no embedding model
            features.extend(np.zeros(128))
        
        # Portfolio state features
        portfolio_features = []
        current_prices = self._get_current_prices()
        
        for symbol in self.symbols:
            # Position info
            portfolio_features.append(self.positions[symbol])
            portfolio_features.append(self.position_values[symbol] / self.initial_balance)
            
            # P&L info
            if abs(self.positions[symbol]) > 1e-6 and self.entry_prices[symbol] > 0:
                pnl = (current_prices[symbol] - self.entry_prices[symbol]) / self.entry_prices[symbol]
                if self.positions[symbol] < 0:  # Short position
                    pnl = -pnl
            else:
                pnl = 0.0
            portfolio_features.append(pnl)
        
        features.extend(portfolio_features)
        
        # Market features for each asset
        market_features = self._get_market_features()
        features.extend(market_features)
        
        # Correlation features (upper triangle of correlation matrix)
        correlation_features = []
        for i in range(self.num_assets):
            for j in range(i+1, self.num_assets):
                correlation_features.append(self.correlation_matrix[i, j])
        features.extend(correlation_features)
        
        # Global features
        global_features = [
            len(self.portfolio_history) / 1000.0,  # Normalized time
            self.balance / self.initial_balance,  # Balance ratio
            self.total_fees / self.initial_balance,  # Cumulative fees
            self.total_trades / 100.0,  # Normalized trade count
            np.mean(self.returns_history[-20:]) if len(self.returns_history) >= 20 else 0.0,  # Recent avg return
            np.std(self.returns_history[-20:]) if len(self.returns_history) >= 20 else 0.0,  # Recent volatility
            sum(1 for pos in self.positions.values() if abs(pos) > 0.1) / self.max_positions,  # Position utilization
            max(self.positions.values()) if self.positions else 0.0,  # Max position
            min(self.positions.values()) if self.positions else 0.0,  # Min position
            np.mean(list(self.positions.values())) if self.positions else 0.0  # Mean position
        ]
        features.extend(global_features)
        
        return np.array(features, dtype=np.float32)
    
    def _get_embedding_features(self) -> List[float]:
        """Get toto embedding features"""
        if self.embedding_model is None:
            return [0.0] * 128
        
        try:
            # Prepare data for embedding model
            idx = self.current_step + self.window_size
            
            # Use first symbol as primary (could be enhanced to use all symbols)
            primary_symbol = self.symbols[0]
            symbol_data = self.market_data[primary_symbol]
            
            if idx >= len(symbol_data):
                return [0.0] * 128
            
            # Get window of price data
            start_idx = max(0, idx - self.window_size)
            window_data = symbol_data.iloc[start_idx:idx]
            
            # Prepare features
            price_features = ['Open', 'High', 'Low', 'Close', 'Returns', 'HL_Ratio', 'OC_Ratio', 
                            'MA_Ratio_5', 'MA_Ratio_10', 'MA_Ratio_20', 'Volatility_20']
            
            price_data = torch.tensor(
                window_data[price_features].values, 
                dtype=torch.float32
            ).unsqueeze(0)  # Add batch dimension
            
            # Symbol ID
            symbol_id = torch.tensor([self.symbol_to_id[primary_symbol]], dtype=torch.long)
            
            # Timestamp features
            current_row = symbol_data.iloc[idx-1]
            timestamps = torch.tensor([[
                current_row.get('Hour', 12),
                current_row.get('DayOfWeek', 1),
                current_row.get('Month', 6)
            ]], dtype=torch.long)
            
            # Market regime (simplified)
            market_regime = torch.tensor([0], dtype=torch.long)  # Neutral regime
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.embedding_model(
                    price_data=price_data,
                    symbol_ids=symbol_id,
                    timestamps=timestamps,
                    market_regime=market_regime
                )
                embeddings = outputs['embeddings'].squeeze(0).numpy()
            
            return embeddings.tolist()
            
        except Exception as e:
            print(f"Error getting embedding features: {e}")
            return [0.0] * 128
    
    def _get_market_features(self) -> List[float]:
        """Get market features for all assets"""
        features = []
        idx = self.current_step + self.window_size
        
        for symbol in self.symbols:
            symbol_data = self.market_data[symbol]
            
            if idx >= len(symbol_data):
                # Use last available data
                row = symbol_data.iloc[-1]
            else:
                row = symbol_data.iloc[idx]
            
            # Price features
            symbol_features = [
                row.get('Returns', 0.0),
                row.get('HL_Ratio', 0.0),
                row.get('OC_Ratio', 0.0),
                row.get('MA_Ratio_5', 1.0),
                row.get('MA_Ratio_10', 1.0),
                row.get('MA_Ratio_20', 1.0),
                row.get('Volatility_5', 0.0),
                row.get('Volatility_20', 0.0),
                row.get('RSI', 50.0) / 100.0,  # Normalize RSI
                row.get('Volume_Ratio', 1.0)
            ]
            
            features.extend(symbol_features)
        
        return features
    
    def get_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        if len(self.portfolio_history) < 2:
            return {}
        
        returns = np.array(self.returns_history)
        portfolio_values = np.array(self.portfolio_history)
        
        total_return = (portfolio_values[-1] - self.initial_balance) / self.initial_balance
        
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            
            # Max drawdown calculation
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - peak) / peak
            max_drawdown = np.min(drawdown)
        else:
            sharpe_ratio = 0.0
            max_drawdown = 0.0
        
        # Win rate
        winning_trades = sum(1 for r in returns if r > 0)
        win_rate = winning_trades / len(returns) if len(returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0,
            'win_rate': win_rate,
            'num_trades': self.total_trades,
            'total_fees': self.total_fees,
            'final_balance': portfolio_values[-1]
        }
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            current_value = self.portfolio_history[-1] if self.portfolio_history else self.initial_balance
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${current_value:,.2f}")
            print(f"Active Positions: {sum(1 for p in self.positions.values() if abs(p) > 0.1)}")
            print(f"Total Trades: {self.total_trades}")
            print(f"Total Fees: ${self.total_fees:.2f}")
            
            # Show top positions
            active_positions = [(sym, pos) for sym, pos in self.positions.items() if abs(pos) > 0.1]
            if active_positions:
                print("Active Positions:")
                for sym, pos in sorted(active_positions, key=lambda x: abs(x[1]), reverse=True)[:5]:
                    print(f"  {sym}: {pos:.3f}")
