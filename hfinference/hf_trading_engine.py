#!/usr/bin/env python3
"""
HuggingFace Trading Engine
Core engine for running inference and executing trades
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import yfinance as yf
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

@dataclass
class TradingSignal:
    """Trading signal from model"""
    timestamp: datetime
    symbol: str
    action: str  # 'buy', 'hold', 'sell'
    confidence: float
    predicted_price: float
    current_price: float
    expected_return: float
    position_size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'action': self.action,
            'confidence': self.confidence,
            'predicted_price': self.predicted_price,
            'current_price': self.current_price,
            'expected_return': self.expected_return,
            'position_size': self.position_size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }


class TransformerTradingModel(nn.Module):
    """Transformer model for inference"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        hidden_size = config['hidden_size']
        num_heads = config['num_heads']
        num_layers = config['num_layers']
        
        # Ensure compatibility
        if hidden_size % num_heads != 0:
            hidden_size = (hidden_size // num_heads) * num_heads
        
        self.hidden_size = hidden_size
        self.config = config
        
        # Model architecture (matching training)
        self.input_projection = nn.Linear(config['input_features'], hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=config.get('intermediate_size', hidden_size * 4),
            dropout=config.get('dropout', 0.1),
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Output heads
        self.price_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(hidden_size // 2, config['prediction_horizon'] * config['input_features'])
        )
        
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(hidden_size // 2, 3)  # Buy, Hold, Sell
        )
        
        # Positional encoding (learnable like optimized model)
        self.positional_encoding = nn.Parameter(
            torch.randn(1, config['sequence_length'], hidden_size) * 0.02,
            requires_grad=True
        )
    
    def _create_positional_encoding(self, seq_len: int, hidden_size: int) -> torch.Tensor:
        pe = torch.zeros(seq_len, hidden_size)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                           -(np.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # Project and add positional encoding
        hidden = self.input_projection(x)
        # Ensure positional encoding is on the same device
        hidden = hidden + self.positional_encoding[:, :seq_len, :].to(x.device)
        hidden = self.layer_norm(hidden)
        
        # Transformer encoding
        hidden = self.transformer(hidden)
        
        # Pool over sequence
        hidden = hidden.mean(dim=1)
        
        # Generate outputs
        price_predictions = self.price_head(hidden)
        action_logits = self.action_head(hidden)
        
        # Reshape price predictions
        price_predictions = price_predictions.view(
            batch_size, 
            self.config['prediction_horizon'], 
            self.config['input_features']
        )
        
        return {
            'price_predictions': price_predictions,
            'action_logits': action_logits,
            'action_probs': torch.softmax(action_logits, dim=-1)
        }


class HFTradingEngine:
    """Main trading engine using HuggingFace models"""
    
    def __init__(self, 
                 checkpoint_path: str,
                 config_path: Optional[str] = None,
                 device: str = 'auto'):
        """
        Initialize trading engine
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to config file (optional)
            device: Device to use ('cuda', 'cpu', 'auto')
        """
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Setup logging
        self.setup_logging()
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Load model
        self.model = self.load_model(checkpoint_path)
        
        # Initialize components
        self.data_processor = DataProcessor(self.config)
        self.risk_manager = RiskManager(self.config)
        self.position_sizer = PositionSizer(self.config)
        
        # Trading state
        self.positions = {}
        self.trade_history = []
        self.current_capital = self.config.get('initial_capital', 10000)
        
        self.logger.info(f"Trading engine initialized on {self.device}")
        
    def setup_logging(self):
        """Setup logging system"""
        log_dir = Path('hfinference/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Default configuration
            config = {
                'input_features': 21,
                'hidden_size': 512,
                'num_heads': 16,
                'num_layers': 8,
                'sequence_length': 60,
                'prediction_horizon': 5,
                'initial_capital': 10000,
                'max_position_size': 0.25,
                'stop_loss': 0.02,
                'take_profit': 0.05,
                'confidence_threshold': 0.6
            }
        
        return config
    
    def load_model(self, checkpoint_path: str) -> TransformerTradingModel:
        """Load trained model from checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Build model config from checkpoint and defaults
        model_config = {}
        
        # Try to get config from checkpoint
        if 'config' in checkpoint:
            if isinstance(checkpoint['config'], dict):
                # Check for nested model config
                if 'model' in checkpoint['config']:
                    model_config.update(checkpoint['config']['model'])
                else:
                    model_config.update(checkpoint['config'])
        
        # Merge with loaded config file
        if 'model' in self.config:
            for key, value in self.config['model'].items():
                if key not in model_config:
                    model_config[key] = value
        
        # Set defaults if still missing
        model_config.setdefault('hidden_size', 512)
        model_config.setdefault('num_heads', 16)
        model_config.setdefault('num_layers', 8)
        model_config.setdefault('intermediate_size', 2048)
        model_config.setdefault('dropout', 0.15)
        model_config.setdefault('input_features', 21)
        model_config.setdefault('sequence_length', 60)
        model_config.setdefault('prediction_horizon', 5)
        
        # Create model
        model = TransformerTradingModel(model_config)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            # Handle DataParallel wrapper
            state_dict = checkpoint['model_state_dict']
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Load with strict=False to handle missing/extra keys
            model.load_state_dict(state_dict, strict=False)
            
            # Log any missing keys
            model_state = model.state_dict()
            missing_keys = set(model_state.keys()) - set(state_dict.keys())
            unexpected_keys = set(state_dict.keys()) - set(model_state.keys())
            
            if missing_keys:
                self.logger.warning(f"Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                self.logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
        
        model.to(self.device)
        model.eval()
        
        self.logger.info(f"Model loaded from {checkpoint_path}")
        if 'global_step' in checkpoint:
            self.logger.info(f"Model trained for {checkpoint['global_step']} steps")
        
        return model
    
    @torch.no_grad()
    def generate_signal(self, symbol: str, data: pd.DataFrame) -> TradingSignal:
        """Generate trading signal for a symbol"""
        
        # Prepare data
        features = self.data_processor.prepare_features(data)
        
        if features is None:
            return None
        
        # Convert to tensor
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Run inference
        outputs = self.model(x)
        
        # Extract predictions
        price_pred = outputs['price_predictions'][0]  # [horizon, features]
        action_probs = outputs['action_probs'][0]  # [3]
        
        # Get current price and predicted price (handle both cases)
        if 'Close' in data.columns:
            current_price = data['Close'].iloc[-1]
        elif 'close' in data.columns:
            current_price = data['close'].iloc[-1]
        else:
            # Fallback to 4th column (typical Close position)
            current_price = data.iloc[-1, 3]
        
        # Use close price prediction (index 3)
        predicted_prices = price_pred[:, 3].cpu().numpy()
        predicted_price = predicted_prices.mean()  # Average over horizon
        
        # Denormalize if needed
        predicted_price = self.data_processor.denormalize_price(predicted_price, current_price)
        
        # Calculate expected return
        expected_return = (predicted_price - current_price) / current_price
        
        # Determine action
        action_idx = torch.argmax(action_probs).item()
        action_map = {0: 'buy', 1: 'hold', 2: 'sell'}
        action = action_map[action_idx]
        
        # Get confidence
        confidence = action_probs[action_idx].item()
        
        # Calculate position size
        position_size = self.position_sizer.calculate_size(
            confidence=confidence,
            expected_return=expected_return,
            current_capital=self.current_capital
        )
        
        # Set risk parameters (check trading config)
        trading_config = self.config.get('trading', {})
        stop_loss_pct = trading_config.get('stop_loss', 0.02)
        take_profit_pct = trading_config.get('take_profit', 0.05)
        
        stop_loss = current_price * (1 - stop_loss_pct) if action == 'buy' else None
        take_profit = current_price * (1 + take_profit_pct) if action == 'buy' else None
        
        signal = TradingSignal(
            timestamp=datetime.now(),
            symbol=symbol,
            action=action,
            confidence=confidence,
            predicted_price=predicted_price,
            current_price=current_price,
            expected_return=expected_return,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        return signal
    
    def execute_trade(self, signal: TradingSignal) -> Dict:
        """Execute trade based on signal"""
        
        # Check risk limits
        if not self.risk_manager.check_risk_limits(signal, self.positions, self.current_capital):
            self.logger.warning(f"Risk limits exceeded for {signal.symbol}")
            return {'status': 'rejected', 'reason': 'risk_limits'}
        
        # Check confidence threshold
        trading_config = self.config.get('trading', {})
        confidence_threshold = trading_config.get('confidence_threshold', 0.6)
        if signal.confidence < confidence_threshold:
            return {'status': 'rejected', 'reason': 'low_confidence'}
        
        trade_result = {
            'timestamp': signal.timestamp,
            'symbol': signal.symbol,
            'action': signal.action,
            'status': 'executed'
        }
        
        if signal.action == 'buy':
            # Calculate shares to buy
            position_value = self.current_capital * signal.position_size
            shares = int(position_value / signal.current_price)
            
            if shares > 0:
                cost = shares * signal.current_price
                self.current_capital -= cost
                
                self.positions[signal.symbol] = {
                    'shares': shares,
                    'entry_price': signal.current_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'entry_time': signal.timestamp
                }
                
                trade_result.update({
                    'shares': shares,
                    'price': signal.current_price,
                    'value': cost
                })
                
                self.logger.info(f"BUY {shares} shares of {signal.symbol} at ${signal.current_price:.2f}")
                
        elif signal.action == 'sell' and signal.symbol in self.positions:
            position = self.positions[signal.symbol]
            proceeds = position['shares'] * signal.current_price
            self.current_capital += proceeds
            
            profit = (signal.current_price - position['entry_price']) * position['shares']
            
            trade_result.update({
                'shares': position['shares'],
                'price': signal.current_price,
                'value': proceeds,
                'profit': profit
            })
            
            del self.positions[signal.symbol]
            
            self.logger.info(f"SELL {position['shares']} shares of {signal.symbol} at ${signal.current_price:.2f} (Profit: ${profit:.2f})")
        
        self.trade_history.append(trade_result)
        return trade_result
    
    def run_backtest(self, symbols: List[str], start_date: str, end_date: str) -> Dict:
        """Run backtest on historical data"""
        
        results = {
            'trades': [],
            'equity_curve': [],
            'metrics': {}
        }
        
        for symbol in symbols:
            self.logger.info(f"Backtesting {symbol}")
            
            # Get historical data
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            # Extract sequence_length from appropriate location
            seq_len = (self.config.get('sequence_length') or 
                      self.config.get('model', {}).get('sequence_length') or 
                      60)
            
            if len(data) < seq_len:
                continue
            
            # Sliding window backtest
            for i in range(seq_len, len(data)):
                window = data.iloc[i-seq_len:i]
                
                # Generate signal
                signal = self.generate_signal(symbol, window)
                
                if signal:
                    # Execute trade
                    trade_result = self.execute_trade(signal)
                    results['trades'].append(trade_result)
                
                # Update positions with current prices
                current_price = data['Close'].iloc[i]
                self.update_positions(symbol, current_price)
                
                # Record equity
                total_value = self.calculate_portfolio_value()
                results['equity_curve'].append({
                    'timestamp': data.index[i],
                    'value': total_value
                })
        
        # Calculate metrics
        results['metrics'] = self.calculate_metrics(results)
        
        return results
    
    def update_positions(self, symbol: str, current_price: float):
        """Update positions with stop loss and take profit"""
        
        if symbol in self.positions:
            position = self.positions[symbol]
            
            # Check stop loss
            if position.get('stop_loss') and current_price <= position['stop_loss']:
                self.logger.info(f"Stop loss triggered for {symbol}")
                signal = TradingSignal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    action='sell',
                    confidence=1.0,
                    predicted_price=current_price,
                    current_price=current_price,
                    expected_return=0,
                    position_size=0
                )
                self.execute_trade(signal)
            
            # Check take profit
            elif position.get('take_profit') and current_price >= position['take_profit']:
                self.logger.info(f"Take profit triggered for {symbol}")
                signal = TradingSignal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    action='sell',
                    confidence=1.0,
                    predicted_price=current_price,
                    current_price=current_price,
                    expected_return=0,
                    position_size=0
                )
                self.execute_trade(signal)
    
    def calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        total = self.current_capital
        
        for symbol, position in self.positions.items():
            # Get current price (in real trading, this would be live)
            # For now, use entry price as approximation
            total += position['shares'] * position['entry_price']
        
        return total
    
    def calculate_metrics(self, results: Dict) -> Dict:
        """Calculate performance metrics"""
        
        if not results['equity_curve']:
            return {}
        
        equity = [e['value'] for e in results['equity_curve']]
        returns = np.diff(equity) / equity[:-1]
        
        metrics = {
            'total_return': (equity[-1] - equity[0]) / equity[0],
            'sharpe_ratio': np.sqrt(252) * (np.mean(returns) / (np.std(returns) + 1e-8)),
            'max_drawdown': self._calculate_max_drawdown(equity),
            'num_trades': len([t for t in results['trades'] if t['status'] == 'executed']),
            'win_rate': self._calculate_win_rate(results['trades'])
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, equity: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = equity[0]
        max_dd = 0
        
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate"""
        profitable = [t for t in trades if t.get('profit', 0) > 0]
        total = [t for t in trades if 'profit' in t]
        
        if not total:
            return 0
        
        return len(profitable) / len(total)


class DataProcessor:
    """Process market data for model input"""
    
    def __init__(self, config: Dict):
        self.config = config
        # Extract sequence_length from various possible locations
        if 'sequence_length' in config:
            self.sequence_length = config['sequence_length']
        elif 'model' in config and 'sequence_length' in config['model']:
            self.sequence_length = config['model']['sequence_length']
        elif 'data' in config and 'sequence_length' in config['data']:
            self.sequence_length = config['data']['sequence_length']
        else:
            self.sequence_length = 60  # Default
        self.scaler = None
    
    def prepare_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare features from raw data"""
        
        if len(data) < self.sequence_length:
            return None
        
        # Take last sequence_length rows
        data = data.tail(self.sequence_length).copy()
        
        # Calculate features
        features = self.calculate_features(data)
        
        # Normalize
        features = self.normalize(features)
        
        return features
    
    def calculate_features(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate technical features"""
        
        df = data.copy()
        
        # Basic OHLCV features
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Price features
        df['Returns'] = df['Close'].pct_change()
        df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['PriceRange'] = (df['High'] - df['Low']) / df['Close']
        df['CloseToOpen'] = (df['Close'] - df['Open']) / df['Open']
        
        feature_cols.extend(['Returns', 'LogReturns', 'PriceRange', 'CloseToOpen'])
        
        # Volume features
        # Handle case where Volume might be multi-dimensional
        if isinstance(df['Volume'], pd.DataFrame):
            volume_col = df['Volume'].iloc[:, 0] if df['Volume'].shape[1] > 0 else df['Volume']
        else:
            volume_col = df['Volume']
        
        df['VolumeMA'] = volume_col.rolling(20).mean()
        df['VolumeRatio'] = volume_col / df['VolumeMA']
        
        feature_cols.extend(['VolumeMA', 'VolumeRatio'])
        
        # Moving averages
        for period in [5, 10, 20]:
            df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
            feature_cols.append(f'SMA_{period}')
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(20).std()
        feature_cols.append('Volatility')
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'])
        feature_cols.append('RSI')
        
        # MACD
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        feature_cols.extend(['MACD', 'MACD_Signal'])
        
        # Add Bollinger Bands to reach 21 features
        bb_period = 20
        bb_std = 2
        df['BB_Middle'] = df['Close'].rolling(bb_period).mean()
        df['BB_Upper'] = df['BB_Middle'] + bb_std * df['Close'].rolling(bb_period).std()
        df['BB_Lower'] = df['BB_Middle'] - bb_std * df['Close'].rolling(bb_period).std()
        feature_cols.extend(['BB_Middle', 'BB_Upper', 'BB_Lower'])
        
        # Fill NaN values (using new pandas syntax)
        df = df.ffill().fillna(0)
        
        # Select features
        features = df[feature_cols].values
        
        return features
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize features"""
        # Simple normalization (in production, use saved scaler from training)
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
        return features
    
    def denormalize_price(self, normalized_price: float, reference_price: float) -> float:
        """Denormalize predicted price"""
        # Simple denormalization
        # In production, this should match the training normalization
        return reference_price * (1 + normalized_price * 0.1)  # Assume 10% scaling


class RiskManager:
    """Manage trading risks"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_position_size = config.get('max_position_size', 0.25)
        self.max_positions = config.get('max_positions', 5)
        self.max_daily_loss = config.get('max_daily_loss', 0.05)
    
    def check_risk_limits(self, signal: TradingSignal, positions: Dict, capital: float) -> bool:
        """Check if trade meets risk limits"""
        
        # Check position size limit
        if signal.position_size > self.max_position_size:
            return False
        
        # Check number of positions
        if signal.action == 'buy' and len(positions) >= self.max_positions:
            return False
        
        # Check if we have enough capital
        if signal.action == 'buy':
            required_capital = signal.current_price * signal.position_size * capital
            if required_capital > capital:
                return False
        
        return True


class PositionSizer:
    """Calculate optimal position sizes"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_position = config.get('max_position_size', 0.25)
        self.kelly_fraction = 0.25  # Use 25% of Kelly for safety
    
    def calculate_size(self, confidence: float, expected_return: float, current_capital: float) -> float:
        """Calculate position size based on Kelly criterion"""
        
        if expected_return <= 0:
            return 0
        
        # Simple Kelly approximation
        # f = p/a - q/b where p=win_prob, q=loss_prob, a=loss_size, b=win_size
        win_prob = confidence
        loss_prob = 1 - confidence
        
        # Assume symmetric wins/losses for simplicity
        kelly = (win_prob - loss_prob) / 1.0
        
        # Apply safety factor
        kelly *= self.kelly_fraction
        
        # Cap at maximum
        position_size = min(max(kelly, 0), self.max_position)
        
        return position_size


if __name__ == "__main__":
    # Example usage
    print("HF Trading Engine initialized")
    print("Use run_trading.py to start trading")