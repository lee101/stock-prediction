#!/usr/bin/env python3
"""
Production Trading Engine - Real-world ready inference system
Matches training setup from hftraining and integrates best practices from predict_stock_e2e
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
# Make yfinance optional; provide a stub so tests can patch yf.download
try:
    import yfinance as yf  # Optional; may be unavailable in restricted envs
except Exception:
    class _YFStub:
        @staticmethod
        def download(*args, **kwargs):
            raise RuntimeError("yfinance unavailable and no local data provided")

        class Ticker:
            def __init__(self, *args, **kwargs):
                pass

            def history(self, *args, **kwargs):
                raise RuntimeError("yfinance unavailable and no local data provided")

    yf = _YFStub
from dataclasses import dataclass, field
try:
    import joblib
except Exception:
    joblib = None
import warnings
import traceback
from collections import deque
import threading
import time

warnings.filterwarnings('ignore')

# Import the model and data processor from training
import sys
sys.path.append(str(Path(__file__).parent.parent))
from hftraining.data_utils import StockDataProcessor
import hfshared

# Import the HF training model directly
try:
    from hftraining.hf_trainer import TransformerTradingModel, HFTrainingConfig
except ImportError:
    # Fallback model definition if training module not available
    HFTrainingConfig = None
    class TransformerTradingModel(nn.Module):
        def __init__(self, config, input_dim=5):
            super().__init__()
            self.config = config
            # Handle both dict and object configs
            if isinstance(config, dict):
                hidden_size = config.get('hidden_size', 512)
                num_heads = config.get('num_heads', 8)
                num_layers = config.get('num_layers', 4)
            else:
                hidden_size = getattr(config, 'hidden_size', 512)
                num_heads = getattr(config, 'num_heads', 8)
                num_layers = getattr(config, 'num_layers', 4)
            
            # Minimal implementation for testing
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    batch_first=True
                ),
                num_layers=num_layers
            )
        
        def forward(self, x):
            # Minimal forward pass for testing
            batch_size = x.shape[0]
            return {
                'price_predictions': torch.randn(batch_size, 5, 30),
                'action_logits': torch.randn(batch_size, 3)
            }


@dataclass
class EnhancedTradingSignal:
    """Enhanced trading signal with production features"""
    timestamp: datetime
    symbol: str
    action: str  # 'buy', 'hold', 'sell'
    confidence: float
    predicted_price: float
    current_price: float
    expected_return: float
    position_size: float
    
    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    
    # Advanced features
    volatility: float = 0.0
    market_regime: str = 'normal'  # 'bullish', 'bearish', 'volatile', 'normal'
    signal_strength: float = 0.0
    risk_score: float = 0.0
    
    # Price predictions
    price_targets: Dict[str, float] = field(default_factory=dict)
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    
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
            'take_profit': self.take_profit,
            'trailing_stop': self.trailing_stop,
            'volatility': self.volatility,
            'market_regime': self.market_regime,
            'signal_strength': self.signal_strength,
            'risk_score': self.risk_score,
            'price_targets': self.price_targets,
            'support_levels': self.support_levels,
            'resistance_levels': self.resistance_levels
        }


@dataclass
class Position:
    """Track open positions"""
    symbol: str
    shares: int
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    high_water_mark: Optional[float] = None
    last_update: Optional[datetime] = None
    
    def update_trailing_stop(self, current_price: float, trail_percent: float = 0.02):
        """Update trailing stop based on high water mark"""
        if current_price > (self.high_water_mark or self.entry_price):
            self.high_water_mark = current_price
            self.trailing_stop = current_price * (1 - trail_percent)
        self.last_update = datetime.now()
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L"""
        return (current_price - self.entry_price) * self.shares
    
    def get_return(self, current_price: float) -> float:
        """Calculate return percentage"""
        return (current_price - self.entry_price) / self.entry_price


class ProductionTradingEngine:
    """Production-ready trading engine with advanced features"""
    
    def __init__(self, 
                 checkpoint_path: str,
                 config: Optional[Dict] = None,
                 config_path: Optional[str] = None,
                 device: str = 'auto',
                 live_trading: bool = False,
                 paper_trading: bool = True,
                 mode: Optional[str] = None):
        """
        Initialize production trading engine
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            config_path: Path to configuration file
            device: Device to use ('cuda', 'cpu', 'auto')
            live_trading: Enable live trading (requires broker integration)
            paper_trading: Enable paper trading mode
        """
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Trading modes
        self.live_trading = live_trading
        self.paper_trading = paper_trading
        
        # Setup logging
        self.setup_logging()
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Load model
        self.model = self.load_model(checkpoint_path)
        
        # Initialize data processor (matches training)
        self.data_processor = StockDataProcessor(
            sequence_length=self.config['model']['sequence_length'],
            prediction_horizon=self.config['model']['prediction_horizon']
        )
        # Try to load training scalers to enforce consistency with training
        self._processor_path = (
            self.config.get('data', {}).get('processor_path')
            or str(Path(checkpoint_path).parent / 'data_processor.pkl')
        )
        self.feature_names: List[str] = []
        if self._processor_path and Path(self._processor_path).exists():
            try:
                processor_data = hfshared.load_processor(self._processor_path)
                if processor_data:
                    # Set scalers on data processor if available
                    self.data_processor.scalers = processor_data.get('scalers', {})
                    self.data_processor.feature_names = processor_data.get('feature_names', [])
                    self.feature_names = list(processor_data.get('feature_names', []))
                    self.logger.info(f"Loaded training scalers from {self._processor_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load training scalers: {e}")
        
        # Portfolio management
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.current_capital = self.config['trading']['initial_capital']
        self.starting_capital = self.current_capital
        
        # Risk management
        self.daily_loss_limit = self.config['trading']['max_daily_loss'] * self.starting_capital
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'current_drawdown': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Market regime detection
        self.market_regime = 'normal'
        self.regime_history = deque(maxlen=100)
        
        # Signal buffer for ensemble/confirmation
        self.signal_buffer: Dict[str, deque] = {}
        
        self.logger.info(f"Production engine initialized on {self.device}")
        self.logger.info(f"Live trading: {live_trading}, Paper trading: {paper_trading}")
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path('hfinference/logs/production')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler for all logs
        file_handler = logging.FileHandler(
            log_dir / f'production_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # File handler for trades only
        trade_handler = logging.FileHandler(
            log_dir / f'trades_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        trade_handler.setFormatter(simple_formatter)
        trade_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(simple_formatter)
        console_handler.setLevel(logging.INFO)
        
        # Setup loggers
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.trade_logger = logging.getLogger(f"{__name__}.trades")
        self.trade_logger.setLevel(logging.INFO)
        self.trade_logger.addHandler(trade_handler)
        
    def load_config(self, config_path: Optional[str]) -> Dict:
        """Load and validate configuration"""
        
        # Start with default config
        config = {
            'model': {
                'input_features': 30,  # More features than base model
                'hidden_size': 512,
                'num_heads': 16,
                'num_layers': 8,
                'intermediate_size': 2048,
                'dropout': 0.15,
                'sequence_length': 60,
                'prediction_horizon': 5
            },
            'trading': {
                'initial_capital': 100000,
                'max_position_size': 0.15,  # Conservative position sizing
                'max_positions': 10,
                'stop_loss': 0.02,
                'take_profit': 0.05,
                'trailing_stop': 0.015,
                'confidence_threshold': 0.65,
                'risk_per_trade': 0.01,
                'max_daily_loss': 0.02,
                'max_drawdown': 0.10,
                'kelly_fraction': 0.25
            },
            'strategy': {
                'use_ensemble': True,
                'ensemble_size': 3,
                'confirmation_required': 2,
                'use_technical_confirmation': True,
                'market_regime_filter': True,
                'volatility_filter': True,
                'volume_filter': True
            },
            'data': {
                'lookback_days': 200,
                'update_interval': 60,
                'use_technical_indicators': True,
                'normalize_features': True,
                'feature_engineering': True
            },
            'risk': {
                'var_confidence': 0.95,
                'stress_test_scenarios': 10,
                'correlation_threshold': 0.7,
                'max_sector_exposure': 0.3
            }
        }
        
        # Override with file config if provided
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                # Deep merge configurations
                for key in file_config:
                    if key in config and isinstance(config[key], dict):
                        config[key].update(file_config[key])
                    else:
                        config[key] = file_config[key]
        
        # Validate configuration
        self._validate_config(config)
        
        return config
    
    def _validate_config(self, config: Dict):
        """Validate configuration parameters"""
        
        # Check required keys
        required = ['model', 'trading', 'strategy', 'data']
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config section: {key}")
        
        # Validate ranges
        if not 0 < config['trading']['max_position_size'] <= 1:
            raise ValueError("max_position_size must be between 0 and 1")
        
        if not 0 < config['trading']['confidence_threshold'] <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        
        self.logger.info("Configuration validated successfully")
    
    def load_model(self, checkpoint_path: str) -> TransformerTradingModel:
        """Load trained model with error handling"""
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            # Try common locations
            alternatives = [
                Path('hftraining/checkpoints/production/final.pt'),
                Path('hftraining/checkpoints/best_model.pt'),
                Path('models/production.pt')
            ]
            
            for alt_path in alternatives:
                if alt_path.exists():
                    self.logger.info(f"Using alternative checkpoint: {alt_path}")
                    checkpoint_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"No checkpoint found at {checkpoint_path} or alternatives")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Extract model config
            model_config = self._extract_model_config(checkpoint)
            
            # Convert dict config to object if needed for HF trainer compatibility
            if HFTrainingConfig and isinstance(model_config, dict):
                config_obj = HFTrainingConfig()
                for key, value in model_config.items():
                    setattr(config_obj, key, value)
                # Set defaults for missing fields
                for field in ['hidden_size', 'num_heads', 'num_layers', 'dropout', 
                             'dropout_rate', 'layer_norm_eps', 'sequence_length', 
                             'prediction_horizon']:
                    if not hasattr(config_obj, field):
                        defaults = {
                            'hidden_size': 512,
                            'num_heads': 8,
                            'num_layers': 4,
                            'dropout': 0.1,
                            'dropout_rate': 0.1,
                            'layer_norm_eps': 1e-12,
                            'sequence_length': 60,
                            'prediction_horizon': 5
                        }
                        setattr(config_obj, field, defaults.get(field, 0))
                
                # Determine input dimension (infer from checkpoint if possible)
                input_dim = model_config.get('input_features', 30)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                if isinstance(state_dict, dict):
                    inferred_dim = hfshared.infer_input_dim_from_state(state_dict)
                    if inferred_dim is not None:
                        input_dim = inferred_dim
                model = TransformerTradingModel(config_obj, input_dim=input_dim)
            else:
                # Use dict config directly (fallback model)
                input_dim = model_config.get('input_features', 30)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                if isinstance(state_dict, dict):
                    inferred_dim = hfshared.infer_input_dim_from_state(state_dict)
                    if inferred_dim is not None:
                        input_dim = inferred_dim
                model = TransformerTradingModel(model_config, input_dim=input_dim)
            
            # Load weights
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            if isinstance(state_dict, dict):
                # Remove module prefix if present (from DataParallel)
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict, strict=False)
            
            model.to(self.device)
            model.eval()
            
            self.logger.info(f"Model loaded successfully from {checkpoint_path}")
            
            # Log training metrics if available
            if 'metrics' in checkpoint:
                self.logger.info(f"Training metrics: {checkpoint['metrics']}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _extract_model_config(self, checkpoint: Dict) -> Dict:
        """Extract model configuration from checkpoint"""
        
        model_config = {}
        
        # Try multiple sources for config
        if 'config' in checkpoint:
            if isinstance(checkpoint['config'], dict):
                if 'model' in checkpoint['config']:
                    model_config.update(checkpoint['config']['model'])
                else:
                    model_config.update(checkpoint['config'])
        
        # Use defaults from main config
        for key, value in self.config['model'].items():
            if key not in model_config:
                model_config[key] = value
        
        return model_config
    
    @torch.no_grad()
    def generate_enhanced_signal(self, 
                                 symbol: str, 
                                 data: pd.DataFrame,
                                 use_ensemble: bool = True) -> EnhancedTradingSignal:
        """Generate enhanced trading signal with multiple confirmations"""
        
        try:
            # Prepare features (matching training data processing)
            features = self.data_processor.prepare_features(data)
            
            if features is None or len(features) < self.config['model']['sequence_length']:
                self.logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Take last sequence_length rows
            seq_len = self.config['model']['sequence_length']
            features = features[-seq_len:]
            
            # Normalize features to match training if configured
            if self.config['data']['normalize_features']:
                features = self._normalize_features(features, data)
            
            # Convert to tensor
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Generate base prediction
            outputs = self.model(x)
            
            # Extract predictions
            price_pred = outputs['price_predictions'][0]  # [horizon, features]
            action_probs = outputs['action_probs'][0]  # [3]
            
            # Get current price
            current_price = data['close'].iloc[-1] if 'close' in data.columns else data['Close'].iloc[-1]
            
            # Calculate predicted prices for different horizons
            price_targets = {}
            horizons = [1, 3, 5]  # 1 day, 3 days, 5 days
            
            for h in horizons:
                if h <= len(price_pred):
                    # Use close price prediction (index 3 in OHLCV)
                    pred = price_pred[h-1, 3].cpu().item()
                    # Denormalize
                    pred_price = self._denormalize_price(pred, current_price)
                    price_targets[f'{h}d'] = pred_price
            
            # Calculate expected return
            avg_predicted_price = np.mean(list(price_targets.values()))
            expected_return = (avg_predicted_price - current_price) / current_price
            
            # Determine action
            action_idx = torch.argmax(action_probs).item()
            action_map = {0: 'buy', 1: 'hold', 2: 'sell'}
            action = action_map[action_idx]
            confidence = action_probs[action_idx].item()
            
            # Calculate technical indicators for confirmation
            tech_signals = self._calculate_technical_signals(data)
            
            # Detect market regime
            market_regime = self._detect_market_regime(data)
            
            # Calculate volatility
            close_col = 'close' if 'close' in data.columns else 'Close'
            volatility = data[close_col].pct_change().rolling(20).std().iloc[-1]
            
            # Calculate signal strength (combination of multiple factors)
            signal_strength = self._calculate_signal_strength(
                confidence, expected_return, tech_signals, market_regime
            )
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(volatility, market_regime, data)
            
            # Calculate support and resistance levels
            support_levels, resistance_levels = self._calculate_support_resistance(data)
            
            # Adjust position size based on Kelly Criterion and risk
            position_size = self._calculate_kelly_position_size(
                confidence, expected_return, volatility, risk_score
            )
            
            # Set dynamic stop-loss and take-profit
            stop_loss, take_profit, trailing_stop = self._calculate_risk_levels(
                current_price, volatility, action, support_levels, resistance_levels
            )
            
            # Create enhanced signal
            signal = EnhancedTradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                action=action,
                confidence=confidence,
                predicted_price=avg_predicted_price,
                current_price=current_price,
                expected_return=expected_return,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop=trailing_stop,
                volatility=volatility,
                market_regime=market_regime,
                signal_strength=signal_strength,
                risk_score=risk_score,
                price_targets=price_targets,
                support_levels=support_levels,
                resistance_levels=resistance_levels
            )
            
            # Apply ensemble confirmation if configured
            if use_ensemble and self.config['strategy']['use_ensemble']:
                signal = self._apply_ensemble_confirmation(symbol, signal)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            self.logger.debug(traceback.format_exc())
            return None
    
    def _normalize_features(self, features: np.ndarray, data: pd.DataFrame) -> np.ndarray:
        """Normalize features to match training using shared utilities."""
        # Use training scaler if loaded via StockDataProcessor
        try:
            scalers = getattr(self.data_processor, 'scalers', {})
            std_scaler = scalers.get('standard') if isinstance(scalers, dict) else None
            if std_scaler is not None:
                feats = np.asarray(features, dtype=np.float32)
                # If we know feature_names from training, recompute features to enforce exact ordering
                if self.feature_names:
                    feats_df = self._calculate_training_style_features(data)
                    feats = hfshared.normalize_with_scaler(
                        feats_df.values.astype(np.float32),
                        std_scaler,
                        self.feature_names,
                        df_for_recompute=data
                    )
                return feats
        except Exception as e:
            self.logger.warning(f"Scaler-based normalization failed; falling back. Error: {e}")

        # Fallback: use shared per-window z-score normalization
        return hfshared.zscore_per_window(features)
    
    def _denormalize_price(self, normalized_price: float, current_price: float) -> float:
        """Denormalize predicted close using shared utility."""
        try:
            scalers = getattr(self.data_processor, 'scalers', {})
            std_scaler = scalers.get('standard') if isinstance(scalers, dict) else None
            if std_scaler is not None:
                return hfshared.denormalize_with_scaler(
                    normalized_price,
                    std_scaler,
                    self.feature_names,
                    column='close'
                )
        except Exception:
            pass
        # Heuristic fallback
        if abs(normalized_price) < 1:
            return float(current_price) * (1 + float(normalized_price))
        return float(normalized_price)

    def _calculate_training_style_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Mirror training feature engineering using shared utilities."""
        return hfshared.compute_training_style_features(data)
    
    def _calculate_technical_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicator signals"""
        
        signals = {}
        
        # RSI signal
        if 'rsi' in data.columns:
            rsi = data['rsi'].iloc[-1]
            if rsi < 30:
                signals['rsi'] = 1.0  # Oversold - buy signal
            elif rsi > 70:
                signals['rsi'] = -1.0  # Overbought - sell signal
            else:
                signals['rsi'] = 0.0
        
        # MACD signal
        if 'macd' in data.columns and 'macd_signal' in data.columns:
            macd = data['macd'].iloc[-1]
            macd_signal = data['macd_signal'].iloc[-1]
            signals['macd'] = 1.0 if macd > macd_signal else -1.0
        
        # Moving average signals
        if 'ma_20' in data.columns and 'ma_50' in data.columns:
            ma_20 = data['ma_20'].iloc[-1]
            ma_50 = data['ma_50'].iloc[-1]
            current_price = data['close'].iloc[-1] if 'close' in data.columns else data['Close'].iloc[-1]
            
            if current_price > ma_20 > ma_50:
                signals['ma_trend'] = 1.0  # Bullish
            elif current_price < ma_20 < ma_50:
                signals['ma_trend'] = -1.0  # Bearish
            else:
                signals['ma_trend'] = 0.0
        
        # Bollinger Bands signal
        if 'bb_position' in data.columns:
            bb_pos = data['bb_position'].iloc[-1]
            if bb_pos < 0.2:
                signals['bb'] = 1.0  # Near lower band
            elif bb_pos > 0.8:
                signals['bb'] = -1.0  # Near upper band
            else:
                signals['bb'] = 0.0
        
        return signals
    
    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime"""
        
        # Calculate recent returns
        returns = data['close'].pct_change() if 'close' in data.columns else data['Close'].pct_change()
        
        # Recent trend
        recent_return = returns.tail(20).mean()
        recent_volatility = returns.tail(20).std()
        
        # Longer term trend
        if len(returns) > 60:
            long_return = returns.tail(60).mean()
            long_volatility = returns.tail(60).std()
        else:
            long_return = recent_return
            long_volatility = recent_volatility
        
        # Classify regime
        if recent_volatility > long_volatility * 1.5:
            regime = 'volatile'
        elif recent_return > long_return * 1.5 and recent_return > 0:
            regime = 'bullish'
        elif recent_return < long_return * 0.5 and recent_return < 0:
            regime = 'bearish'
        else:
            regime = 'normal'
        
        return regime
    
    def _calculate_signal_strength(self, 
                                  confidence: float,
                                  expected_return: float,
                                  tech_signals: Dict[str, float],
                                  market_regime: str) -> float:
        """Calculate overall signal strength"""
        
        strength = confidence
        
        # Adjust for expected return magnitude
        if abs(expected_return) > 0.05:
            strength *= 1.2
        elif abs(expected_return) < 0.01:
            strength *= 0.8
        
        # Technical confirmation
        if tech_signals:
            tech_score = np.mean(list(tech_signals.values()))
            if (expected_return > 0 and tech_score > 0) or (expected_return < 0 and tech_score < 0):
                strength *= 1.1  # Confirmation
            elif (expected_return > 0 and tech_score < 0) or (expected_return < 0 and tech_score > 0):
                strength *= 0.9  # Contradiction
        
        # Market regime adjustment
        regime_multipliers = {
            'bullish': 1.1,
            'bearish': 0.9,
            'volatile': 0.8,
            'normal': 1.0
        }
        strength *= regime_multipliers.get(market_regime, 1.0)
        
        return min(max(strength, 0.0), 1.0)
    
    def _calculate_risk_score(self, 
                            volatility: float,
                            market_regime: str,
                            data: pd.DataFrame) -> float:
        """Calculate risk score for position"""
        
        risk = 0.5  # Base risk
        
        # Volatility risk
        if volatility > 0.03:
            risk += 0.2
        elif volatility > 0.02:
            risk += 0.1
        
        # Market regime risk
        regime_risk = {
            'volatile': 0.3,
            'bearish': 0.2,
            'bullish': -0.1,
            'normal': 0.0
        }
        risk += regime_risk.get(market_regime, 0.0)
        
        # Volume risk (low volume = higher risk)
        if 'Volume' in data.columns or 'volume' in data.columns:
            vol_col = 'Volume' if 'Volume' in data.columns else 'volume'
            recent_volume = data[vol_col].tail(5).mean()
            avg_volume = data[vol_col].tail(20).mean()
            
            if recent_volume < avg_volume * 0.7:
                risk += 0.1
        
        return min(max(risk, 0.0), 1.0)
    
    def _calculate_support_resistance(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Calculate support and resistance levels"""
        
        high_col = 'High' if 'High' in data.columns else 'high'
        low_col = 'Low' if 'Low' in data.columns else 'low'
        close_col = 'Close' if 'Close' in data.columns else 'close'
        
        # Recent highs and lows
        recent_data = data.tail(50)
        
        # Find local maxima and minima
        highs = recent_data[high_col].values
        lows = recent_data[low_col].values
        
        # Simple peak detection
        resistance_levels = []
        support_levels = []
        
        for i in range(2, len(highs) - 2):
            # Resistance (local maxima)
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                resistance_levels.append(float(highs[i]))
            
            # Support (local minima)
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                support_levels.append(float(lows[i]))
        
        # Add psychological levels (round numbers)
        current_price = data[close_col].iloc[-1]
        round_levels = [
            round(current_price, -1),  # Nearest 10
            round(current_price * 0.95, -1),  # 5% below
            round(current_price * 1.05, -1)   # 5% above
        ]
        
        # Sort and limit levels
        support_levels = sorted(set(support_levels + [l for l in round_levels if l < current_price]))[-3:]
        resistance_levels = sorted(set(resistance_levels + [l for l in round_levels if l > current_price]))[:3]
        
        return support_levels, resistance_levels
    
    def _calculate_kelly_position_size(self,
                                      confidence: float,
                                      expected_return: float,
                                      volatility: float,
                                      risk_score: float) -> float:
        """Calculate position size using Kelly Criterion with safety adjustments"""
        
        # Kelly formula: f = (p * b - q) / b
        # where p = probability of win, b = win/loss ratio, q = probability of loss
        
        # Estimate win probability from confidence
        p = confidence
        q = 1 - p
        
        # Estimate win/loss ratio from expected return and volatility
        win_amount = abs(expected_return) if expected_return > 0 else volatility
        loss_amount = volatility * 1.5  # Assume losses can be 1.5x volatility
        
        if loss_amount > 0:
            b = win_amount / loss_amount
        else:
            b = 1.0
        
        # Calculate Kelly fraction
        if b > 0:
            kelly_f = (p * b - q) / b
        else:
            kelly_f = 0
        
        # Apply Kelly fraction limit from config
        kelly_f = min(kelly_f, self.config['trading']['kelly_fraction'])
        
        # Adjust for risk
        kelly_f *= (1 - risk_score * 0.5)
        
        # Apply position size limits
        max_position = self.config['trading']['max_position_size']
        position_size = min(max(kelly_f, 0), max_position)
        
        # Further reduce if we have many positions
        num_positions = len(self.positions)
        max_positions = self.config['trading']['max_positions']
        if num_positions >= max_positions * 0.7:
            position_size *= 0.5
        
        return position_size
    
    def _calculate_risk_levels(self,
                              current_price: float,
                              volatility: float,
                              action: str,
                              support_levels: List[float],
                              resistance_levels: List[float]) -> Tuple[float, float, float]:
        """Calculate dynamic stop-loss, take-profit, and trailing stop"""
        
        # Base levels from config
        base_stop_loss = self.config['trading']['stop_loss']
        base_take_profit = self.config['trading']['take_profit']
        base_trailing = self.config['trading']['trailing_stop']
        
        # Adjust for volatility
        vol_multiplier = max(1.0, volatility / 0.02)  # Increase levels if volatile
        
        if action == 'buy':
            # Stop loss: below nearest support or percentage
            if support_levels:
                support_stop = support_levels[-1] * 0.99  # Just below support
                pct_stop = current_price * (1 - base_stop_loss * vol_multiplier)
                stop_loss = max(support_stop, pct_stop)
            else:
                stop_loss = current_price * (1 - base_stop_loss * vol_multiplier)
            
            # Take profit: at resistance or percentage
            if resistance_levels:
                resistance_target = resistance_levels[0] * 0.99  # Just below resistance
                pct_target = current_price * (1 + base_take_profit * vol_multiplier)
                take_profit = min(resistance_target, pct_target)
            else:
                take_profit = current_price * (1 + base_take_profit * vol_multiplier)
            
            # Trailing stop
            trailing_stop = current_price * (1 - base_trailing * vol_multiplier)
            
        elif action == 'sell':
            # For short positions (if supported)
            stop_loss = current_price * (1 + base_stop_loss * vol_multiplier)
            take_profit = current_price * (1 - base_take_profit * vol_multiplier)
            trailing_stop = current_price * (1 + base_trailing * vol_multiplier)
        else:
            stop_loss = None
            take_profit = None
            trailing_stop = None
        
        return stop_loss, take_profit, trailing_stop
    
    def _apply_ensemble_confirmation(self, symbol: str, signal: EnhancedTradingSignal) -> EnhancedTradingSignal:
        """Apply ensemble voting for signal confirmation"""
        
        # Initialize signal buffer for symbol
        if symbol not in self.signal_buffer:
            self.signal_buffer[symbol] = deque(maxlen=self.config['strategy']['ensemble_size'])
        
        # Add current signal to buffer
        self.signal_buffer[symbol].append(signal)
        
        # Check if we have enough signals
        if len(self.signal_buffer[symbol]) < self.config['strategy']['confirmation_required']:
            # Not enough confirmation yet
            signal.confidence *= 0.5  # Reduce confidence
            return signal
        
        # Vote on action
        actions = [s.action for s in self.signal_buffer[symbol]]
        action_counts = {a: actions.count(a) for a in set(actions)}
        
        # Get majority action
        majority_action = max(action_counts.items(), key=lambda x: x[1])[0]
        majority_count = action_counts[majority_action]
        
        # Update signal if confirmed
        if majority_count >= self.config['strategy']['confirmation_required']:
            signal.action = majority_action
            # Boost confidence based on agreement
            signal.confidence *= (majority_count / len(self.signal_buffer[symbol]))
            signal.signal_strength *= 1.2  # Boost strength for confirmed signals
        else:
            # No clear consensus
            signal.action = 'hold'
            signal.confidence *= 0.3
        
        return signal
    
    def execute_trade(self, signal: EnhancedTradingSignal) -> Dict:
        """Execute trade with comprehensive risk management"""
        
        # Reset daily P&L if new day
        if datetime.now().date() > self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = datetime.now().date()
        
        # Check daily loss limit
        if self.daily_pnl <= -self.daily_loss_limit:
            self.logger.warning(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
            return {'status': 'rejected', 'reason': 'daily_loss_limit'}
        
        # Check confidence threshold
        if signal.confidence < self.config['trading']['confidence_threshold']:
            return {'status': 'rejected', 'reason': 'low_confidence'}
        
        # Check risk score
        if signal.risk_score > 0.8:
            return {'status': 'rejected', 'reason': 'high_risk'}
        
        # Check market regime filter
        if self.config['strategy']['market_regime_filter']:
            if signal.market_regime == 'volatile' and signal.action != 'hold':
                signal.position_size *= 0.5  # Reduce position in volatile markets
        
        trade_result = {
            'timestamp': signal.timestamp,
            'symbol': signal.symbol,
            'action': signal.action,
            'status': 'pending'
        }
        
        try:
            if signal.action == 'buy' and signal.symbol not in self.positions:
                # Calculate shares
                position_value = self.current_capital * signal.position_size
                shares = int(position_value / signal.current_price)
                
                if shares > 0 and position_value <= self.current_capital:
                    # Create position
                    position = Position(
                        symbol=signal.symbol,
                        shares=shares,
                        entry_price=signal.current_price,
                        entry_time=signal.timestamp,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        trailing_stop=signal.trailing_stop,
                        high_water_mark=signal.current_price
                    )
                    
                    # Execute trade (in production, this would call broker API)
                    if self.live_trading:
                        # TODO: Implement broker integration
                        pass
                    
                    # Update portfolio
                    self.positions[signal.symbol] = position
                    self.current_capital -= position_value
                    
                    trade_result.update({
                        'status': 'executed',
                        'shares': shares,
                        'price': signal.current_price,
                        'value': position_value,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit
                    })
                    
                    self.trade_logger.info(
                        f"BUY {shares} {signal.symbol} @ ${signal.current_price:.2f} "
                        f"(SL: ${signal.stop_loss:.2f}, TP: ${signal.take_profit:.2f})"
                    )
                    
                    self.performance_metrics['total_trades'] += 1
                    
            elif signal.action == 'sell' and signal.symbol in self.positions:
                position = self.positions[signal.symbol]
                proceeds = position.shares * signal.current_price
                pnl = position.get_unrealized_pnl(signal.current_price)
                
                # Execute trade
                if self.live_trading:
                    # TODO: Implement broker integration
                    pass
                
                # Update portfolio
                self.current_capital += proceeds
                del self.positions[signal.symbol]
                
                # Update metrics
                self.daily_pnl += pnl
                self.performance_metrics['total_pnl'] += pnl
                
                if pnl > 0:
                    self.performance_metrics['winning_trades'] += 1
                    self.performance_metrics['best_trade'] = max(
                        self.performance_metrics['best_trade'], pnl
                    )
                else:
                    self.performance_metrics['losing_trades'] += 1
                    self.performance_metrics['worst_trade'] = min(
                        self.performance_metrics['worst_trade'], pnl
                    )
                
                trade_result.update({
                    'status': 'executed',
                    'shares': position.shares,
                    'price': signal.current_price,
                    'value': proceeds,
                    'pnl': pnl,
                    'return': position.get_return(signal.current_price)
                })
                
                self.trade_logger.info(
                    f"SELL {position.shares} {signal.symbol} @ ${signal.current_price:.2f} "
                    f"(P&L: ${pnl:.2f}, Return: {position.get_return(signal.current_price):.2%})"
                )
                
        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}")
            trade_result['status'] = 'failed'
            trade_result['error'] = str(e)
        
        self.trade_history.append(trade_result)
        return trade_result
    
    def update_positions(self, market_data: Dict[str, pd.DataFrame]):
        """Update all positions with current market data"""
        
        for symbol, position in list(self.positions.items()):
            if symbol in market_data:
                data = market_data[symbol]
                current_price = data['Close'].iloc[-1] if 'Close' in data.columns else data['close'].iloc[-1]
                
                # Update trailing stop
                position.update_trailing_stop(current_price, self.config['trading']['trailing_stop'])
                
                # Check exit conditions
                should_exit = False
                exit_reason = ""
                
                # Stop loss
                if position.stop_loss and current_price <= position.stop_loss:
                    should_exit = True
                    exit_reason = "stop_loss"
                
                # Trailing stop
                elif position.trailing_stop and current_price <= position.trailing_stop:
                    should_exit = True
                    exit_reason = "trailing_stop"
                
                # Take profit
                elif position.take_profit and current_price >= position.take_profit:
                    should_exit = True
                    exit_reason = "take_profit"
                
                # Time-based exit (optional)
                elif (datetime.now() - position.entry_time).days > 30:
                    should_exit = True
                    exit_reason = "time_limit"
                
                if should_exit:
                    # Create sell signal
                    exit_signal = EnhancedTradingSignal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        action='sell',
                        confidence=1.0,
                        predicted_price=current_price,
                        current_price=current_price,
                        expected_return=0,
                        position_size=0
                    )
                    
                    self.logger.info(f"Exiting {symbol} due to {exit_reason}")
                    self.execute_trade(exit_signal)
    
    def calculate_portfolio_metrics(self) -> Dict:
        """Calculate comprehensive portfolio metrics"""
        
        # Calculate current portfolio value
        portfolio_value = self.current_capital
        for symbol, position in self.positions.items():
            # In production, get real-time price
            portfolio_value += position.shares * position.entry_price
        
        # Calculate returns
        total_return = (portfolio_value - self.starting_capital) / self.starting_capital
        
        # Calculate Sharpe ratio (simplified)
        if self.trade_history:
            returns = []
            for trade in self.trade_history:
                if 'return' in trade:
                    returns.append(trade['return'])
            
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = np.sqrt(252) * (avg_return / (std_return + 1e-8))
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Calculate drawdown
        peak_value = max(portfolio_value, self.starting_capital)
        current_drawdown = (peak_value - portfolio_value) / peak_value
        self.performance_metrics['current_drawdown'] = current_drawdown
        self.performance_metrics['max_drawdown'] = max(
            self.performance_metrics['max_drawdown'],
            current_drawdown
        )
        
        metrics = {
            'portfolio_value': portfolio_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': self._calculate_win_rate(),
            'avg_win': self._calculate_avg_win(),
            'avg_loss': self._calculate_avg_loss(),
            'profit_factor': self._calculate_profit_factor(),
            **self.performance_metrics
        }
        
        return metrics
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate"""
        total = self.performance_metrics['winning_trades'] + self.performance_metrics['losing_trades']
        if total > 0:
            return self.performance_metrics['winning_trades'] / total
        return 0
    
    def _calculate_avg_win(self) -> float:
        """Calculate average winning trade"""
        if self.performance_metrics['winning_trades'] > 0:
            wins = [t['pnl'] for t in self.trade_history if t.get('pnl', 0) > 0]
            return np.mean(wins) if wins else 0
        return 0
    
    def _calculate_avg_loss(self) -> float:
        """Calculate average losing trade"""
        if self.performance_metrics['losing_trades'] > 0:
            losses = [t['pnl'] for t in self.trade_history if t.get('pnl', 0) < 0]
            return np.mean(losses) if losses else 0
        return 0
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = sum(t['pnl'] for t in self.trade_history if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trade_history if t.get('pnl', 0) < 0))
        
        if gross_loss > 0:
            return gross_profit / gross_loss
        elif gross_profit > 0:
            return float('inf')
        else:
            return 0
    
    def save_state(self, filepath: str):
        """Save engine state for recovery"""
        
        state = {
            'positions': {k: v.__dict__ for k, v in self.positions.items()},
            'trade_history': self.trade_history,
            'current_capital': self.current_capital,
            'performance_metrics': self.performance_metrics,
            'daily_pnl': self.daily_pnl,
            'last_reset_date': self.last_reset_date.isoformat(),
            'market_regime': self.market_regime
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        self.logger.info(f"State saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load engine state from file"""
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Restore positions
        self.positions = {}
        for symbol, pos_dict in state['positions'].items():
            position = Position(**pos_dict)
            position.entry_time = datetime.fromisoformat(pos_dict['entry_time'])
            if pos_dict.get('last_update'):
                position.last_update = datetime.fromisoformat(pos_dict['last_update'])
            self.positions[symbol] = position
        
        # Restore other state
        self.trade_history = state['trade_history']
        self.current_capital = state['current_capital']
        self.performance_metrics = state['performance_metrics']
        self.daily_pnl = state['daily_pnl']
        self.last_reset_date = datetime.fromisoformat(state['last_reset_date']).date()
        self.market_regime = state['market_regime']
        
        self.logger.info(f"State loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    engine = ProductionTradingEngine(
        checkpoint_path="hftraining/checkpoints/production/final.pt",
        config_path="hfinference/configs/default_config.json",
        paper_trading=True,
        live_trading=False
    )
    
    # Test with sample data
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    for symbol in symbols:
        # Get recent data
        data = yf.download(symbol, period='6mo', progress=False)
        
        # Generate signal
        signal = engine.generate_enhanced_signal(symbol, data)
        
        if signal:
            print(f"\nSignal for {symbol}:")
            print(f"  Action: {signal.action}")
            print(f"  Confidence: {signal.confidence:.2%}")
            print(f"  Expected Return: {signal.expected_return:.2%}")
            print(f"  Risk Score: {signal.risk_score:.2f}")
            print(f"  Market Regime: {signal.market_regime}")
            
            # Execute trade if confident
            if signal.confidence > 0.65 and signal.action != 'hold':
                result = engine.execute_trade(signal)
                print(f"  Trade Result: {result['status']}")
    
    # Print portfolio metrics
    metrics = engine.calculate_portfolio_metrics()
    print(f"\nPortfolio Metrics:")
    print(f"  Portfolio Value: ${metrics['portfolio_value']:.2f}")
    print(f"  Total Return: {metrics['total_return']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
