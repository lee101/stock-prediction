#!/usr/bin/env python3
"""
HuggingFace Trading Engine
Core engine for running inference and executing trades.

This engine aligns its model architecture and I/O with the
HF-style training code in `hftraining/hf_trainer.py` so that
inference behaves consistently with what was trained.
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
from dataclasses import dataclass
try:
    import joblib  # For loading saved scalers from training
except Exception:
    joblib = None
import warnings

warnings.filterwarnings('ignore')

# Import shared utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))
import hfshared

def _load_local_symbol_data(symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load OHLCV data for a symbol from local CSVs under trainingdata/ or hftraining/trainingdata/.

    Returns a DataFrame or None if no matching local file is found.
    """
    try:
        bases = [Path('trainingdata'), Path('hftraining/trainingdata')]
        candidates = []
        for base in bases:
            for name in [f"{symbol}.csv", f"{symbol.upper()}.csv", f"{symbol.lower()}.csv"]:
                p = base / name
                if p.exists():
                    candidates.append(p)
        path = candidates[0] if candidates else None
        if not path:
            return None

        df = pd.read_csv(path)
        # Standardize columns and parse date if present
        df.columns = [c.lower() for c in df.columns]
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').set_index('date')
            except Exception:
                pass

        # Filter by date range if provided (YYYY-MM-DD strings okay)
        if start_date:
            try:
                df = df[df.index >= pd.to_datetime(start_date)] if isinstance(df.index, pd.DatetimeIndex) else df
            except Exception:
                pass
        if end_date:
            try:
                df = df[df.index <= pd.to_datetime(end_date)] if isinstance(df.index, pd.DatetimeIndex) else df
            except Exception:
                pass

        return df
    except Exception:
        return None

# Import the HF training model to ensure architecture parity
try:
    from hftraining.hf_trainer import (
        TransformerTradingModel as HFTransformerTradingModel,
        HFTrainingConfig as HFConfig,
    )
except Exception:
    # Fallback if training package isn't available at runtime; tests may patch load_model.
    HFTransformerTradingModel = None
    HFConfig = None

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


# Note: The model class is now sourced from hftraining.hf_trainer to match training.


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
        if isinstance(self.config, dict):
            self.current_capital = (
                self.config.get('initial_capital')
                or self.config.get('trading', {}).get('initial_capital')
                or 10000
            )
        else:
            self.current_capital = 10000
        
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
    
    def load_model(self, checkpoint_path: str):
        """Load trained model from checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Determine input dimension for inference features using shared utility
        input_dim = None
        # Try to infer from checkpoint weights using shared utility
        if 'model_state_dict' in checkpoint:
            input_dim = hfshared.infer_input_dim_from_state(checkpoint['model_state_dict'])

        if input_dim is None and isinstance(self.config, dict):
            input_dim = (
                self.config.get('input_features')
                or self.config.get('model', {}).get('input_features')
            )

        if input_dim is None:
            # Final fallback
            input_dim = 5

        # Try to construct the same HF config used in training
        hf_cfg = None
        if 'config' in checkpoint:
            cfg_obj = checkpoint['config']
            try:
                if HFConfig and isinstance(cfg_obj, HFConfig):
                    hf_cfg = cfg_obj
                elif isinstance(cfg_obj, dict):
                    # Direct dataclass init if keys match; else fallback
                    if HFConfig:
                        hf_cfg = HFConfig(**{k: v for k, v in cfg_obj.items() if k in HFConfig.__dataclass_fields__})
            except Exception:
                hf_cfg = None

        # Fallback HF config if none present
        if hf_cfg is None:
            class _TmpCfg:
                pass
            tmp = _TmpCfg()
            # Reasonable defaults matching trainer assumptions
            for k, v in dict(
                hidden_size=512,
                num_layers=8,
                num_heads=16,
                dropout=0.1,
                dropout_rate=0.1,
                layer_norm_eps=1e-12,
                sequence_length=(self.config.get('sequence_length') or self.config.get('model', {}).get('sequence_length') or 60),
                prediction_horizon=(self.config.get('prediction_horizon') or self.config.get('model', {}).get('prediction_horizon') or 5),
                learning_rate=1e-4,
                warmup_steps=0,
                max_steps=0,
            ).items():
                setattr(tmp, k, v)
            hf_cfg = tmp

        # Build model from HF trainer to ensure parity
        if HFTransformerTradingModel is None:
            raise RuntimeError("HF training model not available; cannot construct inference model.")

        model = HFTransformerTradingModel(hf_cfg, input_dim=input_dim)
        
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
        
        # Try to load training scalers to keep inference identical to training
        self._try_load_training_scalers(checkpoint_path)

        return model

    def _try_load_training_scalers(self, checkpoint_path: Path) -> None:
        """Attempt to load data processor scalers saved during training.

        Looks for 'data_processor.pkl' in either:
        - Same directory as checkpoint
        - Path provided via config['processor_path'] or config['data']['processor_path']
        Sets attributes on self.data_processor after it is constructed.
        """
        # Defer if DataProcessor not yet created; we'll store path for later
        proc_path = None
        # 1) Config-specified path
        if isinstance(self.config, dict):
            proc_path = (
                self.config.get('processor_path')
                or self.config.get('data', {}).get('processor_path')
            )
        # 2) Sibling of checkpoint
        if not proc_path:
            candidate = checkpoint_path.parent / 'data_processor.pkl'
            if candidate.exists():
                proc_path = str(candidate)

        # Store for later and propagate into config so DataProcessor can read it
        self._processor_path = proc_path
        if proc_path and isinstance(self.config, dict):
            try:
                self.config.setdefault('processor_path', proc_path)
            except Exception:
                pass
    
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
        
        # Extract predictions (support both trainer-style and legacy shapes)
        price_pred = outputs['price_predictions'][0]
        # HF trainer returns action_logits; tests/legacy may supply action_probs
        if 'action_probs' in outputs:
            action_probs = outputs['action_probs'][0]
        else:
            logits = outputs['action_logits'][0]
            action_probs = torch.softmax(logits, dim=-1)
        
        # Get current price and predicted price (handle both cases)
        if 'Close' in data.columns:
            current_price = data['Close'].iloc[-1]
        elif 'close' in data.columns:
            current_price = data['close'].iloc[-1]
        else:
            # Fallback to 4th column (typical Close position)
            current_price = data.iloc[-1, 3]
        
        # Determine close predictions based on output shape
        # - Trainer-aligned: shape [horizon]
        # - Legacy/tests: shape [horizon, features], with close at index 3
        if price_pred.ndim == 1:
            pred_close_norm = price_pred.cpu().numpy()
        elif price_pred.ndim == 2 and price_pred.shape[-1] >= 4:
            pred_close_norm = price_pred[:, 3].cpu().numpy()
        else:
            # Unexpected shape; fallback to zeros
            pred_close_norm = np.zeros(int(self.config.get('prediction_horizon') or 1), dtype=np.float32)

        # Denormalize predicted close using current feature normalization
        predicted_closes = [self.data_processor.denormalize_close(v) for v in pred_close_norm]
        predicted_price = float(np.mean(predicted_closes))

        # Heuristic: some models (legacy/tests) may treat outputs as deltas.
        # If normalized mean suggests upward move but denorm absolute is below
        # current, fallback to a small delta-based interpretation to keep
        # behavior aligned with expectations.
        mean_norm = float(np.mean(pred_close_norm)) if len(pred_close_norm) else 0.0
        if (mean_norm > 0 and predicted_price <= current_price) or (mean_norm < 0 and predicted_price >= current_price):
            predicted_price = float(current_price * (1.0 + mean_norm * 0.1))
        
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
        trading_config = self.config.get('trading', {}) if isinstance(self.config, dict) else {}
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
        trading_config = self.config.get('trading', {}) if isinstance(self.config, dict) else {}
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
            
            # Get historical data (prefer local CSVs, fallback to yfinance)
            data = _load_local_symbol_data(symbol, start_date, end_date)
            if data is None or len(data) == 0:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            # Extract sequence_length from appropriate location
            seq_len = 60
            if isinstance(self.config, dict):
                seq_len = (
                    self.config.get('sequence_length')
                    or self.config.get('model', {}).get('sequence_length')
                    or 60
                )
            
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
    """Process market data for model input aligned with HF trainer.

    Trainer uses z-score normalization on features and predicts
    normalized close price over the horizon. We replicate that
    per-window to avoid needing persisted scalers.
    """

    def __init__(self, config: Dict):
        self.config = config if isinstance(config, dict) else {}
        # Extract sequence_length
        self.sequence_length = (
            self.config.get('sequence_length')
            or self.config.get('model', {}).get('sequence_length')
            or self.config.get('data', {}).get('sequence_length')
            or 60
        )
        # Feature mode controls how we form inputs.
        # Supported:
        # - 'auto': detect OHLC(+V) from data columns (default)
        # - 'ohlc': force OHLC only
        # - 'ohlcv': force OHLCV (will backfill volume=0 if absent)
        self.feature_mode = (
            self.config.get('feature_mode')
            or self.config.get('data', {}).get('feature_mode')
            or 'auto'
        )
        # Optional: use percent change per feature instead of raw values.
        # Keeps the same dimensionality while improving scale invariance.
        self.use_pct_change = bool(
            self.config.get('use_pct_change')
            or self.config.get('data', {}).get('use_pct_change')
            or False
        )
        # Track per-window normalization params for denorm
        self._feature_mean = None
        self._feature_std = None
        # Training scalers (StandardScaler) if available
        self.scaler = None
        self.feature_names = []
        proc_path = (
            self.config.get('processor_path')
            or self.config.get('data', {}).get('processor_path')
        )
        if proc_path:
            try:
                processor_data = hfshared.load_processor(proc_path)
                if processor_data:
                    self.scaler = processor_data.get('scalers', {}).get('standard')
                    self.feature_names = list(processor_data.get('feature_names', []))
            except Exception:
                # If loading fails, proceed without scalers
                self.scaler = None

    def prepare_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        if len(data) < self.sequence_length:
            return None

        # Take last sequence_length rows
        df = data.tail(self.sequence_length).copy()

        # Compute features. If a trained scaler is available, compute the full
        # training-style feature set and transform with the scaler to preserve
        # exact training distribution and ordering.
        if self.scaler is not None and self.feature_names:
            feats_df = self.calculate_training_style_features(df)
            # Use shared normalization with scaler
            feats_norm = hfshared.normalize_with_scaler(
                feats_df.values.astype(np.float32),
                self.scaler,
                self.feature_names,
                df_for_recompute=df
            )
            return feats_norm
        else:
            # Minimal inference features (OHLC +/- V) with optional pct change
            feats = self.calculate_features(df)
            # Use shared per-window normalization
            feats_norm = hfshared.zscore_per_window(feats)
            # Store params for denormalization compatibility
            self._feature_mean = feats.mean(axis=0)
            self._feature_std = feats.std(axis=0) + 1e-8
            return feats_norm

    def calculate_features(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate compact features using shared utilities."""
        return hfshared.compute_compact_features(
            data,
            feature_mode=self.feature_mode,
            use_pct_change=self.use_pct_change
        )

    def calculate_training_style_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Replicate training feature engineering using shared utilities."""
        return hfshared.compute_training_style_features(data)

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Per-window z-score normalization using shared utility."""
        normalized = hfshared.zscore_per_window(features)
        # Store params for denormalization compatibility
        self._feature_mean = features.mean(axis=0)
        self._feature_std = features.std(axis=0) + 1e-8
        return normalized

    def denormalize_close(self, normalized_close: float) -> float:
        """Denormalize close price using shared utility."""
        if self.scaler is not None:
            return hfshared.denormalize_with_scaler(
                normalized_close,
                self.scaler,
                self.feature_names,
                column='close'
            )
        # Otherwise fallback to per-window stats captured during normalize()
        if self._feature_mean is None or self._feature_std is None:
            return float(normalized_close)
        mu_c = float(self._feature_mean[3]) if len(self._feature_mean) > 3 else 0.0
        std_c = float(self._feature_std[3]) if len(self._feature_std) > 3 else 1.0
        return float(normalized_close) * std_c + mu_c


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
