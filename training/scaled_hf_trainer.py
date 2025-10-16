#!/usr/bin/env python3
"""
Scaled HuggingFace Training Pipeline with Advanced Features
- Full dataset support (130+ symbols)
- Larger model architecture
- PEFT/LoRA for efficient training
- Advanced features and preprocessing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    get_cosine_schedule_with_warmup,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScaledStockConfig(PretrainedConfig):
    """Configuration for scaled stock transformer"""
    model_type = "scaled_stock_transformer"
    
    # Scaled up architecture
    hidden_size: int = 512  # Doubled from before
    num_hidden_layers: int = 12  # Deeper network
    num_attention_heads: int = 16  # More attention heads
    intermediate_size: int = 2048  # Larger FFN
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 1024
    layer_norm_eps: float = 1e-12
    
    # Stock-specific parameters
    num_features: int = 30  # More features
    sequence_length: int = 100  # Longer sequences
    prediction_horizon: int = 10  # Longer prediction
    num_actions: int = 5  # More granular actions: Strong Buy, Buy, Hold, Sell, Strong Sell
    
    # Advanced features
    use_rotary_embeddings: bool = True
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    use_mixture_of_experts: bool = False
    num_experts: int = 4
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


class AdvancedStockDataset(Dataset):
    """Advanced dataset with sophisticated feature engineering"""
    
    def __init__(
        self, 
        data_dir: str,
        symbols: List[str] = None,
        sequence_length: int = 100,
        prediction_horizon: int = 10,
        augmentation: bool = True,
        max_samples_per_symbol: int = 1000,
        use_cache: bool = True
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.augmentation = augmentation
        self.max_samples_per_symbol = max_samples_per_symbol
        self.use_cache = use_cache
        
        # Cache directory
        self.cache_dir = Path(data_dir).parent / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load all available symbols if not specified
        data_path = Path(data_dir)
        if symbols is None:
            symbols = [f.stem for f in data_path.glob('*.csv')]
            # Filter out non-stock files
            symbols = [s for s in symbols if not any(x in s for x in ['metadata', 'combined', 'summary'])]
        
        logger.info(f"Loading data for {len(symbols)} symbols")
        
        # Load and preprocess all stock data
        self.data_samples = []
        self.load_all_stock_data(data_dir, symbols)
        
        logger.info(f"Total samples created: {len(self.data_samples)}")
    
    def load_all_stock_data(self, data_dir: str, symbols: List[str]):
        """Load data for all symbols with caching"""
        data_path = Path(data_dir)
        
        for symbol in symbols:
            # Check cache first
            cache_file = self.cache_dir / f"{symbol}_processed.npz"
            
            if self.use_cache and cache_file.exists():
                try:
                    cached_data = np.load(cache_file, allow_pickle=True)
                    samples = cached_data['samples'].tolist()
                    self.data_samples.extend(samples[:self.max_samples_per_symbol])
                    logger.info(f"Loaded {len(samples)} cached samples for {symbol}")
                    continue
                except Exception as e:
                    logger.warning(f"Cache load failed for {symbol}: {e}")
            
            # Load fresh data
            file_path = data_path / f"{symbol}.csv"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    
                    # Extract advanced features
                    features = self.extract_advanced_features(df, symbol)
                    
                    if features is not None and len(features) > self.sequence_length + self.prediction_horizon:
                        # Create sequences
                        symbol_samples = self.create_sequences(features, symbol)
                        
                        # Cache the processed data
                        if self.use_cache and symbol_samples:
                            np.savez_compressed(cache_file, samples=symbol_samples)
                        
                        # Add to dataset (with limit)
                        self.data_samples.extend(symbol_samples[:self.max_samples_per_symbol])
                        logger.info(f"Processed {len(symbol_samples)} samples for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to process {symbol}: {e}")
    
    def extract_advanced_features(self, df: pd.DataFrame, symbol: str) -> Optional[np.ndarray]:
        """Extract sophisticated features including technical indicators"""
        try:
            features_list = []
            
            # Get OHLC columns (handle case variations)
            price_cols = []
            for col in ['open', 'high', 'low', 'close', 'Open', 'High', 'Low', 'Close']:
                if col in df.columns:
                    price_cols.append(col)
                    if len(price_cols) == 4:
                        break
            
            if len(price_cols) < 4:
                logger.warning(f"Missing price columns for {symbol}")
                return None
            
            # Extract prices
            prices = df[price_cols].values
            
            # Normalize prices
            prices_norm = (prices - prices.mean(axis=0)) / (prices.std(axis=0) + 1e-8)
            features_list.append(prices_norm)
            
            # Volume (synthetic if not available)
            if 'volume' in df.columns or 'Volume' in df.columns:
                vol_col = 'volume' if 'volume' in df.columns else 'Volume'
                volume = df[vol_col].values
            else:
                # Synthetic volume based on price volatility
                volume = np.abs(np.diff(prices[:, 3], prepend=prices[0, 3])) * 1e6
            
            volume_norm = (volume - volume.mean()) / (volume.std() + 1e-8)
            features_list.append(volume_norm.reshape(-1, 1))
            
            # Close price for technical indicators
            close = prices[:, 3]
            
            # 1. Returns (multiple timeframes)
            for lag in [1, 5, 10, 20]:
                returns = np.zeros_like(close)
                if len(close) > lag:
                    returns[lag:] = (close[lag:] - close[:-lag]) / (close[:-lag] + 1e-8)
                features_list.append(returns.reshape(-1, 1))
            
            # 2. Moving averages
            for window in [5, 10, 20, 50]:
                ma = pd.Series(close).rolling(window, min_periods=1).mean().values
                ma_ratio = close / (ma + 1e-8)
                features_list.append(ma_ratio.reshape(-1, 1))
            
            # 3. Exponential moving averages
            for span in [12, 26]:
                ema = pd.Series(close).ewm(span=span, adjust=False).mean().values
                ema_ratio = close / (ema + 1e-8)
                features_list.append(ema_ratio.reshape(-1, 1))
            
            # 4. Bollinger Bands
            bb_window = 20
            bb_std = pd.Series(close).rolling(bb_window, min_periods=1).std().values
            bb_mean = pd.Series(close).rolling(bb_window, min_periods=1).mean().values
            bb_upper = bb_mean + 2 * bb_std
            bb_lower = bb_mean - 2 * bb_std
            bb_position = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
            features_list.append(bb_position.reshape(-1, 1))
            
            # 5. RSI
            rsi = self.calculate_rsi(close, 14)
            features_list.append(rsi.reshape(-1, 1))
            
            # 6. MACD
            ema_12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
            ema_26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
            macd = ema_12 - ema_26
            signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values
            macd_hist = macd - signal
            macd_norm = macd_hist / (np.std(macd_hist) + 1e-8)
            features_list.append(macd_norm.reshape(-1, 1))
            
            # 7. ATR (Average True Range)
            high = prices[:, 1]
            low = prices[:, 2]
            atr = self.calculate_atr(high, low, close, 14)
            atr_norm = atr / (close + 1e-8)
            features_list.append(atr_norm.reshape(-1, 1))
            
            # 8. Stochastic Oscillator
            stoch_k, stoch_d = self.calculate_stochastic(high, low, close, 14)
            features_list.append(stoch_k.reshape(-1, 1))
            features_list.append(stoch_d.reshape(-1, 1))
            
            # 9. Volume indicators
            if volume is not None:
                # OBV (On Balance Volume)
                obv = self.calculate_obv(close, volume)
                obv_norm = (obv - obv.mean()) / (obv.std() + 1e-8)
                features_list.append(obv_norm.reshape(-1, 1))
                
                # Volume SMA ratio
                vol_sma = pd.Series(volume).rolling(20, min_periods=1).mean().values
                vol_ratio = volume / (vol_sma + 1e-8)
                features_list.append(vol_ratio.reshape(-1, 1))
            
            # 10. Market microstructure
            # Spread proxy (high - low)
            spread = (high - low) / (close + 1e-8)
            features_list.append(spread.reshape(-1, 1))
            
            # Combine all features
            features = np.concatenate(features_list, axis=1)
            
            # Handle NaN and Inf
            features = np.nan_to_num(features, nan=0, posinf=1, neginf=-1)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {symbol}: {e}")
            return None
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        deltas = np.diff(prices, prepend=prices[0])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(period, min_periods=1).mean().values
        avg_losses = pd.Series(losses).rolling(period, min_periods=1).mean().values
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi / 100.0
    
    def calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value doesn't have previous close
        
        atr = pd.Series(tr).rolling(period, min_periods=1).mean().values
        return atr
    
    def calculate_stochastic(self, high, low, close, period=14):
        """Calculate Stochastic Oscillator"""
        k_values = []
        
        for i in range(len(close)):
            if i < period - 1:
                k_values.append(50)  # Neutral value for initial period
            else:
                period_high = high[i-period+1:i+1].max()
                period_low = low[i-period+1:i+1].min()
                
                if period_high - period_low > 0:
                    k = 100 * (close[i] - period_low) / (period_high - period_low)
                else:
                    k = 50
                k_values.append(k)
        
        k_values = np.array(k_values)
        d_values = pd.Series(k_values).rolling(3, min_periods=1).mean().values
        
        return k_values / 100.0, d_values / 100.0
    
    def calculate_obv(self, close, volume):
        """Calculate On Balance Volume"""
        obv = np.zeros_like(volume)
        obv[0] = volume[0]
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        
        return obv
    
    def create_sequences(self, features: np.ndarray, symbol: str) -> List[Dict]:
        """Create training sequences with advanced labeling"""
        sequences = []
        total_len = self.sequence_length + self.prediction_horizon
        
        for i in range(len(features) - total_len + 1):
            seq = features[i:i + self.sequence_length]
            targets = features[i + self.sequence_length:i + total_len]
            
            # Advanced action labeling based on future returns
            # Use close price (column 3) for return calculation
            future_prices = targets[:, 3]
            current_price = seq[-1, 3]
            
            # Calculate various return horizons
            returns_1d = (targets[0, 3] - current_price) / (abs(current_price) + 1e-8)
            returns_5d = (targets[min(4, len(targets)-1), 3] - current_price) / (abs(current_price) + 1e-8)
            returns_10d = (targets[-1, 3] - current_price) / (abs(current_price) + 1e-8)
            
            # Multi-class action based on return thresholds
            if returns_1d > 0.02:  # +2%
                action = 0  # Strong Buy
            elif returns_1d > 0.005:  # +0.5%
                action = 1  # Buy
            elif returns_1d < -0.02:  # -2%
                action = 4  # Strong Sell
            elif returns_1d < -0.005:  # -0.5%
                action = 3  # Sell
            else:
                action = 2  # Hold
            
            sequences.append({
                'sequence': seq,
                'targets': targets,
                'action': action,
                'symbol': symbol,
                'returns_1d': returns_1d,
                'returns_5d': returns_5d,
                'returns_10d': returns_10d
            })
        
        return sequences
    
    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        
        sequence = torch.FloatTensor(sample['sequence'])
        targets = torch.FloatTensor(sample['targets'])
        
        # Apply augmentation if training
        if self.augmentation and np.random.random() < 0.3:
            # Noise injection
            noise = torch.randn_like(sequence) * 0.02
            sequence = sequence + noise
            
            # Random scaling
            scale = 1.0 + (np.random.random() - 0.5) * 0.1
            sequence = sequence * scale
            targets = targets * scale
            
            # Dropout (randomly zero out some features)
            if np.random.random() < 0.1:
                dropout_mask = torch.rand(sequence.shape[1]) > 0.1
                sequence[:, dropout_mask] = sequence[:, dropout_mask] * 0
        
        return {
            'input_ids': sequence,
            'labels': targets,
            'action_labels': torch.tensor(sample['action'], dtype=torch.long),
            'attention_mask': torch.ones(self.sequence_length)
        }


class ScaledStockTransformer(PreTrainedModel):
    """Scaled transformer with advanced architecture"""
    
    config_class = ScaledStockConfig
    
    def __init__(self, config: ScaledStockConfig):
        super().__init__(config)
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.num_features, config.hidden_size)
        
        # Positional embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Transformer encoder with scaled architecture
        encoder_config = {
            'd_model': config.hidden_size,
            'nhead': config.num_attention_heads,
            'dim_feedforward': config.intermediate_size,
            'dropout': config.hidden_dropout_prob,
            'activation': 'gelu',
            'layer_norm_eps': config.layer_norm_eps,
            'batch_first': True,
            'norm_first': True
        }
        
        encoder_layer = nn.TransformerEncoderLayer(**encoder_config)
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_hidden_layers,
            enable_nested_tensor=False
        )
        
        # Pooler
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        
        # Output heads
        self.price_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.LayerNorm(config.intermediate_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.intermediate_size // 2),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size // 2, config.prediction_horizon * config.num_features)
        )
        
        self.action_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.LayerNorm(config.intermediate_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.num_actions)
        )
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        action_labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
    ):
        batch_size, seq_len, _ = input_ids.shape
        device = input_ids.device
        
        # Input embeddings
        hidden_states = self.input_projection(input_ids)
        
        # Add positional embeddings
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        hidden_states = hidden_states + position_embeddings
        
        # Layer norm and dropout
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Transformer encoder
        if self.config.gradient_checkpointing and self.training:
            hidden_states = torch.utils.checkpoint.checkpoint(
                self.encoder, hidden_states
            )
        else:
            hidden_states = self.encoder(hidden_states)
        
        # Pooling (mean pooling with attention mask)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = hidden_states.mean(dim=1)
        
        pooled_output = self.pooler(pooled_output)
        
        # Predictions
        price_predictions = self.price_predictor(pooled_output)
        action_logits = self.action_classifier(pooled_output)
        
        # Calculate losses
        loss = None
        if labels is not None or action_labels is not None:
            loss = 0.0
            
            if labels is not None:
                # Reshape predictions
                price_predictions_reshaped = price_predictions.view(
                    batch_size, self.config.prediction_horizon, self.config.num_features
                )
                
                # Weighted MSE loss (emphasize close price prediction)
                weights = torch.ones_like(labels)
                weights[:, :, 3] = 2.0  # Double weight for close price
                
                price_loss = F.mse_loss(price_predictions_reshaped, labels, reduction='none')
                price_loss = (price_loss * weights).mean()
                loss += price_loss
            
            if action_labels is not None:
                # Class-weighted cross-entropy
                action_loss = F.cross_entropy(action_logits, action_labels)
                loss += action_loss * 0.5  # Balance with price loss
        
        if not return_dict:
            output = (action_logits, price_predictions)
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=action_logits,
            hidden_states=hidden_states,
            attentions=None
        )


def create_scaled_trainer(
    model: ScaledStockTransformer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    config: ScaledStockConfig,
    output_dir: str = "./scaled_stock_model"
) -> Trainer:
    """Create trainer with optimized settings for scaled model"""
    
    # Apply LoRA if configured
    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=["input_projection", "encoder", "price_predictor", "action_classifier"],
            lora_dropout=config.lora_dropout,
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, lora_config)
        logger.info(f"Applied LoRA. Trainable params: {model.print_trainable_parameters()}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Training parameters
        num_train_epochs=20,
        per_device_train_batch_size=16,  # Adjust based on GPU memory
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=8,  # Effective batch size = 128
        
        # Learning rate schedule
        learning_rate=2e-5,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        
        # Optimization
        optim="adamw_torch",
        adam_epsilon=1e-8,
        adam_beta1=0.9,
        adam_beta2=0.999,
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Evaluation and checkpointing
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Logging
        logging_dir=f"{output_dir}/logs",
        logging_steps=20,
        report_to=["tensorboard"],
        
        # Performance optimizations
        fp16=torch.cuda.is_available(),
        bf16=False,  # Use if supported
        dataloader_num_workers=4,
        gradient_checkpointing=config.gradient_checkpointing,
        
        # Other
        remove_unused_columns=False,
        push_to_hub=False,
        seed=42,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)
        ],
    )
    
    return trainer


def main():
    """Main training function for scaled model"""
    logger.info("="*80)
    logger.info("SCALED HUGGINGFACE TRAINING PIPELINE")
    logger.info("="*80)
    
    # Configuration
    config = ScaledStockConfig(
        hidden_size=512,
        num_hidden_layers=8,  # Start with 8 layers for testing
        num_attention_heads=16,
        intermediate_size=2048,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_features=30,  # Advanced features
        sequence_length=100,
        prediction_horizon=10,
        num_actions=5,
        use_rotary_embeddings=True,
        gradient_checkpointing=True,
        use_lora=True,
        lora_r=16,
        lora_alpha=32
    )
    
    # Load full dataset
    logger.info("Loading training dataset...")
    train_dataset = AdvancedStockDataset(
        data_dir="../trainingdata/train",
        symbols=None,  # Use all available symbols
        sequence_length=config.sequence_length,
        prediction_horizon=config.prediction_horizon,
        augmentation=True,
        max_samples_per_symbol=500,  # Limit for memory
        use_cache=True
    )
    
    logger.info("Loading validation dataset...")
    # Use different subset for validation
    val_symbols = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'AAPL', 'GOOGL', 'MSFT']
    eval_dataset = AdvancedStockDataset(
        data_dir="../trainingdata/train",
        symbols=val_symbols,
        sequence_length=config.sequence_length,
        prediction_horizon=config.prediction_horizon,
        augmentation=False,
        max_samples_per_symbol=200,
        use_cache=True
    )
    
    logger.info(f"Dataset sizes - Train: {len(train_dataset):,}, Eval: {len(eval_dataset):,}")
    
    # Create model
    model = ScaledStockTransformer(config)
    
    # Log model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Create trainer
    trainer = create_scaled_trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
        output_dir="./scaled_stock_model"
    )
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save model
    trainer.save_model()
    logger.info("Model saved!")
    
    # Final evaluation
    eval_results = trainer.evaluate()
    logger.info(f"Final evaluation results: {eval_results}")
    
    # Save training results
    results = {
        'train_result': train_result.metrics,
        'eval_result': eval_results,
        'config': config.to_dict(),
        'dataset_info': {
            'train_size': len(train_dataset),
            'eval_size': len(eval_dataset),
            'num_features': config.num_features,
            'sequence_length': config.sequence_length
        }
    }
    
    with open("./scaled_stock_model/training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)


if __name__ == "__main__":
    main()