#!/usr/bin/env python3
"""
Modern HuggingFace Training Pipeline for Stock Prediction
Uses latest transformers, efficient training techniques, and multi-stock support
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
    get_cosine_schedule_with_warmup
)
from transformers.modeling_outputs import SequenceClassifierOutput
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StockTransformerConfig(PretrainedConfig):
    """Configuration for Stock Transformer model"""
    model_type = "stock_transformer"
    
    hidden_size: int = 256
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 1024
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    layer_norm_eps: float = 1e-12
    
    # Stock-specific parameters
    num_features: int = 15  # OHLCV + technical indicators
    sequence_length: int = 60
    prediction_horizon: int = 5
    num_actions: int = 3  # Buy, Hold, Sell
    
    # Advanced features
    use_rotary_embeddings: bool = True
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for better long-range modeling"""
    
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('cos', freqs.cos())
        self.register_buffer('sin', freqs.sin())
    
    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        cos = self.cos[:seq_len].unsqueeze(0)
        sin = self.sin[:seq_len].unsqueeze(0)
        
        # Apply rotary embedding
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.stack([-x2, x1], dim=-1).flatten(-2)
        x_pos = torch.stack([x1, x2], dim=-1).flatten(-2)
        
        return x_pos * cos + x_rot * sin


class StockTransformerModel(PreTrainedModel):
    """Modern Transformer for Stock Prediction with HuggingFace compatibility"""
    
    config_class = StockTransformerConfig
    
    def __init__(self, config: StockTransformerConfig):
        super().__init__(config)
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.num_features, config.hidden_size)
        
        # Positional embeddings
        if config.use_rotary_embeddings:
            self.pos_embedding = RotaryPositionalEmbedding(
                config.hidden_size, 
                config.max_position_embeddings
            )
        else:
            self.pos_embedding = nn.Embedding(
                config.max_position_embeddings, 
                config.hidden_size
            )
        
        # Transformer blocks with modern improvements
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Output heads
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Multi-task heads
        self.price_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.prediction_horizon * config.num_features)
        )
        
        self.action_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
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
    ) -> SequenceClassifierOutput:
        """
        Forward pass with multi-task learning
        
        Args:
            input_ids: [batch, seq_len, features]
            attention_mask: [batch, seq_len]
            labels: Price prediction targets [batch, horizon, features]
            action_labels: Action classification targets [batch]
        """
        batch_size, seq_len, _ = input_ids.shape
        device = input_ids.device
        
        # Input projection
        hidden_states = self.input_projection(input_ids)
        
        # Add positional embeddings
        if self.config.use_rotary_embeddings:
            hidden_states = self.pos_embedding(hidden_states)
        else:
            position_ids = torch.arange(seq_len, device=device).expand(batch_size, -1)
            hidden_states = hidden_states + self.pos_embedding(position_ids)
        
        # Create attention mask if needed
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        # Expand attention mask for transformer
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_ids.shape[:2], device
        )
        
        # Apply transformer layers
        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, extended_attention_mask
                )
            else:
                hidden_states = layer(hidden_states, extended_attention_mask)
        
        # Apply final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        # Pool to get sequence representation (use last token)
        pooled_output = hidden_states[:, -1]
        
        # Get predictions
        price_predictions = self.price_predictor(pooled_output)
        action_logits = self.action_classifier(pooled_output)
        
        # Calculate losses if labels provided
        loss = None
        if labels is not None or action_labels is not None:
            loss = 0.0
            
            if labels is not None:
                # Reshape predictions and labels
                price_predictions_reshaped = price_predictions.view(
                    batch_size, self.config.prediction_horizon, self.config.num_features
                )
                # MSE loss for price prediction
                price_loss = F.mse_loss(price_predictions_reshaped, labels)
                loss += price_loss
            
            if action_labels is not None:
                # Cross-entropy loss for action classification
                action_loss = F.cross_entropy(action_logits, action_labels)
                loss += action_loss
        
        if not return_dict:
            output = (action_logits,) + (price_predictions,)
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=action_logits,
            hidden_states=hidden_states,
            attentions=None
        )
    
    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        """Create extended attention mask for transformer"""
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return extended_attention_mask


class TransformerBlock(nn.Module):
    """Single Transformer block with modern improvements"""
    
    def __init__(self, config: StockTransformerConfig):
        super().__init__()
        
        # Multi-head attention with optional flash attention
        self.attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        
        # Feed-forward network with SwiGLU activation
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size * 2)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        
        # Layer norms
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention with residual
        normed_hidden_states = self.layer_norm1(hidden_states)
        attention_output, _ = self.attention(
            normed_hidden_states, 
            normed_hidden_states, 
            normed_hidden_states,
            attn_mask=attention_mask
        )
        hidden_states = hidden_states + self.dropout(attention_output)
        
        # Feed-forward with SwiGLU and residual
        normed_hidden_states = self.layer_norm2(hidden_states)
        
        # SwiGLU activation
        ff_output = self.intermediate(normed_hidden_states)
        x1, x2 = ff_output.chunk(2, dim=-1)
        ff_output = x1 * F.silu(x2)
        ff_output = self.output(ff_output)
        
        hidden_states = hidden_states + self.dropout(ff_output)
        
        return hidden_states


class MultiStockDataset(Dataset):
    """Dataset for multiple stock symbols with advanced preprocessing"""
    
    def __init__(
        self, 
        data_dir: str,
        symbols: List[str],
        sequence_length: int = 60,
        prediction_horizon: int = 5,
        augmentation: bool = True
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.augmentation = augmentation
        
        # Load and preprocess all stock data
        self.data_samples = []
        self.load_stock_data(data_dir, symbols)
    
    def load_stock_data(self, data_dir: str, symbols: List[str]):
        """Load data for all symbols"""
        data_path = Path(data_dir)
        
        for symbol in symbols:
            # Try different file patterns
            for pattern in [f"{symbol}.csv", f"{symbol}*.csv"]:
                files = list(data_path.glob(pattern))
                if files:
                    df = pd.read_csv(files[0], index_col=0, parse_dates=True)
                    
                    # Preprocess features
                    features = self.extract_features(df)
                    
                    # Create sequences
                    self.create_sequences(features, symbol)
                    break
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and normalize features"""
        features = []
        
        # Price features
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                values = df[col].values
                # Normalize using rolling statistics
                values = (values - np.mean(values)) / (np.std(values) + 1e-8)
                features.append(values)
        
        # Add Volume if available, otherwise use synthetic volume
        if 'Volume' in df.columns:
            values = df['Volume'].values
            values = (values - np.mean(values)) / (np.std(values) + 1e-8)
            features.append(values)
        else:
            # Synthetic volume based on price movement
            if 'Close' in df.columns:
                close = df['Close'].values
                volume = np.abs(np.diff(close, prepend=close[0])) * 1000000
                volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)
                features.append(volume)
        
        # Technical indicators
        if 'Close' in df.columns:
            close = df['Close'].values
            
            # Returns
            returns = np.diff(close) / close[:-1]
            returns = np.concatenate([[0], returns])
            features.append(returns)
            
            # Moving averages
            for window in [5, 10, 20]:
                ma = pd.Series(close).rolling(window).mean().fillna(method='bfill').values
                ma_ratio = close / (ma + 1e-8)
                features.append(ma_ratio)
            
            # RSI
            rsi = self.calculate_rsi(close)
            features.append(rsi)
            
            # Volatility
            volatility = pd.Series(returns).rolling(20).std().fillna(0).values
            features.append(volatility)
        
        return np.stack(features, axis=1)
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 100
        rsi = np.zeros_like(prices)
        rsi[:period] = 50  # neutral
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 100
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi / 100.0  # Normalize to 0-1
    
    def create_sequences(self, features: np.ndarray, symbol: str):
        """Create training sequences from features"""
        total_len = self.sequence_length + self.prediction_horizon
        
        for i in range(len(features) - total_len + 1):
            sequence = features[i:i + self.sequence_length]
            targets = features[i + self.sequence_length:i + total_len]
            
            # Determine action label
            future_return = (targets[0, 3] - sequence[-1, 3]) / sequence[-1, 3]
            
            if future_return > 0.01:
                action = 0  # Buy
            elif future_return < -0.01:
                action = 2  # Sell
            else:
                action = 1  # Hold
            
            self.data_samples.append({
                'sequence': sequence,
                'targets': targets,
                'action': action,
                'symbol': symbol
            })
    
    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        
        sequence = torch.FloatTensor(sample['sequence'])
        targets = torch.FloatTensor(sample['targets'])
        
        # Apply augmentation if training
        if self.augmentation and np.random.random() < 0.5:
            # Add noise
            noise = torch.randn_like(sequence) * 0.01
            sequence = sequence + noise
            
            # Random scaling
            scale = 1.0 + (np.random.random() - 0.5) * 0.1
            sequence = sequence * scale
            targets = targets * scale
        
        return {
            'input_ids': sequence,
            'labels': targets,
            'action_labels': torch.tensor(sample['action'], dtype=torch.long),
            'attention_mask': torch.ones(self.sequence_length)
        }


def create_hf_trainer(
    model: StockTransformerModel,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    output_dir: str = "./hf_stock_model"
) -> Trainer:
    """Create HuggingFace Trainer with optimized settings"""
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Training parameters
        num_train_epochs=50,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=4,
        
        # Learning rate schedule
        learning_rate=5e-5,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        
        # Optimization
        optim="adamw_torch",
        adam_epsilon=1e-8,
        adam_beta1=0.9,
        adam_beta2=0.999,
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Evaluation
        evaluation_strategy="steps",
        eval_steps=100,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Checkpointing
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        
        # Logging
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        report_to=["tensorboard"],
        
        # Performance
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        
        # Debugging
        disable_tqdm=False,
        seed=42,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5)
        ],
    )
    
    return trainer


def main():
    """Main training function"""
    logger.info("Starting HuggingFace Modern Training Pipeline")
    
    # Configuration
    config = StockTransformerConfig(
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_features=15,
        sequence_length=60,
        prediction_horizon=5,
        use_rotary_embeddings=True,
        gradient_checkpointing=True
    )
    
    # Load datasets
    train_dataset = MultiStockDataset(
        data_dir="../trainingdata/train",
        symbols=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 'TSLA', 'META', 'SPY', 'QQQ'],
        sequence_length=config.sequence_length,
        prediction_horizon=config.prediction_horizon,
        augmentation=True
    )
    
    eval_dataset = MultiStockDataset(
        data_dir="../trainingdata/test",
        symbols=['AAPL', 'GOOGL', 'MSFT', 'SPY'],
        sequence_length=config.sequence_length,
        prediction_horizon=config.prediction_horizon,
        augmentation=False
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Create model
    model = StockTransformerModel(config)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = create_hf_trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir="./hf_modern_stock_model"
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model()
    logger.info("Training complete! Model saved.")
    
    # Evaluate
    eval_results = trainer.evaluate()
    logger.info(f"Final evaluation results: {eval_results}")
    
    # Save results
    with open("./hf_modern_stock_model/results.json", "w") as f:
        json.dump(eval_results, f, indent=2)


if __name__ == "__main__":
    main()