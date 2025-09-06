#!/usr/bin/env python3
"""
Advanced Training System V2
State-of-the-art techniques for better model performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import numpy as np
import pandas as pd
try:
    import yfinance as yf  # Optional; may be unavailable in restricted envs
except Exception:
    class _YFStub:
        @staticmethod
        def download(*args, **kwargs):
            raise RuntimeError("yfinance unavailable; use local trainingdata instead")

        class Ticker:
            def __init__(self, *args, **kwargs):
                pass

            def history(self, *args, **kwargs):
                raise RuntimeError("yfinance unavailable; use local trainingdata instead")

    yf = _YFStub
from pathlib import Path
from datetime import datetime
import json
import logging
import math
from typing import Dict, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from robust_data_pipeline import create_robust_dataloader


class AdvancedTransformerModel(nn.Module):
    """Advanced Transformer with modern techniques"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Enhanced model dimensions
        hidden_size = config['hidden_size']
        num_heads = config['num_heads']
        num_layers = config['num_layers']
        
        # Ensure compatibility
        if hidden_size % num_heads != 0:
            hidden_size = (hidden_size // num_heads) * num_heads
        
        self.hidden_size = hidden_size
        self.config = config
        
        # Advanced input processing
        self.input_projection = nn.Sequential(
            nn.Linear(config['input_features'], hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1))
        )
        
        # Learnable positional encoding (better than fixed)
        self.positional_encoding = nn.Parameter(
            torch.randn(1, config['sequence_length'], hidden_size) * 0.02
        )
        
        # Multi-scale feature extraction
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv1d(hidden_size, hidden_size//4, kernel_size=k, padding=k//2)
            for k in [1, 3, 5, 7]
        ])
        
        # Advanced transformer layers with modifications
        self.transformer_layers = nn.ModuleList([
            AdvancedTransformerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=config.get('intermediate_size', hidden_size * 4),
                dropout=config.get('dropout', 0.1),
                layer_idx=i
            ) for i in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_size)
        
        # Multi-head attention pooling
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=config.get('dropout', 0.1),
            batch_first=True
        )
        
        # Enhanced output heads with residual connections
        self.price_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(hidden_size // 2, config['prediction_horizon'] * config['input_features'])
        )
        
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(hidden_size // 4, 3)  # Buy, Hold, Sell
        )
        
        # Confidence head for uncertainty estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights with advanced technique
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Advanced weight initialization"""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # He initialization for ReLU-like activations
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.MultiheadAttention):
            # Xavier initialization for attention
            torch.nn.init.xavier_uniform_(module.in_proj_weight)
            torch.nn.init.xavier_uniform_(module.out_proj.weight)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, features = x.shape
        
        # Advanced input processing
        hidden = self.input_projection(x)
        
        # Add learnable positional encoding
        hidden = hidden + self.positional_encoding[:, :seq_len, :]
        
        # Multi-scale feature extraction
        conv_features = []
        hidden_conv = hidden.transpose(1, 2)  # [B, H, T]
        for conv in self.multi_scale_conv:
            conv_out = F.gelu(conv(hidden_conv))
            conv_features.append(conv_out)
        
        # Combine multi-scale features
        multi_scale = torch.cat(conv_features, dim=1).transpose(1, 2)  # [B, T, H]
        hidden = hidden + multi_scale
        
        # Advanced transformer processing
        for layer in self.transformer_layers:
            hidden = layer(hidden)
        
        hidden = self.final_norm(hidden)
        
        # Attention-based pooling instead of mean
        query = hidden.mean(dim=1, keepdim=True)  # Global representation as query
        pooled, attention_weights = self.attention_pooling(
            query, hidden, hidden
        )
        pooled = pooled.squeeze(1)  # [B, H]
        
        # Generate enhanced outputs
        price_predictions = self.price_head(pooled)
        action_logits = self.action_head(pooled)
        confidence = self.confidence_head(pooled)
        
        # Reshape price predictions
        price_predictions = price_predictions.view(
            batch_size, 
            self.config['prediction_horizon'], 
            self.config['input_features']
        )
        
        return {
            'price_predictions': price_predictions,
            'action_logits': action_logits,
            'action_probs': torch.softmax(action_logits, dim=-1),
            'confidence': confidence,
            'attention_weights': attention_weights
        }


class AdvancedTransformerLayer(nn.Module):
    """Advanced Transformer layer with improvements"""
    
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, 
                 dropout: float, layer_idx: int):
        super().__init__()
        
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        
        # Multi-head attention with improvements
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization (pre-norm style)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Enhanced feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Stochastic depth for regularization
        self.stochastic_depth_prob = 0.1 * (layer_idx + 1) / 8  # Increasing with depth
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        norm_x = self.norm1(x)
        attn_out, _ = self.attention(norm_x, norm_x, norm_x)
        
        # Stochastic depth
        if self.training and torch.rand(1).item() < self.stochastic_depth_prob:
            attn_out = attn_out * 0
        
        x = x + attn_out
        
        # Pre-norm FFN
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        
        # Stochastic depth
        if self.training and torch.rand(1).item() < self.stochastic_depth_prob:
            ffn_out = ffn_out * 0
        
        x = x + ffn_out
        
        return x


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()


class AdvancedLoss(nn.Module):
    """Advanced multi-component loss function"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mse = nn.MSELoss()
        self.focal = FocalLoss(alpha=1, gamma=2)
        self.huber = nn.SmoothL1Loss()
        
    def forward(self, outputs, targets, action_labels):
        # Price prediction loss (Huber loss is more robust)
        price_loss = self.huber(outputs['price_predictions'], targets)
        
        # Action prediction loss (Focal loss for imbalanced classes)
        action_loss = self.focal(outputs['action_logits'], action_labels.squeeze())
        
        # Confidence regularization (encourage confident predictions)
        confidence = outputs['confidence']
        confidence_loss = -torch.log(confidence + 1e-8).mean()
        
        # Total loss with adaptive weighting
        total_loss = (
            price_loss + 
            0.5 * action_loss + 
            0.1 * confidence_loss
        )
        
        return {
            'total_loss': total_loss,
            'price_loss': price_loss,
            'action_loss': action_loss,
            'confidence_loss': confidence_loss
        }


class AdvancedTrainer:
    """Advanced trainer with state-of-the-art techniques"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup paths
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.patience = 0
        self.max_patience = config.get('early_stopping_patience', 10)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.loss_fn = None
        
        # EMA for model weights
        self.ema_model = None
        self.ema_decay = config.get('ema_decay', 0.999)
        
    def setup_logging(self):
        """Setup advanced logging"""
        log_dir = Path('hftraining/logs/advanced_v2')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'advanced_training_{timestamp}.log'
        
        # Clear handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_model(self, input_features: int):
        """Setup advanced model"""
        self.config['input_features'] = input_features
        
        self.model = AdvancedTransformerModel(self.config).to(self.device)
        
        # Setup EMA
        self.ema_model = AdvancedTransformerModel(self.config).to(self.device)
        self.ema_model.load_state_dict(self.model.state_dict())
        
        # Setup loss function
        self.loss_fn = AdvancedLoss(self.config)
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Advanced model with {total_params:,} parameters ({trainable_params:,} trainable)")
        
    def setup_optimizer(self):
        """Setup advanced optimizer with sophisticated scheduling"""
        
        # Parameter groups for different learning rates
        no_decay = ["bias", "LayerNorm.weight", "positional_encoding"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.get('weight_decay', 0.01),
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # Use AdamW with advanced settings
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config['learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Advanced learning rate scheduling
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config['learning_rate'],
            total_steps=self.config['max_steps'],
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=100.0
        )
        
        # Setup mixed precision
        if self.device.type == 'cuda':
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        self.logger.info(f"Advanced optimizer: AdamW with OneCycleLR")
        self.logger.info(f"Max LR: {self.config['learning_rate']}, Total steps: {self.config['max_steps']}")
        
    def update_ema(self):
        """Update EMA model weights"""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data = ema_param.data * self.ema_decay + param.data * (1 - self.ema_decay)
    
    def train_step(self, batch) -> Dict[str, float]:
        """Advanced training step"""
        self.model.train()
        
        # Extract batch data
        inputs = batch['input_ids'].to(self.device)
        targets = batch['labels'].to(self.device)
        action_labels = batch['action_labels'].to(self.device)
        
        # Forward pass with mixed precision
        if self.scaler is not None:
            with autocast():
                outputs = self.model(inputs)
                losses = self.loss_fn(outputs, targets, action_labels)
                total_loss = losses['total_loss']
            
            # Backward pass
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            
            # Advanced gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(inputs)
            losses = self.loss_fn(outputs, targets, action_labels)
            total_loss = losses['total_loss']
            
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        # Update EMA
        self.update_ema()
        
        return {
            'total_loss': total_loss.item(),
            'price_loss': losses['price_loss'].item(),
            'action_loss': losses['action_loss'].item(),
            'confidence_loss': losses['confidence_loss'].item(),
            'learning_rate': self.scheduler.get_last_lr()[0],
            'grad_norm': grad_norm.item()
        }
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Advanced validation with EMA model"""
        self.ema_model.eval()  # Use EMA model for validation
        
        total_losses = {'total': 0, 'price': 0, 'action': 0, 'confidence': 0}
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input_ids'].to(self.device)
                targets = batch['labels'].to(self.device)
                action_labels = batch['action_labels'].to(self.device)
                
                outputs = self.ema_model(inputs)
                losses = self.loss_fn(outputs, targets, action_labels)
                
                total_losses['total'] += losses['total_loss'].item()
                total_losses['price'] += losses['price_loss'].item()
                total_losses['action'] += losses['action_loss'].item()
                total_losses['confidence'] += losses['confidence_loss'].item()
                
                # Calculate accuracy
                predicted_actions = torch.argmax(outputs['action_logits'], dim=1)
                correct_predictions += (predicted_actions == action_labels.squeeze()).sum().item()
                total_predictions += action_labels.size(0)
                
                num_batches += 1
        
        avg_losses = {k: v / max(num_batches, 1) for k, v in total_losses.items()}
        accuracy = correct_predictions / max(total_predictions, 1)
        
        return {
            'val_loss': avg_losses['total'],
            'val_price_loss': avg_losses['price'],
            'val_action_loss': avg_losses['action'],
            'val_confidence_loss': avg_losses['confidence'],
            'val_accuracy': accuracy
        }
    
    def train(self, train_loader, val_loader):
        """Advanced training loop"""
        self.logger.info("="*80)
        self.logger.info("🚀 STARTING ADVANCED TRAINING V2")
        self.logger.info("="*80)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Max Steps: {self.config['max_steps']}")
        self.logger.info(f"EMA Decay: {self.ema_decay}")
        
        best_val_accuracy = 0.0
        
        for epoch in range(self.config['num_epochs']):
            self.epoch = epoch
            self.logger.info(f"\n📈 EPOCH {epoch+1}/{self.config['num_epochs']}")
            self.logger.info("-"*50)
            
            epoch_metrics = []
            
            for batch_idx, batch in enumerate(train_loader):
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)
                
                self.global_step += 1
                
                # Log progress
                if self.global_step % self.config.get('log_interval', 50) == 0:
                    avg_metrics = {k: np.mean([m[k] for m in epoch_metrics[-50:]]) 
                                 for k in metrics.keys()}
                    
                    self.logger.info(
                        f"Step {self.global_step:6d} | "
                        f"Loss: {avg_metrics['total_loss']:.4f} | "
                        f"LR: {avg_metrics['learning_rate']:.2e} | "
                        f"Price: {avg_metrics['price_loss']:.4f} | "
                        f"Action: {avg_metrics['action_loss']:.4f} | "
                        f"Conf: {avg_metrics['confidence_loss']:.4f} | "
                        f"Grad: {avg_metrics['grad_norm']:.3f}"
                    )
                
                # Validation
                if self.global_step % self.config['val_interval'] == 0:
                    val_metrics = self.validate(val_loader)
                    
                    self.logger.info(
                        f"🔍 Val Loss: {val_metrics['val_loss']:.4f} | "
                        f"Val Acc: {val_metrics['val_accuracy']:.3f} | "
                        f"Price: {val_metrics['val_price_loss']:.4f} | "
                        f"Action: {val_metrics['val_action_loss']:.4f}"
                    )
                    
                    # Save best models
                    if val_metrics['val_loss'] < self.best_loss:
                        self.best_loss = val_metrics['val_loss']
                        self.save_checkpoint('best_loss')
                        self.logger.info(f"🏆 New best loss model! {val_metrics['val_loss']:.4f}")
                        self.patience = 0
                    else:
                        self.patience += 1
                    
                    if val_metrics['val_accuracy'] > best_val_accuracy:
                        best_val_accuracy = val_metrics['val_accuracy']
                        self.save_checkpoint('best_accuracy')
                        self.logger.info(f"🎯 New best accuracy model! {val_metrics['val_accuracy']:.3f}")
                    
                    # Early stopping
                    if self.patience >= self.max_patience:
                        self.logger.info(f"⏰ Early stopping after {self.patience} steps without improvement")
                        self.save_checkpoint('final')
                        return
                
                # Max steps check
                if self.global_step >= self.config['max_steps']:
                    self.logger.info(f"✅ Reached max steps: {self.config['max_steps']}")
                    self.save_checkpoint('final')
                    return
            
            # End of epoch
            avg_epoch_metrics = {k: np.mean([m[k] for m in epoch_metrics]) 
                               for k in epoch_metrics[0].keys()}
            self.logger.info(f"📊 Epoch {epoch+1} complete. Avg Loss: {avg_epoch_metrics['total_loss']:.4f}")
            self.save_checkpoint(f'epoch_{epoch+1}')
        
        self.save_checkpoint('final')
        self.logger.info("✅ Advanced training completed!")
    
    def save_checkpoint(self, name: str):
        """Save checkpoint with EMA model"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = self.checkpoint_dir / f'{name}.pt'
        torch.save(checkpoint, path)
        self.logger.info(f"💾 Advanced checkpoint saved: {path}")


def load_enhanced_data():
    """Load data with enhanced preprocessing"""
    try:
        # More diverse symbols for better generalization
        symbols = [
            'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NFLX', 'NVDA',
            'JPM', 'BAC', 'WMT', 'JNJ', 'V', 'PG', 'DIS', 'ADBE'
        ]
        print("📊 Downloading enhanced dataset...")
        
        all_data = []
        
        for symbol in symbols:
            print(f"  • {symbol}")
            try:
                data = yf.download(symbol, start='2019-01-01', end='2025-01-01', progress=False)
                
                if len(data) > 100:  # Ensure sufficient data
                    # Basic OHLCV
                    df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                    
                    # Enhanced technical features
                    df['returns'] = df['Close'].pct_change()
                    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
                    df['price_range'] = (df['High'] - df['Low']) / df['Close']
                    df['close_to_open'] = (df['Close'] - df['Open']) / df['Open']
                    
                    # Advanced volume features
                    volume_col = df['Volume']
                    df['volume_sma'] = volume_col.rolling(20).mean()
                    df['volume_ratio'] = volume_col / (df['volume_sma'] + 1e-8)
                    df['volume_rsi'] = calculate_rsi(volume_col, 14)
                    
                    # Multiple timeframe moving averages
                    for period in [5, 10, 20, 50]:
                        df[f'sma_{period}'] = df['Close'].rolling(period).mean()
                        df[f'sma_{period}_ratio'] = df['Close'] / (df[f'sma_{period}'] + 1e-8)
                    
                    # Advanced technical indicators
                    df['volatility'] = df['returns'].rolling(20).std()
                    df['rsi'] = calculate_rsi(df['Close'], 14)
                    df['rsi_fast'] = calculate_rsi(df['Close'], 7)
                    
                    # MACD with signal
                    df['macd'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
                    df['macd_signal'] = df['macd'].ewm(span=9).mean()
                    df['macd_histogram'] = df['macd'] - df['macd_signal']
                    
                    # Bollinger Bands
                    bb_period = 20
                    bb_std = 2
                    df['bb_middle'] = df['Close'].rolling(bb_period).mean()
                    df['bb_upper'] = df['bb_middle'] + bb_std * df['Close'].rolling(bb_period).std()
                    df['bb_lower'] = df['bb_middle'] - bb_std * df['Close'].rolling(bb_period).std()
                    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
                    
                    # Clean and normalize
                    df = df.ffill().fillna(0)
                    df = df.replace([np.inf, -np.inf], 0)
                    
                    # Select final 21 features to match expected input
                    feature_cols = [
                        'Open', 'High', 'Low', 'Close', 'Volume',
                        'returns', 'log_returns', 'price_range', 'close_to_open',
                        'volume_sma', 'volume_ratio', 'volume_rsi',
                        'sma_5_ratio', 'sma_20_ratio', 'volatility', 'rsi', 'rsi_fast',
                        'macd', 'macd_signal', 'macd_histogram', 'bb_position'
                    ]
                    
                    if len(feature_cols) == 21:  # Ensure exactly 21 features
                        selected_data = df[feature_cols].values
                        if not np.isnan(selected_data).any():
                            all_data.append(selected_data)
            except Exception as e:
                print(f"    Warning: Failed to process {symbol}: {e}")
                continue
        
        if not all_data:
            print("⚠️ No data loaded, using fallback")
            return np.random.randn(10000, 21)
        
        # Combine and normalize
        combined_data = np.vstack(all_data)
        print(f"📈 Combined data shape: {combined_data.shape}")
        
        # Advanced normalization
        from sklearn.preprocessing import RobustScaler, QuantileTransformer
        
        # Use QuantileTransformer for better handling of outliers
        scaler = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
        normalized_data = scaler.fit_transform(combined_data)
        
        return normalized_data
        
    except Exception as e:
        print(f"❌ Error loading enhanced data: {e}")
        return np.random.randn(10000, 21)


def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def main():
    """Main advanced training function"""
    
    # Advanced configuration
    config = {
        # Enhanced model architecture
        'hidden_size': 1024,        # Larger model
        'num_heads': 16,
        'num_layers': 12,           # Deeper model
        'intermediate_size': 4096,   # Larger FFN
        'dropout': 0.15,            # More dropout for regularization
        'sequence_length': 60,
        'prediction_horizon': 5,
        
        # Advanced training parameters
        'batch_size': 16,           # Smaller batch for larger model
        'learning_rate': 1e-4,      # Higher learning rate
        'weight_decay': 0.01,
        'num_epochs': 100,
        'max_steps': 20000,         # More training steps
        'val_interval': 150,
        'log_interval': 50,
        'early_stopping_patience': 15,
        
        # EMA and regularization
        'ema_decay': 0.9999,
        
        # Data loading
        'num_workers': 6,
        
        # Paths
        'checkpoint_dir': 'hftraining/checkpoints/advanced_v2'
    }
    
    print("🚀 Starting ADVANCED TRAINING SYSTEM V2")
    print("="*80)
    print("🎯 State-of-the-art techniques for maximum performance")
    print(json.dumps(config, indent=2))
    
    # Load enhanced data
    print("\n📊 Loading enhanced dataset...")
    data = load_enhanced_data()
    
    # Advanced data splitting
    train_size = int(0.85 * len(data))  # More training data
    val_size = int(0.10 * len(data))
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    
    print(f"📈 Data splits: Train={train_data.shape}, Val={val_data.shape}, Test={test_data.shape}")
    
    # Create advanced data loaders
    print("\n🔄 Creating enhanced data loaders...")
    train_loader = create_robust_dataloader(
        train_data,
        batch_size=config['batch_size'],
        sequence_length=config['sequence_length'],
        prediction_horizon=config['prediction_horizon'],
        shuffle=True,
        num_workers=config['num_workers'],
        augment=True  # Enable augmentation
    )
    
    val_loader = create_robust_dataloader(
        val_data,
        batch_size=config['batch_size'],
        sequence_length=config['sequence_length'],
        prediction_horizon=config['prediction_horizon'],
        shuffle=False,
        num_workers=config['num_workers']//2,
        augment=False
    )
    
    # Initialize advanced trainer
    print("\n⚙️  Setting up advanced trainer...")
    trainer = AdvancedTrainer(config)
    
    # Setup advanced model
    input_features = data.shape[1]
    trainer.setup_model(input_features)
    
    # Setup advanced optimizer
    trainer.setup_optimizer()
    
    # Start advanced training
    print("\n🎯 Starting advanced training...")
    trainer.train(train_loader, val_loader)
    
    print("\n✅ ADVANCED TRAINING V2 COMPLETED!")
    print("🎉 Your model is ready for superior trading performance!")


if __name__ == "__main__":
    main()
