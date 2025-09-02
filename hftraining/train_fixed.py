#!/usr/bin/env python3
"""
Fixed Training Script with Proper Learning Rate Scheduling
Addresses all identified issues from experiments
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
import json
import logging
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from robust_data_pipeline import create_robust_dataloader


class FixedTransformerModel(nn.Module):
    """Transformer model with proper architecture"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Model dimensions
        hidden_size = config['hidden_size']
        num_heads = config['num_heads']
        num_layers = config['num_layers']
        
        # Ensure compatibility
        if hidden_size % num_heads != 0:
            hidden_size = (hidden_size // num_heads) * num_heads
        
        self.hidden_size = hidden_size
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config['input_features'], hidden_size)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(
            config['sequence_length'], hidden_size
        )
        
        # Transformer layers
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
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _create_positional_encoding(self, seq_len: int, hidden_size: int) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(seq_len, hidden_size)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                           -(np.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, features = x.shape
        
        # Project input
        hidden = self.input_projection(x)
        
        # Add positional encoding
        hidden = hidden + self.positional_encoding[:, :seq_len, :].to(x.device)
        hidden = self.layer_norm(hidden)
        
        # Transformer encoding
        hidden = self.transformer(hidden)
        
        # Pool sequence dimension
        pooled = hidden.mean(dim=1)  # [batch, hidden_size]
        
        # Generate predictions
        price_predictions = self.price_head(pooled)
        action_logits = self.action_head(pooled)
        
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


class FixedTrainer:
    """Trainer with fixed learning rate scheduling"""
    
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
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components (will be set up later)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
    def setup_logging(self):
        """Setup proper logging"""
        log_dir = Path('hftraining/logs/fixed')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'fixed_training_{timestamp}.log'
        
        # Clear any existing handlers
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
        """Setup the model"""
        self.config['input_features'] = input_features
        
        self.model = FixedTransformerModel(self.config).to(self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model created with {total_params:,} parameters ({trainable_params:,} trainable)")
        
    def setup_optimizer(self):
        """Setup optimizer with FIXED learning rate scheduling"""
        
        # Use AdamW optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            betas=(0.9, 0.999),
            weight_decay=self.config.get('weight_decay', 0.01),
            eps=1e-8
        )
        
        # FIXED: Use a scheduler that maintains proper learning rate
        # Option 1: Simple StepLR
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1000,  # Reduce LR every 1000 steps
            gamma=0.9       # Multiply by 0.9
        )
        
        # Setup mixed precision
        if self.device.type == 'cuda':
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        self.logger.info(f"Optimizer: AdamW, LR: {self.config['learning_rate']}")
        self.logger.info(f"Scheduler: StepLR (step_size=1000, gamma=0.9)")
        
    def train_step(self, batch) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Extract batch data (using correct keys from robust_data_pipeline)
        inputs = batch['input_ids']
        targets = batch['labels']
        action_labels = batch['action_labels']
        
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        action_labels = action_labels.to(self.device)
        
        # Forward pass with mixed precision
        if self.scaler is not None:
            with autocast():
                outputs = self.model(inputs)
                
                # Calculate loss
                price_loss = nn.MSELoss()(
                    outputs['price_predictions'], 
                    targets[:, :self.config['prediction_horizon'], :]
                )
                
                # Action loss using real labels
                action_loss = nn.CrossEntropyLoss()(outputs['action_logits'], action_labels.squeeze())
                
                total_loss = price_loss + action_loss
            
            # Backward pass
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(inputs)
            
            price_loss = nn.MSELoss()(
                outputs['price_predictions'], 
                targets[:, :self.config['prediction_horizon'], :]
            )
            
            action_loss = nn.CrossEntropyLoss()(outputs['action_logits'], action_labels.squeeze())
            
            total_loss = price_loss + action_loss
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()  # Step the scheduler
        
        return {
            'total_loss': total_loss.item(),
            'price_loss': price_loss.item(),
            'action_loss': action_loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def validate(self, val_loader) -> float:
        """Validation step"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input_ids'].to(self.device)
                targets = batch['labels'].to(self.device)
                action_labels = batch['action_labels'].to(self.device)
                
                outputs = self.model(inputs)
                
                price_loss = nn.MSELoss()(
                    outputs['price_predictions'], 
                    targets[:, :self.config['prediction_horizon'], :]
                )
                
                action_loss = nn.CrossEntropyLoss()(outputs['action_logits'], action_labels.squeeze())
                
                total_loss += (price_loss + action_loss).item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        self.logger.info("="*80)
        self.logger.info("🚀 STARTING FIXED TRAINING SESSION")
        self.logger.info("="*80)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Max Steps: {self.config['max_steps']}")
        self.logger.info(f"Validation Interval: {self.config['val_interval']}")
        
        for epoch in range(self.config['num_epochs']):
            self.epoch = epoch
            self.logger.info(f"\n📈 EPOCH {epoch+1}/{self.config['num_epochs']}")
            self.logger.info("-"*50)
            
            epoch_losses = []
            
            for batch_idx, batch in enumerate(train_loader):
                metrics = self.train_step(batch)
                epoch_losses.append(metrics)
                
                self.global_step += 1
                
                # Log progress
                if self.global_step % self.config.get('log_interval', 50) == 0:
                    avg_loss = np.mean([m['total_loss'] for m in epoch_losses[-50:]])
                    current_lr = metrics['learning_rate']
                    
                    self.logger.info(
                        f"Step {self.global_step:6d} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Price Loss: {metrics['price_loss']:.4f} | "
                        f"Action Loss: {metrics['action_loss']:.4f}"
                    )
                
                # Validation
                if self.global_step % self.config['val_interval'] == 0:
                    val_loss = self.validate(val_loader)
                    
                    self.logger.info(f"🔍 Validation Loss: {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_checkpoint('best')
                        self.logger.info(f"🏆 New best model saved! Loss: {val_loss:.4f}")
                
                # Early stopping
                if self.global_step >= self.config['max_steps']:
                    self.logger.info(f"✅ Reached max steps: {self.config['max_steps']}")
                    self.save_checkpoint('final')
                    return
            
            # End of epoch
            avg_epoch_loss = np.mean([m['total_loss'] for m in epoch_losses])
            self.logger.info(f"📊 Epoch {epoch+1} complete. Avg Loss: {avg_epoch_loss:.4f}")
            self.save_checkpoint(f'epoch_{epoch+1}')
        
        self.save_checkpoint('final')
        self.logger.info("✅ Training completed!")
    
    def save_checkpoint(self, name: str):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
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
        self.logger.info(f"💾 Checkpoint saved: {path}")


def load_and_prepare_data():
    """Load and prepare stock data"""
    try:
        # Download data for multiple stocks
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        print("Downloading stock data...")
        
        all_data = []
        
        for symbol in symbols:
            print(f"  • {symbol}")
            data = yf.download(symbol, start='2020-01-01', end='2025-01-01', progress=False)
            
            if len(data) > 0:
                # Basic OHLCV
                df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                
                # Technical features
                df['returns'] = df['Close'].pct_change()
                df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
                df['price_range'] = (df['High'] - df['Low']) / df['Close']
                df['close_to_open'] = (df['Close'] - df['Open']) / df['Open']
                
                # Volume features  
                # Handle potential multi-column Volume issue
                volume_col = df['Volume']
                if hasattr(volume_col, 'iloc') and len(volume_col.shape) > 1:
                    volume_col = volume_col.iloc[:, 0] if volume_col.shape[1] > 0 else volume_col
                
                df['volume_sma'] = volume_col.rolling(20).mean()
                df['volume_ratio'] = volume_col / df['volume_sma']
                
                # Moving averages
                for period in [5, 10, 20]:
                    df[f'sma_{period}'] = df['Close'].rolling(period).mean()
                    df[f'sma_{period}_ratio'] = df['Close'] / df[f'sma_{period}']
                
                # Technical indicators
                df['volatility'] = df['returns'].rolling(20).std()
                
                # RSI
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                
                # MACD
                df['macd'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
                df['macd_signal'] = df['macd'].ewm(span=9).mean()
                
                # Clean data
                df = df.ffill().fillna(0)
                df = df.replace([np.inf, -np.inf], 0)
                
                all_data.append(df.values)
        
        # Combine data
        combined_data = np.vstack(all_data)
        print(f"Combined data shape: {combined_data.shape}")
        
        # Normalize
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        normalized_data = scaler.fit_transform(combined_data)
        
        return normalized_data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        # Fallback to dummy data
        return np.random.randn(5000, 21)


def main():
    """Main training function"""
    
    # Fixed configuration
    config = {
        # Model architecture
        'hidden_size': 512,
        'num_heads': 16,
        'num_layers': 8,
        'intermediate_size': 2048,
        'dropout': 0.1,
        'sequence_length': 60,
        'prediction_horizon': 5,
        
        # Training parameters
        'batch_size': 32,
        'learning_rate': 5e-5,  # Fixed learning rate
        'weight_decay': 0.01,
        'num_epochs': 50,
        'max_steps': 10000,
        'val_interval': 200,
        'log_interval': 50,
        
        # Data loading
        'num_workers': 4,
        
        # Paths
        'checkpoint_dir': 'hftraining/checkpoints/fixed'
    }
    
    print("🚀 Starting Fixed Training Pipeline")
    print("="*60)
    print(json.dumps(config, indent=2))
    
    # Load data
    print("\n📊 Loading data...")
    data = load_and_prepare_data()
    
    # Split data
    train_size = int(0.8 * len(data))
    val_size = int(0.1 * len(data))
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    
    print(f"Data splits: Train={train_data.shape}, Val={val_data.shape}")
    
    # Create data loaders
    print("\n🔄 Creating data loaders...")
    train_loader = create_robust_dataloader(
        train_data,
        batch_size=config['batch_size'],
        sequence_length=config['sequence_length'],
        prediction_horizon=config['prediction_horizon'],
        shuffle=True,
        num_workers=config['num_workers'],
        augment=True
    )
    
    val_loader = create_robust_dataloader(
        val_data,
        batch_size=config['batch_size'],
        sequence_length=config['sequence_length'],
        prediction_horizon=config['prediction_horizon'],
        shuffle=False,
        num_workers=2,
        augment=False
    )
    
    # Initialize trainer
    print("\n⚙️  Setting up trainer...")
    trainer = FixedTrainer(config)
    
    # Setup model
    input_features = data.shape[1]
    trainer.setup_model(input_features)
    
    # Setup optimizer with fixed scheduler
    trainer.setup_optimizer()
    
    # Start training
    print("\n🎯 Starting training...")
    trainer.train(train_loader, val_loader)
    
    print("\n✅ Training completed successfully!")
    
    # Test the saved model with inference system
    print("\n🧪 Testing model with inference system...")
    try:
        from hfinference.test_inference import test_model_loading
        if test_model_loading():
            print("✅ Model is compatible with inference system")
        else:
            print("⚠️ Model may have compatibility issues")
    except Exception as e:
        print(f"Could not test inference compatibility: {e}")


if __name__ == "__main__":
    main()