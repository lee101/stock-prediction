#!/usr/bin/env python3
"""
Optimized Training System
Practical improvements for better performance without complexity
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import logging
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from robust_data_pipeline import create_robust_dataloader


class OptimizedTransformerModel(nn.Module):
    """Optimized Transformer with practical improvements"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Optimized model dimensions
        hidden_size = config['hidden_size']
        num_heads = config['num_heads']
        num_layers = config['num_layers']
        
        # Ensure compatibility
        if hidden_size % num_heads != 0:
            hidden_size = (hidden_size // num_heads) * num_heads
        
        self.hidden_size = hidden_size
        self.config = config
        
        # Enhanced input processing
        self.input_projection = nn.Sequential(
            nn.Linear(config['input_features'], hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1))
        )
        
        # Learnable positional encoding (better than sinusoidal)
        self.positional_encoding = nn.Parameter(
            torch.randn(1, config['sequence_length'], hidden_size) * 0.02
        )
        
        # Improved transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=config.get('intermediate_size', hidden_size * 4),
            dropout=config.get('dropout', 0.1),
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LayerNorm for better training
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.final_norm = nn.LayerNorm(hidden_size)
        
        # Attention pooling instead of mean pooling
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads//2,  # Fewer heads for pooling
            dropout=config.get('dropout', 0.1),
            batch_first=True
        )
        
        # Improved output heads
        self.price_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.LayerNorm(hidden_size//2),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(hidden_size//2, config['prediction_horizon'] * config['input_features'])
        )
        
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//4),
            nn.LayerNorm(hidden_size//4),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(hidden_size//4, 3)  # Buy, Hold, Sell
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Proper weight initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, features = x.shape
        
        # Enhanced input processing
        hidden = self.input_projection(x)
        
        # Add learnable positional encoding
        hidden = hidden + self.positional_encoding[:, :seq_len, :]
        
        # Transformer processing
        hidden = self.transformer(hidden)
        hidden = self.final_norm(hidden)
        
        # Attention-based pooling
        query = hidden.mean(dim=1, keepdim=True)  # Global query
        pooled, attention_weights = self.attention_pooling(
            query, hidden, hidden
        )
        pooled = pooled.squeeze(1)  # [B, H]
        
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
            'action_probs': torch.softmax(action_logits, dim=-1),
            'attention_weights': attention_weights
        }


class ImprovedLoss(nn.Module):
    """Improved multi-objective loss function"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.huber = nn.SmoothL1Loss()  # More robust than MSE
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, outputs, targets, action_labels):
        # Price prediction loss (Huber loss is more robust to outliers)
        price_loss = self.huber(outputs['price_predictions'], targets)
        
        # Action prediction loss
        action_loss = self.ce(outputs['action_logits'], action_labels.squeeze())
        
        # Weighted combination
        total_loss = price_loss + 0.5 * action_loss
        
        return {
            'total_loss': total_loss,
            'price_loss': price_loss,
            'action_loss': action_loss
        }


class OptimizedTrainer:
    """Optimized trainer with practical improvements"""
    
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
        self.best_accuracy = 0.0
        self.patience = 0
        self.max_patience = config.get('early_stopping_patience', 8)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.loss_fn = None
        
    def setup_logging(self):
        """Setup optimized logging"""
        log_dir = Path('hftraining/logs/optimized')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'optimized_training_{timestamp}.log'
        
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
        """Setup optimized model"""
        self.config['input_features'] = input_features
        
        self.model = OptimizedTransformerModel(self.config).to(self.device)
        self.loss_fn = ImprovedLoss(self.config)
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Optimized model: {total_params:,} parameters ({trainable_params:,} trainable)")
        
    def setup_optimizer(self):
        """Setup optimized optimizer and scheduler"""
        
        # Parameter groups for differential learning rates
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
        
        # Use Shampoo optimizer for better convergence
        try:
            from modern_optimizers import Shampoo
            self.optimizer = Shampoo(
                optimizer_grouped_parameters,
                lr=self.config['learning_rate'],
                betas=(0.9, 0.999),
                eps=1e-10,
                weight_decay=0.0  # Already handled in param groups
            )
            self.logger.info("Using Shampoo optimizer")
        except ImportError:
            self.logger.warning("Shampoo not available, falling back to AdamW")
            self.optimizer = optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.config['learning_rate'],
                betas=(0.9, 0.999),
                eps=1e-8
            )
        
        # Use CosineAnnealingWarmRestarts for better convergence
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=500,  # Restart every 500 steps
            T_mult=1,  # Keep same period
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        # Setup mixed precision
        if self.device.type == 'cuda':
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        self.logger.info(f"Optimizer: AdamW with CosineAnnealingWarmRestarts")
        self.logger.info(f"LR: {self.config['learning_rate']}, T_0: 500")
        
    def train_step(self, batch) -> Dict[str, float]:
        """Optimized training step"""
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
            
            # Gradient clipping
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
        
        return {
            'total_loss': total_loss.item(),
            'price_loss': losses['price_loss'].item(),
            'action_loss': losses['action_loss'].item(),
            'learning_rate': self.scheduler.get_last_lr()[0],
            'grad_norm': grad_norm.item()
        }
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Optimized validation"""
        self.model.eval()
        
        total_losses = {'total': 0, 'price': 0, 'action': 0}
        correct_predictions = 0
        total_predictions = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input_ids'].to(self.device)
                targets = batch['labels'].to(self.device)
                action_labels = batch['action_labels'].to(self.device)
                
                outputs = self.model(inputs)
                losses = self.loss_fn(outputs, targets, action_labels)
                
                total_losses['total'] += losses['total_loss'].item()
                total_losses['price'] += losses['price_loss'].item()
                total_losses['action'] += losses['action_loss'].item()
                
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
            'val_accuracy': accuracy
        }
    
    def train(self, train_loader, val_loader):
        """Optimized training loop"""
        self.logger.info("="*80)
        self.logger.info("🚀 STARTING OPTIMIZED TRAINING")
        self.logger.info("="*80)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Max Steps: {self.config['max_steps']}")
        
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
                    improved = False
                    if val_metrics['val_loss'] < self.best_loss:
                        self.best_loss = val_metrics['val_loss']
                        self.save_checkpoint('best_loss')
                        self.logger.info(f"🏆 New best loss! {val_metrics['val_loss']:.4f}")
                        improved = True
                    
                    if val_metrics['val_accuracy'] > self.best_accuracy:
                        self.best_accuracy = val_metrics['val_accuracy']
                        self.save_checkpoint('best_accuracy')
                        self.logger.info(f"🎯 New best accuracy! {val_metrics['val_accuracy']:.3f}")
                        improved = True
                    
                    if improved:
                        self.patience = 0
                    else:
                        self.patience += 1
                    
                    # Early stopping
                    if self.patience >= self.max_patience:
                        self.logger.info(f"⏰ Early stopping after {self.patience} validations without improvement")
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
        self.logger.info("✅ Optimized training completed!")
    
    def save_checkpoint(self, name: str):
        """Save optimized checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'best_accuracy': self.best_accuracy,
            'config': self.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = self.checkpoint_dir / f'{name}.pt'
        torch.save(checkpoint, path)
        self.logger.info(f"💾 Checkpoint saved: {path}")


def load_optimized_data():
    """Load data with optimized preprocessing from trainingdata/ CSVs (no downloads)."""
    try:
        data_dir = Path("trainingdata")
        print("Loading optimized dataset from local CSVs...")

        csv_files = list(data_dir.glob("*.csv"))
        all_data = []
        for csv in csv_files:
            try:
                df = pd.read_csv(csv)
                # Normalize to capitalized column names for consistency
                df.columns = df.columns.str.title()
                if not set(['Open','High','Low','Close','Volume']).issubset(df.columns):
                    continue

                # Technical features mirroring previous pipeline
                df['returns'] = df['Close'].pct_change()
                df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
                df['price_range'] = (df['High'] - df['Low']) / (df['Close'] + 1e-8)
                df['close_to_open'] = (df['Close'] - df['Open']) / (df['Open'] + 1e-8)
                df['volume_sma'] = df['Volume'].rolling(20).mean()
                df['volume_ratio'] = df['Volume'] / (df['volume_sma'] + 1e-8)
                for period in [5, 10, 20]:
                    sma_col = f'sma_{period}'
                    df[sma_col] = df['Close'].rolling(period).mean()
                    df[f'{sma_col}_ratio'] = df['Close'] / (df[sma_col] + 1e-8)
                df['volatility'] = df['returns'].rolling(20).std()
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
                df['macd'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
                df['macd_signal'] = df['macd'].ewm(span=9).mean()

                feature_cols = [
                    'Open', 'High', 'Low', 'Close', 'Volume',
                    'returns', 'log_returns', 'price_range', 'close_to_open',
                    'volume_sma', 'volume_ratio',
                    'sma_5', 'sma_5_ratio', 'sma_10', 'sma_10_ratio',
                    'sma_20', 'sma_20_ratio', 'volatility', 'rsi', 'macd', 'macd_signal'
                ]

                df = df.ffill().fillna(0).replace([np.inf, -np.inf], 0)
                if set(feature_cols).issubset(df.columns):
                    arr = df[feature_cols].values
                    if not np.isnan(arr).any():
                        all_data.append(arr)
                        print(f"    {csv.name}: {arr.shape[0]} samples")
            except Exception as e:
                print(f"    Failed {csv.name}: {e}")
                continue
        
        if not all_data:
            print("⚠️ Using fallback random data")
            return np.random.randn(10000, 21)
        
        # Combine data
        combined_data = np.vstack(all_data)
        print(f"📈 Total combined data: {combined_data.shape}")
        
        # Robust normalization
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        normalized_data = scaler.fit_transform(combined_data)
        
        return normalized_data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return np.random.randn(10000, 21)


def main():
    """Main optimized training function"""
    
    # Optimized configuration
    config = {
        # Model architecture - balanced size
        'hidden_size': 768,         # Good balance
        'num_heads': 16,
        'num_layers': 10,           # Deep but manageable
        'intermediate_size': 3072,  # 4x hidden_size
        'dropout': 0.1,
        'sequence_length': 60,
        'prediction_horizon': 5,
        
        # Training parameters - optimized for convergence
        'batch_size': 24,           # Good compromise
        'learning_rate': 8e-5,      # Slightly higher
        'weight_decay': 0.01,
        'num_epochs': 80,
        'max_steps': 15000,
        'val_interval': 100,
        'log_interval': 50,
        'early_stopping_patience': 12,
        
        # Data loading
        'num_workers': 4,
        
        # Paths
        'checkpoint_dir': 'hftraining/checkpoints/optimized'
    }
    
    print("Starting optimized training system")
    print("="*80)
    print(json.dumps(config, indent=2))
    
    # Load optimized data
    data = load_optimized_data()
    
    # Data splitting
    train_size = int(0.85 * len(data))
    val_size = int(0.10 * len(data))
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    
    print(f"Data splits: Train={train_data.shape}, Val={val_data.shape}, Test={test_data.shape}")
    
    # Create data loaders
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
    trainer = OptimizedTrainer(config)
    
    # Setup model
    input_features = data.shape[1]
    trainer.setup_model(input_features)
    
    # Setup optimizer
    trainer.setup_optimizer()
    
    # Start training
    trainer.train(train_loader, val_loader)
    
    print("\nOptimized training completed")
    
    # Profit calculation over 30 days
    print("\nCalculating 30-day profit simulation...")
    profit = calculate_trading_profit(trainer.model, test_data, config, days=30)
    print(f"Estimated 30-day profit: ${profit:.2f}")
    print(f"Return: {profit/config.get('initial_capital', 10000)*100:.2f}%")


def calculate_trading_profit(model, test_data, config, days=30, initial_capital=10000):
    """Calculate estimated trading profit over specified days"""
    
    model.eval()
    
    # Simulate trading over test period - use more samples for better estimation
    sequence_length = config['sequence_length']
    available_samples = len(test_data) - sequence_length
    step_size = max(1, available_samples // (days * 2))  # 2 decisions per day
    
    capital = initial_capital
    shares = 0
    buy_price = 0
    trades = 0
    successful_trades = 0
    trade_log = []
    
    with torch.no_grad():
        for i in range(0, available_samples, step_size):
            if i + sequence_length >= len(test_data):
                break
                
            # Get sequence
            sequence = torch.FloatTensor(test_data[i:i + sequence_length]).unsqueeze(0)
            
            # Get prediction
            try:
                output = model(sequence)
                action_probs = output['action_probs']
                predicted_action = torch.argmax(action_probs, dim=-1).item()
                confidence = torch.max(action_probs).item()
                
                # Use normalized price change as proxy (close price at index 3)
                current_normalized = test_data[i + sequence_length - 1, 3]
                
                # Get next price if available
                next_normalized = None
                if i + sequence_length < len(test_data):
                    next_normalized = test_data[i + sequence_length, 3]
                
                # Convert normalized price to reasonable dollar values (assume avg stock ~$100)
                base_price = 100
                current_price = base_price * (1 + current_normalized)  # normalized around 0
                
                # Trade execution with relaxed constraints
                if confidence > 0.35:  # Lower confidence threshold
                    if predicted_action == 0 and shares == 0:  # Buy signal
                        max_shares = int(capital * 0.8 / current_price)  # Use 80% of capital
                        if max_shares > 0:
                            shares = max_shares
                            buy_cost = shares * current_price
                            capital -= buy_cost
                            buy_price = current_price
                            trades += 1
                            trade_log.append(f'BUY {shares} @ ${current_price:.2f}')
                    
                    elif predicted_action == 2 and shares > 0:  # Sell signal
                        sell_value = shares * current_price
                        capital += sell_value
                        profit_on_trade = sell_value - (shares * buy_price)
                        if profit_on_trade > 0:
                            successful_trades += 1
                        trade_log.append(f'SELL {shares} @ ${current_price:.2f} (P/L: ${profit_on_trade:.2f})')
                        shares = 0
                        trades += 1
                        
            except Exception as e:
                continue
    
    # Close any remaining positions at final price
    if shares > 0:
        final_normalized = test_data[-1, 3]
        final_price = base_price * (1 + final_normalized)
        sell_value = shares * final_price
        capital += sell_value
        profit_on_trade = sell_value - (shares * buy_price)
        if profit_on_trade > 0:
            successful_trades += 1
        trade_log.append(f'FINAL SELL {shares} @ ${final_price:.2f} (P/L: ${profit_on_trade:.2f})')
        shares = 0
    
    profit = capital - initial_capital
    success_rate = successful_trades / max(trades, 1) * 100
    
    print(f"Total trades: {trades}")
    print(f"Successful trades: {successful_trades}")
    print(f"Success rate: {success_rate:.1f}%")
    
    # Show last few trades
    if trade_log:
        print("Recent trades:")
        for trade in trade_log[-5:]:
            print(f"  {trade}")
    
    return profit


if __name__ == "__main__":
    main()
