#!/usr/bin/env python3
"""
Advanced Training Pipeline with Comprehensive Logging and Analysis
Implements an improvement cycle for better loss optimization
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
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training/training_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingMetricsLogger:
    """Comprehensive metrics logger for training analysis"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / 'metrics.jsonl'
        self.summary_file = self.log_dir / 'summary.json'
        
        self.metrics_history = defaultdict(list)
        self.current_epoch = 0
        self.start_time = time.time()
        
    def log_batch(self, batch_idx: int, metrics: Dict[str, float]):
        """Log batch-level metrics"""
        entry = {
            'epoch': self.current_epoch,
            'batch': batch_idx,
            'timestamp': time.time() - self.start_time,
            **metrics
        }
        
        # Save to file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        # Update history
        for key, value in metrics.items():
            self.metrics_history[f'batch_{key}'].append(value)
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch-level metrics"""
        self.current_epoch = epoch
        
        for key, value in metrics.items():
            self.metrics_history[f'epoch_{key}'].append(value)
        
        # Calculate improvement metrics
        if len(self.metrics_history['epoch_loss']) > 1:
            prev_loss = self.metrics_history['epoch_loss'][-2]
            curr_loss = self.metrics_history['epoch_loss'][-1]
            improvement = (prev_loss - curr_loss) / prev_loss * 100
            self.metrics_history['loss_improvement'].append(improvement)
            logger.info(f"Loss improvement: {improvement:.2f}%")
    
    def analyze_training(self) -> Dict[str, Any]:
        """Analyze training metrics and provide insights"""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'total_training_time': float(time.time() - self.start_time),
            'epochs_trained': int(self.current_epoch),
        }
        
        # Loss analysis
        if 'epoch_loss' in self.metrics_history:
            losses = self.metrics_history['epoch_loss']
            # Filter out NaN values
            valid_losses = [l for l in losses if not np.isnan(l)]
            
            if valid_losses:
                analysis['loss_stats'] = {
                    'initial': float(valid_losses[0]) if valid_losses else 0,
                    'final': float(valid_losses[-1]) if valid_losses else 0,
                    'best': float(min(valid_losses)) if valid_losses else 0,
                    'worst': float(max(valid_losses)) if valid_losses else 0,
                    'mean': float(np.mean(valid_losses)) if valid_losses else 0,
                    'std': float(np.std(valid_losses)) if valid_losses else 0,
                    'total_reduction': float(valid_losses[0] - valid_losses[-1]) if len(valid_losses) > 1 else 0,
                    'percent_reduction': float((valid_losses[0] - valid_losses[-1]) / valid_losses[0] * 100) if len(valid_losses) > 1 and valid_losses[0] != 0 else 0
                }
                
                # Detect plateaus
                if len(valid_losses) > 10:
                    recent_std = np.std(valid_losses[-10:])
                    analysis['plateau_detected'] = bool(recent_std < 0.001)
                
                # Learning rate effectiveness
                if 'epoch_lr' in self.metrics_history:
                    lrs = self.metrics_history['epoch_lr']
                    if len(valid_losses) > 1 and len(lrs) > 1:
                        try:
                            analysis['lr_correlation'] = float(np.corrcoef(valid_losses[:len(lrs)], lrs[:len(valid_losses)])[0, 1])
                        except:
                            analysis['lr_correlation'] = 0.0
        
        # Gradient analysis
        if 'batch_grad_norm' in self.metrics_history:
            grad_norms = self.metrics_history['batch_grad_norm']
            valid_grads = [g for g in grad_norms if not np.isnan(g)]
            
            if valid_grads:
                analysis['gradient_stats'] = {
                    'mean': float(np.mean(valid_grads)),
                    'std': float(np.std(valid_grads)),
                    'max': float(max(valid_grads)),
                    'exploding_gradients': bool(max(valid_grads) > 100)
                }
        
        # Save analysis
        with open(self.summary_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    def plot_metrics(self):
        """Generate training visualization plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Loss curve
        if 'epoch_loss' in self.metrics_history:
            axes[0, 0].plot(self.metrics_history['epoch_loss'])
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
        
        # Learning rate schedule
        if 'epoch_lr' in self.metrics_history:
            axes[0, 1].plot(self.metrics_history['epoch_lr'])
            axes[0, 1].set_title('Learning Rate')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('LR')
            axes[0, 1].grid(True)
        
        # Loss improvement
        if 'loss_improvement' in self.metrics_history:
            axes[0, 2].bar(range(len(self.metrics_history['loss_improvement'])), 
                          self.metrics_history['loss_improvement'])
            axes[0, 2].set_title('Loss Improvement per Epoch')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Improvement (%)')
            axes[0, 2].grid(True)
        
        # Gradient norms
        if 'batch_grad_norm' in self.metrics_history:
            axes[1, 0].hist(self.metrics_history['batch_grad_norm'], bins=50)
            axes[1, 0].set_title('Gradient Norm Distribution')
            axes[1, 0].set_xlabel('Gradient Norm')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True)
        
        # Accuracy if available
        if 'epoch_accuracy' in self.metrics_history:
            axes[1, 1].plot(self.metrics_history['epoch_accuracy'])
            axes[1, 1].set_title('Training Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].grid(True)
        
        # Loss vs LR scatter
        if 'epoch_loss' in self.metrics_history and 'epoch_lr' in self.metrics_history:
            axes[1, 2].scatter(self.metrics_history['epoch_lr'][:len(self.metrics_history['epoch_loss'])],
                              self.metrics_history['epoch_loss'][:len(self.metrics_history['epoch_lr'])])
            axes[1, 2].set_title('Loss vs Learning Rate')
            axes[1, 2].set_xlabel('Learning Rate')
            axes[1, 2].set_ylabel('Loss')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_analysis.png', dpi=150)
        plt.close()


class ImprovedStockDataset(Dataset):
    """Enhanced dataset with better preprocessing"""
    
    def __init__(self, data_path: str, sequence_length: int = 60, augment: bool = True):
        self.sequence_length = sequence_length
        self.augment = augment
        
        # Load data
        if Path(data_path).exists():
            self.data = pd.read_csv(data_path)
        else:
            # Generate synthetic data for testing
            logger.warning(f"Data file not found: {data_path}. Using synthetic data.")
            self.data = self._generate_synthetic_data()
        
        # Preprocess
        self.features = self._prepare_features()
        self.targets = self._prepare_targets()
        
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic stock data for testing"""
        n_samples = 10000
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='1h')
        
        # Generate realistic price movement
        returns = np.random.normal(0.0001, 0.02, n_samples)
        price = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': price * (1 + np.random.normal(0, 0.001, n_samples)),
            'high': price * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
            'low': price * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
            'close': price,
            'volume': np.random.lognormal(15, 1, n_samples)
        })
        
        # Add technical indicators
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()
        data['rsi'] = self._calculate_rsi(data['close'])
        
        return data.dropna()
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _prepare_features(self) -> torch.Tensor:
        """Prepare and normalize features"""
        feature_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Add if available
        for col in ['sma_20', 'sma_50', 'rsi']:
            if col in self.data.columns:
                feature_cols.append(col)
        
        features = self.data[feature_cols].values
        
        # Normalize
        self.feature_mean = features.mean(axis=0)
        self.feature_std = features.std(axis=0) + 1e-8
        features = (features - self.feature_mean) / self.feature_std
        
        return torch.FloatTensor(features)
    
    def _prepare_targets(self) -> torch.Tensor:
        """Prepare targets (next price movement)"""
        if 'close' in self.data.columns:
            prices = self.data['close'].values
            returns = np.diff(prices) / prices[:-1]
            
            # Classification: 0=down, 1=neutral, 2=up
            targets = np.zeros(len(returns))
            targets[returns < -0.001] = 0
            targets[returns > 0.001] = 2
            targets[(returns >= -0.001) & (returns <= 0.001)] = 1
            
            # Pad to match features length
            targets = np.concatenate([[1], targets])  # Add neutral for first sample
        else:
            targets = np.random.randint(0, 3, len(self.features))
        
        return torch.LongTensor(targets)
    
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        # Get sequence
        x = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        
        # Data augmentation
        if self.augment and torch.rand(1).item() > 0.5:
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        
        return x, y


class ImprovedTransformerModel(nn.Module):
    """Enhanced Transformer with modern techniques"""
    
    def __init__(self, input_dim=8, hidden_dim=128, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer layers with improvements
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for better stability
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output heads
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # 3 classes: down, neutral, up
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.5)  # Reduced gain for stability
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Project input
        x = self.input_projection(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Use last timestep for classification
        x = x[:, -1, :]
        
        # Classification
        return self.classifier(x)


class AdaptiveOptimizer:
    """Adaptive optimizer that adjusts based on training progress"""
    
    def __init__(self, model, initial_lr=1e-3):
        self.model = model
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        
        # Try different optimizers
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=initial_lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        self.loss_history = []
        self.patience_counter = 0
        
    def step(self, loss):
        """Optimizer step with adaptive adjustments"""
        self.optimizer.step()
        self.scheduler.step()
        
        # Track loss
        self.loss_history.append(loss)
        
        # Adaptive adjustments
        if len(self.loss_history) > 20:
            recent_losses = self.loss_history[-20:]
            
            # Check for plateau
            if np.std(recent_losses) < 1e-4:
                self.patience_counter += 1
                
                if self.patience_counter > 5:
                    # Restart with new learning rate
                    logger.info("Plateau detected, adjusting learning rate")
                    new_lr = self.current_lr * 0.5
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    self.current_lr = new_lr
                    self.patience_counter = 0
            else:
                self.patience_counter = 0
        
        return self.optimizer.param_groups[0]['lr']
    
    def zero_grad(self):
        self.optimizer.zero_grad()


def train_with_analysis(config: Dict[str, Any]):
    """Main training function with comprehensive analysis"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(f'training/runs/run_{timestamp}')
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize logger
    metrics_logger = TrainingMetricsLogger(run_dir)
    
    # Data
    logger.info("Loading data...")
    train_dataset = ImprovedStockDataset(
        config.get('data_path', 'data/train.csv'),
        sequence_length=config.get('sequence_length', 60),
        augment=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Model
    logger.info("Initializing model...")
    model = ImprovedTransformerModel(
        input_dim=train_dataset.features.shape[1],
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 4),
        num_heads=config.get('num_heads', 8),
        dropout=config.get('dropout', 0.1)
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdaptiveOptimizer(model, initial_lr=config.get('learning_rate', 1e-3))
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training loop
    logger.info("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(config.get('num_epochs', 100)):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            # Check for NaN
            if torch.isnan(loss):
                logger.warning(f"NaN loss detected at epoch {epoch}, batch {batch_idx}. Skipping...")
                continue
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Check for NaN gradients
            if torch.isnan(grad_norm):
                logger.warning(f"NaN gradients detected. Skipping update...")
                optimizer.zero_grad()
                continue
            
            # Optimizer step
            scaler.step(optimizer.optimizer)
            scaler.update()
            current_lr = optimizer.step(loss.item())
            
            # Metrics
            epoch_loss += loss.item()
            pred = output.argmax(dim=1)
            epoch_correct += (pred == target).sum().item()
            epoch_total += target.size(0)
            
            # Log batch metrics
            if batch_idx % 10 == 0:
                batch_metrics = {
                    'loss': loss.item(),
                    'grad_norm': grad_norm.item(),
                    'lr': current_lr
                }
                metrics_logger.log_batch(batch_idx, batch_metrics)
        
        # Epoch metrics
        avg_loss = epoch_loss / len(train_loader)
        accuracy = epoch_correct / epoch_total
        
        epoch_metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'lr': current_lr
        }
        metrics_logger.log_epoch(epoch, epoch_metrics)
        
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']}: "
                   f"Loss={avg_loss:.4f}, Acc={accuracy:.4f}, LR={current_lr:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.optimizer.state_dict(),
                'loss': best_loss,
            }, run_dir / 'best_model.pth')
            logger.info(f"Saved best model with loss {best_loss:.4f}")
        
        # Periodic analysis
        if (epoch + 1) % 10 == 0:
            analysis = metrics_logger.analyze_training()
            logger.info(f"Training Analysis: {json.dumps(analysis, indent=2)}")
            
            # Suggest improvements
            if analysis.get('plateau_detected', False):
                logger.warning("Training plateau detected! Consider:")
                logger.warning("- Reducing learning rate")
                logger.warning("- Increasing model capacity")
                logger.warning("- Adding more data augmentation")
    
    # Final analysis
    logger.info("Training completed! Generating final analysis...")
    final_analysis = metrics_logger.analyze_training()
    metrics_logger.plot_metrics()
    
    # Generate improvement recommendations
    recommendations = generate_improvement_recommendations(final_analysis)
    
    with open(run_dir / 'recommendations.json', 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    logger.info(f"Training complete! Results saved to {run_dir}")
    logger.info(f"Final loss: {final_analysis['loss_stats']['final']:.4f}")
    logger.info(f"Improvement: {final_analysis['loss_stats']['percent_reduction']:.2f}%")
    
    return run_dir, final_analysis


def generate_improvement_recommendations(analysis: Dict[str, Any]) -> Dict[str, List[str]]:
    """Generate recommendations based on training analysis"""
    recommendations = {
        'immediate': [],
        'next_run': [],
        'long_term': []
    }
    
    # Loss-based recommendations
    if 'loss_stats' in analysis:
        loss_stats = analysis['loss_stats']
        
        if loss_stats['percent_reduction'] < 10:
            recommendations['immediate'].append("Low loss reduction - increase learning rate or epochs")
        
        if loss_stats['std'] > 0.1:
            recommendations['immediate'].append("High loss variance - reduce learning rate or add gradient clipping")
    
    # Plateau detection
    if analysis.get('plateau_detected', False):
        recommendations['next_run'].append("Plateau detected - try cyclical learning rates")
        recommendations['next_run'].append("Consider adding dropout or weight decay")
    
    # Gradient analysis
    if 'gradient_stats' in analysis:
        grad_stats = analysis['gradient_stats']
        
        if grad_stats.get('exploding_gradients', False):
            recommendations['immediate'].append("Exploding gradients detected - reduce learning rate")
        
        if grad_stats['mean'] < 0.001:
            recommendations['next_run'].append("Vanishing gradients - check model architecture")
    
    # Learning rate effectiveness
    if 'lr_correlation' in analysis:
        if abs(analysis['lr_correlation']) < 0.3:
            recommendations['long_term'].append("Weak LR-loss correlation - experiment with different optimizers")
    
    return recommendations


if __name__ == "__main__":
    # Configuration
    config = {
        'data_path': 'data/stock_data.csv',
        'sequence_length': 60,
        'batch_size': 32,
        'hidden_dim': 128,
        'num_layers': 4,
        'num_heads': 8,
        'dropout': 0.1,
        'learning_rate': 1e-4,  # Reduced for stability
        'num_epochs': 30  # Reduced for faster testing
    }
    
    # Run training
    run_dir, analysis = train_with_analysis(config)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"Results saved to: {run_dir}")
    print(f"Final loss: {analysis['loss_stats']['final']:.4f}")
    print(f"Total improvement: {analysis['loss_stats']['percent_reduction']:.2f}%")
    print("\nCheck recommendations.json for improvement suggestions!")