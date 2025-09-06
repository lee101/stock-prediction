#!/usr/bin/env python3
"""
Differentiable Training Pipeline with Best Practices
- Ensures all operations are differentiable
- Proper gradient flow throughout the network
- Mixed precision training support
- Gradient accumulation and clipping
- Comprehensive gradient monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for differentiable training"""
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 4
    mixed_precision: bool = True
    warmup_steps: int = 100
    weight_decay: float = 1e-4
    dropout_rate: float = 0.1
    label_smoothing: float = 0.1
    use_gradient_checkpointing: bool = False
    monitor_gradients: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class DifferentiableAttention(nn.Module):
    """Fully differentiable attention mechanism"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "Hidden dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape for multi-head attention
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention (all differentiable operations)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)
        
        return output


class DifferentiableTransformerBlock(nn.Module):
    """Transformer block with guaranteed differentiability"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = DifferentiableAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),  # GELU is smooth and differentiable everywhere
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture for better gradient flow
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x


class DifferentiableTradingModel(nn.Module):
    """Trading model with fully differentiable operations"""
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 256, num_layers: int = 6, 
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, hidden_dim) * 0.02)
        
        self.transformer_blocks = nn.ModuleList([
            DifferentiableTransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Multiple output heads for different trading decisions
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # Buy, Hold, Sell
        )
        
        self.position_size_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Position size in [-1, 1]
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Confidence in [0, 1]
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        # Project input and add positional encoding
        x = self.input_projection(x)
        if seq_len <= self.positional_encoding.size(1):
            x = x + self.positional_encoding[:, :seq_len, :]
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        x = self.norm(x)
        
        # Use the last timestep for predictions
        last_hidden = x[:, -1, :]
        
        # Get outputs from different heads
        actions = self.action_head(last_hidden)
        position_sizes = self.position_size_head(last_hidden)
        confidences = self.confidence_head(last_hidden)
        
        return {
            'actions': actions,
            'position_sizes': position_sizes,
            'confidences': confidences,
            'hidden_states': x
        }


class DifferentiableLoss(nn.Module):
    """Custom differentiable loss function for trading"""
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        super().__init__()
        self.alpha = alpha  # Weight for action loss
        self.beta = beta    # Weight for position size loss
        self.gamma = gamma  # Weight for confidence calibration
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        losses = {}
        
        # Action classification loss with label smoothing
        if 'actions' in targets:
            action_logits = predictions['actions']
            action_targets = targets['actions']
            
            # Apply label smoothing for better generalization
            num_classes = action_logits.size(-1)
            smooth_targets = torch.zeros_like(action_logits)
            smooth_targets.scatter_(1, action_targets.unsqueeze(1), 1.0)
            smooth_targets = smooth_targets * 0.9 + 0.1 / num_classes
            
            losses['action_loss'] = F.cross_entropy(action_logits, action_targets)
        
        # Position size regression loss (smooth L1 for robustness)
        if 'position_sizes' in targets:
            position_pred = predictions['position_sizes']
            position_target = targets['position_sizes']
            losses['position_loss'] = F.smooth_l1_loss(position_pred, position_target)
        
        # Confidence calibration loss
        if 'confidences' in predictions and 'returns' in targets:
            confidences = predictions['confidences']
            returns = targets['returns']
            
            # Confidence should correlate with actual returns
            confidence_target = torch.sigmoid(returns * 10)  # Scale returns to [0, 1]
            losses['confidence_loss'] = F.mse_loss(confidences, confidence_target)
        
        # Combine losses with weights
        total_loss = torch.tensor(0.0, device=predictions['actions'].device)
        if 'action_loss' in losses:
            total_loss = total_loss + self.alpha * losses['action_loss']
        if 'position_loss' in losses:
            total_loss = total_loss + self.beta * losses['position_loss']
        if 'confidence_loss' in losses:
            total_loss = total_loss + self.gamma * losses['confidence_loss']
        
        return total_loss, losses


class GradientMonitor:
    """Monitor gradient flow through the network"""
    
    def __init__(self):
        self.gradient_stats = defaultdict(list)
        self.hooks = []
        
    def register_hooks(self, model: nn.Module):
        """Register backward hooks to monitor gradients"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(lambda grad, name=name: self._store_gradient(name, grad))
                self.hooks.append(hook)
    
    def _store_gradient(self, name: str, grad: torch.Tensor):
        """Store gradient statistics"""
        if grad is not None:
            self.gradient_stats[name].append({
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'max': grad.max().item(),
                'min': grad.min().item(),
                'norm': grad.norm().item()
            })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get gradient statistics"""
        stats = {}
        for name, grad_list in self.gradient_stats.items():
            if grad_list:
                latest = grad_list[-1]
                stats[name] = latest
        return stats
    
    def check_gradient_health(self) -> Dict[str, bool]:
        """Check for gradient issues"""
        issues = {}
        for name, grad_list in self.gradient_stats.items():
            if grad_list:
                latest = grad_list[-1]
                issues[name] = {
                    'vanishing': abs(latest['mean']) < 1e-7,
                    'exploding': abs(latest['max']) > 100,
                    'nan': np.isnan(latest['mean']),
                    'inf': np.isinf(latest['mean'])
                }
        return issues
    
    def clear(self):
        """Clear stored gradients"""
        self.gradient_stats.clear()
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class DifferentiableTrainer:
    """Trainer with best practices for differentiable training"""
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model.to(config.device)
        self.config = config
        self.device = torch.device(config.device)
        
        # Optimizer with weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self.get_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Loss function
        self.criterion = DifferentiableLoss()
        
        # Gradient monitor
        self.grad_monitor = GradientMonitor() if config.monitor_gradients else None
        
        # Training history
        self.history = defaultdict(list)
        
        logger.info(f"Initialized DifferentiableTrainer on {config.device}")
        
    def get_scheduler(self):
        """Create learning rate scheduler with warmup"""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            else:
                return 1.0
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with proper gradient handling"""
        
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Mixed precision training
        if self.config.mixed_precision and self.scaler is not None:
            with autocast():
                outputs = self.model(batch['inputs'])
                loss, loss_components = self.criterion(outputs, batch)
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
        else:
            outputs = self.model(batch['inputs'])
            loss, loss_components = self.criterion(outputs, batch)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Standard backward pass
            loss.backward()
        
        # Store loss components
        metrics = {
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            **{k: v.item() for k, v in loss_components.items()}
        }
        
        return metrics
    
    def optimization_step(self, step: int):
        """Perform optimization with gradient clipping and updates"""
        
        if self.config.mixed_precision and self.scaler is not None:
            # Unscale gradients for clipping
            self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping to prevent exploding gradients
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.gradient_clip_norm
        )
        
        # Check gradient health
        if self.grad_monitor:
            grad_issues = self.grad_monitor.check_gradient_health()
            unhealthy = sum(any(v.values()) for v in grad_issues.values())
            if unhealthy > 0:
                logger.warning(f"Gradient issues detected in {unhealthy} parameters")
        
        # Optimizer step
        if self.config.mixed_precision and self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Clear gradients
        self.optimizer.zero_grad()
        
        # Update learning rate
        self.scheduler.step()
        
        return total_norm
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        
        epoch_metrics = defaultdict(list)
        accumulation_counter = 0
        
        # Register gradient hooks
        if self.grad_monitor and epoch == 0:
            self.grad_monitor.register_hooks(self.model)
        
        for step, batch in enumerate(dataloader):
            # Forward and backward pass
            metrics = self.train_step(batch)
            
            for k, v in metrics.items():
                epoch_metrics[k].append(v)
            
            accumulation_counter += 1
            
            # Perform optimization step after accumulation
            if accumulation_counter % self.config.gradient_accumulation_steps == 0:
                grad_norm = self.optimization_step(step)
                epoch_metrics['grad_norm'].append(grad_norm.item())
                accumulation_counter = 0
            
            # Log progress
            if step % 10 == 0:
                avg_loss = np.mean(epoch_metrics['loss'][-10:])
                lr = self.scheduler.get_last_lr()[0]
                logger.info(f"Epoch {epoch}, Step {step}, Loss: {avg_loss:.4f}, LR: {lr:.6f}")
        
        # Final optimization step if needed
        if accumulation_counter > 0:
            grad_norm = self.optimization_step(len(dataloader))
            epoch_metrics['grad_norm'].append(grad_norm.item())
        
        # Compute epoch averages
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        
        return avg_metrics
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validation with gradient checking disabled"""
        
        self.model.eval()
        val_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(batch['inputs'])
                loss, loss_components = self.criterion(outputs, batch)
                
                val_metrics['val_loss'].append(loss.item())
                for k, v in loss_components.items():
                    val_metrics[f'val_{k}'].append(v.item())
                
                # Calculate accuracy
                if 'actions' in outputs and 'actions' in batch:
                    preds = outputs['actions'].argmax(dim=-1)
                    correct = (preds == batch['actions']).float().mean()
                    val_metrics['val_accuracy'].append(correct.item())
        
        avg_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
        
        return avg_metrics
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, 
              num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """Full training loop"""
        
        num_epochs = num_epochs or self.config.num_epochs
        best_val_loss = float('inf')
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            if val_loader:
                val_metrics = self.validate(val_loader)
                train_metrics.update(val_metrics)
                
                # Save best model
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint(f'best_model_epoch_{epoch}.pt')
            
            # Store history
            for k, v in train_metrics.items():
                self.history[k].append(v)
            
            # Log epoch summary
            logger.info(f"Epoch {epoch} Summary:")
            for k, v in train_metrics.items():
                logger.info(f"  {k}: {v:.4f}")
            
            # Check for NaN
            if np.isnan(train_metrics['loss']):
                logger.error("NaN loss detected, stopping training")
                break
        
        # Clean up gradient monitor
        if self.grad_monitor:
            self.grad_monitor.remove_hooks()
        
        return dict(self.history)
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'history': dict(self.history)
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.history = defaultdict(list, checkpoint.get('history', {}))
        
        logger.info(f"Loaded checkpoint from {path}")


class TradingDataset(Dataset):
    """Dataset for trading data"""
    
    def __init__(self, data: pd.DataFrame, seq_len: int = 20):
        self.data = data
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data) - self.seq_len - 1
    
    def __getitem__(self, idx):
        # Get sequence of features
        seq_data = self.data.iloc[idx:idx + self.seq_len]
        
        # Normalize features
        features = torch.FloatTensor(seq_data[['open', 'high', 'low', 'close', 'volume', 'returns']].values)
        
        # Get target (next day's action)
        next_return = self.data.iloc[idx + self.seq_len]['returns']
        
        if next_return > 0.01:
            action = 0  # Buy
        elif next_return < -0.01:
            action = 2  # Sell
        else:
            action = 1  # Hold
        
        position_size = np.clip(next_return * 10, -1, 1)  # Scale return to position size
        
        return {
            'inputs': features,
            'actions': torch.LongTensor([action]).squeeze(),
            'position_sizes': torch.FloatTensor([position_size]).squeeze(),
            'returns': torch.FloatTensor([next_return]).squeeze()
        }


def create_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create synthetic trading data for testing"""
    
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate synthetic price data
    returns = np.random.normal(0.001, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.01, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.02, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.02, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(15, 1, n_samples),
        'returns': returns
    })
    
    return data


def main():
    """Main training pipeline"""
    
    # Create configuration
    config = TrainingConfig(
        learning_rate=1e-3,
        batch_size=32,
        num_epochs=50,
        gradient_clip_norm=1.0,
        gradient_accumulation_steps=4,
        mixed_precision=torch.cuda.is_available(),
        warmup_steps=100,
        weight_decay=1e-4,
        dropout_rate=0.1,
        monitor_gradients=True
    )
    
    # Create model
    model = DifferentiableTradingModel(
        input_dim=6,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        dropout=config.dropout_rate
    )
    
    # Create synthetic data
    data = create_synthetic_data(5000)
    
    # Split data
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    # Create datasets and dataloaders
    train_dataset = TradingDataset(train_data)
    val_dataset = TradingDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Create trainer
    trainer = DifferentiableTrainer(model, config)
    
    # Train model
    logger.info("Starting differentiable training pipeline")
    history = trainer.train(train_loader, val_loader, num_epochs=config.num_epochs)
    
    # Save final model
    trainer.save_checkpoint('final_model.pt')
    
    # Plot training history
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(history['loss'], label='Train Loss')
    if 'val_loss' in history:
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].set_title('Training Loss')
    
    if 'grad_norm' in history:
        axes[0, 1].plot(history['grad_norm'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Gradient Norm')
        axes[0, 1].set_title('Gradient Norm')
    
    if 'val_accuracy' in history:
        axes[1, 0].plot(history['val_accuracy'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Validation Accuracy')
    
    if 'action_loss' in history:
        axes[1, 1].plot(history['action_loss'], label='Action Loss')
        if 'position_loss' in history:
            axes[1, 1].plot(history['position_loss'], label='Position Loss')
        if 'confidence_loss' in history:
            axes[1, 1].plot(history['confidence_loss'], label='Confidence Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].set_title('Loss Components')
    
    plt.tight_layout()
    plt.savefig('training/differentiable_training_history.png')
    plt.close()
    
    logger.info("Training complete! Results saved to training/differentiable_training_history.png")
    
    return model, trainer, history


if __name__ == "__main__":
    model, trainer, history = main()