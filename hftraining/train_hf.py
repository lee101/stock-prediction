#!/usr/bin/env python3
"""
HuggingFace-style Training Script Entry Point
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import sys
import os
import time

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
# Add parent directory to path
sys.path.append(os.path.dirname(current_dir))

from hf_trainer import (
    HFTrainingConfig,
    TransformerTradingModel,
    GPro,
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    MixedPrecisionTrainer,
    EarlyStopping
)
from logging_utils import get_logger, MetricsTracker


class StockDataset(Dataset):
    """Dataset for stock trading data"""
    
    def __init__(self, data, sequence_length=60, prediction_horizon=5):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Ensure we have enough data for sequences
        if len(data) < sequence_length + prediction_horizon:
            raise ValueError(f"Dataset too small: {len(data)} < {sequence_length + prediction_horizon}")
    
    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1
    
    def __getitem__(self, idx):
        # Input sequence
        start_idx = idx
        end_idx = idx + self.sequence_length
        
        sequence = self.data[start_idx:end_idx]
        
        # Target: next price movements
        target_start = end_idx
        target_end = target_start + self.prediction_horizon
        targets = self.data[target_start:target_end]
        
        # Convert to tensors
        sequence = torch.FloatTensor(sequence)
        targets = torch.FloatTensor(targets)
        
        # Generate action labels (simplified: based on next price movement)
        next_price = targets[0, 3] if len(targets) > 0 else sequence[-1, 3]  # Close price
        current_price = sequence[-1, 3]
        
        if next_price > current_price * 1.01:  # 1% threshold
            action_label = 0  # Buy
        elif next_price < current_price * 0.99:
            action_label = 2  # Sell
        else:
            action_label = 1  # Hold
        
        return {
            'input_ids': sequence,
            'labels': targets,
            'action_labels': torch.tensor(action_label, dtype=torch.long),
            'attention_mask': torch.ones(self.sequence_length)
        }


class HFTrainer:
    """HuggingFace-style trainer for stock prediction"""
    
    def __init__(self, model, config: HFTrainingConfig, train_dataset, eval_dataset=None):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup data parallel if available
        if torch.cuda.device_count() > 1 and config.use_data_parallel:
            self.model = nn.DataParallel(self.model)
        
        # Setup mixed precision
        self.mp_trainer = MixedPrecisionTrainer(config.use_mixed_precision)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            threshold=config.early_stopping_threshold,
            greater_is_better=config.greater_is_better
        )
        
        # Enhanced logging (initialize first)
        self.training_logger = get_logger(config.logging_dir, "training")
        self.metrics_tracker = MetricsTracker()
        
        # Setup logging (TensorBoard)
        self.setup_logging()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = None
        self.start_time = None
        
    def _create_optimizer(self):
        """Create optimizer based on config"""
        if self.config.optimizer_name.lower() == "gpro":
            return GPro(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_name.lower() == "adamw":
            return AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay
            )
        else:
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay
            )
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.max_steps
        )
    
    def setup_logging(self):
        """Setup logging directories and tensorboard"""
        # Create directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.logging_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup tensorboard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f"{self.config.logging_dir}/hf_training_{timestamp}"
        self.writer = SummaryWriter(log_dir)
        
        self.training_logger.info(f"TensorBoard logging to: {log_dir}")
        self.training_logger.info(f"Output directory: {self.config.output_dir}")
    
    def train(self):
        """Main training loop"""
        
        # Start timing
        self.start_time = time.time()
        
        # Log training start
        model_info = {
            'hidden_size': self.config.hidden_size,
            'num_layers': self.config.num_layers,
            'total_params': sum(p.numel() for p in self.model.parameters())
        }
        
        config_dict = {
            'experiment_name': 'hf_training',
            'description': 'HuggingFace-style stock prediction training',
            'optimizer': self.config.optimizer_name,
            'learning_rate': self.config.learning_rate,
            'batch_size': self.config.batch_size,
            'max_steps': self.config.max_steps,
            'device': str(self.device)
        }
        
        self.training_logger.log_training_start(config_dict, model_info)
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        eval_loader = None
        if self.eval_dataset:
            eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # Training loop
        self.model.train()
        pbar = tqdm(total=self.config.max_steps, desc="Training", ncols=120)
        
        epoch = 0
        while self.global_step < self.config.max_steps:
            epoch += 1
            self.current_epoch = epoch
            
            # Log epoch start
            if self.config.max_steps // len(train_loader) > 1:
                total_epochs = self.config.max_steps // len(train_loader)
                self.training_logger.log_epoch_start(epoch, total_epochs)
            else:
                self.training_logger.log_epoch_start(epoch)
            
            epoch_start_time = time.time()
            epoch_loss = 0
            epoch_steps = 0
            
            for batch_idx, batch in enumerate(train_loader):
                if self.global_step >= self.config.max_steps:
                    break
                
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Training step
                loss = self.training_step(batch)
                epoch_loss += loss
                epoch_steps += 1
                
                # Enhanced Logging
                if self.global_step % self.config.logging_steps == 0:
                    # Log to TensorBoard
                    self.log_metrics({
                        'train/loss': loss,
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'train/epoch': epoch
                    })
                    
                    # Log to file and console
                    metrics = {
                        'loss': loss,
                        'learning_rate': self.scheduler.get_last_lr()[0],
                        'epoch': epoch
                    }
                    self.training_logger.log_step_metrics(self.global_step, metrics, "train")
                    self.metrics_tracker.add_metric(self.global_step, "train", **metrics)
                    
                    # Update progress bar
                    pbar.set_description(
                        self.training_logger.create_progress_bar_desc(
                            self.global_step, loss, self.scheduler.get_last_lr()[0]
                        )
                    )
                
                # Evaluation
                if (self.eval_dataset and 
                    self.global_step % self.config.eval_steps == 0 and 
                    self.global_step > 0):
                    eval_metrics = self.evaluate(eval_loader)
                    
                    # Log to TensorBoard
                    self.log_metrics(eval_metrics, prefix='eval')
                    
                    # Enhanced evaluation logging
                    self.training_logger.log_step_metrics(self.global_step, eval_metrics, "eval")
                    self.metrics_tracker.add_metric(self.global_step, "eval", **eval_metrics)
                    
                    # Early stopping check
                    metric_value = eval_metrics.get('loss', loss)
                    self.early_stopping(metric_value)
                    
                    if self.early_stopping.should_stop:
                        self.training_logger.log_early_stopping(
                            self.global_step, 
                            self.config.early_stopping_patience
                        )
                        break
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0 and self.global_step > 0:
                    checkpoint_path = self.save_checkpoint()
                    self.training_logger.log_checkpoint_saved(self.global_step, checkpoint_path)
                
                pbar.update(1)
            
            if self.early_stopping.should_stop:
                break
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
            self.training_logger.log_epoch_summary(epoch, avg_epoch_loss, epoch_time)
        
        pbar.close()
        
        # Calculate total training time
        total_training_time = time.time() - self.start_time
        
        # Save final model
        final_checkpoint_path = self.save_checkpoint(is_final=True)
        
        # Get final metrics
        final_metrics = {
            'final_loss': self.metrics_tracker.get_recent_avg('loss', 10),
            'total_steps': self.global_step,
            'total_epochs': epoch
        }
        
        # Log training completion
        self.training_logger.log_training_complete(total_training_time, final_metrics)
        
        return self.model
    
    def training_step(self, batch):
        """Single training step"""
        self.optimizer.zero_grad()
        
        with self.mp_trainer.autocast():
            # Forward pass
            outputs = self.model(
                batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # Calculate losses
            action_loss = F.cross_entropy(
                outputs['action_logits'],
                batch['action_labels']
            )
            
            price_loss = F.mse_loss(
                outputs['price_predictions'],
                batch['labels'][:, :self.config.prediction_horizon, 3]  # Close prices
            )
            
            # Combined loss
            total_loss = action_loss + 0.5 * price_loss
        
        # Backward pass
        scaled_loss = self.mp_trainer.scale_loss(total_loss)
        scaled_loss.backward()
        
        # Gradient clipping
        if self.config.max_grad_norm > 0:
            if self.mp_trainer.enabled:
                self.mp_trainer.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        
        # Optimizer step
        self.mp_trainer.step_optimizer(self.optimizer)
        self.scheduler.step()
        
        self.global_step += 1
        
        return total_loss.item()
    
    def evaluate(self, eval_loader):
        """Evaluation loop"""
        self.model.eval()
        
        total_loss = 0
        total_action_loss = 0
        total_price_loss = 0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                action_loss = F.cross_entropy(
                    outputs['action_logits'],
                    batch['action_labels']
                )
                
                price_loss = F.mse_loss(
                    outputs['price_predictions'],
                    batch['labels'][:, :self.config.prediction_horizon, 3]
                )
                
                total_loss += action_loss.item() + 0.5 * price_loss.item()
                total_action_loss += action_loss.item()
                total_price_loss += price_loss.item()
                total_steps += 1
        
        self.model.train()
        
        return {
            'loss': total_loss / total_steps,
            'action_loss': total_action_loss / total_steps,
            'price_loss': total_price_loss / total_steps
        }
    
    def log_metrics(self, metrics, prefix='train'):
        """Log metrics to tensorboard"""
        for key, value in metrics.items():
            if not key.startswith(prefix):
                key = f"{prefix}/{key}"
            self.writer.add_scalar(key, value, self.global_step)
    
    def save_checkpoint(self, is_final=False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'config': self.config
        }
        
        if is_final:
            checkpoint_path = Path(self.config.output_dir) / "final_model.pth"
        else:
            checkpoint_path = Path(self.config.output_dir) / f"checkpoint_step_{self.global_step}.pth"
        
        torch.save(checkpoint, checkpoint_path)
        return str(checkpoint_path)


def load_data():
    """Load and prepare training data"""
    # This should be adapted to load your specific stock data
    # For now, we'll create dummy data
    print("Loading stock data...")
    
    # Try to load real data first
    data_path = Path("../trainingdata")
    if data_path.exists():
        # Look for CSV files
        csv_files = list(data_path.glob("*.csv"))
        if csv_files:
            print(f"Found {len(csv_files)} CSV files")
            # Load first CSV as example
            df = pd.read_csv(csv_files[0])
            
            # Assume columns: [timestamp, open, high, low, close, volume, ...]
            if len(df.columns) >= 5:
                # Take OHLCV columns
                data = df.iloc[:, 1:6].values  # Skip timestamp
                print(f"Loaded real data: {data.shape}")
                return data
    
    # Generate synthetic data if no real data found
    print("Generating synthetic stock data...")
    np.random.seed(42)
    
    # Generate realistic stock price movements
    length = 10000
    initial_price = 100.0
    
    # Random walk with drift
    returns = np.random.normal(0.0005, 0.02, length)  # 0.05% daily return, 2% volatility
    prices = [initial_price]
    
    for i in range(1, length):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(max(new_price, 0.01))  # Prevent negative prices
    
    prices = np.array(prices)
    
    # Generate OHLCV data
    data = []
    for i in range(len(prices)):
        price = prices[i]
        
        # Generate realistic OHLC from close price
        volatility = abs(np.random.normal(0, 0.01))
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        open_price = np.random.uniform(low, high)
        
        # Volume (random but realistic)
        volume = np.random.exponential(1000000)
        
        data.append([open_price, high, low, price, volume])
    
    data = np.array(data)
    print(f"Generated synthetic data: {data.shape}")
    
    return data


def main():
    """Main training function"""
    # Configuration
    config = HFTrainingConfig(
        # Model
        hidden_size=256,
        num_layers=6,
        num_heads=8,
        
        # Training
        learning_rate=1e-4,
        warmup_steps=500,
        max_steps=10000,
        batch_size=16,
        
        # Optimizer
        optimizer_name="gpro",
        weight_decay=0.01,
        
        # Evaluation
        eval_steps=250,
        save_steps=500,
        logging_steps=50,
        
        # Output
        output_dir="hftraining/output",
        logging_dir="hftraining/logs"
    )
    
    # Load data
    data = load_data()
    
    # Normalize data
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    data_normalized = (data - data_mean) / (data_std + 1e-8)
    
    # Split data
    train_size = int(0.8 * len(data_normalized))
    val_size = int(0.1 * len(data_normalized))
    
    train_data = data_normalized[:train_size]
    val_data = data_normalized[train_size:train_size + val_size]
    test_data = data_normalized[train_size + val_size:]
    
    print(f"Data splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Create datasets
    train_dataset = StockDataset(
        train_data,
        sequence_length=config.sequence_length,
        prediction_horizon=config.prediction_horizon
    )
    
    val_dataset = StockDataset(
        val_data,
        sequence_length=config.sequence_length,
        prediction_horizon=config.prediction_horizon
    ) if len(val_data) > config.sequence_length + config.prediction_horizon else None
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset) if val_dataset else 0}")
    
    # Create model
    model = TransformerTradingModel(config, input_dim=data.shape[1])
    
    # Create trainer
    trainer = HFTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Train model with error handling
    try:
        trained_model = trainer.train()
        return trained_model
    except KeyboardInterrupt:
        trainer.training_logger.warning("Training interrupted by user")
        checkpoint_path = trainer.save_checkpoint()
        trainer.training_logger.log_checkpoint_saved(trainer.global_step, checkpoint_path)
        return None
    except Exception as e:
        trainer.training_logger.log_error(e, trainer.global_step)
        return None


if __name__ == "__main__":
    main()