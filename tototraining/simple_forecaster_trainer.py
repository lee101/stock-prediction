#!/usr/bin/env python3
"""
Simple Forecaster Training Pipeline
A basic training script for time series forecasting that uses the OHLC dataloader
and a simple transformer-based forecaster model.
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import time
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import our dataloader
from toto_ohlc_dataloader import TotoOHLCDataLoader, DataLoaderConfig

# Simple Transformer Forecaster Model
class SimpleTransformerForecaster(nn.Module):
    """A simple transformer-based forecaster for time series data."""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 prediction_length: int = 24,
                 dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.prediction_length = prediction_length

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding - larger for long sequences
        self.pos_encoding = nn.Parameter(torch.randn(1, 2048, hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, prediction_length)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            predictions: Tensor of shape (batch_size, prediction_length)
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Apply transformer
        x = self.transformer(x)  # (batch_size, seq_len, hidden_dim)

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (batch_size, hidden_dim)

        # Apply dropout
        x = self.dropout(x)

        # Output projection
        predictions = self.output_projection(x)  # (batch_size, prediction_length)

        return predictions


@dataclass
class SimpleTrainerConfig:
    """Configuration for simple trainer"""

    # Model parameters
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1

    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    max_epochs: int = 50
    warmup_epochs: int = 5

    # Optimization
    use_mixed_precision: bool = True
    gradient_clip_val: float = 1.0

    # Validation
    validation_frequency: int = 1
    early_stopping_patience: int = 10

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "simple_training.log"

    # Checkpointing
    save_dir: str = "simple_checkpoints"
    save_frequency: int = 5


class SimpleForecasterTrainer:
    """Simple trainer for forecasting models"""

    def __init__(self, config: SimpleTrainerConfig, dataloader_config: DataLoaderConfig):
        self.config = config
        self.dataloader_config = dataloader_config

        # Setup logging
        self._setup_logging()

        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Model and optimizer (to be initialized)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if config.use_mixed_precision else None

        self.logger.info("SimpleForecasterTrainer initialized")

    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper())

        handlers = [logging.StreamHandler()]
        if self.config.log_file:
            handlers.append(logging.FileHandler(self.config.log_file))

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers,
            force=True
        )

        self.logger = logging.getLogger(__name__)

    def prepare_data(self):
        """Prepare data loaders"""
        self.logger.info("Preparing data loaders...")

        # Create OHLC data loader
        dataloader = TotoOHLCDataLoader(self.dataloader_config)
        self.dataloaders = dataloader.prepare_dataloaders()

        if not self.dataloaders:
            raise ValueError("No data loaders created!")

        self.logger.info(f"Created data loaders: {list(self.dataloaders.keys())}")

        # Log dataset sizes
        for split, loader in self.dataloaders.items():
            self.logger.info(f"{split}: {len(loader.dataset)} samples, {len(loader)} batches")

    def setup_model(self):
        """Setup model, optimizer, and scheduler"""
        self.logger.info("Setting up model...")

        if not self.dataloaders:
            raise ValueError("Data loaders not prepared! Call prepare_data() first.")

        # Determine input dimension from data loader
        sample_batch = next(iter(self.dataloaders['train']))
        input_dim = sample_batch.series.shape[1]  # Number of features

        self.logger.info(f"Input dimension: {input_dim}")
        self.logger.info(f"Prediction length: {self.dataloader_config.prediction_length}")

        # Create model
        self.model = SimpleTransformerForecaster(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            prediction_length=self.dataloader_config.prediction_length,
            dropout=self.config.dropout
        )

        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.logger.info(f"Model moved to device: {device}")

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Create scheduler
        total_steps = len(self.dataloaders['train']) * self.config.max_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.learning_rate * 0.01
        )

        self.logger.info("Model setup completed")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        device = next(self.model.parameters()).device

        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.dataloaders['train']):
            batch_start_time = time.time()

            # Move batch to device
            series = batch.series.to(device)  # (batch_size, features, time)
            batch_size, features, seq_len = series.shape

            # Transpose to (batch_size, time, features) for transformer
            x = series.transpose(1, 2)  # (batch_size, seq_len, features)

            # Create target: predict the last prediction_length values of the first feature (Close price)
            target_feature_idx = 0  # Assuming first feature is what we want to predict
            if seq_len >= self.dataloader_config.prediction_length:
                y = series[:, target_feature_idx, -self.dataloader_config.prediction_length:]
            else:
                # Fallback: repeat last value
                y = series[:, target_feature_idx, -1:].repeat(1, self.dataloader_config.prediction_length)

            # Forward pass with mixed precision
            with autocast(enabled=self.config.use_mixed_precision):
                predictions = self.model(x)
                loss = F.mse_loss(predictions, y)

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.scheduler.step()

            # Track metrics
            total_loss += loss.item()
            num_batches += 1

            # Log progress
            if batch_idx % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.dataloaders['train'])}, "
                    f"Loss: {loss.item():.6f}, LR: {current_lr:.8f}"
                )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'loss': avg_loss}

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        if 'val' not in self.dataloaders:
            return {}

        self.model.eval()
        device = next(self.model.parameters()).device

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.dataloaders['val']:
                # Move batch to device
                series = batch.series.to(device)
                batch_size, features, seq_len = series.shape

                # Transpose to (batch_size, time, features)
                x = series.transpose(1, 2)

                # Create target
                target_feature_idx = 0
                if seq_len >= self.dataloader_config.prediction_length:
                    y = series[:, target_feature_idx, -self.dataloader_config.prediction_length:]
                else:
                    y = series[:, target_feature_idx, -1:].repeat(1, self.dataloader_config.prediction_length)

                # Forward pass
                with autocast(enabled=self.config.use_mixed_precision):
                    predictions = self.model(x)
                    loss = F.mse_loss(predictions, y)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'loss': avg_loss}

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__,
            'timestamp': datetime.now().isoformat()
        }

        # Save regular checkpoint
        checkpoint_path = Path(self.config.save_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = Path(self.config.save_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with validation loss: {self.best_val_loss:.6f}")

        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load training state
        self.current_epoch = checkpoint['epoch'] + 1  # Start from next epoch
        self.best_val_loss = checkpoint['best_val_loss']

        self.logger.info(f"Checkpoint loaded: resuming from epoch {self.current_epoch}, best val loss: {self.best_val_loss:.6f}")

    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")

        # Start fresh training for large context model
        # (Skip checkpoint loading to train from scratch)

        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch

            self.logger.info(f"Epoch {epoch + 1}/{self.config.max_epochs}")

            # Train epoch
            train_metrics = self.train_epoch()

            # Validation epoch
            val_metrics = {}
            if epoch % self.config.validation_frequency == 0:
                val_metrics = self.validate_epoch()

            # Log metrics
            log_msg = f"Epoch {epoch + 1} - Train Loss: {train_metrics['loss']:.6f}"
            if val_metrics:
                log_msg += f", Val Loss: {val_metrics['loss']:.6f}"
            self.logger.info(log_msg)

            # Check for best model
            is_best = False
            if val_metrics and 'loss' in val_metrics:
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                    is_best = True
                else:
                    self.patience_counter += 1

            # Save checkpoint
            if epoch % self.config.save_frequency == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

            # Early stopping
            if (self.patience_counter >= self.config.early_stopping_patience and
                val_metrics and self.config.early_stopping_patience > 0):
                self.logger.info(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                break

        self.logger.info("Training completed!")


def main():
    """Main function to run training"""
    print("🚀 Simple Forecaster Training Pipeline")

    # Training configuration - Large context training
    trainer_config = SimpleTrainerConfig(
        hidden_dim=512,  # Larger model for longer sequences
        num_layers=6,    # Deeper model
        num_heads=8,
        dropout=0.1,
        learning_rate=1e-4,
        weight_decay=0.01,
        batch_size=8,   # Match dataloader batch size
        max_epochs=100,
        warmup_epochs=5,
        use_mixed_precision=True,
        validation_frequency=1,
        early_stopping_patience=15,
        save_frequency=5,
        log_level="INFO",
        log_file="large_context_training.log",
        save_dir="large_context_checkpoints"
    )

    # Dataloader configuration - Large context window
    dataloader_config = DataLoaderConfig(
        train_data_path="trainingdata/train",
        test_data_path="trainingdata/test",
        batch_size=8,  # Smaller batch size for larger sequences
        sequence_length=512,  # Much larger context window
        prediction_length=48,  # Longer prediction horizon
        validation_split=0.2,
        add_technical_indicators=True,
        normalization_method="robust",
        max_symbols=10  # Limit for faster training
    )

    # Create trainer
    trainer = SimpleForecasterTrainer(trainer_config, dataloader_config)

    try:
        # Prepare data and setup model
        trainer.prepare_data()
        trainer.setup_model()

        # Start training
        trainer.train()

        print("✅ Training completed successfully!")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()