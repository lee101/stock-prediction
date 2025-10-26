#!/usr/bin/env python3
"""
Toto OHLC Training Script
Trains the Datadog Toto model specifically on OHLC data with proper validation split.
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

# Add the toto directory to sys.path
toto_path = Path(__file__).parent.parent / "toto"
sys.path.insert(0, str(toto_path))

try:
    from toto.model.toto import Toto
    from toto.model.scaler import StdMeanScaler
except Exception as exc:  # pragma: no cover - fallback for tests/sandboxes
    logging.getLogger(__name__).warning(
        "Falling back to lightweight Toto stub for testing: %s", exc
    )

    class StdMeanScaler:
        pass

    class Toto(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.model = nn.Identity()


@dataclass
class TotoOHLCConfig:
    """Configuration for Toto OHLC training"""
    patch_size: int = 12
    stride: int = 6
    embed_dim: int = 256
    num_layers: int = 8
    num_heads: int = 8
    mlp_hidden_dim: int = 512
    dropout: float = 0.1
    spacewise_every_n_layers: int = 2
    scaler_cls: str = "<class 'model.scaler.StdMeanScaler'>"
    output_distribution_classes: List[str] = None
    sequence_length: int = 96  # Number of time steps to use as input
    prediction_length: int = 24  # Number of time steps to predict
    validation_days: int = 30  # Last N days for validation
    
    def __post_init__(self):
        if self.output_distribution_classes is None:
            self.output_distribution_classes = ["<class 'model.distribution.StudentTOutput'>"]


class OHLCDataset(torch.utils.data.Dataset):
    """Dataset for OHLC data"""
    
    def __init__(self, data: pd.DataFrame, config: TotoOHLCConfig):
        self.config = config
        self.data = self.prepare_data(data)
        
    def prepare_data(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare OHLC data for training"""
        # Ensure we have the expected columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        # Convert to numpy array and normalize
        ohlc_data = data[required_cols].values.astype(np.float32)
        
        # Add volume if available, otherwise create dummy volume
        if 'Volume' in data.columns:
            volume = data['Volume'].values.astype(np.float32).reshape(-1, 1)
        else:
            volume = np.ones((len(ohlc_data), 1), dtype=np.float32)
        
        # Combine OHLC + Volume = 5 features
        return np.concatenate([ohlc_data, volume], axis=1)
    
    def __len__(self):
        return max(0, len(self.data) - self.config.sequence_length - self.config.prediction_length + 1)
    
    def __getitem__(self, idx):
        # Get input sequence
        start_idx = idx
        end_idx = start_idx + self.config.sequence_length
        pred_end_idx = end_idx + self.config.prediction_length
        
        if pred_end_idx > len(self.data):
            raise IndexError(f"Index {idx} out of range")
        
        # Input features (past sequence)
        x = torch.from_numpy(self.data[start_idx:end_idx])  # Shape: (seq_len, 5)
        
        # Target (future values to predict) - use Close prices
        y = torch.from_numpy(self.data[end_idx:pred_end_idx, 3])  # Shape: (pred_len,) - Close prices
        
        return x, y


class TotoOHLCTrainer:
    """Trainer for Toto model on OHLC data"""
    
    def __init__(self, config: TotoOHLCConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('tototraining/training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.model = None
        self.optimizer = None
        self.scaler = None
        
    def initialize_model(self, input_dim: int):
        """Initialize the Toto model"""
        model = Toto(
            patch_size=self.config.patch_size,
            stride=self.config.stride,
            embed_dim=self.config.embed_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            mlp_hidden_dim=self.config.mlp_hidden_dim,
            dropout=self.config.dropout,
            spacewise_every_n_layers=self.config.spacewise_every_n_layers,
            scaler_cls=self.config.scaler_cls,
            output_distribution_classes=self.config.output_distribution_classes,
            use_memory_efficient_attention=False,  # Disable since xformers not available
        )
        model.to(self.device)
        self.model = model
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        self.logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def load_data(self) -> Tuple[Dict[str, OHLCDataset], Dict[str, torch.utils.data.DataLoader]]:
        """Load and split OHLC data"""
        data_dir = Path('data')
        datasets = {}
        dataloaders = {}
        
        # Find all CSV files
        csv_files = []
        for timestamp_dir in data_dir.iterdir():
            if timestamp_dir.is_dir() and timestamp_dir.name.startswith('2024'):
                csv_files.extend(list(timestamp_dir.glob('*.csv')))
        
        if not csv_files:
            # Fallback to root data directory
            csv_files = list(data_dir.glob('*.csv'))
        
        self.logger.info(f"Found {len(csv_files)} CSV files")
        
        all_train_data = []
        all_val_data = []
        
        for csv_file in csv_files[:50]:  # Limit for initial training
            try:
                df = pd.read_csv(csv_file)
                
                # Parse timestamp if it exists
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')
                
                # Split into train/validation (last 30 days for validation)
                if len(df) < self.config.sequence_length + self.config.prediction_length:
                    continue
                
                # Simple split: last validation_days worth of data for validation
                val_size = min(len(df) // 10, self.config.validation_days * 24 * 4)  # Assume 15min intervals
                val_size = max(val_size, self.config.sequence_length + self.config.prediction_length)
                
                train_df = df.iloc[:-val_size]
                val_df = df.iloc[-val_size:]
                
                if len(train_df) >= self.config.sequence_length + self.config.prediction_length:
                    all_train_data.append(train_df)
                if len(val_df) >= self.config.sequence_length + self.config.prediction_length:
                    all_val_data.append(val_df)
                    
            except Exception as e:
                self.logger.warning(f"Error loading {csv_file}: {e}")
                continue
        
        # Combine all data
        if all_train_data:
            combined_train_df = pd.concat(all_train_data, ignore_index=True)
            datasets['train'] = OHLCDataset(combined_train_df, self.config)
            dataloaders['train'] = torch.utils.data.DataLoader(
                datasets['train'], 
                batch_size=32, 
                shuffle=True,
                num_workers=2,
                drop_last=True
            )
        
        if all_val_data:
            combined_val_df = pd.concat(all_val_data, ignore_index=True)
            datasets['val'] = OHLCDataset(combined_val_df, self.config)
            dataloaders['val'] = torch.utils.data.DataLoader(
                datasets['val'], 
                batch_size=32, 
                shuffle=False,
                num_workers=2,
                drop_last=True
            )
        
        self.logger.info(f"Train samples: {len(datasets.get('train', []))}")
        self.logger.info(f"Val samples: {len(datasets.get('val', []))}")
        
        return datasets, dataloaders
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass - provide required masks
            try:
                # Prepare masks for the Toto model
                batch_size, seq_len, features = x.shape
                
                # Create input_padding_mask (no padding in our case)
                input_padding_mask = torch.zeros(batch_size, 1, seq_len, dtype=torch.bool, device=x.device)
                
                # Create id_mask (all different time series, so all ones)
                id_mask = torch.ones(batch_size, 1, seq_len, dtype=torch.float32, device=x.device)
                
                # Reshape input to match expected format (batch, variate, time_steps)
                x_reshaped = x.transpose(1, 2).contiguous()  # From (batch, time, features) to (batch, features, time)
                
                # Call the backbone model with proper arguments
                output = self.model.model(x_reshaped, input_padding_mask, id_mask)
                
                # Handle the TotoOutput which has distribution, loc, scale
                if hasattr(output, 'loc'):
                    predictions = output.loc  # Use location parameter as prediction
                elif isinstance(output, dict) and 'prediction' in output:
                    predictions = output['prediction']
                else:
                    predictions = output
                
                # Ensure shapes match
                if predictions.dim() == 3:  # (batch, seq, features)
                    predictions = predictions[:, -1, 0]  # Take last timestep, first feature
                elif predictions.dim() == 2:
                    predictions = predictions[:, 0]  # First feature
                
                loss = torch.nn.functional.mse_loss(predictions, y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    self.logger.info(f"Batch {batch_idx}, Loss: {loss.item():.6f}")
                    
            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx}: {e}")
                raise RuntimeError(f"Model training error: {e}") from e
        
        return total_loss / max(num_batches, 1)
    
    def validate(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                
                try:
                    # Prepare masks for the Toto model
                    batch_size, seq_len, features = x.shape
                    
                    # Create input_padding_mask (no padding in our case)
                    input_padding_mask = torch.zeros(batch_size, 1, seq_len, dtype=torch.bool, device=x.device)
                    
                    # Create id_mask (all different time series, so all ones)
                    id_mask = torch.ones(batch_size, 1, seq_len, dtype=torch.float32, device=x.device)
                    
                    # Reshape input to match expected format (batch, variate, time_steps)
                    x_reshaped = x.transpose(1, 2).contiguous()  # From (batch, time, features) to (batch, features, time)
                    
                    # Call the backbone model with proper arguments
                    output = self.model.model(x_reshaped, input_padding_mask, id_mask)
                    
                    if hasattr(output, 'loc'):
                        predictions = output.loc  # Use location parameter as prediction
                    elif isinstance(output, dict) and 'prediction' in output:
                        predictions = output['prediction']
                    else:
                        predictions = output
                    
                    # Ensure shapes match
                    if predictions.dim() == 3:
                        predictions = predictions[:, -1, 0]
                    elif predictions.dim() == 2:
                        predictions = predictions[:, 0]
                    
                    loss = torch.nn.functional.mse_loss(predictions, y)
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    self.logger.error(f"Error in validation: {e}")
                    raise RuntimeError(f"Model validation error: {e}") from e
        
        return total_loss / max(num_batches, 1)
    
    def train(self, num_epochs: int = 50):
        """Main training loop"""
        self.logger.info("Starting Toto OHLC training...")
        
        # Load data
        datasets, dataloaders = self.load_data()
        
        if 'train' not in dataloaders:
            self.logger.error("No training data found!")
            return
        
        # Initialize model with correct input dimension (5 for OHLCV)
        self.initialize_model(input_dim=5)
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(dataloaders['train'])
            self.logger.info(f"Train Loss: {train_loss:.6f}")
            
            # Validate
            if 'val' in dataloaders:
                val_loss = self.validate(dataloaders['val'])
                self.logger.info(f"Val Loss: {val_loss:.6f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'tototraining/best_model.pth')
                    self.logger.info(f"New best model saved! Val Loss: {val_loss:.6f}")
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    self.logger.info("Early stopping triggered!")
                    break
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss if 'val' in dataloaders else None,
                }, f'tototraining/checkpoint_epoch_{epoch + 1}.pth')
        
        self.logger.info("Training completed!")


def main():
    """Main training function"""
    print("🚀 Starting Toto OHLC Training")
    
    # Create config
    config = TotoOHLCConfig(
        patch_size=12,
        stride=6,
        embed_dim=128,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        sequence_length=96,
        prediction_length=24,
        validation_days=30
    )
    
    # Initialize trainer
    trainer = TotoOHLCTrainer(config)
    
    # Start training
    trainer.train(num_epochs=100)
    
    print("✅ Training completed! Check tototraining/training.log for details.")


if __name__ == "__main__":
    main()
