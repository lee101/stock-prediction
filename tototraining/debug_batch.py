#!/usr/bin/env python3
"""Debug the batch type issue"""

import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import warnings
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

from toto_trainer import TotoTrainer, TrainerConfig
from toto_ohlc_dataloader import DataLoaderConfig, MaskedTimeseries


def debug_batch_type():
    """Debug what type of batch we're getting"""
    
    temp_dir = tempfile.mkdtemp()
    try:
        train_dir = Path(temp_dir) / "train_data"
        train_dir.mkdir(parents=True, exist_ok=True)
        
        # Create simple data
        dates = pd.date_range('2023-01-01', periods=200, freq='H')
        data = pd.DataFrame({
            'timestamp': dates,
            'Open': np.random.uniform(90, 110, 200),
            'High': np.random.uniform(95, 115, 200),
            'Low': np.random.uniform(85, 105, 200),
            'Close': np.random.uniform(90, 110, 200),
            'Volume': np.random.randint(1000, 10000, 200)
        })
        data.to_csv(train_dir / "TEST.csv", index=False)
        
        # Configure
        trainer_config = TrainerConfig(
            batch_size=4, max_epochs=1, save_dir=str(Path(temp_dir) / "checkpoints")
        )
        dataloader_config = DataLoaderConfig(
            train_data_path=str(train_dir),
            test_data_path="nonexistent",
            batch_size=4,
            validation_split=0.2,
            test_split_days=1,  # Smaller split
            num_workers=0,
            min_sequence_length=100,
            drop_last=False
        )
        
        # Create trainer
        trainer = TotoTrainer(trainer_config, dataloader_config)
        trainer.prepare_data()
        
        # Get a batch and examine it
        train_loader = trainer.dataloaders['train']
        batch = next(iter(train_loader))
        
        print(f"Batch type: {type(batch)}")
        print(f"Batch type name: {type(batch).__name__}")
        print(f"Batch module: {type(batch).__module__}")
        print(f"Is MaskedTimeseries: {isinstance(batch, MaskedTimeseries)}")
        print(f"MaskedTimeseries module: {MaskedTimeseries.__module__}")
        print(f"MaskedTimeseries from trainer: {trainer.__class__.__module__}")
        
        # Check attributes
        if hasattr(batch, 'series'):
            print(f"Has series attribute: {batch.series.shape}")
        if hasattr(batch, 'padding_mask'):
            print(f"Has padding_mask attribute: {batch.padding_mask.shape}")
        if hasattr(batch, 'id_mask'):
            print(f"Has id_mask attribute: {batch.id_mask.shape}")
        
        # Try importing from trainer module
        try:
            from toto_trainer import MaskedTimeseries as TrainerMaskedTimeseries
            print(f"Trainer MaskedTimeseries: {TrainerMaskedTimeseries}")
            print(f"Is trainer MaskedTimeseries: {isinstance(batch, TrainerMaskedTimeseries)}")
        except ImportError as e:
            print(f"Cannot import MaskedTimeseries from toto_trainer: {e}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    debug_batch_type()