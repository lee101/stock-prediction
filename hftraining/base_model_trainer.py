#!/usr/bin/env python3
"""
Base Model Training Pipeline
Trains a base model on multiple stock pairs, then allows fine-tuning for individual stocks
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
from tqdm import tqdm

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.append(os.path.dirname(current_dir))

from config import create_config, ExperimentConfig
from train_hf import HFTrainer, StockDataset
from hf_trainer import TransformerTradingModel, HFTrainingConfig
from data_utils import StockDataProcessor, split_data, load_local_stock_data
from profit_tracker import ProfitTracker, integrate_profit_tracking
from logging_utils import get_logger


class MultiStockDataset(torch.utils.data.Dataset):
    """Dataset that combines multiple stock pairs for base model training"""
    
    def __init__(
        self,
        stock_data: Dict[str, np.ndarray],
        sequence_length: int = 60,
        prediction_horizon: int = 5,
        processor: Optional[StockDataProcessor] = None
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.processor = processor
        
        # Combine all stock data
        self.stock_names = list(stock_data.keys())
        self.stock_indices = []  # Track which stock each sample belongs to
        self.all_sequences = []
        self.all_targets = []
        self.all_actions = []
        
        # Process each stock
        for stock_name, data in stock_data.items():
            if len(data) < sequence_length + prediction_horizon:
                print(f"Skipping {stock_name}: insufficient data")
                continue
            
            # Create sequences for this stock
            n_samples = len(data) - sequence_length - prediction_horizon + 1
            
            for i in range(n_samples):
                # Input sequence
                seq = data[i:i + sequence_length]
                
                # Target sequence
                target_start = i + sequence_length
                target_end = target_start + prediction_horizon
                target = data[target_start:target_end]
                
                # Action label based on price movement
                current_price = data[i + sequence_length - 1, 3]  # Close price
                next_price = data[i + sequence_length, 3]
                price_change = (next_price - current_price) / current_price
                
                if price_change > 0.01:
                    action = 0  # Buy
                elif price_change < -0.01:
                    action = 2  # Sell
                else:
                    action = 1  # Hold
                
                self.all_sequences.append(seq)
                self.all_targets.append(target)
                self.all_actions.append(action)
                self.stock_indices.append(self.stock_names.index(stock_name))
        
        print(f"Created dataset with {len(self.all_sequences)} samples from {len(self.stock_names)} stocks")
    
    def __len__(self):
        return len(self.all_sequences)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.FloatTensor(self.all_sequences[idx]),
            'labels': torch.FloatTensor(self.all_targets[idx]),
            'action_labels': torch.tensor(self.all_actions[idx], dtype=torch.long),
            'attention_mask': torch.ones(self.sequence_length),
            'stock_idx': torch.tensor(self.stock_indices[idx], dtype=torch.long)
        }


class BaseModelTrainer:
    """Manages base model training and fine-tuning pipeline"""
    
    def __init__(
        self,
        base_stocks: List[str] = None,
        output_dir: str = "hftraining/models",
        tensorboard_dir: str = "hftraining/tensorboard"
    ):
        self.base_stocks = base_stocks or [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
            'META', 'NVDA', 'JPM', 'V', 'JNJ'
        ]
        self.output_dir = Path(output_dir)
        self.tensorboard_dir = Path(tensorboard_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.base_model_dir = self.output_dir / "base_models"
        self.finetuned_dir = self.output_dir / "finetuned"
        self.base_model_dir.mkdir(exist_ok=True)
        self.finetuned_dir.mkdir(exist_ok=True)
        
        # Data processor
        self.processor = StockDataProcessor()
        
        # Logger
        self.logger = get_logger(str(self.output_dir / "logs"), "base_model_training")
    
    def download_all_stock_data(
        self,
        start_date: str = '2018-01-01',
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Load local CSV data for all base stocks (no external download)"""
        self.logger.info(f"Loading local data for {len(self.base_stocks)} stocks from trainingdata/")
        stock_data = load_local_stock_data(self.base_stocks, data_dir="trainingdata")
        if not stock_data:
            self.logger.error("No local CSVs found under trainingdata/ for requested symbols")
            return {}
        # Log statistics (if date column exists)
        for symbol, df in stock_data.items():
            n = len(df)
            if n > 0:
                self.logger.info(f"{symbol}: {n} records")
        return stock_data
    
    def prepare_multi_stock_data(
        self,
        stock_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, np.ndarray]:
        """Process and normalize data for all stocks"""
        
        processed_data = {}
        all_data_for_scaling = []
        
        # First pass: collect all data for fitting scalers
        for symbol, df in stock_data.items():
            features = self.processor.prepare_features(df)
            all_data_for_scaling.append(features)
        
        # Fit scalers on combined data
        combined_data = np.vstack(all_data_for_scaling)
        self.processor.fit_scalers(combined_data)
        
        # Save processor
        processor_path = self.base_model_dir / "data_processor.pkl"
        self.processor.save_scalers(str(processor_path))
        self.logger.info(f"Saved data processor to {processor_path}")
        
        # Second pass: transform all data
        for symbol, df in stock_data.items():
            features = self.processor.prepare_features(df)
            normalized = self.processor.transform(features)
            processed_data[symbol] = normalized
        
        return processed_data
    
    def train_base_model(
        self,
        config: Optional[ExperimentConfig] = None,
        max_steps: int = 10000,
        batch_size: int = 32,
        learning_rate: float = 1e-4
    ) -> Tuple[TransformerTradingModel, str]:
        """Train base model on all stocks"""
        
        self.logger.info("=" * 80)
        self.logger.info("🚀 STARTING BASE MODEL TRAINING")
        self.logger.info("=" * 80)
        
        # Configuration
        if config is None:
            config = create_config("production")
            config.training.max_steps = max_steps
            config.training.batch_size = batch_size
            config.training.learning_rate = learning_rate
        
        # Update paths for base model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.output.output_dir = str(self.base_model_dir / f"base_{timestamp}")
        config.output.logging_dir = str(self.tensorboard_dir / f"base_{timestamp}")
        config.experiment_name = f"base_model_{timestamp}"
        
        # Download and prepare data
        stock_data = self.download_all_stock_data()
        processed_data = self.prepare_multi_stock_data(stock_data)
        
        # Create multi-stock dataset
        dataset = MultiStockDataset(
            processed_data,
            sequence_length=config.data.sequence_length,
            prediction_horizon=config.data.prediction_horizon,
            processor=self.processor
        )
        
        # Split into train/val
        train_size = int(0.85 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        self.logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        # Create model
        input_dim = list(processed_data.values())[0].shape[1]
        hf_config = HFTrainingConfig(
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            sequence_length=config.data.sequence_length,
            prediction_horizon=config.data.prediction_horizon,
            learning_rate=config.training.learning_rate,
            batch_size=config.training.batch_size,
            max_steps=config.training.max_steps,
            warmup_steps=config.training.warmup_steps,
            output_dir=config.output.output_dir,
            logging_dir=config.output.logging_dir
        )
        
        model = TransformerTradingModel(hf_config, input_dim=input_dim)
        
        self.logger.info(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create trainer with profit tracking
        trainer = HFTrainer(
            model=model,
            config=hf_config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        # Add profit tracking
        profit_tracker = ProfitTracker()
        trainer = integrate_profit_tracking(trainer, profit_tracker)
        
        # Train model
        self.logger.info("Starting training...")
        trained_model = trainer.train()
        
        # Save base model checkpoint
        checkpoint_path = self.base_model_dir / f"base_checkpoint_{timestamp}.pth"
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'config': hf_config,
            'input_dim': input_dim,
            'stock_symbols': self.base_stocks,
            'training_metrics': trainer.metrics_tracker.best_metrics if hasattr(trainer, 'metrics_tracker') else {}
        }, checkpoint_path)
        
        self.logger.info(f"✅ Base model saved to {checkpoint_path}")
        
        # Save config
        config.save(str(self.base_model_dir / f"base_config_{timestamp}.json"))
        
        return trained_model, str(checkpoint_path)
    
    def finetune_for_stock(
        self,
        stock_symbol: str,
        base_checkpoint_path: str,
        num_epochs: int = 10,
        learning_rate: float = 5e-5,
        start_date: str = '2020-01-01'
    ) -> Tuple[TransformerTradingModel, str]:
        """Fine-tune base model for a specific stock"""
        
        self.logger.info(f"Fine-tuning for {stock_symbol}")
        
        # Load base model
        checkpoint = torch.load(base_checkpoint_path, weights_only=False)
        base_config = checkpoint['config']
        input_dim = checkpoint['input_dim']
        
        # Create model and load weights
        model = TransformerTradingModel(base_config, input_dim=input_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        self.logger.info(f"Loaded base model from {base_checkpoint_path}")
        
        # Load stock-specific data locally
        stock_map = load_local_stock_data([stock_symbol], data_dir="trainingdata")
        if stock_symbol not in stock_map or len(stock_map[stock_symbol]) == 0:
            self.logger.error(f"No local CSV found for {stock_symbol} under trainingdata/")
            return None, None
        df = stock_map[stock_symbol]
        features = self.processor.prepare_features(df)
        normalized_data = self.processor.transform(features)
        
        # Create dataset
        dataset = StockDataset(
            normalized_data,
            sequence_length=base_config.sequence_length,
            prediction_horizon=base_config.prediction_horizon
        )
        
        # Split data
        train_size = int(0.85 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        self.logger.info(f"{stock_symbol} dataset - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        # Update config for fine-tuning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        finetune_config = base_config
        finetune_config.learning_rate = learning_rate
        finetune_config.max_steps = num_epochs * len(train_dataset) // finetune_config.batch_size
        finetune_config.warmup_steps = min(500, finetune_config.max_steps // 10)
        finetune_config.output_dir = str(self.finetuned_dir / f"{stock_symbol}_{timestamp}")
        finetune_config.logging_dir = str(self.tensorboard_dir / f"finetune_{stock_symbol}_{timestamp}")
        
        # Create trainer
        trainer = HFTrainer(
            model=model,
            config=finetune_config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        # Add profit tracking
        profit_tracker = ProfitTracker()
        trainer = integrate_profit_tracking(trainer, profit_tracker)
        
        # Fine-tune
        self.logger.info(f"Starting fine-tuning for {stock_symbol}...")
        finetuned_model = trainer.train()
        
        # Save fine-tuned model
        finetuned_path = self.finetuned_dir / f"{stock_symbol}_finetuned_{timestamp}.pth"
        torch.save({
            'model_state_dict': finetuned_model.state_dict(),
            'config': finetune_config,
            'input_dim': input_dim,
            'stock_symbol': stock_symbol,
            'base_checkpoint': base_checkpoint_path,
            'training_metrics': trainer.metrics_tracker.best_metrics if hasattr(trainer, 'metrics_tracker') else {}
        }, finetuned_path)
        
        self.logger.info(f"✅ Fine-tuned model for {stock_symbol} saved to {finetuned_path}")
        
        return finetuned_model, str(finetuned_path)
    
    def run_complete_pipeline(
        self,
        stocks_to_finetune: Optional[List[str]] = None,
        base_training_steps: int = 10000,
        finetune_epochs: int = 10
    ):
        """Run complete base training + fine-tuning pipeline"""
        
        self.logger.info("🚀 Starting Complete Training Pipeline")
        
        # Train base model
        base_model, base_checkpoint = self.train_base_model(
            max_steps=base_training_steps
        )
        
        # Fine-tune for each specified stock
        if stocks_to_finetune is None:
            stocks_to_finetune = self.base_stocks[:3]  # Default to first 3
        
        finetuned_models = {}
        
        for stock in stocks_to_finetune:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Fine-tuning for {stock}")
            self.logger.info(f"{'='*60}")
            
            finetuned_model, finetuned_path = self.finetune_for_stock(
                stock_symbol=stock,
                base_checkpoint_path=base_checkpoint,
                num_epochs=finetune_epochs
            )
            
            if finetuned_model is not None:
                finetuned_models[stock] = {
                    'model': finetuned_model,
                    'path': finetuned_path
                }
        
        # Generate summary report
        self.generate_pipeline_report(base_checkpoint, finetuned_models)
        
        return base_checkpoint, finetuned_models
    
    def generate_pipeline_report(
        self,
        base_checkpoint: str,
        finetuned_models: Dict[str, Dict]
    ):
        """Generate comprehensive training report"""
        
        report_path = self.output_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        report = f"""# Base Model + Fine-tuning Training Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Base Model Training

- **Checkpoint:** {base_checkpoint}
- **Base Stocks:** {', '.join(self.base_stocks)}
- **Model Architecture:** Transformer-based trading model

## Fine-tuned Models

"""
        
        for stock, info in finetuned_models.items():
            report += f"""
### {stock}
- **Model Path:** {info['path']}
- **Status:** ✅ Completed
"""
        
        report += f"""
## Directory Structure

```
{self.output_dir}/
├── base_models/          # Base model checkpoints
├── finetuned/           # Fine-tuned models per stock
└── logs/                # Training logs
```

## TensorBoard

View training metrics:
```bash
tensorboard --logdir {self.tensorboard_dir}
```

## Next Steps

1. Evaluate models on test data
2. Run backtesting simulations
3. Deploy best performing models
4. Monitor live performance

---
*Generated by BaseModelTrainer*
"""
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"📄 Report saved to {report_path}")


def main():
    """Main entry point for base model training"""
    
    # Configuration
    trainer = BaseModelTrainer(
        base_stocks=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
        output_dir="hftraining/models",
        tensorboard_dir="hftraining/tensorboard"
    )
    
    # Run complete pipeline
    base_checkpoint, finetuned_models = trainer.run_complete_pipeline(
        stocks_to_finetune=['AAPL', 'GOOGL'],
        base_training_steps=5000,  # Reduced for testing
        finetune_epochs=5
    )
    
    print(f"\n✅ Training Complete!")
    print(f"Base Model: {base_checkpoint}")
    print(f"Fine-tuned Models: {list(finetuned_models.keys())}")


if __name__ == "__main__":
    main()
