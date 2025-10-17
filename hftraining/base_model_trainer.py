#!/usr/bin/env python3
"""
Base Model Training Pipeline
Trains a base model on multiple stock pairs, then allows fine-tuning for individual stocks
"""

import os
import sys
import torch
from torch.serialization import add_safe_globals
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from typing import Dict, List, Optional, Sequence, Tuple
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
from data_utils import (
    StockDataProcessor,
    split_data,
    load_local_stock_data,
    PairStockDataset,
    MultiAssetPortfolioDataset,
    align_on_timestamp,
    load_toto_prediction_history,
)
from profit_tracker import ProfitTracker, integrate_profit_tracking
from logging_utils import get_logger
from toto_features import TotoOptions
from portfolio_rl_trainer import (
    PortfolioAllocationModel,
    PortfolioRLConfig,
    DifferentiablePortfolioTrainer,
)


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
        tensorboard_dir: str = "hftraining/tensorboard",
        use_toto_forecasts: bool = True,
        toto_options: Optional[TotoOptions] = None,
        data_dir: str = "trainingdata",
        max_rows: Optional[int] = None,
        toto_predictions_dir: Optional[str] = None,
    ):
        self.base_stocks = base_stocks or [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
            'META', 'NVDA', 'JPM', 'V', 'JNJ'
        ]
        self.output_dir = Path(output_dir)
        self.tensorboard_dir = Path(tensorboard_dir)
        self.use_toto_forecasts = use_toto_forecasts
        self.toto_options = toto_options or TotoOptions()
        self.data_dir = Path(data_dir)
        self.max_rows = max_rows
        self.toto_predictions_dir = Path(toto_predictions_dir).expanduser() if toto_predictions_dir else None
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.base_model_dir = self.output_dir / "base_models"
        self.finetuned_dir = self.output_dir / "finetuned"
        self.base_model_dir.mkdir(exist_ok=True)
        self.finetuned_dir.mkdir(exist_ok=True)

        # Logger
        self.logger = get_logger(str(self.output_dir / "logs"), "base_model_training")

        self._toto_prediction_features: Dict[str, pd.DataFrame] = {}
        self._toto_prediction_columns: List[str] = []
        if self.toto_predictions_dir:
            try:
                features, columns = load_toto_prediction_history(self.toto_predictions_dir)
                self._toto_prediction_features = features
                self._toto_prediction_columns = columns
                if not features:
                    self.logger.warning(
                        "No Toto prediction rows found in %s; continuing without precomputed features",
                        self.toto_predictions_dir,
                    )
                else:
                    self.logger.info(
                        "Loaded Toto prediction features for %d symbols from %s",
                        len(features),
                        self.toto_predictions_dir,
                    )
            except FileNotFoundError:
                self.logger.warning(
                    "Toto prediction directory '%s' not found; skipping precomputed features",
                    self.toto_predictions_dir,
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(
                    "Failed to load Toto prediction features from %s: %s",
                    self.toto_predictions_dir,
                    exc,
                )

        # Data processor
        self.processor = StockDataProcessor(
            sequence_length=self.toto_options.context_length,
            prediction_horizon=self.toto_options.horizon,
            use_toto_forecasts=self.use_toto_forecasts,
            toto_options=self.toto_options,
            toto_prediction_features=self._toto_prediction_features,
            toto_prediction_columns=self._toto_prediction_columns,
        )

    def _configure_processor_from_config(self, data_config):
        """Ensure processor follows the latest data configuration."""
        toto_opts = TotoOptions(
            use_toto=data_config.use_toto_forecasts,
            horizon=data_config.toto_horizon,
            context_length=data_config.sequence_length,
            num_samples=data_config.toto_num_samples,
            toto_model_id=data_config.toto_model_id,
            toto_device=data_config.toto_device,
        )
        self.use_toto_forecasts = data_config.use_toto_forecasts
        self.toto_options = toto_opts
        self.processor = StockDataProcessor(
            sequence_length=data_config.sequence_length,
            prediction_horizon=data_config.prediction_horizon,
            use_toto_forecasts=self.use_toto_forecasts,
            toto_options=toto_opts,
            toto_prediction_features=self._toto_prediction_features,
            toto_prediction_columns=self._toto_prediction_columns,
        )
    
    def download_all_stock_data(
        self,
        start_date: str = '2018-01-01',
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Load local CSV data for all base stocks (no external download)"""
        self.logger.info(
            f"Loading local data for {len(self.base_stocks)} stocks from {self.data_dir}/"
        )
        stock_data = load_local_stock_data(self.base_stocks, data_dir=str(self.data_dir))
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
        
        processed_data: Dict[str, np.ndarray] = {}
        all_data_for_scaling: List[np.ndarray] = []
        
        # First pass: collect all data for fitting scalers
        for symbol, df in stock_data.items():
            if self.max_rows is not None:
                df = df.tail(self.max_rows).copy()
            features = self.processor.prepare_features(df, symbol=symbol)
            all_data_for_scaling.append(features)

        if not all_data_for_scaling:
            raise ValueError("No features produced for scaling; check data preparation.")

        feature_dims = [arr.shape[1] for arr in all_data_for_scaling]
        target_dim = max(feature_dims)

        def _pad_features(array: np.ndarray, dim: int) -> np.ndarray:
            if array.shape[1] == dim:
                return array
            pad_width = dim - array.shape[1]
            if pad_width <= 0:
                return array
            padding = np.zeros((array.shape[0], pad_width), dtype=array.dtype)
            return np.concatenate([array, padding], axis=1)

        aligned_for_scaling = [_pad_features(arr, target_dim) for arr in all_data_for_scaling]

        # Fit scalers on combined data
        combined_data = np.vstack(aligned_for_scaling)
        self.processor.fit_scalers(combined_data)
        self._feature_dim = target_dim
        
        # Save processor
        processor_path = self.base_model_dir / "data_processor.pkl"
        self.processor.save_scalers(str(processor_path))
        self.logger.info(f"Saved data processor to {processor_path}")
        
        # Second pass: transform all data
        for symbol, df in stock_data.items():
            if self.max_rows is not None:
                df = df.tail(self.max_rows).copy()
            features = self.processor.prepare_features(df, symbol=symbol)
            features = _pad_features(features, target_dim)
            normalized = self.processor.transform(features)
            processed_data[symbol] = normalized
        
        return processed_data
    
    def train_base_model(
        self,
        config: Optional[ExperimentConfig] = None,
        max_steps: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        progressive_schedule: Optional[Sequence[int]] = None,
    ) -> Tuple[TransformerTradingModel, str]:
        """Train base model on all stocks"""
        
        self.logger.info("=" * 80)
        self.logger.info("🚀 STARTING BASE MODEL TRAINING")
        self.logger.info("=" * 80)
        
        # Configuration
        if config is None:
            config = create_config("production")

        if max_steps is not None:
            config.training.max_steps = max_steps
        if batch_size is not None:
            config.training.batch_size = batch_size
        if learning_rate is not None:
            config.training.learning_rate = learning_rate

        schedule: Optional[List[int]] = None
        if progressive_schedule:
            schedule = [int(step) for step in progressive_schedule if int(step) > 0]
            if schedule:
                config.training.max_steps = schedule[0]
            else:
                schedule = None

        # Ensure data config reflects current Toto/settings regardless of source.
        config.data.use_toto_forecasts = self.use_toto_forecasts
        config.data.toto_horizon = self.toto_options.horizon
        config.data.sequence_length = self.toto_options.context_length
        config.data.prediction_horizon = self.toto_options.horizon
        config.data.toto_num_samples = self.toto_options.num_samples
        config.data.toto_model_id = self.toto_options.toto_model_id
        config.data.toto_device = self.toto_options.toto_device
        config.training.use_mixed_precision = False
        config.training.gradient_checkpointing = False
        
        # Update paths for base model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.output.output_dir = str(self.base_model_dir / f"base_{timestamp}")
        config.output.logging_dir = str(self.tensorboard_dir / f"base_{timestamp}")
        config.experiment_name = f"base_model_{timestamp}"

        # Ensure processor matches configuration (Toto, sequence length, etc.)
        self._configure_processor_from_config(config.data)
        
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
            logging_dir=config.output.logging_dir,
            profit_loss_weight=config.training.profit_loss_weight,
            transaction_cost_bps=config.training.transaction_cost_bps,
            use_mixed_precision=config.training.use_mixed_precision,
            use_gradient_checkpointing=config.training.gradient_checkpointing,
        )
        hf_config.dataloader_num_workers = 0
        hf_config.persistent_workers = False
        hf_config.prefetch_factor = 2

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
        
        # Train model (optionally in progressive stages)
        self.logger.info("Starting training...")

        if schedule:
            cumulative = 0
            for stage_idx, stage_steps in enumerate(schedule, start=1):
                cumulative += stage_steps
                trainer.config.max_steps = cumulative
                trainer.config.warmup_steps = min(
                    trainer.config.warmup_steps,
                    max(1, cumulative // 10),
                )
                trainer.training_logger.info(
                    f"Progressive base stage {stage_idx}/{len(schedule)} -> max_steps {cumulative:,}"
                )
                trainer.train()
            trained_model = trainer.model
        else:
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
        stock_map = load_local_stock_data([stock_symbol], data_dir=str(self.data_dir))
        if stock_symbol not in stock_map or len(stock_map[stock_symbol]) == 0:
            self.logger.error(f"No local CSV found for {stock_symbol} under trainingdata/")
            return None, None
        df = stock_map[stock_symbol]
        if self.max_rows is not None:
            df = df.tail(self.max_rows).copy()
        features = self.processor.prepare_features(df, symbol=stock_symbol)
        normalized_data = self.processor.transform(features)
        
        # Create dataset
        dataset = StockDataset(
            normalized_data,
            sequence_length=base_config.sequence_length,
            prediction_horizon=base_config.prediction_horizon,
            processor=self.processor,
            symbol=stock_symbol,
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
        finetune_config.dataloader_num_workers = 0
        finetune_config.persistent_workers = False
        
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
        finetune_epochs: int = 10,
        pair_symbols: Optional[List[Tuple[str, str]]] = None,
        base_config: Optional[ExperimentConfig] = None,
        rl_config: Optional[PortfolioRLConfig] = None,
    ):
        """Run complete base training + fine-tuning pipeline"""
        
        self.logger.info("🚀 Starting Complete Training Pipeline")
        
        # Train base model
        base_model, base_checkpoint = self.train_base_model(
            config=base_config,
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

        # Train portfolio allocation policies for selected symbol groups
        if pair_symbols is None:
            default_slice = min(4, len(self.base_stocks))
            pair_symbols = [tuple(self.base_stocks[:default_slice])]

        pair_metrics: Dict[Tuple[str, ...], Dict[str, float]] = {}
        for symbol_group in pair_symbols:
            symbols_tuple = tuple(symbol_group)
            try:
                self.logger.info(f"Training portfolio RL for symbols {symbols_tuple}")
                metrics = self.train_pair_portfolio(symbols_tuple, rl_config=rl_config)
                pair_metrics[symbols_tuple] = metrics
            except Exception as exc:
                self.logger.error(f"Failed portfolio RL for {symbols_tuple}: {exc}")

        # Generate summary report
        self.generate_pipeline_report(base_checkpoint, finetuned_models, pair_metrics)
        
        return base_checkpoint, finetuned_models, pair_metrics
    
    def generate_pipeline_report(
        self,
        base_checkpoint: str,
        finetuned_models: Dict[str, Dict],
        pair_metrics: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None,
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

        if pair_metrics:
            report += """
## Portfolio RL Pairs

"""
            for symbols_tuple, metrics in pair_metrics.items():
                summary = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                title = "/".join(symbols_tuple)
                report += f"""
### {title}
- **Metrics:** {summary}
"""
        
        report += f"""
## Directory Structure

```
{self.output_dir}/
├── base_models/          # Base model checkpoints
├── finetuned/           # Fine-tuned models per stock
├── finetuned/portfolio_pairs/  # Differentiable portfolio RL checkpoints
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

    def train_pair_portfolio(
        self,
        symbols: Sequence[str],
        rl_config: Optional[PortfolioRLConfig] = None,
        initial_checkpoint: Optional[Path] = None,
    ) -> Dict[str, float]:
        """Train differentiable portfolio allocation policy for one or more symbols."""

        symbols = list(symbols)
        if len(symbols) < 2:
            raise ValueError("At least two symbols are required for portfolio training")
        self.logger.info(f"Preparing portfolio RL training for symbols {symbols}")

        if not self.processor.scalers:
            processor_path = self.base_model_dir / "data_processor.pkl"
            if processor_path.exists():
                self.processor.load_scalers(str(processor_path))
            else:
                raise RuntimeError("Data processor scalers are not initialised. Run base training first.")

        stock_map = load_local_stock_data(symbols, data_dir=str(self.data_dir))
        for sym in symbols:
            if sym not in stock_map:
                raise RuntimeError(f"Missing CSV data for {sym} under {self.data_dir}/")

        # Align timestamps across all assets
        aligned_frames = []
        for sym in symbols:
            df = stock_map[sym]
            if self.max_rows is not None:
                df = df.tail(self.max_rows).copy()
            aligned_frames.append(df)

        candidate_keys = ['date', 'timestamp', 'Datetime', 'datetime']
        key = next((col for col in candidate_keys if all(col in df.columns for df in aligned_frames)), None)
        if key is None:
            raise ValueError("Unable to align portfolio symbols – no shared timestamp column")

        common_index = None
        normalised_frames = []
        for df in aligned_frames:
            col = df[key]
            if not pd.api.types.is_datetime64_any_dtype(col):
                col = pd.to_datetime(col)
            try:
                col = col.dt.tz_localize(None)
            except AttributeError:
                pass
            df = df.copy()
            align_col = col.dt.floor("T")
            df["__align_key"] = align_col
            idx = pd.Index(align_col)
            common_index = idx if common_index is None else common_index.intersection(idx)
            normalised_frames.append(df)

        if common_index is None or len(common_index) == 0:
            raise RuntimeError("No overlapping timestamps across portfolio symbols")

        common_index = pd.Index(sorted(common_index))
        if self.max_rows is not None and len(common_index) > self.max_rows:
            common_index = common_index[-self.max_rows:]

        aligned_data = []
        for df in normalised_frames:
            df = df.set_index("__align_key").loc[common_index].reset_index()
            df = df.rename(columns={"__align_key": "date"})
            aligned_data.append(df)

        asset_arrays: List[np.ndarray] = []
        asset_close_prices: List[np.ndarray] = []
        close_feature_index: Optional[int] = None
        for sym, df in zip(symbols, aligned_data):
            features = self.processor.prepare_features(df, symbol=sym)
            feature_names = list(self.processor.feature_names)
            if not feature_names:
                raise RuntimeError("Processor returned empty feature name list.")
            current_close_idx = feature_names.index('close') if 'close' in feature_names else 3
            if close_feature_index is None:
                close_feature_index = current_close_idx
            elif current_close_idx != close_feature_index:
                raise RuntimeError(
                    f"Inconsistent close feature index across assets: {current_close_idx} vs {close_feature_index}"
                )

            normalized = self.processor.transform(features).astype(np.float32, copy=False)
            asset_arrays.append(normalized)
            asset_close_prices.append(features[:, close_feature_index].astype(np.float32, copy=False))

        if close_feature_index is None:
            raise RuntimeError("Unable to determine close feature index for portfolio dataset.")

        dataset = MultiAssetPortfolioDataset(
            asset_arrays,
            symbols,
            asset_close_prices,
            sequence_length=self.processor.sequence_length,
            prediction_horizon=self.processor.prediction_horizon,
            close_feature_index=close_feature_index,
        )

        if len(dataset) < 10:
            raise RuntimeError("Portfolio dataset too small for RL training")

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        rl_config = rl_config or PortfolioRLConfig()
        rl_config.logging_dir = str(self.tensorboard_dir / "portfolio")
        if getattr(rl_config, "wandb_group", None) is None:
            rl_config.wandb_group = "portfolio_rl"
        train_loader = DataLoader(train_ds, batch_size=rl_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=rl_config.batch_size) if val_size > 0 else None

        sample = dataset[0]['input_ids']
        input_dim = sample.shape[-1]
        num_assets = len(symbols)
        model = PortfolioAllocationModel(input_dim=input_dim, config=rl_config, num_assets=num_assets)
        if initial_checkpoint:
            add_safe_globals([PortfolioRLConfig])
            ckpt_payload = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
            state_dict = ckpt_payload.get("model_state_dict", ckpt_payload)
            cleaned_state = {
                k.replace("_orig_mod.", ""): v
                for k, v in state_dict.items()
            }
            model.load_state_dict(cleaned_state)
        trainer = DifferentiablePortfolioTrainer(model, rl_config, train_loader, val_loader)
        metrics = trainer.train()
        final_state = trainer.export_state_dict()

        checkpoint_dir = self.finetuned_dir / "portfolio_pairs"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_name = "_".join(symbols)
        ckpt_path = checkpoint_dir / f"{ckpt_name}_portfolio.pt"
        payload = {
            'model_state_dict': final_state,
            'config': rl_config,
            'symbols': symbols,
            'metrics': metrics,
        }
        torch.save(payload, ckpt_path)

        best_state = trainer.best_state_dict()
        best_path: Optional[Path] = None
        if best_state is not None:
            best_path = checkpoint_dir / f"{ckpt_name}_portfolio_best.pt"
            best_payload = {
                'model_state_dict': best_state,
                'config': rl_config,
                'symbols': symbols,
                'metrics': metrics,
                'best_epoch': metrics.get("best_epoch", -1),
                'best_val_profit': metrics.get("best_val_profit"),
            }
            torch.save(best_payload, best_path)
            metrics["best_checkpoint"] = str(best_path)
        else:
            metrics["best_checkpoint"] = None

        self.logger.info(f"Portfolio RL model saved to {ckpt_path}")
        if best_path:
            self.logger.info(f"Best validation checkpoint saved to {best_path}")
        return metrics


def main():
    """Main entry point for base model training"""
    
    # Configuration
    trainer = BaseModelTrainer(
        base_stocks=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
        output_dir="hftraining/models",
        tensorboard_dir="hftraining/tensorboard"
    )
    
    # Run complete pipeline
    base_checkpoint, finetuned_models, pair_metrics = trainer.run_complete_pipeline(
        stocks_to_finetune=['AAPL', 'GOOGL'],
        base_training_steps=5000,  # Reduced for testing
        finetune_epochs=5,
        pair_symbols=[('AAPL', 'GOOGL', 'TSLA')]
    )
    
    print(f"\n✅ Training Complete!")
    print(f"Base Model: {base_checkpoint}")
    print(f"Fine-tuned Models: {list(finetuned_models.keys())}")
    print(f"Portfolio Pairs: {list(pair_metrics.keys())}")


if __name__ == "__main__":
    main()
