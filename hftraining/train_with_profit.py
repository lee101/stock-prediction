#!/usr/bin/env python3
"""
Enhanced Training Script with Profit Tracking
Runs training with profit metrics logged to TensorBoard
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.append(os.path.dirname(current_dir))

from config import create_config
from train_hf import HFTrainer, StockDataset
from hf_trainer import TransformerTradingModel, HFTrainingConfig
from data_utils import StockDataProcessor, split_data, load_local_stock_data
from profit_tracker import ProfitTracker, integrate_profit_tracking, ProfitAwareLoss
from base_model_trainer import BaseModelTrainer
from logging_utils import get_logger


def setup_directories():
    """Setup organized directory structure"""
    
    base_dir = Path("hftraining")
    
    # Main directories
    dirs = {
        'models': base_dir / "models",
        'models_base': base_dir / "models" / "base",
        'models_finetuned': base_dir / "models" / "finetuned",
        'tensorboard': base_dir / "tensorboard",
        'tensorboard_base': base_dir / "tensorboard" / "base",
        'tensorboard_finetuned': base_dir / "tensorboard" / "finetuned",
        'logs': base_dir / "logs",
        'data': base_dir / "data",
        'data_processed': base_dir / "data" / "processed",
        'data_raw': base_dir / "data" / "raw",
        'reports': base_dir / "reports",
        'checkpoints': base_dir / "checkpoints",
    }
    
    # Create all directories
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
    
    print("📁 Directory structure created:")
    print(f"""
    hftraining/
    ├── models/
    │   ├── base/           # Base models trained on all stocks
    │   └── finetuned/      # Stock-specific fine-tuned models
    ├── tensorboard/
    │   ├── base/           # Base model training logs
    │   └── finetuned/      # Fine-tuning logs per stock
    ├── logs/               # Text logs
    ├── data/
    │   ├── raw/           # Downloaded stock data
    │   └── processed/     # Processed features
    ├── reports/           # Training reports
    └── checkpoints/       # Training checkpoints
    """)
    
    return dirs


def train_single_stock_with_profit(
    stock_symbol: str,
    config: dict = None,
    dirs: dict = None,
    data_dir: str = "trainingdata"
):
    """Train model for single stock with profit tracking"""
    
    if dirs is None:
        dirs = setup_directories()
    
    logger = get_logger(str(dirs['logs']), f"train_{stock_symbol}")
    
    logger.info(f"🚀 Training model for {stock_symbol} with profit tracking")
    
    # Configuration
    if config is None:
        config = create_config("quick_test")
        config.model.hidden_size = 256
        config.model.num_layers = 6
        config.training.max_steps = 5000
        config.training.batch_size = 16
    
    # Set paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.output.output_dir = str(dirs['models'] / f"{stock_symbol}_{timestamp}")
    config.output.logging_dir = str(dirs['tensorboard'] / f"{stock_symbol}_{timestamp}")
    config.experiment_name = f"{stock_symbol}_{timestamp}"
    
    # Load local CSV data
    logger.info(f"Loading local CSV for {stock_symbol}...")
    stock_map = load_local_stock_data([stock_symbol], data_dir=data_dir)
    if stock_symbol not in stock_map:
        logger.error(f"No local CSV found for {stock_symbol} under {data_dir} (with fallbacks)")
        return None
    df = stock_map[stock_symbol]
    logger.info(f"Loaded {len(df)} records for {stock_symbol}")
    
    # Process data
    processor = StockDataProcessor()
    features = processor.prepare_features(df)
    processor.fit_scalers(features)
    normalized_data = processor.transform(features)
    
    # Save processor
    processor_path = Path(config.output.output_dir) / "data_processor.pkl"
    Path(config.output.output_dir).mkdir(parents=True, exist_ok=True)
    processor.save_scalers(str(processor_path))
    
    # Create datasets
    train_data, val_data, test_data = split_data(normalized_data, 0.7, 0.15, 0.15)
    
    train_dataset = StockDataset(
        train_data,
        sequence_length=config.data.sequence_length,
        prediction_horizon=config.data.prediction_horizon,
        processor=processor,
    )

    val_dataset = StockDataset(
        val_data,
        sequence_length=config.data.sequence_length,
        prediction_horizon=config.data.prediction_horizon,
        processor=processor,
    )
    
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create model
    input_dim = normalized_data.shape[1]
    hf_config = HFTrainingConfig(
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        sequence_length=config.data.sequence_length,
        prediction_horizon=config.data.prediction_horizon,
        learning_rate=config.training.learning_rate,
        batch_size=config.training.batch_size,
        max_steps=config.training.max_steps,
        output_dir=config.output.output_dir,
        logging_dir=config.output.logging_dir,
        profit_loss_weight=config.training.profit_loss_weight,
        transaction_cost_bps=config.training.transaction_cost_bps,
    )
    
    model = TransformerTradingModel(hf_config, input_dim=input_dim)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    trainer = HFTrainer(
        model=model,
        config=hf_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Add profit tracking
    profit_tracker = ProfitTracker(
        initial_capital=10000,
        commission=0.001,
        max_position_size=0.3
    )
    
    trainer = integrate_profit_tracking(trainer, profit_tracker)
    
    # Train model
    logger.info("Starting training with profit tracking...")
    trained_model = trainer.train()
    
    # Save final model
    final_path = Path(config.output.output_dir) / "final_model.pth"
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': hf_config,
        'input_dim': input_dim,
        'stock_symbol': stock_symbol,
        'training_complete': True
    }, final_path)
    
    logger.info(f"✅ Training complete! Model saved to {final_path}")
    
    # Generate profit report
    generate_profit_report(trainer, stock_symbol, dirs['reports'])
    
    return trained_model, str(final_path)


def generate_profit_report(trainer, stock_symbol: str, report_dir: Path):
    """Generate profit tracking report"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"profit_report_{stock_symbol}_{timestamp}.md"
    
    # Get final metrics if available
    if hasattr(trainer, 'profit_tracker'):
        # Get the last tracked metrics
        report = f"""# Profit Tracking Report - {stock_symbol}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Training Configuration
- **Model:** {trainer.config.hidden_size}d, {trainer.config.num_layers} layers
- **Steps:** {trainer.global_step}
- **Stock:** {stock_symbol}

## Profit Metrics
(Metrics tracked during training - simulated trading)

- **Initial Capital:** $10,000
- **Commission:** 0.1%
- **Max Position Size:** 30% of capital

## TensorBoard

To view detailed profit metrics over time:
```bash
tensorboard --logdir {trainer.config.logging_dir}
```

Look for the `profit/` metrics:
- `profit/total_return` - Cumulative return
- `profit/sharpe_ratio` - Risk-adjusted return
- `profit/max_drawdown` - Maximum drawdown
- `profit/win_rate` - Percentage of profitable trades
- `profit/total_trades` - Number of trades executed

## Notes

Profit tracking during training provides insights into:
1. How well the model's predictions translate to profits
2. Risk-adjusted performance (Sharpe ratio)
3. Trading frequency and win rate
4. Maximum drawdown and risk metrics

These metrics help optimize for profitability, not just prediction accuracy.
"""
    else:
        report = f"# Profit Report - {stock_symbol}\n\nProfit tracking not available for this training run."
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"📄 Profit report saved to {report_path}")


def run_multi_stock_experiment(
    stocks: list = None,
    use_base_model: bool = True
):
    """Run experiment with multiple stocks"""
    
    if stocks is None:
        stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    dirs = setup_directories()
    
    if use_base_model:
        # Train base model then fine-tune
        print("🎯 Training base model on all stocks...")
        trainer = BaseModelTrainer(
            base_stocks=stocks,
            output_dir=str(dirs['models_base']),
            tensorboard_dir=str(dirs['tensorboard_base'])
        )
        
        base_checkpoint, finetuned_models = trainer.run_complete_pipeline(
            stocks_to_finetune=stocks[:2],  # Fine-tune first 2
            base_training_steps=5000,
            finetune_epochs=5
        )
        
        print(f"\n✅ Base model: {base_checkpoint}")
        print(f"✅ Fine-tuned: {list(finetuned_models.keys())}")
        
    else:
        # Train individual models
        print("🎯 Training individual models per stock...")
        
        results = {}
        for stock in stocks:
            print(f"\n{'='*60}")
            print(f"Training {stock}")
            print(f"{'='*60}")
            
            model, path = train_single_stock_with_profit(
                stock_symbol=stock,
                dirs=dirs
            )
            
            if model is not None:
                results[stock] = path
        
        print(f"\n✅ Trained {len(results)} models:")
        for stock, path in results.items():
            print(f"  - {stock}: {path}")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Train models with profit tracking')
    parser.add_argument('--stock', type=str, help='Single stock to train')
    parser.add_argument('--stocks', nargs='+', help='Multiple stocks to train')
    parser.add_argument('--base-model', action='store_true', help='Use base model approach')
    parser.add_argument('--steps', type=int, default=5000, help='Training steps')
    
    args = parser.parse_args()
    
    if args.stock:
        # Train single stock
        dirs = setup_directories()
        model, path = train_single_stock_with_profit(args.stock, dirs=dirs)
        print(f"\n✅ Model trained and saved to: {path}")
        
    elif args.stocks:
        # Train multiple stocks
        run_multi_stock_experiment(args.stocks, use_base_model=args.base_model)
        
    else:
        # Default: train with profit tracking on a single stock
        print("🚀 Running default training with profit tracking...")
        dirs = setup_directories()
        model, path = train_single_stock_with_profit('AAPL', dirs=dirs)
        
        print(f"\n✅ Training complete!")
        print(f"📊 View metrics in TensorBoard:")
        print(f"   tensorboard --logdir hftraining/tensorboard")
        print(f"\n💾 Model saved to: {path}")


if __name__ == "__main__":
    main()
