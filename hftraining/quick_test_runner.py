#!/usr/bin/env python3
"""
Quick Test Runner for Small Data Experiments
Tests the training system with minimal resources and saves models
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import tempfile

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.append(os.path.dirname(current_dir))

from config import create_config
from run_training import run_training, setup_environment, load_and_process_data, create_model
from data_utils import generate_synthetic_data, split_data, StockDataProcessor
from train_hf import StockDataset


def create_small_test_data(length=2000, n_features=15, seed=42):
    """Create small synthetic dataset for testing"""
    np.random.seed(seed)
    
    print(f"Generating {length} samples with {n_features} features...")
    
    # Create more realistic stock-like data
    data = []
    current_price = 100.0
    
    for i in range(length):
        # Random walk with mean reversion
        price_change = np.random.normal(0, 0.02) - 0.001 * (current_price - 100) / 100
        current_price *= (1 + price_change)
        current_price = max(current_price, 1.0)  # Prevent negative prices
        
        # Generate OHLCV
        volatility = abs(np.random.normal(0, 0.01))
        high = current_price * (1 + volatility)
        low = current_price * (1 - volatility * 0.8)
        open_price = np.random.uniform(low, high)
        volume = np.random.lognormal(15, 1)  # Log-normal volume
        
        # Add basic features
        row = [open_price, high, low, current_price, volume]
        
        # Add technical indicators (simplified)
        if i >= 20:  # Need history for indicators
            recent_prices = [data[j][3] for j in range(max(0, i-20), i)]
            ma_5 = np.mean(recent_prices[-5:]) if len(recent_prices) >= 5 else current_price
            ma_20 = np.mean(recent_prices) if len(recent_prices) >= 20 else current_price
            
            # RSI simplified
            gains = [max(0, recent_prices[j] - recent_prices[j-1]) 
                    for j in range(1, len(recent_prices))]
            losses = [max(0, recent_prices[j-1] - recent_prices[j]) 
                     for j in range(1, len(recent_prices))]
            
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0.5
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0.5
            
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100
            
            # Price momentum
            momentum = current_price - recent_prices[0] if len(recent_prices) > 0 else 0
            
            # Bollinger band position
            if len(recent_prices) >= 20:
                bb_mean = np.mean(recent_prices)
                bb_std = np.std(recent_prices)
                bb_pos = (current_price - bb_mean) / (2 * bb_std) if bb_std > 0 else 0
            else:
                bb_pos = 0
            
            # Volume features
            if i >= 10:
                recent_volumes = [data[j][4] for j in range(max(0, i-10), i)]
                vol_ma = np.mean(recent_volumes)
                vol_ratio = volume / vol_ma if vol_ma > 0 else 1.0
            else:
                vol_ratio = 1.0
            
            # Add features to reach n_features
            additional_features = [
                ma_5, ma_20, rsi, momentum, bb_pos, vol_ratio,
                high/low, current_price/open_price, 
                np.log(volume) if volume > 0 else 0,
                volatility
            ]
            
            row.extend(additional_features[:n_features-5])  # Limit to n_features total
        else:
            # Fill with zeros/defaults for early samples
            row.extend([current_price] * (n_features - 5))
        
        data.append(row[:n_features])  # Ensure exactly n_features
    
    data = np.array(data, dtype=np.float32)
    
    print(f"Generated data shape: {data.shape}")
    print(f"Price range: {data[:, 3].min():.2f} - {data[:, 3].max():.2f}")
    print(f"No NaN values: {not np.any(np.isnan(data))}")
    
    return data


def save_model_and_artifacts(trainer, config, processor, output_dir):
    """Save model, processor, and training artifacts"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving artifacts to {output_path}")
    
    # Save final model
    model_path = output_path / "trained_model.pth"
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'model_config': config,
        'input_dim': trainer.model.input_dim if hasattr(trainer.model, 'input_dim') else None
    }, model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Save data processor
    if processor:
        processor_path = output_path / "data_processor.pkl"
        processor.save_scalers(str(processor_path))
        print(f"✅ Data processor saved to {processor_path}")
    
    # Save configuration
    config_path = output_path / "experiment_config.json"
    config.save(str(config_path))
    print(f"✅ Configuration saved to {config_path}")
    
    # Save training metrics if available
    if hasattr(trainer, 'metrics_tracker') and trainer.metrics_tracker.metrics:
        metrics_path = output_path / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump([
                {k: (v.isoformat() if hasattr(v, 'isoformat') else v) 
                 for k, v in metric.items()}
                for metric in trainer.metrics_tracker.metrics
            ], f, indent=2)
        print(f"✅ Training metrics saved to {metrics_path}")
    
    return output_path


def load_saved_model(model_path, config_path=None):
    """Load a saved model and configuration"""
    print(f"📂 Loading model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if config_path:
        from config import ExperimentConfig
        config = ExperimentConfig.load(config_path)
    else:
        config = checkpoint.get('model_config')
    
    # Recreate model
    from hf_trainer import TransformerTradingModel, HFTrainingConfig
    
    if hasattr(config, 'model'):
        # New config format
        hf_config = HFTrainingConfig(
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
            sequence_length=config.data.sequence_length,
            prediction_horizon=config.data.prediction_horizon
        )
    else:
        # Old config format
        hf_config = config
    
    input_dim = checkpoint.get('input_dim', 15)
    model = TransformerTradingModel(hf_config, input_dim=input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Model loaded successfully")
    return model, config


def run_quick_experiment():
    """Run a quick training experiment with small data"""
    
    print("🚀 Starting Quick Training Experiment")
    print("=" * 60)
    
    # Create test configuration
    config = create_config("quick_test")
    
    # Override for even smaller/faster test
    config.model.hidden_size = 64
    config.model.num_layers = 2
    config.model.num_heads = 4
    
    config.training.max_steps = 500
    config.training.batch_size = 4
    config.training.learning_rate = 1e-3
    config.training.warmup_steps = 50
    
    config.evaluation.eval_steps = 100
    config.evaluation.save_steps = 250  
    config.evaluation.logging_steps = 25
    
    config.data.sequence_length = 20
    config.data.prediction_horizon = 3
    
    # Set up output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.output.output_dir = f"hftraining/quick_test_output_{timestamp}"
    config.output.logging_dir = f"hftraining/quick_test_logs_{timestamp}"
    
    config.experiment_name = f"quick_test_{timestamp}"
    config.description = "Quick training test with synthetic data"
    
    print(f"📋 Configuration:")
    print(f"  Model: {config.model.hidden_size}d, {config.model.num_layers} layers")
    print(f"  Training: {config.training.max_steps} steps, batch={config.training.batch_size}")
    print(f"  Output: {config.output.output_dir}")
    
    # Generate small test data
    raw_data = create_small_test_data(length=1000, n_features=15)
    
    # Process data
    processor = StockDataProcessor(
        sequence_length=config.data.sequence_length,
        prediction_horizon=config.data.prediction_horizon
    )
    
    # Fit processor and transform data
    train_end = int(len(raw_data) * 0.8)
    processor.fit_scalers(raw_data[:train_end])
    processed_data = processor.transform(raw_data)
    
    # Split data
    train_data, val_data, _ = split_data(processed_data, 0.7, 0.2, 0.1)
    
    print(f"📊 Data splits: Train={len(train_data)}, Val={len(val_data)}")
    
    # Create datasets
    train_dataset = StockDataset(
        train_data,
        sequence_length=config.data.sequence_length,
        prediction_horizon=config.data.prediction_horizon
    )
    
    val_dataset = StockDataset(
        val_data,
        sequence_length=config.data.sequence_length,
        prediction_horizon=config.data.prediction_horizon
    ) if len(val_data) > config.data.sequence_length + config.data.prediction_horizon else None
    
    print(f"📈 Datasets: Train={len(train_dataset)}, Val={len(val_dataset) if val_dataset else 0}")
    
    # Setup environment
    device = setup_environment(config)
    
    # Create model
    model, hf_config = create_model(config, input_dim=15)
    
    # Create trainer with improved scheduler
    from train_hf import HFTrainer
    trainer = HFTrainer(
        model=model,
        config=hf_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Replace scheduler with improved version that doesn't get stuck at 0
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        trainer.optimizer,
        T_0=100,  # Restart every 100 steps
        T_mult=2,  # Double the period after each restart
        eta_min=1e-6  # Minimum learning rate to prevent getting stuck
    )
    
    print(f"\n🎯 Starting training...")
    
    # Train model
    try:
        trained_model = trainer.train()
        
        if trained_model is not None:
            print(f"\n🎉 Training completed successfully!")
            
            # Save model and artifacts
            output_path = save_model_and_artifacts(trainer, config, processor, config.output.output_dir)
            
            # Test model loading
            model_path = output_path / "trained_model.pth" 
            config_path = output_path / "experiment_config.json"
            
            print(f"\n🔄 Testing model loading...")
            loaded_model, loaded_config = load_saved_model(model_path, config_path)
            
            print(f"✅ Model loading test successful!")
            
            # Generate final report
            generate_experiment_report(trainer, config, output_path)
            
            return output_path, trainer
        else:
            print(f"❌ Training failed")
            return None, None
            
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def generate_experiment_report(trainer, config, output_dir):
    """Generate a comprehensive experiment report"""
    report_path = Path(output_dir) / "experiment_report.md"
    
    # Collect metrics
    if hasattr(trainer, 'metrics_tracker') and trainer.metrics_tracker.metrics:
        final_metrics = trainer.metrics_tracker.metrics[-1] if trainer.metrics_tracker.metrics else {}
        best_metrics = trainer.metrics_tracker.best_metrics
    else:
        final_metrics = {}
        best_metrics = {}
    
    # Calculate training time
    total_time = getattr(trainer, 'total_training_time', 0)
    
    report_content = f"""# Quick Training Experiment Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Experiment Configuration

- **Name:** {config.experiment_name}
- **Description:** {config.description}
- **Model:** {config.model.hidden_size}d, {config.model.num_layers} layers, {config.model.num_heads} heads
- **Training:** {config.training.max_steps} steps, batch size {config.training.batch_size}
- **Learning Rate:** {config.training.learning_rate}
- **Optimizer:** {config.training.optimizer}

## Training Results

- **Total Steps:** {trainer.global_step if hasattr(trainer, 'global_step') else 'Unknown'}
- **Training Time:** {total_time:.1f}s
- **Final Loss:** {final_metrics.get('loss', 'Unknown')}
- **Best Metrics:** {best_metrics}

## Model Information

- **Total Parameters:** {sum(p.numel() for p in trainer.model.parameters()):,}
- **Trainable Parameters:** {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad):,}
- **Model Size:** ~{sum(p.numel() for p in trainer.model.parameters()) * 4 / 1024 / 1024:.1f} MB

## Files Generated

- `trained_model.pth` - Complete model checkpoint
- `data_processor.pkl` - Data preprocessing pipeline
- `experiment_config.json` - Full experiment configuration
- `training_metrics.json` - Detailed training metrics
- `experiment_report.md` - This report

## Usage

```python
# Load the trained model
from hftraining.quick_test_runner import load_saved_model

model_path = "{output_dir}/trained_model.pth"
config_path = "{output_dir}/experiment_config.json"

model, config = load_saved_model(model_path, config_path)

# Use for inference
model.eval()
with torch.no_grad():
    outputs = model(input_sequences)
    actions = outputs['action_logits']
    prices = outputs['price_predictions']
```

## Next Steps

1. **Scale Up**: Use larger models and datasets for production
2. **Hyperparameter Tuning**: Experiment with different optimizers and learning rates
3. **Feature Engineering**: Add more sophisticated technical indicators
4. **Ensemble Methods**: Combine multiple models for better predictions
5. **Backtesting**: Test on historical market data

---
*Generated by Quick Test Runner*
"""

    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"📄 Experiment report saved to {report_path}")


def main():
    """Main entry point"""
    print("🧪 HF Training System - Quick Test Runner")
    print("=========================================")
    
    try:
        output_path, trainer = run_quick_experiment()
        
        if output_path:
            print(f"\n✅ Quick test completed successfully!")
            print(f"📁 Results saved to: {output_path}")
            print(f"\n📋 Summary:")
            print(f"   • Model trained and saved")
            print(f"   • Data processor saved")
            print(f"   • Configuration saved")
            print(f"   • Training metrics logged")
            print(f"   • Model loading tested")
            print(f"\n🚀 Ready for larger experiments!")
            return 0
        else:
            print(f"\n❌ Quick test failed")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())