#!/usr/bin/env python3
"""
Main training runner script
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import random
import numpy as np
from datetime import datetime

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
# Add parent directory to path
sys.path.append(os.path.dirname(current_dir))

from config import create_config, ExperimentConfig
from data_utils import load_training_data, StockDataProcessor, create_sequences, split_data
from train_hf import StockDataset, HFTrainer
from hf_trainer import TransformerTradingModel
from modern_optimizers import get_optimizer


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_environment(config: ExperimentConfig):
    """Setup training environment"""
    
    # Set seed
    set_seed(config.system.seed)
    
    # Create output directories
    Path(config.output.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output.logging_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output.cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = Path(config.output.output_dir) / "config.json"
    config.save(str(config_path))
    print(f"Configuration saved to: {config_path}")
    
    # Device setup
    if config.system.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = config.system.device
    
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    
    return device


def load_and_process_data(config: ExperimentConfig):
    """Load and process training data"""
    
    print("Loading training data...")
    
    # Load raw data
    raw_data = load_training_data(
        data_dir=config.data.data_dir,
        symbols=config.data.symbols,
        start_date=config.data.start_date
    )
    
    print(f"Raw data shape: {raw_data.shape}")
    
    # Initialize data processor
    processor = StockDataProcessor(
        sequence_length=config.data.sequence_length,
        prediction_horizon=config.data.prediction_horizon
    )
    
    # Fit scalers on training data
    train_end = int(len(raw_data) * config.data.train_ratio)
    processor.fit_scalers(raw_data[:train_end])
    
    # Transform data
    normalized_data = processor.transform(raw_data)
    
    # Save processor
    processor_path = Path(config.output.output_dir) / "data_processor.pkl"
    processor.save_scalers(str(processor_path))
    print(f"Data processor saved to: {processor_path}")
    
    # Split data
    train_data, val_data, test_data = split_data(
        normalized_data,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio
    )
    
    print(f"Data splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Create sequences
    print("Creating sequences...")
    
    train_sequences, train_targets, train_actions = create_sequences(
        train_data,
        config.data.sequence_length,
        config.data.prediction_horizon
    )
    
    val_sequences, val_targets, val_actions = create_sequences(
        val_data,
        config.data.sequence_length,
        config.data.prediction_horizon
    )
    
    print(f"Training sequences: {train_sequences.shape}")
    print(f"Validation sequences: {val_sequences.shape}")
    
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
    
    return train_dataset, val_dataset, processor


def create_model(config: ExperimentConfig, input_dim: int):
    """Create and initialize model"""
    
    print("Creating model...")
    
    # Convert config to HFTrainingConfig format
    from hf_trainer import HFTrainingConfig
    
    hf_config = HFTrainingConfig(
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.max_steps,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_grad_norm=config.training.max_grad_norm,
        
        optimizer_name=config.training.optimizer,
        weight_decay=config.training.weight_decay,
        adam_beta1=config.training.adam_beta1,
        adam_beta2=config.training.adam_beta2,
        adam_epsilon=config.training.adam_epsilon,
        
        batch_size=config.training.batch_size,
        eval_steps=config.evaluation.eval_steps,
        save_steps=config.evaluation.save_steps,
        logging_steps=config.evaluation.logging_steps,
        
        sequence_length=config.data.sequence_length,
        prediction_horizon=config.data.prediction_horizon,
        
        use_mixed_precision=config.training.use_mixed_precision,
        use_gradient_checkpointing=config.training.gradient_checkpointing,
        use_data_parallel=config.system.use_data_parallel,
        
        output_dir=config.output.output_dir,
        logging_dir=config.output.logging_dir,
        cache_dir=config.output.cache_dir,
        
        evaluation_strategy=config.evaluation.evaluation_strategy,
        metric_for_best_model=config.evaluation.metric_for_best_model,
        greater_is_better=config.evaluation.greater_is_better,
        load_best_model_at_end=config.evaluation.load_best_model_at_end,
        
        early_stopping_patience=config.training.early_stopping_patience,
        early_stopping_threshold=config.training.early_stopping_threshold
    )
    
    model = TransformerTradingModel(hf_config, input_dim=input_dim)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    return model, hf_config


def run_training(config: ExperimentConfig):
    """Run the complete training pipeline"""
    
    print(f"Starting training experiment: {config.experiment_name}")
    print(f"Description: {config.description}")
    print("=" * 80)
    
    # Setup environment
    device = setup_environment(config)
    
    # Load and process data
    train_dataset, val_dataset, processor = load_and_process_data(config)
    
    # Create model
    if hasattr(processor, 'feature_names') and processor.feature_names:
        input_dim = len(processor.feature_names)
    else:
        # Get input dimension from a sample
        sample = train_dataset[0]
        input_dim = sample['input_ids'].shape[1]
    
    print(f"Input dimension: {input_dim}")
    model, hf_config = create_model(config, input_dim)
    
    # Create trainer
    print("Creating trainer...")
    trainer = HFTrainer(
        model=model,
        config=hf_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Start training
    print("Starting training...")
    print("=" * 80)
    
    start_time = datetime.now()
    
    try:
        trained_model = trainer.train()
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        print("=" * 80)
        print(f"Training completed successfully!")
        print(f"Training time: {training_time}")
        print(f"Final model saved to: {config.output.output_dir}")
        
        return trained_model, trainer
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save checkpoint
        trainer.save_checkpoint()
        return None, trainer
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None, trainer


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="HuggingFace-style Stock Prediction Training")
    
    parser.add_argument(
        "--config_type",
        type=str,
        default="default",
        choices=["default", "quick_test", "production", "research"],
        help="Type of configuration to use"
    )
    
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to custom configuration file"
    )
    
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name for this experiment"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_file:
        print(f"Loading configuration from: {args.config_file}")
        config = ExperimentConfig.load(args.config_file)
    else:
        print(f"Using {args.config_type} configuration")
        config = create_config(args.config_type)
    
    # Apply command line overrides
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    
    if args.output_dir:
        config.output.output_dir = args.output_dir
    
    if args.resume_from_checkpoint:
        config.output.resume_from_checkpoint = args.resume_from_checkpoint
    
    if args.debug:
        config.system.debug_mode = True
        config.training.max_steps = 100
        config.evaluation.eval_steps = 20
        config.evaluation.save_steps = 50
        config.evaluation.logging_steps = 10
    
    # Run training
    model, trainer = run_training(config)
    
    if model is not None:
        print("Training completed successfully!")
        return 0
    else:
        print("Training failed or was interrupted")
        return 1


if __name__ == "__main__":
    exit(main())