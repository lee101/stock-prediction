#!/usr/bin/env python3
"""
Launch script for HuggingFace-style training with enhanced logging
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from config import create_config
from run_training import run_training


def main():
    """Main launch function with enhanced options"""
    
    parser = argparse.ArgumentParser(
        description="🚀 Launch HuggingFace-style Stock Prediction Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run
  python launch_training.py --config quick_test --steps 100
  
  # Production training with custom name
  python launch_training.py --config production --name "my_production_run"
  
  # Research experiment with specific optimizer
  python launch_training.py --config research --optimizer gpro --lr 1e-4
  
  # Custom configuration
  python launch_training.py --config_file my_config.json
        """
    )
    
    # Configuration options
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="quick_test",
        choices=["default", "quick_test", "production", "research"],
        help="Predefined configuration to use"
    )
    
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to custom configuration file"
    )
    
    # Experiment settings
    parser.add_argument(
        "--name", "-n",
        type=str,
        help="Experiment name (default: auto-generated)"
    )
    
    parser.add_argument(
        "--description",
        type=str,
        help="Experiment description"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="hftraining/output",
        help="Output directory for model checkpoints"
    )
    
    parser.add_argument(
        "--log_dir", 
        type=str,
        default="hftraining/logs",
        help="Directory for logs and TensorBoard"
    )
    
    # Training parameters
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["gpro", "lion", "adamw", "adafactor", "lamb", "sophia", "adan"],
        help="Optimizer to use"
    )
    
    parser.add_argument(
        "--lr", "--learning_rate",
        type=float,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--steps", "--max_steps",
        type=int,
        help="Maximum training steps"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs"
    )
    
    # Model parameters
    parser.add_argument(
        "--hidden_size",
        type=int,
        help="Hidden size of the model"
    )
    
    parser.add_argument(
        "--num_layers",
        type=int,
        help="Number of transformer layers"
    )
    
    parser.add_argument(
        "--num_heads",
        type=int,
        help="Number of attention heads"
    )
    
    # Data parameters
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Stock symbols to use for training (e.g., AAPL GOOGL MSFT)"
    )
    
    parser.add_argument(
        "--sequence_length",
        type=int,
        help="Input sequence length"
    )
    
    parser.add_argument(
        "--prediction_horizon",
        type=int,
        help="Number of steps to predict"
    )
    
    # System settings
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training"
    )
    
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Enable mixed precision training"
    )
    
    parser.add_argument(
        "--no_mixed_precision",
        action="store_true",
        help="Disable mixed precision training"
    )
    
    # Debug and testing
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (short training)"
    )
    
    parser.add_argument(
        "--test_pipeline",
        action="store_true",
        help="Run pipeline test instead of training"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Logging verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal logging"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Load or create configuration
    if args.config_file:
        print(f"📋 Loading configuration from: {args.config_file}")
        from config import ExperimentConfig
        config = ExperimentConfig.load(args.config_file)
    else:
        print(f"📋 Using '{args.config}' configuration")
        config = create_config(args.config)
    
    # Apply command line overrides
    apply_overrides(config, args)
    
    # Set debug mode
    if args.debug:
        print("🐛 Debug mode enabled")
        config.training.max_steps = 50
        config.evaluation.eval_steps = 15
        config.evaluation.save_steps = 25
        config.evaluation.logging_steps = 5
        config.system.debug_mode = True
    
    # Run pipeline test
    if args.test_pipeline:
        print("🧪 Running pipeline test...")
        from test_pipeline import test_pipeline
        return 0 if test_pipeline() else 1
    
    # Display configuration summary
    print_config_summary(config)
    
    # Confirm before starting (unless in debug mode)
    if not args.debug and not args.quiet:
        response = input("\n🚀 Start training? [Y/n]: ").strip().lower()
        if response in ['n', 'no']:
            print("❌ Training cancelled")
            return 0
    
    # Run training
    print("\n" + "="*80)
    model, trainer = run_training(config)
    
    if model is not None:
        print("\n✅ Training completed successfully!")
        return 0
    else:
        print("\n❌ Training failed or was interrupted")
        return 1


def print_banner():
    """Print startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                      🤗 HuggingFace-Style Stock Training                     ║
    ║                         Modern ML for Financial Prediction                   ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def apply_overrides(config, args):
    """Apply command line argument overrides to config"""
    
    # Experiment settings
    if args.name:
        config.experiment_name = args.name
    if args.description:
        config.description = args.description
    if args.output_dir:
        config.output.output_dir = args.output_dir
    if args.log_dir:
        config.output.logging_dir = args.log_dir
    
    # Training parameters
    if args.optimizer:
        config.training.optimizer = args.optimizer
    if args.lr:
        config.training.learning_rate = args.lr
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.steps:
        config.training.max_steps = args.steps
    if args.epochs:
        config.training.num_epochs = args.epochs
    
    # Model parameters
    if args.hidden_size:
        config.model.hidden_size = args.hidden_size
    if args.num_layers:
        config.model.num_layers = args.num_layers
    if args.num_heads:
        config.model.num_heads = args.num_heads
    
    # Data parameters
    if args.symbols:
        config.data.symbols = args.symbols
    if args.sequence_length:
        config.data.sequence_length = args.sequence_length
    if args.prediction_horizon:
        config.data.prediction_horizon = args.prediction_horizon
    
    # System settings
    if args.device:
        config.system.device = args.device
    if args.mixed_precision:
        config.training.use_mixed_precision = True
    if args.no_mixed_precision:
        config.training.use_mixed_precision = False
    if args.resume:
        config.output.resume_from_checkpoint = args.resume
    if args.seed:
        config.system.seed = args.seed


def print_config_summary(config):
    """Print configuration summary"""
    print(f"\n📊 EXPERIMENT CONFIGURATION")
    print(f"{'─' * 50}")
    print(f"🏷️  Name: {config.experiment_name}")
    print(f"📝 Description: {config.description}")
    print(f"🎯 Optimizer: {config.training.optimizer}")
    print(f"📈 Learning Rate: {config.training.learning_rate}")
    print(f"🔢 Batch Size: {config.training.batch_size}")
    print(f"⏭️  Max Steps: {config.training.max_steps:,}")
    print(f"🏗️  Model: {config.model.hidden_size}d, {config.model.num_layers} layers, {config.model.num_heads} heads")
    print(f"📊 Data: {config.data.sequence_length} seq len, {len(config.data.symbols)} symbols")
    print(f"💾 Output: {config.output.output_dir}")
    print(f"📋 Logs: {config.output.logging_dir}")


if __name__ == "__main__":
    exit(main())