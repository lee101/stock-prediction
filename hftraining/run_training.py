#!/usr/bin/env python3
"""
Main training runner script
"""

import argparse
import sys
import os
from pathlib import Path
import random
from datetime import datetime
from typing import Optional

try:  # Prefer injected heavy dependencies when available.
    from .injection import get_numpy, get_torch
except Exception:  # pragma: no cover - script execution fallback
    try:
        from injection import get_numpy, get_torch  # type: ignore
    except Exception:  # pragma: no cover - direct imports as last resort
        def get_torch():
            import torch as _torch  # type: ignore

            return _torch

        def get_numpy():
            import numpy as _np  # type: ignore

            return _np

torch = get_torch()
np = get_numpy()

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
# Add parent directory to path
sys.path.append(os.path.dirname(current_dir))

from config import create_config, ExperimentConfig, TrainingConfig
from data_utils import load_training_data, StockDataProcessor, create_sequences, split_data
from train_hf import StockDataset, HFTrainer
from src.torch_backend import configure_tf32_backends, maybe_set_float32_precision
from src.gpu_utils import cli_flag_was_provided, detect_total_vram_bytes, recommend_batch_size
from hf_trainer import TransformerTradingModel
from modern_optimizers import get_optimizer
from toto_features import TotoOptions


HF_BATCH_THRESHOLDS = [(8, 8), (16, 16), (24, 24), (40, 32), (64, 48), (80, 64)]


def set_seed(seed: int, deterministic: bool = True):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    try:
        torch.use_deterministic_algorithms(deterministic)
    except Exception:
        pass


def setup_environment(config: ExperimentConfig):
    """Setup training environment"""

    # Adopt nanochat fast-allocation trick if user hasn't set a custom config
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    # Respect distributed launchers that set LOCAL_RANK
    if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
        try:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        except Exception:
            pass
    
    # Set seed
    set_seed(config.system.seed, deterministic=config.system.deterministic)
    
    # Resolve dirs relative to this file to avoid hftraining/hftraining nesting
    base_dir = Path(__file__).parent
    def _resolve_dir(path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        parts = p.parts
        if parts and parts[0].lower() == 'hftraining':
            p = Path(*parts[1:]) if len(parts) > 1 else Path('.')
        return base_dir / p

    # Normalize paths back into config so downstream consumers use resolved paths
    config.output.output_dir = str(_resolve_dir(config.output.output_dir))
    config.output.logging_dir = str(_resolve_dir(config.output.logging_dir))
    config.output.cache_dir = str(_resolve_dir(config.output.cache_dir))

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
        # Optional TF32 for faster matmul on Ampere+
        allow_tf32 = getattr(config.system, "allow_tf32", True)
        if allow_tf32:
            try:
                state = configure_tf32_backends(torch)
                surface = "new" if state["new_api"] else "legacy"
                print(f"TF32 enabled via {surface} precision controls")
            except Exception as exc:
                print(f"Failed to enable TF32 optimisations: {exc}")
        else:
            try:
                matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
                if matmul is not None:
                    if hasattr(matmul, "fp32_precision"):
                        matmul.fp32_precision = "ieee"
                    elif hasattr(matmul, "allow_tf32"):
                        matmul.allow_tf32 = False
                cudnn_backend = getattr(torch.backends, "cudnn", None)
                if cudnn_backend is not None:
                    conv = getattr(cudnn_backend, "conv", None)
                    if conv is not None and hasattr(conv, "fp32_precision"):
                        conv.fp32_precision = "ieee"
                    elif hasattr(cudnn_backend, "allow_tf32"):
                        cudnn_backend.allow_tf32 = False
                print("TF32 fast paths disabled per configuration")
            except Exception:
                pass
        if torch.cuda.is_available():
            maybe_set_float32_precision(torch, mode="high")
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        pass
    
    return device


def maybe_autotune_batch_size(config: ExperimentConfig, device: str) -> None:
    """Adjust batch size heuristically based on detected GPU memory."""

    if not getattr(config.system, "auto_batch_size", True):
        return

    if device == "cpu":
        return

    query_device: Optional[str] = None
    if device == "auto":
        if not torch.cuda.is_available():
            return
    elif device.startswith("cuda"):
        query_device = device
    else:
        return

    total_vram = detect_total_vram_bytes(query_device)
    if total_vram is None:
        return

    allow_increase = bool(getattr(config.system, "auto_batch_allow_increase", True))
    if cli_flag_was_provided("--batch_size") or cli_flag_was_provided("--batch-size"):
        allow_increase = False

    default_batch = TrainingConfig().batch_size
    if config.training.batch_size > default_batch:
        # Only cap when overrides request a larger batch than defaults.
        allow_increase = False

    recommended = recommend_batch_size(
        total_vram,
        config.training.batch_size,
        HF_BATCH_THRESHOLDS,
        allow_increase=allow_increase,
    )

    max_auto = getattr(config.training, "max_auto_batch_size", None)
    if max_auto is not None:
        recommended = min(recommended, max_auto)

    if recommended != config.training.batch_size:
        gb = total_vram / (1024 ** 3)
        print(
            f"[hftraining] adjusting batch size from {config.training.batch_size} to {recommended}"
            f" for detected {gb:.1f} GiB VRAM"
        )
        config.training.batch_size = recommended


def load_and_process_data(config: ExperimentConfig):
    """Load and process training data"""
    
    print("Loading training data...")

    toto_options = TotoOptions(
        use_toto=config.data.use_toto_forecasts,
        horizon=config.data.toto_horizon,
        context_length=config.data.sequence_length,
        num_samples=config.data.toto_num_samples,
        toto_model_id=config.data.toto_model_id,
        toto_device=config.data.toto_device,
    )
    
    # Load raw training data
    raw_train_data = load_training_data(
        data_dir=config.data.data_dir,
        symbols=config.data.symbols,
        start_date=config.data.start_date,
        use_toto_forecasts=config.data.use_toto_forecasts,
        toto_options=toto_options,
        sequence_length=config.data.sequence_length,
        prediction_horizon=config.data.prediction_horizon,
    )
    
    print(f"Training raw data shape: {raw_train_data.shape}")
    
    # Initialize data processor
    processor = StockDataProcessor(
        sequence_length=config.data.sequence_length,
        prediction_horizon=config.data.prediction_horizon,
        use_toto_forecasts=config.data.use_toto_forecasts,
        toto_options=toto_options,
    )
    
    # Fit scalers on training data
    total_train_len = len(raw_train_data)
    min_required = config.data.sequence_length + config.data.prediction_horizon
    train_cutoff = int(total_train_len * config.data.train_ratio)
    if train_cutoff <= 0:
        train_cutoff = total_train_len
    if total_train_len >= min_required:
        train_cutoff = max(train_cutoff, min_required)
    else:
        train_cutoff = total_train_len
    train_cutoff = min(train_cutoff, total_train_len)
    processor.fit_scalers(raw_train_data[:train_cutoff])
    
    # Transform data
    normalized_train_data = processor.transform(raw_train_data)
    
    # Save processor
    processor_path = Path(config.output.output_dir) / "data_processor.pkl"
    processor.save_scalers(str(processor_path))
    print(f"Data processor saved to: {processor_path}")
    
    val_data = None
    if config.data.validation_data_dir:
        raw_val_data = load_training_data(
            data_dir=config.data.validation_data_dir,
            symbols=config.data.symbols,
            start_date=config.data.start_date,
            use_toto_forecasts=config.data.use_toto_forecasts,
            toto_options=toto_options,
            sequence_length=config.data.sequence_length,
            prediction_horizon=config.data.prediction_horizon,
        )
        print(f"Validation raw data shape: {raw_val_data.shape}")
        val_data = processor.transform(raw_val_data)
        train_data = normalized_train_data[:train_cutoff]
        print(f"Data splits - Train: {len(train_data)}, External Val: {len(val_data)}")
    else:
        train_data, val_data, _ = split_data(
            normalized_train_data,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            test_ratio=config.data.test_ratio
        )
        print(f"Data splits - Train: {len(train_data)}, Val: {len(val_data)} (split)")
    
    # Create sequences
    print("Creating sequences...")
    
    train_sequences, train_targets, train_actions = create_sequences(
        train_data,
        config.data.sequence_length,
        config.data.prediction_horizon
    )
    
    print(f"Training sequences: {train_sequences.shape}")
    val_dataset = None
    if val_data is not None and len(val_data) > config.data.sequence_length + config.data.prediction_horizon:
        val_sequences, val_targets, val_actions = create_sequences(
            val_data,
            config.data.sequence_length,
            config.data.prediction_horizon
        )
        print(f"Validation sequences: {val_sequences.shape}")
    else:
        print("Validation set too small to create sequences; skipping validation dataset.")
        val_data = None
    
    # Create datasets
    train_dataset = StockDataset(
        train_data,
        sequence_length=config.data.sequence_length,
        prediction_horizon=config.data.prediction_horizon,
        processor=processor,
    )

    if val_data is not None:
        val_dataset = StockDataset(
            val_data,
            sequence_length=config.data.sequence_length,
            prediction_horizon=config.data.prediction_horizon,
            processor=processor,
        )
    
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
        muon_momentum=config.training.muon_momentum,
        muon_nesterov=config.training.muon_nesterov,
        muon_ns_steps=config.training.muon_ns_steps,
        muon_adamw_lr=config.training.muon_adamw_lr,
        
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
        early_stopping_threshold=config.training.early_stopping_threshold,
        profit_loss_weight=config.training.profit_loss_weight,
        transaction_cost_bps=config.training.transaction_cost_bps,
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
    maybe_autotune_batch_size(config, device)

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
