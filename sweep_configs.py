#!/usr/bin/env python3
"""
Hyperparameter Sweep Configurations for All Models

Defines systematic sweep grids for Toto, Kronos, and Chronos2 to find
optimal configurations based on pct_mae.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import itertools


# =============================================================================
# TOTO SWEEP CONFIGS
# =============================================================================

TOTO_SWEEP_GRID = {
    # Core architecture (aligned with paper recommendations)
    "patch_size": [32, 64],  # Paper uses 32
    "stride": [None],  # None = same as patch_size
    "context_length": [512, 1024],  # Paper recommends 512+
    "prediction_length": [64, 128],

    # Model size
    "embed_dim": [768],  # Keep fixed for now
    "num_layers": [12],
    "num_heads": [12],

    # Optimization
    "learning_rate": [1e-4, 3e-4, 5e-4],
    "warmup_steps": [2000, 5000],
    "weight_decay": [0.01],
    "gradient_clip_val": [1.0],  # Fixed from 0.1

    # Training
    "batch_size": [4, 8],
    "grad_accum": [8, 16],
    "max_epochs": [50, 100],

    # Loss
    "loss_type": ["huber", "quantile"],
    "huber_delta": [0.01, 0.05],
}

TOTO_QUICK_SWEEP = {
    # Minimal sweep for quick testing
    "patch_size": [32],
    "context_length": [512],
    "learning_rate": [3e-4],
    "max_epochs": [30],
    "loss_type": ["huber"],
}


# =============================================================================
# KRONOS SWEEP CONFIGS
# =============================================================================

KRONOS_SWEEP_GRID = {
    # Core parameters
    "context_length": [256, 512, 1024],
    "prediction_length": [64, 128],

    # Model architecture
    "n_heads": [8, 16],
    "n_layers": [12, 16],
    "d_model": [512, 768],

    # Optimization
    "learning_rate": [1e-4, 3e-4, 5e-4],
    "warmup_steps": [1000, 2000],
    "weight_decay": [0.01, 0.05],
    "gradient_clip": [1.0, 5.0],

    # Training
    "batch_size": [16, 32],
    "epochs": [50, 100],

    # Loss
    "loss": ["huber", "mse"],
    "huber_delta": [0.01, 0.05],
}

KRONOS_QUICK_SWEEP = {
    "context_length": [512],
    "learning_rate": [3e-4],
    "epochs": [30],
    "loss": ["huber"],
}


# =============================================================================
# CHRONOS2 SWEEP CONFIGS
# =============================================================================

CHRONOS2_SWEEP_GRID = {
    # Context window
    "context_length": [512, 1024],
    "prediction_length": [64, 128],

    # Model parameters
    "model_size": ["small", "base", "large"],  # If using different model sizes

    # Fine-tuning parameters
    "learning_rate": [1e-5, 5e-5, 1e-4],  # Lower for fine-tuning
    "warmup_steps": [500, 1000],
    "weight_decay": [0.01],
    "gradient_clip": [1.0],

    # Training
    "batch_size": [8, 16],
    "epochs": [20, 50],

    # Fine-tuning strategy
    "freeze_backbone": [True, False],
    "lora_r": [8, 16, 32],  # If using LoRA
}

CHRONOS2_QUICK_SWEEP = {
    "context_length": [512],
    "learning_rate": [5e-5],
    "epochs": [20],
    "freeze_backbone": [True],
}


# =============================================================================
# SWEEP GENERATOR
# =============================================================================

def generate_sweep_configs(
    grid: Dict[str, List[Any]],
    max_configs: int = 100,
    random_sample: bool = False
) -> List[Dict[str, Any]]:
    """
    Generate all hyperparameter combinations from grid

    Args:
        grid: Dict of {param_name: [values]}
        max_configs: Maximum number of configs to generate
        random_sample: If True, randomly sample configs; else use grid search

    Returns:
        List of config dicts
    """
    import random

    # Generate all combinations
    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    all_combinations = list(itertools.product(*values))

    # Convert to list of dicts
    configs = []
    for combo in all_combinations:
        config = dict(zip(keys, combo))
        # Handle stride=None case for Toto
        if "stride" in config and config["stride"] is None:
            config["stride"] = config.get("patch_size", 32)
        configs.append(config)

    # Limit number of configs
    if len(configs) > max_configs:
        if random_sample:
            configs = random.sample(configs, max_configs)
        else:
            configs = configs[:max_configs]

    return configs


def get_priority_configs(model_name: str) -> List[Dict[str, Any]]:
    """
    Get priority configurations to try first based on research/best practices

    These are hand-picked configs that are likely to perform well based on:
    - Paper recommendations
    - Previous successful runs
    - Common best practices
    """
    if model_name == "toto":
        return [
            # Paper-aligned config
            {
                "patch_size": 32,
                "stride": 32,
                "context_length": 512,
                "prediction_length": 64,
                "learning_rate": 3e-4,
                "warmup_steps": 5000,
                "max_epochs": 100,
                "batch_size": 8,
                "grad_accum": 8,
                "loss_type": "quantile",
                "gradient_clip_val": 1.0,
            },
            # Longer context
            {
                "patch_size": 32,
                "stride": 32,
                "context_length": 1024,
                "prediction_length": 128,
                "learning_rate": 3e-4,
                "warmup_steps": 5000,
                "max_epochs": 100,
                "batch_size": 4,
                "grad_accum": 16,
                "loss_type": "quantile",
                "gradient_clip_val": 1.0,
            },
            # Lower LR
            {
                "patch_size": 32,
                "stride": 32,
                "context_length": 512,
                "prediction_length": 64,
                "learning_rate": 1e-4,
                "warmup_steps": 5000,
                "max_epochs": 100,
                "batch_size": 8,
                "grad_accum": 8,
                "loss_type": "huber",
                "huber_delta": 0.01,
                "gradient_clip_val": 1.0,
            },
        ]

    elif model_name == "kronos":
        return [
            # Balanced config
            {
                "context_length": 512,
                "prediction_length": 64,
                "learning_rate": 3e-4,
                "warmup_steps": 2000,
                "epochs": 100,
                "batch_size": 32,
                "loss": "huber",
                "huber_delta": 0.05,
            },
            # Larger context
            {
                "context_length": 1024,
                "prediction_length": 128,
                "learning_rate": 1e-4,
                "warmup_steps": 2000,
                "epochs": 100,
                "batch_size": 16,
                "loss": "huber",
                "huber_delta": 0.01,
            },
        ]

    elif model_name == "chronos2":
        return [
            # Fine-tuning with frozen backbone
            {
                "context_length": 512,
                "prediction_length": 64,
                "learning_rate": 5e-5,
                "warmup_steps": 1000,
                "epochs": 50,
                "batch_size": 16,
                "freeze_backbone": True,
                "lora_r": 16,
            },
            # Full fine-tuning
            {
                "context_length": 512,
                "prediction_length": 64,
                "learning_rate": 1e-4,
                "warmup_steps": 1000,
                "epochs": 50,
                "batch_size": 8,
                "freeze_backbone": False,
            },
        ]

    return []


# =============================================================================
# SWEEP EXECUTION HELPERS
# =============================================================================

def format_config_for_cli(config: Dict[str, Any], model_name: str) -> List[str]:
    """Convert config dict to CLI arguments"""
    args = []

    if model_name == "toto":
        # Map to toto CLI args
        arg_mapping = {
            "learning_rate": "--lr",
            "max_epochs": "--max-epochs",
            "warmup_steps": "--warmup-steps",
            "batch_size": "--device-bs",
            "grad_accum": "--grad-accum",
            "context_length": "--context-length",
            "patch_size": "--patch-size",
            "stride": "--stride",
        }
    elif model_name == "kronos":
        arg_mapping = {
            "learning_rate": "--lr",
            "epochs": "--epochs",
            "context_length": "--context-length",
            "batch_size": "--batch-size",
        }
    elif model_name == "chronos2":
        arg_mapping = {
            "learning_rate": "--lr",
            "epochs": "--epochs",
            "context_length": "--context-length",
            "batch_size": "--batch-size",
            "freeze_backbone": "--freeze-backbone",
        }
    else:
        arg_mapping = {}

    for key, value in config.items():
        cli_arg = arg_mapping.get(key)
        if cli_arg:
            if isinstance(value, bool):
                if value:
                    args.append(cli_arg)
            else:
                args.extend([cli_arg, str(value)])

    return args


def main():
    """Example usage"""
    print("=" * 80)
    print("HYPERPARAMETER SWEEP CONFIGURATIONS")
    print("=" * 80)

    # Toto sweeps
    print("\n📊 TOTO PRIORITY CONFIGS:")
    for i, config in enumerate(get_priority_configs("toto"), 1):
        print(f"\n  Config {i}:")
        for k, v in config.items():
            print(f"    {k}: {v}")

    # Generate full grid
    print("\n📊 TOTO FULL GRID:")
    toto_configs = generate_sweep_configs(TOTO_SWEEP_GRID, max_configs=20)
    print(f"  Generated {len(toto_configs)} configurations")

    # Kronos
    print("\n📊 KRONOS PRIORITY CONFIGS:")
    for i, config in enumerate(get_priority_configs("kronos"), 1):
        print(f"\n  Config {i}:")
        for k, v in config.items():
            print(f"    {k}: {v}")

    # Chronos2
    print("\n📊 CHRONOS2 PRIORITY CONFIGS:")
    for i, config in enumerate(get_priority_configs("chronos2"), 1):
        print(f"\n  Config {i}:")
        for k, v in config.items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
