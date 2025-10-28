#!/usr/bin/env python3
"""
Configuration management for HuggingFace-style training
"""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 16
    intermediate_size: int = 2048
    dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    activation: str = "gelu"
    use_bias: bool = True
    tie_word_embeddings: bool = False


@dataclass
class DataConfig:
    """Data processing configuration"""
    sequence_length: int = 60
    prediction_horizon: int = 5
    overlap_ratio: float = 0.5
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Data sources
    symbols: List[str] = field(default_factory=lambda: ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"])
    start_date: str = "2015-01-01"
    end_date: Optional[str] = None
    data_dir: str = "trainingdata"
    validation_data_dir: Optional[str] = None
    
    # Preprocessing
    use_technical_indicators: bool = True
    normalize_data: bool = True
    augment_data: bool = True
    noise_factor: float = 0.01
    scaling_factor: float = 0.05
    augmentation_multiplier: int = 0  # extra augmented copies of training split

    # Datadog Toto integration
    use_toto_forecasts: bool = True
    toto_model_id: str = "Datadog/Toto-Open-Base-1.0"
    toto_device: str = "cuda"
    toto_horizon: int = 8
    toto_num_samples: int = 2048


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Optimization
    learning_rate: float = 1e-4
    optimizer: str = "gpro"  # gpro, lion, adamw, adafactor, lamb, sophia, adan
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    use_fused_optimizer: bool = True
    # Stability
    use_adaptive_grad_clip: bool = False
    agc_clip_factor: float = 0.01
    agc_eps: float = 1e-3
    skip_non_finite_grads: bool = True
    
    # Learning rate scheduling
    warmup_steps: int = 1000
    scheduler_type: str = "cosine"  # linear, cosine, polynomial
    num_cycles: float = 0.5
    
    # Optimizer-specific parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_adamw_lr: Optional[float] = None
    
    # Training dynamics
    num_epochs: int = 50
    max_steps: Optional[int] = None
    batch_size: int = 32
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    max_tokens_per_batch: int = 0
    length_bucketing: List[int] = field(default_factory=lambda: [60])
    horizon_bucketing: List[int] = field(default_factory=lambda: [5])
    window_stride: int = 1
    pack_windows: bool = True
    bucket_warmup_steps: int = 0
    
    # Mixed precision
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True
    precision: str = "bf16"
    
    # Regularization
    dropout_rate: float = 0.1
    label_smoothing: float = 0.0
    profit_loss_weight: float = 0.2
    transaction_cost_bps: float = 10.0  # 1 bps = 0.0001
    profit_curriculum_warmup_steps: int = 0
    profit_curriculum_steps: int = 0
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.0001
    # Metrics (moved from training config)
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    evaluation_strategy: str = "steps"  # no, steps, epoch
    eval_steps: int = 500
    eval_accumulation_steps: Optional[int] = None
    
    # Metrics
    compute_metrics: bool = True
    prediction_loss_only: bool = False
    
    # Logging
    logging_strategy: str = "steps"  # no, steps, epoch
    logging_steps: int = 100
    logging_first_step: bool = True
    
    # Saving
    save_strategy: str = "steps"  # no, steps, epoch
    save_steps: int = 1000
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    
    # Best model selection
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False


@dataclass
class SystemConfig:
    """System and hardware configuration"""
    # Device settings
    device: str = "auto"  # auto, cpu, cuda, mps
    use_data_parallel: bool = True
    use_distributed: bool = False
    
    # Memory management
    dataloader_drop_last: bool = False
    remove_unused_columns: bool = True
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    # Debugging
    debug_mode: bool = False
    profile_training: bool = False
    allow_tf32: bool = True


@dataclass
class OutputConfig:
    """Output and logging configuration"""
    # Directories
    output_dir: str = "hftraining/output"
    logging_dir: str = "hftraining/logs"
    cache_dir: str = "hftraining/cache"
    
    # Reporting
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Checkpointing
    resume_from_checkpoint: Optional[str] = None
    ignore_data_skip: bool = False
    
    # Prediction outputs
    prediction_loss_only: bool = False
    include_inputs_for_metrics: bool = False


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Experiment metadata
    experiment_name: str = "stock_prediction_experiment"
    description: str = "HuggingFace-style stock prediction training"
    version: str = "1.0.0"
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        config_dict = asdict(self)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Convert nested dictionaries back to dataclasses
        model_config = ModelConfig(**config_dict['model'])
        data_config = DataConfig(**config_dict['data'])
        training_config = TrainingConfig(**config_dict['training'])
        evaluation_config = EvaluationConfig(**config_dict['evaluation'])
        system_config = SystemConfig(**config_dict['system'])
        output_config = OutputConfig(**config_dict['output'])
        
        # Remove nested configs from main dict
        for key in ['model', 'data', 'training', 'evaluation', 'system', 'output']:
            config_dict.pop(key, None)
        
        return cls(
            model=model_config,
            data=data_config,
            training=training_config,
            evaluation=evaluation_config,
            system=system_config,
            output=output_config,
            **config_dict
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def update(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), dict):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)
            else:
                # Try to update nested configs
                for config_name in ['model', 'data', 'training', 'evaluation', 'system', 'output']:
                    config = getattr(self, config_name)
                    if hasattr(config, key):
                        setattr(config, key, value)
                        break


# Predefined configurations
def get_default_config() -> ExperimentConfig:
    """Get default configuration"""
    return ExperimentConfig()


def get_quick_test_config() -> ExperimentConfig:
    """Get configuration for quick testing"""
    config = ExperimentConfig()
    
    # Smaller model for quick testing
    config.model.hidden_size = 128
    config.model.num_layers = 4
    config.model.num_heads = 8
    
    # Less data and shorter training
    config.data.sequence_length = 30
    config.data.symbols = ["AAPL"]
    
    config.training.max_steps = 1000
    config.training.batch_size = 8
    config.training.warmup_steps = 100
    
    config.evaluation.eval_steps = 100
    config.evaluation.save_steps = 200
    config.evaluation.logging_steps = 20
    
    config.experiment_name = "quick_test"
    
    return config


def get_production_config() -> ExperimentConfig:
    """Get configuration for production training"""
    config = ExperimentConfig()
    
    # Larger model for production
    config.model.hidden_size = 768
    config.model.num_layers = 12
    config.model.num_heads = 12
    
    # More data and longer training
    config.data.sequence_length = 120
    config.data.symbols = [
        "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", 
        "META", "NVDA", "NFLX", "CRM", "ORCL"
    ]
    
    config.training.num_epochs = 100
    config.training.max_steps = 50000
    config.training.batch_size = 16
    config.training.learning_rate = 5e-5
    config.training.warmup_steps = 2000
    
    config.evaluation.eval_steps = 1000
    config.evaluation.save_steps = 2000
    config.evaluation.logging_steps = 100
    
    config.experiment_name = "production_training"
    
    return config


def get_research_config() -> ExperimentConfig:
    """Get configuration for research experiments"""
    config = ExperimentConfig()
    
    # Research-focused settings
    config.model.hidden_size = 512
    config.model.num_layers = 8
    
    config.data.sequence_length = 90
    config.data.prediction_horizon = 10
    config.data.augment_data = True
    
    config.training.optimizer = "sophia"  # Try different optimizers
    config.training.learning_rate = 1e-4
    config.training.gradient_checkpointing = True
    
    config.system.profile_training = True
    # Use TensorBoard only
    config.output.report_to = ["tensorboard"]
    
    config.experiment_name = "research_experiment"
    
    return config


# Configuration factory
def create_config(config_type: str = "default", **kwargs) -> ExperimentConfig:
    """
    Create configuration based on type
    
    Args:
        config_type: Type of configuration (default, quick_test, production, research)
        **kwargs: Additional parameters to override
        
    Returns:
        ExperimentConfig instance
    """
    
    config_factories = {
        "default": get_default_config,
        "quick_test": get_quick_test_config,
        "production": get_production_config,
        "research": get_research_config,
    }
    
    if config_type not in config_factories:
        raise ValueError(f"Unknown config type: {config_type}. Available: {list(config_factories.keys())}")
    
    config = config_factories[config_type]()
    
    # Apply any overrides
    if kwargs:
        config.update(**kwargs)
    
    return config
