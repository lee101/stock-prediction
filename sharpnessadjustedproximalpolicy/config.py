"""Experiment configs for sharpness-adjusted proximal policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SAPConfig:
    """Sharpness-adjusted proximal policy config, extends binanceneural TrainingConfig."""

    # SAM mode: "periodic" (fast), "full_sam", "looksam", "none" (baseline)
    sam_mode: str = "periodic"

    # Perturbation radius
    rho: float = 0.05

    # How often to probe sharpness (periodic mode)
    probe_every: int = 10

    # EMA smoothing for sharpness signal
    ema_beta: float = 0.9

    # Target sharpness (scale=1.0 at this value)
    target_sharpness: float = 1.0

    # LR scale bounds
    min_lr_scale: float = 0.3
    max_lr_scale: float = 3.0
    scale_mode: str = "linear"  # "linear" or "log"

    # Full SAM specific
    adaptive_sam: bool = True  # weight-normalized perturbation
    rho_min: float = 0.01
    rho_max: float = 0.2

    # LookSAM specific
    looksam_every: int = 5
    looksam_alpha: float = 0.5

    # Early stopping: stop if val score degrades N epochs in a row
    early_stop_patience: int = 8
    # Min epochs before early stopping kicks in
    early_stop_min_epochs: int = 3

    # Sharpness logging
    log_sharpness: bool = True


# Experiment sweep configurations
EXPERIMENTS = [
    # Baselines
    {"name": "baseline_adamw", "sam_mode": "none"},
    {"name": "baseline_cosine", "sam_mode": "none", "lr_schedule": "cosine"},

    # Periodic SAM (fast -- 10% overhead)
    {"name": "periodic_rho005", "sam_mode": "periodic", "rho": 0.05, "probe_every": 10},
    {"name": "periodic_rho01", "sam_mode": "periodic", "rho": 0.1, "probe_every": 10},
    {"name": "periodic_rho02", "sam_mode": "periodic", "rho": 0.2, "probe_every": 10},
    {"name": "periodic_rho005_fast", "sam_mode": "periodic", "rho": 0.05, "probe_every": 20},
    {"name": "periodic_rho005_freq", "sam_mode": "periodic", "rho": 0.05, "probe_every": 5},

    # Periodic + cosine LR
    {"name": "periodic_cosine_rho005", "sam_mode": "periodic", "rho": 0.05, "probe_every": 10, "lr_schedule": "cosine"},
    {"name": "periodic_cosine_rho01", "sam_mode": "periodic", "rho": 0.1, "probe_every": 10, "lr_schedule": "cosine"},

    # Asymmetric scaling ranges
    {"name": "periodic_wide_scale", "sam_mode": "periodic", "rho": 0.05, "min_lr_scale": 0.1, "max_lr_scale": 5.0},
    {"name": "periodic_narrow_scale", "sam_mode": "periodic", "rho": 0.05, "min_lr_scale": 0.5, "max_lr_scale": 2.0},
    {"name": "periodic_log_scale", "sam_mode": "periodic", "rho": 0.05, "scale_mode": "log"},

    # Different target sharpness
    {"name": "periodic_target_05", "sam_mode": "periodic", "rho": 0.05, "target_sharpness": 0.5},
    {"name": "periodic_target_2", "sam_mode": "periodic", "rho": 0.05, "target_sharpness": 2.0},
    {"name": "periodic_target_5", "sam_mode": "periodic", "rho": 0.05, "target_sharpness": 5.0},

    # Full SAM (2x overhead but strongest signal)
    {"name": "fullsam_rho005", "sam_mode": "full_sam", "rho": 0.05},
    {"name": "fullsam_rho01", "sam_mode": "full_sam", "rho": 0.1},
    {"name": "fullsam_adaptive", "sam_mode": "full_sam", "rho": 0.05, "adaptive_sam": True},

    # LookSAM (amortized full SAM)
    {"name": "looksam_k5", "sam_mode": "looksam", "rho": 0.05, "looksam_every": 5},
    {"name": "looksam_k10", "sam_mode": "looksam", "rho": 0.05, "looksam_every": 10},
    {"name": "looksam_k5_rho01", "sam_mode": "looksam", "rho": 0.1, "looksam_every": 5},

    # Larger models (test if SAM helps bigger nets generalize)
    {"name": "h512_periodic_rho005", "sam_mode": "periodic", "rho": 0.05, "transformer_dim": 512, "transformer_layers": 6},
    {"name": "h512_baseline", "sam_mode": "none", "transformer_dim": 512, "transformer_layers": 6},
    {"name": "h768_periodic_rho005", "sam_mode": "periodic", "rho": 0.05, "transformer_dim": 768, "transformer_layers": 8},
    {"name": "h768_baseline", "sam_mode": "none", "transformer_dim": 768, "transformer_layers": 8},

    # Weight decay interactions
    {"name": "periodic_wd003", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.03},
    {"name": "periodic_wd005", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.05},
    {"name": "periodic_wd01", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1},

    # Feature noise + SAM
    {"name": "periodic_fn01", "sam_mode": "periodic", "rho": 0.05, "feature_noise_std": 0.01},
    {"name": "periodic_fn02", "sam_mode": "periodic", "rho": 0.05, "feature_noise_std": 0.02},

    # Multi-lag + SAM
    {"name": "periodic_multilag", "sam_mode": "periodic", "rho": 0.05, "decision_lag_range": "0,1,2"},
    {"name": "periodic_lag2", "sam_mode": "periodic", "rho": 0.05, "decision_lag_bars": 2},
]


SYMBOLS_CORE = [
    "BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "AAVEUSD",
    "LINKUSD", "XRPUSD", "AVAXUSD", "DOTUSD", "LTCUSD",
]

SYMBOLS_EXTENDED = SYMBOLS_CORE + [
    "ADAUSD", "BNBUSD", "UNIUSD", "NEARUSD", "MATICUSD",
]

DEFAULT_TRAINING_OVERRIDES = {
    "epochs": 30,
    "batch_size": 16,
    "sequence_length": 72,
    "learning_rate": 3e-4,
    "weight_decay": 0.04,
    "maker_fee": 0.001,
    "max_leverage": 1.0,
    "margin_annual_rate": 0.0625,
    "fill_temperature": 0.01,
    "decision_lag_bars": 2,
    "fill_buffer_pct": 0.0005,
    "loss_type": "sortino",
    "return_weight": 0.08,
    "validation_use_binary_fills": True,
    "use_compile": True,
    "use_tf32": True,
    "transformer_dim": 256,
    "transformer_layers": 4,
    "transformer_heads": 8,
}
