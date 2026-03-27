"""Experiment configs for sharpness-adjusted proximal policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SAPConfig:
    """Sharpness-adjusted proximal policy config, extends binanceneural TrainingConfig."""

    # Mode: directional proximal sharpness control, full SAM, LookSAM, or baseline
    sam_mode: str = "periodic"

    # Probe radius for directional curvature estimation / SAM perturbation
    rho: float = 0.05

    # How often to measure sharpness-aware control signals
    probe_every: int = 10

    # EMA smoothing for sharpness tracking
    ema_beta: float = 0.9

    # Target directional sharpness; if warmup_probes > 0 this autocalibrates
    target_sharpness: float = 1.0
    warmup_probes: int = 8

    # Actual step interpolation bounds and controller gains
    min_step_scale: float = 0.35
    max_step_scale: float = 1.15
    step_scale_beta: float = 0.6
    flat_bonus: float = 0.15
    sharp_penalty: float = 1.0
    loss_penalty: float = 0.5

    # Optional sanity-run caps (0 = unlimited)
    max_train_batches: int = 0
    max_val_batches: int = 0

    # Backwards-compatible legacy names kept for old sweep dicts
    min_lr_scale: float = 0.35
    max_lr_scale: float = 1.15
    scale_mode: str = "directional"

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

    # SWA (Stochastic Weight Averaging)
    use_swa: bool = False
    swa_start_frac: float = 0.5  # start averaging after 50% of training

    # Gradient noise injection
    use_grad_noise: bool = False
    grad_noise_sigma: float = 0.01
    grad_noise_gamma: float = 0.55  # decay exponent

    # Sharpness-adaptive feature noise (scale input noise by sharpness)
    use_adaptive_feature_noise: bool = False
    adaptive_fn_base: float = 0.005
    adaptive_fn_max: float = 0.05

    # Sharpness logging
    log_sharpness: bool = True

    def __post_init__(self) -> None:
        if self.min_step_scale == 0.35 and self.min_lr_scale != 0.35:
            self.min_step_scale = self.min_lr_scale
        if self.max_step_scale == 1.15 and self.max_lr_scale != 1.15:
            self.max_step_scale = self.max_lr_scale


# Experiment sweep configurations
EXPERIMENTS = [
    # Baselines
    {"name": "baseline_adamw", "sam_mode": "none"},
    {"name": "baseline_cosine", "sam_mode": "none", "lr_schedule": "cosine"},

    # Directional proximal sharpness control
    {"name": "proximal_rho005", "sam_mode": "periodic", "rho": 0.05, "probe_every": 10},
    {"name": "proximal_rho01", "sam_mode": "periodic", "rho": 0.1, "probe_every": 10},
    {"name": "proximal_rho02", "sam_mode": "periodic", "rho": 0.2, "probe_every": 10},
    {"name": "proximal_rho005_fast", "sam_mode": "periodic", "rho": 0.05, "probe_every": 20},
    {"name": "proximal_rho005_freq", "sam_mode": "periodic", "rho": 0.05, "probe_every": 5},
    {"name": "proximal_cautious", "sam_mode": "periodic", "rho": 0.05, "probe_every": 8, "max_step_scale": 1.0, "sharp_penalty": 1.5},
    {"name": "proximal_escape", "sam_mode": "periodic", "rho": 0.05, "probe_every": 8, "max_step_scale": 1.2, "flat_bonus": 0.25},
    {"name": "proximal_nowarmup", "sam_mode": "periodic", "rho": 0.05, "probe_every": 4, "warmup_probes": 0},
    {"name": "proximal_escape_nowarmup", "sam_mode": "periodic", "rho": 0.05, "probe_every": 4, "warmup_probes": 0, "max_step_scale": 1.2, "flat_bonus": 0.25},

    # Periodic + cosine LR
    {"name": "periodic_cosine_rho005", "sam_mode": "periodic", "rho": 0.05, "probe_every": 10, "lr_schedule": "cosine"},
    {"name": "periodic_cosine_rho01", "sam_mode": "periodic", "rho": 0.1, "probe_every": 10, "lr_schedule": "cosine"},

    # Asymmetric scaling ranges
    {"name": "periodic_wide_scale", "sam_mode": "periodic", "rho": 0.05, "min_step_scale": 0.3, "max_step_scale": 2.0},
    {"name": "periodic_narrow_scale", "sam_mode": "periodic", "rho": 0.05, "min_step_scale": 0.7, "max_step_scale": 1.3},
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

    # --- Round 2: informed by initial results ---
    # Higher wd baselines (to compare with SAM wd scaling)
    {"name": "baseline_wd01", "sam_mode": "none", "weight_decay": 0.1},
    {"name": "baseline_wd02", "sam_mode": "none", "weight_decay": 0.2},

    # SAM + higher wd with wider scale cap
    {"name": "periodic_wd01_wide", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1, "max_step_scale": 3.0},
    {"name": "periodic_wd015", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.15},
    {"name": "periodic_wd02", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.2},

    # Cosine LR + high wd (combine best regularization strategies)
    {"name": "periodic_wd01_cosine", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1, "lr_schedule": "cosine"},
    {"name": "periodic_wd005_cosine", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.05, "lr_schedule": "cosine"},

    # h512 with higher lr (fix dead model)
    {"name": "h512_lr1e3_periodic", "sam_mode": "periodic", "rho": 0.05, "transformer_dim": 512, "transformer_layers": 6, "learning_rate": 1e-3},
    {"name": "h512_lr1e3_baseline", "sam_mode": "none", "transformer_dim": 512, "transformer_layers": 6, "learning_rate": 1e-3},

    # --- Round 3: new approaches ---
    # SWA (average weights over second half of training)
    {"name": "swa_wd01", "sam_mode": "none", "weight_decay": 0.1, "use_swa": True, "swa_start_frac": 0.5},
    {"name": "swa_wd004", "sam_mode": "none", "weight_decay": 0.04, "use_swa": True, "swa_start_frac": 0.5},
    {"name": "swa_periodic_wd01", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1, "use_swa": True},

    # Gradient noise injection (Neelakantan et al.)
    {"name": "gradnoise_s001", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1, "use_grad_noise": True, "grad_noise_sigma": 0.01},
    {"name": "gradnoise_s005", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1, "use_grad_noise": True, "grad_noise_sigma": 0.05},
    {"name": "gradnoise_baseline", "sam_mode": "none", "weight_decay": 0.1, "use_grad_noise": True, "grad_noise_sigma": 0.01},

    # Adaptive feature noise (scale input noise by sharpness)
    {"name": "afn_periodic", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1, "use_adaptive_feature_noise": True},
    {"name": "afn_baseline", "sam_mode": "none", "weight_decay": 0.1, "use_adaptive_feature_noise": True},

    # Kitchen sink: SAM + SWA + grad noise + high wd
    {"name": "kitchen_sink", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1, "use_swa": True, "use_grad_noise": True, "grad_noise_sigma": 0.01},

    # Higher lr sweep (SAM may stabilize higher lr)
    {"name": "lr5e4_periodic", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1, "learning_rate": 5e-4},
    {"name": "lr5e4_baseline", "sam_mode": "none", "weight_decay": 0.1, "learning_rate": 5e-4},
    {"name": "lr1e3_periodic", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1, "learning_rate": 1e-3},
    {"name": "lr1e3_baseline", "sam_mode": "none", "weight_decay": 0.1, "learning_rate": 1e-3},

    # --- Round 4: temporal bar-shift augmentation (user insight: jitter start index ±N bars) ---
    # Cheap augmentation: each training window starts at idx + Uniform(-N, N) — more diverse views
    {"name": "barshift_1_baseline", "sam_mode": "none", "weight_decay": 0.1, "bar_shift_range": 1},
    {"name": "barshift_2_baseline", "sam_mode": "none", "weight_decay": 0.1, "bar_shift_range": 2},
    {"name": "barshift_5_baseline", "sam_mode": "none", "weight_decay": 0.1, "bar_shift_range": 5},
    {"name": "barshift_12_baseline", "sam_mode": "none", "weight_decay": 0.1, "bar_shift_range": 12},
    {"name": "barshift_2_periodic", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1, "bar_shift_range": 2},
    {"name": "barshift_5_periodic", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1, "bar_shift_range": 5},
    {"name": "barshift_12_periodic", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1, "bar_shift_range": 12},
    # Bar shift + cosine LR + SAM (should regularize from multiple angles)
    {"name": "barshift_5_cosine_periodic", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1,
     "bar_shift_range": 5, "lr_schedule": "cosine"},
    # Bar shift + SWA (weight averaging smooths out shift-induced variance)
    {"name": "barshift_5_swa", "sam_mode": "none", "weight_decay": 0.1, "bar_shift_range": 5, "use_swa": True},

    # --- Round 4b: Sparse MoE FFN (fine-grained experts, soft routing, same param count) ---
    # model_arch="moe" uses BinanceHourlyPolicyNano with SparseMoEFeedForward (n_experts=8)
    # Each expert inner_dim = hidden_dim * mlp_ratio / n_experts — same total params as dense FFN
    {"name": "moe8_baseline", "sam_mode": "none", "weight_decay": 0.1,
     "model_arch": "moe", "moe_num_experts": 8},
    {"name": "moe8_periodic", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1,
     "model_arch": "moe", "moe_num_experts": 8},
    {"name": "moe4_baseline", "sam_mode": "none", "weight_decay": 0.1,
     "model_arch": "moe", "moe_num_experts": 4},
    {"name": "moe16_baseline", "sam_mode": "none", "weight_decay": 0.1,
     "model_arch": "moe", "moe_num_experts": 16},
    # MoE + bar shift (should compound: more diverse data + more specialized capacity)
    {"name": "moe8_barshift5_baseline", "sam_mode": "none", "weight_decay": 0.1,
     "model_arch": "moe", "moe_num_experts": 8, "bar_shift_range": 5},
    {"name": "moe8_barshift5_periodic", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1,
     "model_arch": "moe", "moe_num_experts": 8, "bar_shift_range": 5},
    # Larger MoE model (h512, 6L, 8 experts) — more capacity
    {"name": "moe8_h512_baseline", "sam_mode": "none", "weight_decay": 0.1,
     "model_arch": "moe", "moe_num_experts": 8, "transformer_dim": 512, "transformer_layers": 6},
    {"name": "moe8_h512_periodic", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1,
     "model_arch": "moe", "moe_num_experts": 8, "transformer_dim": 512, "transformer_layers": 6},
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
    "use_compile": False,
    "use_tf32": True,
    "transformer_dim": 256,
    "transformer_layers": 4,
    "transformer_heads": 8,
}
