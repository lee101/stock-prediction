"""V4 Configuration - Chronos-2 inspired architecture.

Key changes from V3:
- Patching: 5-day patches for faster processing
- Multi-window: Direct prediction of multiple future windows
- Quantile outputs: Price distribution instead of point estimates
- Trimmed mean: Robust aggregation across windows
- Learned position sizing: Dynamic sizing based on uncertainty
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class TemperatureSchedule:
    """Temperature annealing for differentiable fills."""

    initial_temp: float = 0.01
    final_temp: float = 0.0001
    warmup_epochs: int = 10
    anneal_epochs: int = 150
    anneal_type: str = "linear"

    def get_temperature(self, epoch: int, total_epochs: int = 200) -> float:
        if epoch < self.warmup_epochs:
            return self.initial_temp

        progress = min(1.0, (epoch - self.warmup_epochs) / max(1, self.anneal_epochs))

        if self.anneal_type == "cosine":
            progress = 0.5 * (1 - math.cos(math.pi * progress))

        return self.initial_temp + (self.final_temp - self.initial_temp) * progress


@dataclass
class SimulationConfigV4:
    """Configuration for V4 multi-window trade simulation."""

    maker_fee: float = 0.0008
    initial_cash: float = 1.0
    initial_inventory: float = 0.0
    equity_max_leverage: float = 2.0
    crypto_max_leverage: float = 1.0
    leverage_fee_rate: float = 0.065

    # V4: Multi-window parameters
    num_windows: int = 4           # Number of prediction windows
    window_size: int = 5           # Days per window
    max_hold_days: int = 20        # Total max hold (4 windows * 5 days)
    min_hold_days: int = 1
    forced_exit_slippage: float = 0.001
    use_trading_days: bool = True

    # Aggregation
    trim_fraction: float = 0.25    # Trim top/bottom 25% for trimmed mean
    use_confidence_weighting: bool = True  # Weight windows by confidence


@dataclass
class PolicyConfigV4:
    """Model architecture configuration for V4 with patching and multi-window."""

    input_dim: int = 20
    hidden_dim: int = 512          # Wide (from model size experiments)
    num_layers: int = 4            # Shallow (from model size experiments)
    num_heads: int = 8
    num_kv_heads: int = 4
    dropout: float = 0.1
    max_len: int = 512
    equity_max_leverage: float = 2.0
    crypto_max_leverage: float = 1.0
    use_cross_attention: bool = True
    logits_softcap: float = 15.0

    # V4: Patching
    patch_size: int = 5            # 5-day patches
    use_patch_embedding: bool = True
    patch_residual: bool = True    # Residual connection in patch embedding

    # V4: Multi-window output
    num_windows: int = 4           # Number of prediction windows
    window_size: int = 5           # Days per window

    # V4: Quantile outputs
    num_quantiles: int = 3         # q25, q50, q75
    quantile_levels: Tuple[float, ...] = (0.25, 0.5, 0.75)

    # V4: Position sizing
    min_position: float = 0.05     # Min 5% of capital
    max_position: float = 1.0      # Max 100% of capital (per asset class leverage)
    position_from_confidence: bool = True  # Scale position by model confidence
    position_from_uncertainty: bool = True  # Scale position by inverse uncertainty

    # Price constraints
    price_offset_pct: float = 0.05
    min_price_gap_pct: float = 0.005

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def num_patches(self) -> int:
        """Number of patches for default sequence length."""
        return 256 // self.patch_size  # 51 patches

    @property
    def outputs_per_window(self) -> int:
        """Outputs per prediction window."""
        # buy_quantiles(3) + sell_quantiles(3) + position(1) + confidence(1) + exit_day(1)
        return self.num_quantiles * 2 + 3


@dataclass
class DailyDatasetConfigV4:
    """Dataset configuration for V4 training."""

    symbols: Tuple[str, ...] = (
        "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
        "BTCUSD", "ETHUSD",
    )
    data_root: Path = field(default_factory=lambda: Path("trainingdata/train"))
    forecast_cache_dir: Path = field(default_factory=lambda: Path("strategytraining/forecast_cache"))
    sequence_length: int = 256
    lookahead_days: int = 20
    val_fraction: float = 0.2
    min_history_days: int = 300
    require_forecasts: bool = False
    forecast_fill_strategy: str = "persistence"
    forecast_cache_writeback: bool = True
    feature_columns: Optional[List[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    validation_days: int = 80  # Increased from 40 to reduce overfitting
    symbol_dropout_rate: float = 0.1
    exclude_symbols: Optional[List[str]] = None
    exclude_symbols_file: Optional[str] = None
    crypto_only: bool = False
    include_weekly_features: bool = True
    grouping_strategy: str = "correlation"
    correlation_min_corr: float = 0.6
    correlation_max_group_size: int = 12
    correlation_window_days: int = 252
    correlation_min_overlap: int = 60
    split_crypto_groups: bool = True

    @property
    def total_length(self) -> int:
        return self.sequence_length + self.lookahead_days


@dataclass
class DailyTrainingConfigV4:
    """V4 training configuration."""

    # Training parameters
    epochs: int = 200
    batch_size: int = 32
    sequence_length: int = 256
    lookahead_days: int = 20
    learning_rate: float = 0.0003
    weight_decay: float = 0.01  # Added L2 regularization to reduce overfitting
    grad_clip: float = 1.0

    # V4: Patching
    patch_size: int = 5

    # V4: Multi-window
    num_windows: int = 4
    window_size: int = 5

    # V4: Quantiles
    num_quantiles: int = 3
    quantile_levels: Tuple[float, ...] = (0.25, 0.5, 0.75)

    # V4: Aggregation
    trim_fraction: float = 0.25
    use_confidence_weighting: bool = True

    # Simulation parameters
    maker_fee: float = 0.0008
    return_weight: float = 0.08
    equity_max_leverage: float = 2.0
    crypto_max_leverage: float = 1.0
    leverage_fee_rate: float = 0.065
    steps_per_year: int = 252
    max_hold_days: int = 20
    min_hold_days: int = 1
    forced_exit_slippage: float = 0.001

    # Model architecture (wide & shallow from experiments)
    transformer_dim: int = 512
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_kv_heads: int = 4
    transformer_dropout: float = 0.1
    logits_softcap: float = 15.0
    price_offset_pct: float = 0.05
    min_price_gap_pct: float = 0.005

    # Position sizing
    min_position: float = 0.05
    max_position: float = 1.0
    position_from_confidence: bool = True
    position_from_uncertainty: bool = True

    # Temperature annealing
    initial_temperature: float = 0.01
    final_temperature: float = 0.0001
    temp_warmup_epochs: int = 10
    temp_anneal_epochs: int = 150
    temp_anneal_type: str = "linear"

    # Dual optimizer settings
    optimizer_name: str = "dual"
    matrix_lr: float = 0.02
    embed_lr: float = 0.2
    head_lr: float = 0.004
    muon_momentum: float = 0.95
    adamw_betas: Tuple[float, float] = (0.8, 0.95)
    adamw_eps: float = 1e-10

    # Scheduler
    warmup_steps: int = 100
    use_cosine_schedule: bool = True

    # EMA
    ema_decay: float = 0.0

    # V4 Loss weights
    return_loss_weight: float = 1.0          # Maximize returns
    sharpe_loss_weight: float = 0.1          # Risk-adjusted returns
    forced_exit_penalty: float = 0.1         # Penalize deadline exits
    quantile_calibration_weight: float = 0.05  # Calibrated uncertainty
    position_regularization: float = 0.01    # Prevent extreme positions

    # V7+ Loss weights (based on backtest findings)
    quantile_ordering_weight: float = 0.0    # Enforce q10 < q50 < q90 (set >0 to enable)
    exit_days_penalty_weight: float = 0.0    # Prefer shorter holds (set >0 to enable)

    # Portfolio utilization (V11+)
    utilization_loss_weight: float = 0.1     # Encourage portfolio utilization
    utilization_target: float = 0.5          # Target 50% portfolio utilization

    # Training settings
    dry_train_steps: Optional[int] = None
    device: Optional[str] = None
    run_name: str = "neuraldailyv4"
    wandb_project: str = "neuraldailyv4"
    wandb_entity: Optional[str] = None
    log_dir: str = "tensorboard_logs/neuraldailyv4"
    checkpoint_root: str = "neuraldailyv4/checkpoints"
    top_k_checkpoints: int = 50
    preload_checkpoint_path: Optional[str] = None
    force_retrain: bool = False
    seed: int = 1337
    num_workers: int = 0
    use_compile: bool = False
    compile_mode: str = "max-autotune"
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    use_tf32: bool = True
    use_cross_attention: bool = True

    # Data augmentation
    permutation_rate: float = 0.5
    price_scale_range: Tuple[float, float] = (0.9, 1.1)
    price_scale_probability: float = 0.2

    # Dataset config
    dataset: Optional[DailyDatasetConfigV4] = None

    def get_temperature_schedule(self) -> TemperatureSchedule:
        return TemperatureSchedule(
            initial_temp=self.initial_temperature,
            final_temp=self.final_temperature,
            warmup_epochs=self.temp_warmup_epochs,
            anneal_epochs=self.temp_anneal_epochs,
            anneal_type=self.temp_anneal_type,
        )

    def get_simulation_config(self) -> SimulationConfigV4:
        return SimulationConfigV4(
            maker_fee=self.maker_fee,
            equity_max_leverage=self.equity_max_leverage,
            crypto_max_leverage=self.crypto_max_leverage,
            leverage_fee_rate=self.leverage_fee_rate,
            num_windows=self.num_windows,
            window_size=self.window_size,
            max_hold_days=self.max_hold_days,
            min_hold_days=self.min_hold_days,
            forced_exit_slippage=self.forced_exit_slippage,
            trim_fraction=self.trim_fraction,
            use_confidence_weighting=self.use_confidence_weighting,
        )

    def get_policy_config(self, input_dim: int) -> PolicyConfigV4:
        return PolicyConfigV4(
            input_dim=input_dim,
            hidden_dim=self.transformer_dim,
            num_layers=self.transformer_layers,
            num_heads=self.transformer_heads,
            num_kv_heads=self.transformer_kv_heads,
            dropout=self.transformer_dropout,
            max_len=self.sequence_length + 64,
            equity_max_leverage=self.equity_max_leverage,
            crypto_max_leverage=self.crypto_max_leverage,
            use_cross_attention=self.use_cross_attention,
            logits_softcap=self.logits_softcap,
            patch_size=self.patch_size,
            num_windows=self.num_windows,
            window_size=self.window_size,
            num_quantiles=self.num_quantiles,
            quantile_levels=self.quantile_levels,
            min_position=self.min_position,
            max_position=self.max_position,
            position_from_confidence=self.position_from_confidence,
            position_from_uncertainty=self.position_from_uncertainty,
            price_offset_pct=self.price_offset_pct,
            min_price_gap_pct=self.min_price_gap_pct,
        )
