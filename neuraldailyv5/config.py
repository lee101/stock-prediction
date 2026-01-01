"""V5 Configuration - NEPA-style Portfolio Latent Prediction.

Key changes from V4:
- Portfolio latents: Predict sequence of portfolio embeddings
- Portfolio weights: Output target allocations per asset
- Ramp-into-position: Gradual portfolio matching throughout day
- NEPA loss: Cosine similarity for sequence coherence
- Sortino focus: Optimize downside risk
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class TemperatureSchedule:
    """Temperature annealing for differentiable rebalancing."""

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
class SimulationConfigV5:
    """Configuration for V5 portfolio simulation."""

    # Fees
    maker_fee: float = 0.0008
    taker_fee: float = 0.001
    slippage_bps: float = 5.0  # Basis points of slippage per trade

    # Portfolio constraints
    equity_max_leverage: float = 2.0
    crypto_max_leverage: float = 1.0
    max_single_asset_weight: float = 0.5  # Max 50% in single asset
    min_position_weight: float = 0.01  # Min 1% to open position
    rebalance_threshold: float = 0.02  # Rebalance if drift > 2%

    # Ramp-into-position
    ramp_periods: int = 4  # Split orders into N periods throughout day
    market_impact_bps: float = 2.0  # Per-period market impact

    # Time
    lookahead_days: int = 20
    use_trading_days: bool = True

    # Risk metrics
    risk_free_rate: float = 0.04  # Annual risk-free rate for Sortino
    min_acceptable_return: float = 0.0  # MAR for Sortino


@dataclass
class PolicyConfigV5:
    """Model architecture configuration for V5 with portfolio latents."""

    input_dim: int = 20
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    num_kv_heads: int = 4
    dropout: float = 0.1
    max_len: int = 512
    logits_softcap: float = 15.0

    # Patching (from V4)
    patch_size: int = 5
    use_patch_embedding: bool = True
    patch_residual: bool = True

    # V5: Portfolio latent prediction
    latent_dim: int = 64  # Dimension of portfolio latents
    num_assets: int = 11  # Number of assets to allocate (SPY, QQQ, ..., BTCUSD, ETHUSD)
    use_nepa_loss: bool = True  # NEPA cosine similarity loss
    nepa_loss_weight: float = 0.1

    # V5: Atrous convolution for long-range (optional)
    use_atrous_conv: bool = True
    atrous_rates: Tuple[int, ...] = (1, 2, 4, 8)  # Dilation rates
    atrous_channels: int = 128

    # V5: Multi-resolution
    use_multi_resolution: bool = True
    resolution_scales: Tuple[int, ...] = (1, 2, 5)  # Aggregate at 1d, 2d, 5d scales
    trim_fraction: float = 0.25  # Trimmed mean for multi-res aggregation

    # Portfolio output
    output_volatility: bool = True  # Predict per-asset volatility
    output_confidence: bool = True  # Predict allocation confidence

    # Leverage
    equity_max_leverage: float = 2.0
    crypto_max_leverage: float = 1.0

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def num_patches(self) -> int:
        return 256 // self.patch_size


@dataclass
class DailyDatasetConfigV5:
    """Dataset configuration for V5 training."""

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
    validation_days: int = 40
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
class DailyTrainingConfigV5:
    """V5 training configuration."""

    # Training parameters
    epochs: int = 200
    batch_size: int = 32
    sequence_length: int = 256
    lookahead_days: int = 20
    learning_rate: float = 0.0003
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # V5: Patching
    patch_size: int = 5

    # V5: Portfolio latent
    latent_dim: int = 64
    use_nepa_loss: bool = True
    nepa_loss_weight: float = 0.1

    # V5: Atrous
    use_atrous_conv: bool = True
    atrous_rates: Tuple[int, ...] = (1, 2, 4, 8)

    # V5: Multi-resolution
    use_multi_resolution: bool = True
    resolution_scales: Tuple[int, ...] = (1, 2, 5)
    trim_fraction: float = 0.25

    # Simulation parameters
    maker_fee: float = 0.0008
    taker_fee: float = 0.001
    slippage_bps: float = 5.0
    equity_max_leverage: float = 2.0
    crypto_max_leverage: float = 1.0
    max_single_asset_weight: float = 0.5
    rebalance_threshold: float = 0.02
    ramp_periods: int = 4

    # Model architecture
    transformer_dim: int = 512
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_kv_heads: int = 4
    transformer_dropout: float = 0.1
    logits_softcap: float = 15.0

    # Temperature annealing
    initial_temperature: float = 0.01
    final_temperature: float = 0.0001
    temp_warmup_epochs: int = 10
    temp_anneal_epochs: int = 150
    temp_anneal_type: str = "linear"

    # Dual optimizer settings (from nanochat)
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

    # V5 Loss weights
    return_loss_weight: float = 1.0          # Maximize returns
    sortino_loss_weight: float = 0.2         # Sortino ratio (downside risk)
    nepa_loss_weight: float = 0.1            # NEPA coherence loss
    turnover_penalty: float = 0.05           # Penalize excessive rebalancing
    concentration_penalty: float = 0.05      # Penalize concentrated portfolios
    volatility_calibration_weight: float = 0.05  # Calibrated vol estimates

    # Training settings
    dry_train_steps: Optional[int] = None
    device: Optional[str] = None
    run_name: str = "neuraldailyv5"
    wandb_project: str = "neuraldailyv5"
    wandb_entity: Optional[str] = None
    log_dir: str = "tensorboard_logs/neuraldailyv5"
    checkpoint_root: str = "neuraldailyv5/checkpoints"
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

    # Dataset config
    dataset: Optional[DailyDatasetConfigV5] = None

    def get_temperature_schedule(self) -> TemperatureSchedule:
        return TemperatureSchedule(
            initial_temp=self.initial_temperature,
            final_temp=self.final_temperature,
            warmup_epochs=self.temp_warmup_epochs,
            anneal_epochs=self.temp_anneal_epochs,
            anneal_type=self.temp_anneal_type,
        )

    def get_simulation_config(self) -> SimulationConfigV5:
        return SimulationConfigV5(
            maker_fee=self.maker_fee,
            taker_fee=self.taker_fee,
            slippage_bps=self.slippage_bps,
            equity_max_leverage=self.equity_max_leverage,
            crypto_max_leverage=self.crypto_max_leverage,
            max_single_asset_weight=self.max_single_asset_weight,
            rebalance_threshold=self.rebalance_threshold,
            ramp_periods=self.ramp_periods,
            lookahead_days=self.lookahead_days,
        )

    def get_policy_config(self, input_dim: int, num_assets: int) -> PolicyConfigV5:
        return PolicyConfigV5(
            input_dim=input_dim,
            hidden_dim=self.transformer_dim,
            num_layers=self.transformer_layers,
            num_heads=self.transformer_heads,
            num_kv_heads=self.transformer_kv_heads,
            dropout=self.transformer_dropout,
            max_len=self.sequence_length + 64,
            logits_softcap=self.logits_softcap,
            patch_size=self.patch_size,
            latent_dim=self.latent_dim,
            num_assets=num_assets,
            use_nepa_loss=self.use_nepa_loss,
            nepa_loss_weight=self.nepa_loss_weight,
            use_atrous_conv=self.use_atrous_conv,
            atrous_rates=self.atrous_rates,
            use_multi_resolution=self.use_multi_resolution,
            resolution_scales=self.resolution_scales,
            trim_fraction=self.trim_fraction,
            equity_max_leverage=self.equity_max_leverage,
            crypto_max_leverage=self.crypto_max_leverage,
        )
