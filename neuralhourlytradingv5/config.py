"""V5 Configuration - Hourly crypto trading with learned position timing.

Key features:
- Learned position length (0-24 hours): 0 = skip, 1-24 = hold hours
- Fee-aware price clamping with 8bps dead zone around midpoint
- 168-hour sequences with 4-hour patching
- Multi-scale Chronos2 integration
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class TemperatureScheduleV5:
    """Temperature annealing for differentiable fills."""

    initial_temp: float = 0.02
    final_temp: float = 0.001
    warmup_epochs: int = 5
    anneal_epochs: int = 50
    anneal_type: str = "linear"

    def get_temperature(self, epoch: int, total_epochs: int = 100) -> float:
        if epoch < self.warmup_epochs:
            return self.initial_temp

        progress = min(1.0, (epoch - self.warmup_epochs) / max(1, self.anneal_epochs))

        if self.anneal_type == "cosine":
            progress = 0.5 * (1 - math.cos(math.pi * progress))

        return self.initial_temp + (self.final_temp - self.initial_temp) * progress


@dataclass
class SimulationConfigV5:
    """Configuration for V5 hourly trade simulation."""

    maker_fee: float = 0.0008  # 8 bps per leg
    initial_cash: float = 1.0
    initial_inventory: float = 0.0
    max_leverage: float = 1.0  # Crypto typically 1x
    forced_exit_slippage: float = 0.001  # 10 bps slippage on forced exit
    max_position_hours: int = 24  # Maximum hours to hold a position
    stop_loss_pct: float = 0.0  # 0 disables stop-loss
    stop_loss_slippage: float = 0.001  # Slippage applied on stop-loss exits


@dataclass
class PolicyConfigV5:
    """Model architecture configuration for V5 hourly crypto."""

    input_dim: int = 19  # Will be computed from feature columns
    hidden_dim: int = 384  # Smaller than V4 (512) since more hourly data
    num_layers: int = 4
    num_heads: int = 6
    num_kv_heads: int = 3  # MQA with 2x head sharing
    dropout: float = 0.1
    max_len: int = 256

    # Patching (V4-style, adapted for hourly)
    patch_size: int = 4  # 4-hour patches
    use_patch_embedding: bool = True
    patch_residual: bool = True

    # Output heads
    max_position_hours: int = 24  # Max hours to hold
    num_position_classes: int = 25  # 0 = skip, 1-24 = hold hours

    # CRITICAL: Price safety with fee buffer
    # Prices are always offset from midpoint (reference_close)
    # min_offset ensures we don't trade inside the fee zone
    maker_fee: float = 0.0008  # 8 bps
    min_price_offset_pct: float = 0.0008  # == maker_fee (dead zone)
    max_price_offset_pct: float = 0.03  # Max 3% from midpoint

    # Position sizing
    min_position: float = 0.0  # 0 = no trade
    max_position: float = 1.0  # Full capital

    logits_softcap: float = 15.0

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def num_patches(self) -> int:
        """Number of patches for default sequence length (168 hours)."""
        return 168 // self.patch_size  # 42 patches


@dataclass
class DefaultStrategyConfig:
    """Default strategy configuration based on backtesting results.

    Benchmark results (2025-12-28) - Symbol-specific models vs BTCUSD-trained:

    | Symbol  | BTCUSD-trained | Symbol-specific | Improvement |
    |---------|----------------|-----------------|-------------|
    | LINKUSD | +19.80%        | +29.86%         | +10.06%     |
    | UNIUSD  | +4.15%         | +18.92%         | +14.77%     |
    | BTCUSD  | +1.16%         | N/A             | baseline    |

    Key findings:
    - Symbol-specific training significantly improves performance
    - LINKUSD model: 70.6% win rate, 64.7% TP rate
    - UNIUSD model: 73.7% win rate, 68.4% TP rate
    - Use dedicated checkpoints per symbol for best results
    """

    # Symbol-specific checkpoints (best performers)
    symbol_checkpoints: dict = field(default_factory=lambda: {
        "LINKUSD": "neuralhourlytradingv5/checkpoints_link/best_hourlyv5_epoch021_20251228_064138.pt",
        "UNIUSD": "neuralhourlytradingv5/checkpoints_uni/best_hourlyv5_epoch017_20251228_071935.pt",
        "BTCUSD": "neuralhourlytradingv5/checkpoints/best_hourlyv5_epoch028_20251227_074345.pt",
    })

    # Default checkpoint (BTCUSD model as fallback)
    checkpoint_path: str = "neuralhourlytradingv5/checkpoints/best_hourlyv5_epoch028_20251227_074345.pt"

    # Recommended trading symbols (sorted by performance)
    recommended_symbols: Tuple[str, ...] = ("LINKUSD", "UNIUSD")

    # All benchmarked symbols
    all_symbols: Tuple[str, ...] = ("BTCUSD", "ETHUSD", "SOLUSD", "LINKUSD", "UNIUSD")

    def get_checkpoint_for_symbol(self, symbol: str) -> str:
        """Get the best checkpoint for a specific symbol."""
        return self.symbol_checkpoints.get(symbol, self.checkpoint_path)


@dataclass
class DatasetConfigV5:
    """Dataset configuration for V5 hourly training."""

    symbols: Tuple[str, ...] = ("BTCUSD", "ETHUSD", "SOLUSD", "LINKUSD")
    data_root: Path = field(default_factory=lambda: Path("trainingdatahourly/crypto"))
    forecast_cache_dir: Path = field(
        default_factory=lambda: Path("neuralhourlytradingv5/forecast_cache")
    )
    sequence_length: int = 168  # 1 week of hourly data
    lookahead_hours: int = 24  # Max position length for simulation
    validation_hours: int = 240  # 10 days as specified
    min_history_hours: int = 24 * 60  # 60 days minimum
    max_history_hours: Optional[int] = None  # Limit history for memory safety
    max_feature_lookback_hours: int = 24 * 7  # 1 week lookback for features
    chronos_skip_rates: Tuple[int, ...] = (1, 2, 4)  # Multi-scale forecasting
    val_fraction: float = 0.15
    feature_columns: Optional[List[str]] = None
    refresh_hours: int = 72  # Refresh data every 72 hours


@dataclass
class TrainingConfigV5:
    """V5 training configuration with fee-aware simulation."""

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    sequence_length: int = 168  # 1 week
    lookahead_hours: int = 24
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Patching
    patch_size: int = 4  # 4-hour patches

    # Simulation parameters
    maker_fee: float = 0.0008  # 8 bps
    max_leverage: float = 1.0  # Crypto 1x
    forced_exit_slippage: float = 0.001
    stop_loss_pct: float = 0.0  # 0 disables stop-loss
    stop_loss_slippage: float = 0.001

    # Price constraints (CRITICAL for safety)
    min_price_offset_pct: float = 0.0008  # == maker_fee (dead zone)
    max_price_offset_pct: float = 0.03  # Max 3% from midpoint

    # Model architecture
    transformer_dim: int = 384
    transformer_layers: int = 4
    transformer_heads: int = 6
    transformer_kv_heads: int = 3
    transformer_dropout: float = 0.1
    logits_softcap: float = 15.0

    # Position sizing
    min_position: float = 0.0
    max_position: float = 1.0

    # Temperature annealing for differentiable fills
    initial_temperature: float = 0.02
    final_temperature: float = 0.001
    temp_warmup_epochs: int = 5
    temp_anneal_epochs: int = 50
    temp_anneal_type: str = "linear"

    # Optimizer settings
    optimizer_name: str = "dual"  # Muon + AdamW
    matrix_lr: float = 0.02
    embed_lr: float = 0.2
    head_lr: float = 0.004
    muon_momentum: float = 0.95
    adamw_betas: Tuple[float, float] = (0.8, 0.95)
    adamw_eps: float = 1e-10
    warmup_steps: int = 100
    use_cosine_schedule: bool = True

    # Loss weights
    sortino_weight: float = 1.0  # Primary: risk-adjusted returns
    return_weight: float = 0.1  # Raw returns
    forced_exit_penalty: float = 0.2  # Discourage hitting max hold time
    stop_loss_penalty: float = 0.1  # Penalize stop-loss exits when enabled
    no_trade_penalty: float = 0.01  # Prevent always skipping
    spread_utilization: float = 0.05  # Encourage using offset range

    # EMA
    ema_decay: float = 0.0

    # Training settings
    dry_train_steps: Optional[int] = None
    device: Optional[str] = None
    run_name: str = "neuralhourlytradingv5"
    wandb_project: str = "neuralhourlytradingv5"
    wandb_entity: Optional[str] = None
    log_dir: str = "tensorboard_logs/neuralhourlytradingv5"
    checkpoint_root: str = "neuralhourlytradingv5/checkpoints"
    top_k_checkpoints: int = 50
    preload_checkpoint_path: Optional[str] = None
    force_retrain: bool = False
    seed: int = 1337
    num_workers: int = 0
    use_compile: bool = True
    compile_mode: str = "max-autotune"
    use_amp: bool = False  # Disabled for hourly (smaller batches)
    amp_dtype: str = "bfloat16"
    use_tf32: bool = True

    # Dataset config
    dataset: Optional[DatasetConfigV5] = None

    def get_temperature_schedule(self) -> TemperatureScheduleV5:
        return TemperatureScheduleV5(
            initial_temp=self.initial_temperature,
            final_temp=self.final_temperature,
            warmup_epochs=self.temp_warmup_epochs,
            anneal_epochs=self.temp_anneal_epochs,
            anneal_type=self.temp_anneal_type,
        )

    def get_simulation_config(self) -> SimulationConfigV5:
        return SimulationConfigV5(
            maker_fee=self.maker_fee,
            max_leverage=self.max_leverage,
            forced_exit_slippage=self.forced_exit_slippage,
            max_position_hours=self.lookahead_hours,
            stop_loss_pct=self.stop_loss_pct,
            stop_loss_slippage=self.stop_loss_slippage,
        )

    def get_policy_config(self, input_dim: int) -> PolicyConfigV5:
        return PolicyConfigV5(
            input_dim=input_dim,
            hidden_dim=self.transformer_dim,
            num_layers=self.transformer_layers,
            num_heads=self.transformer_heads,
            num_kv_heads=self.transformer_kv_heads,
            dropout=self.transformer_dropout,
            max_len=self.sequence_length + 64,
            patch_size=self.patch_size,
            maker_fee=self.maker_fee,
            min_price_offset_pct=self.min_price_offset_pct,
            max_price_offset_pct=self.max_price_offset_pct,
            min_position=self.min_position,
            max_position=self.max_position,
            logits_softcap=self.logits_softcap,
        )
