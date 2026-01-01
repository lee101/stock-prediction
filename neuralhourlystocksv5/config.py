"""V5 Stock Configuration - Hourly stock trading with learned position timing.

Key differences from crypto V5:
- Lower fees: 2bps maker fee (vs 8bps for crypto)
- Market hours filtering: Only 9:30 AM - 4:00 PM ET
- Data source: trainingdatahourly/stocks/
- 7 trading hours per day (vs 24 for crypto)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# Re-use temperature schedule from crypto (same annealing logic)
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
class SimulationConfigStocksV5:
    """Configuration for V5 stock hourly trade simulation."""

    # STOCK-SPECIFIC: Lower fees than crypto
    maker_fee: float = 0.0002  # 2 bps per leg (typical retail broker)
    initial_cash: float = 1.0
    initial_inventory: float = 0.0
    max_leverage: float = 2.0  # Stocks allow 2x margin
    forced_exit_slippage: float = 0.0005  # 5 bps slippage (tighter than crypto)
    max_position_hours: int = 24  # Maximum hours to hold (within market hours)


@dataclass
class PolicyConfigStocksV5:
    """Model architecture configuration for V5 hourly stocks."""

    input_dim: int = 19  # Will be computed from feature columns
    hidden_dim: int = 384  # Same as crypto V5
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

    # STOCK-SPECIFIC: Price safety with lower fee buffer
    maker_fee: float = 0.0002  # 2 bps (stock retail)
    min_price_offset_pct: float = 0.0002  # == maker_fee (dead zone)
    max_price_offset_pct: float = 0.02  # Max 2% from midpoint (less volatile than crypto)

    # Position sizing
    min_position: float = 0.0  # 0 = no trade
    max_position: float = 1.0  # Full capital

    logits_softcap: float = 15.0

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def num_patches(self) -> int:
        """Number of patches for default sequence length (168 bars)."""
        return 168 // self.patch_size  # 42 patches


@dataclass
class DefaultStockStrategyConfig:
    """Default strategy configuration for stock trading.

    Stocks have ~7 tradable hours per day (9:30 AM - 4:00 PM ET).
    A 168-bar sequence covers ~24 trading days.
    """

    # Default checkpoint (will be trained)
    checkpoint_path: str = "neuralhourlystocksv5/checkpoints/best_stocksv5.pt"

    # All stock symbols (loaded dynamically from data directory)
    data_root: Path = field(default_factory=lambda: Path("trainingdatahourly/stocks"))

    def get_all_symbols(self) -> List[str]:
        """Get all available stock symbols from data directory."""
        if not self.data_root.exists():
            return []
        return sorted([
            f.stem for f in self.data_root.glob("*.csv")
        ])


@dataclass
class DatasetConfigStocksV5:
    """Dataset configuration for V5 stock hourly training."""

    # Symbols will be loaded dynamically
    symbols: Optional[Tuple[str, ...]] = None
    data_root: Path = field(default_factory=lambda: Path("trainingdatahourly/stocks"))
    forecast_cache_dir: Path = field(
        default_factory=lambda: Path("neuralhourlystocksv5/forecast_cache")
    )
    sequence_length: int = 168  # ~24 trading days of hourly bars
    lookahead_hours: int = 24  # Max position length (within market hours)
    validation_hours: int = 70  # ~10 trading days (7 hours/day)
    min_history_hours: int = 7 * 60  # ~60 trading days
    max_feature_lookback_hours: int = 7 * 7  # ~7 trading days
    chronos_skip_rates: Tuple[int, ...] = (1, 2, 4)  # Multi-scale forecasting
    val_fraction: float = 0.15
    feature_columns: Optional[List[str]] = None
    refresh_hours: int = 24  # Refresh data daily (stocks update less frequently)

    # STOCK-SPECIFIC: Market hours filtering
    market_hours_only: bool = True  # Only include bars during NYSE hours
    market_open_hour: int = 9   # 9:30 AM ET
    market_open_minute: int = 30
    market_close_hour: int = 16  # 4:00 PM ET


@dataclass
class TrainingConfigStocksV5:
    """V5 stock training configuration with fee-aware simulation."""

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    sequence_length: int = 168  # ~24 trading days
    lookahead_hours: int = 24
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Patching
    patch_size: int = 4  # 4-hour patches

    # STOCK-SPECIFIC: Simulation parameters
    maker_fee: float = 0.0002  # 2 bps (retail stock broker)
    max_leverage: float = 2.0  # Stocks allow 2x margin
    forced_exit_slippage: float = 0.0005

    # Price constraints (CRITICAL for safety)
    min_price_offset_pct: float = 0.0002  # == maker_fee (dead zone)
    max_price_offset_pct: float = 0.02  # Max 2% from midpoint

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
    no_trade_penalty: float = 0.01  # Prevent always skipping
    spread_utilization: float = 0.05  # Encourage using offset range

    # EMA
    ema_decay: float = 0.0

    # Training settings
    dry_train_steps: Optional[int] = None
    device: Optional[str] = None
    run_name: str = "neuralhourlystocksv5"
    wandb_project: str = "neuralhourlystocksv5"
    wandb_entity: Optional[str] = None
    log_dir: str = "tensorboard_logs/neuralhourlystocksv5"
    checkpoint_root: str = "neuralhourlystocksv5/checkpoints"
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
    dataset: Optional[DatasetConfigStocksV5] = None

    def get_temperature_schedule(self) -> TemperatureScheduleV5:
        return TemperatureScheduleV5(
            initial_temp=self.initial_temperature,
            final_temp=self.final_temperature,
            warmup_epochs=self.temp_warmup_epochs,
            anneal_epochs=self.temp_anneal_epochs,
            anneal_type=self.temp_anneal_type,
        )

    def get_simulation_config(self) -> SimulationConfigStocksV5:
        return SimulationConfigStocksV5(
            maker_fee=self.maker_fee,
            max_leverage=self.max_leverage,
            forced_exit_slippage=self.forced_exit_slippage,
            max_position_hours=self.lookahead_hours,
        )

    def get_policy_config(self, input_dim: int) -> PolicyConfigStocksV5:
        return PolicyConfigStocksV5(
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
