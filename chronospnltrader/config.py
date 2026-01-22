"""Configuration for ChronosPnL Trader.

Key innovation: Uses Chronos2 both for price forecasting AND as a "judge"
to predict whether the strategy's PnL will be profitable tomorrow.
The neural model is trained to maximize predicted next-day profitability.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class DataConfig:
    """Configuration for hourly data loading."""

    data_root: Path = field(default_factory=lambda: Path("trainingdatahourly/stocks"))
    forecast_cache_dir: Path = field(
        default_factory=lambda: Path("chronospnltrader/forecast_cache")
    )

    # Sequence configuration
    sequence_length: int = 168  # ~24 trading days of hourly bars (7 hrs/day)
    lookahead_hours: int = 24  # Max hours to hold
    pnl_history_days: int = 30  # Days of PnL history to track
    pnl_history_hours: int = 210  # ~30 trading days * 7 hours/day

    # Validation
    validation_hours: int = 70  # ~10 trading days
    min_history_hours: int = 420  # ~60 trading days minimum
    val_fraction: float = 0.15

    # Market hours filtering
    market_hours_only: bool = True
    market_open_hour: int = 9
    market_open_minute: int = 30
    market_close_hour: int = 16

    # Symbols
    symbols: Optional[Tuple[str, ...]] = None

    def get_all_symbols(self) -> List[str]:
        """Get all available stock symbols from data directory."""
        if not self.data_root.exists():
            return []
        return sorted([f.stem for f in self.data_root.glob("*.csv")])


@dataclass
class ForecastConfig:
    """Configuration for Chronos2 forecasting."""

    model_id: str = "amazon/chronos-2"
    device_map: str = "cuda"
    context_length: int = 168  # Hourly context
    prediction_length: int = 7  # Predict 7 hours ahead (1 trading day)
    quantile_levels: Tuple[float, ...] = (0.1, 0.5, 0.9)
    batch_size: int = 24  # Smaller for hourly

    # Multi-scale forecasting
    skip_rates: Tuple[int, ...] = (1, 2, 4)  # 1h, 2h, 4h forecasts


@dataclass
class SimulationConfig:
    """Configuration for hourly market simulation."""

    # Fee structure (stock retail)
    maker_fee: float = 0.0002  # 2 bps per leg
    taker_fee: float = 0.0003  # 3 bps
    forced_exit_slippage: float = 0.0005  # 5 bps slippage

    # Capital
    initial_cash: float = 1.0  # Normalized
    max_leverage: float = 2.0

    # Position constraints
    max_position_hours: int = 24
    min_position_size: float = 0.0
    max_position_size: float = 1.0

    # Price constraints
    min_price_offset_pct: float = 0.0002  # == maker_fee (dead zone)
    max_price_offset_pct: float = 0.02  # Max 2% from reference


@dataclass
class SimpleAlgoConfig:
    """Configuration for the simple Chronos2-based algorithm."""

    # Entry: use Chronos predicted low with safety buffer
    entry_buffer_pct: float = 0.001  # 10 bps below predicted low

    # Exit: use Chronos predicted high with safety buffer
    exit_buffer_pct: float = 0.001  # 10 bps below predicted high

    # Position sizing based on forecast confidence
    confidence_threshold: float = 0.5  # Min forecast confidence
    position_scale: float = 1.0  # Scale position by confidence

    # PnL forecasting
    pnl_forecast_horizon: int = 7  # Forecast next 7 hours of PnL
    pnl_min_profit_threshold: float = 0.0001  # 1 bp minimum predicted profit

    # Trade filtering
    min_predicted_return: float = 0.001  # 10 bps minimum expected return
    max_predicted_drawdown: float = 0.02  # 2% max predicted drawdown


@dataclass
class PolicyConfig:
    """Model architecture configuration for neural price optimizer."""

    input_dim: int = 25  # Features + PnL history
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 4
    num_kv_heads: int = 2
    dropout: float = 0.1
    max_len: int = 256

    # Patching
    patch_size: int = 4  # 4-hour patches
    use_patch_embedding: bool = True
    patch_residual: bool = True

    # Output constraints (fee-aware)
    maker_fee: float = 0.0002
    min_price_offset_pct: float = 0.0002
    max_price_offset_pct: float = 0.02

    # Position outputs
    min_position: float = 0.0
    max_position: float = 1.0
    max_position_hours: int = 24
    num_position_classes: int = 25  # 0=skip, 1-24=hours

    logits_softcap: float = 15.0

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads


@dataclass
class TemperatureSchedule:
    """Temperature annealing for differentiable simulation."""

    initial_temp: float = 0.02
    final_temp: float = 0.001
    warmup_epochs: int = 5
    anneal_epochs: int = 50
    anneal_type: str = "linear"  # "linear" or "cosine"

    def get_temperature(self, epoch: int, total_epochs: int = 100) -> float:
        if epoch < self.warmup_epochs:
            return self.initial_temp

        progress = min(1.0, (epoch - self.warmup_epochs) / max(1, self.anneal_epochs))

        if self.anneal_type == "cosine":
            progress = 0.5 * (1 - math.cos(math.pi * progress))

        return self.initial_temp + (self.final_temp - self.initial_temp) * progress


@dataclass
class TrainingConfig:
    """Full training configuration."""

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Dataset
    sequence_length: int = 168
    lookahead_hours: int = 24
    pnl_history_hours: int = 210  # 30 days of PnL

    # Simulation
    maker_fee: float = 0.0002
    forced_exit_slippage: float = 0.0005

    # Price constraints
    min_price_offset_pct: float = 0.0002
    max_price_offset_pct: float = 0.02

    # Model architecture
    transformer_dim: int = 256
    transformer_layers: int = 3
    transformer_heads: int = 4
    transformer_kv_heads: int = 2
    transformer_dropout: float = 0.1
    logits_softcap: float = 15.0
    patch_size: int = 4

    # Temperature annealing
    initial_temperature: float = 0.02
    final_temperature: float = 0.001
    temp_warmup_epochs: int = 5
    temp_anneal_epochs: int = 50
    temp_anneal_type: str = "linear"

    # Loss weights
    sortino_weight: float = 0.5  # Standard Sortino
    pnl_forecast_weight: float = 1.0  # Chronos2 predicted profitability
    return_weight: float = 0.1
    forced_exit_penalty: float = 0.2
    no_trade_penalty: float = 0.01
    spread_utilization: float = 0.05

    # Optimizer
    optimizer_name: str = "adamw"
    adamw_betas: Tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-8
    warmup_steps: int = 100
    use_cosine_schedule: bool = True

    # Training settings
    device: Optional[str] = None
    run_name: str = "chronospnltrader"
    wandb_project: str = "chronospnltrader"
    wandb_entity: Optional[str] = None
    log_dir: str = "tensorboard_logs/chronospnltrader"
    checkpoint_root: str = "chronospnltrader/checkpoints"
    top_k_checkpoints: int = 10
    preload_checkpoint_path: Optional[str] = None
    seed: int = 1337
    num_workers: int = 0
    use_compile: bool = False  # Disable for simpler debugging
    use_amp: bool = False
    amp_dtype: str = "bfloat16"
    use_tf32: bool = True

    # Dataset config
    data: Optional[DataConfig] = None
    forecast: Optional[ForecastConfig] = None
    simulation: Optional[SimulationConfig] = None
    simple_algo: Optional[SimpleAlgoConfig] = None

    def get_temperature_schedule(self) -> TemperatureSchedule:
        return TemperatureSchedule(
            initial_temp=self.initial_temperature,
            final_temp=self.final_temperature,
            warmup_epochs=self.temp_warmup_epochs,
            anneal_epochs=self.temp_anneal_epochs,
            anneal_type=self.temp_anneal_type,
        )

    def get_simulation_config(self) -> SimulationConfig:
        return SimulationConfig(
            maker_fee=self.maker_fee,
            forced_exit_slippage=self.forced_exit_slippage,
            min_price_offset_pct=self.min_price_offset_pct,
            max_price_offset_pct=self.max_price_offset_pct,
        )

    def get_policy_config(self, input_dim: int) -> PolicyConfig:
        return PolicyConfig(
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
            logits_softcap=self.logits_softcap,
        )
