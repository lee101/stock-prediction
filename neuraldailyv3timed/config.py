"""V3 Timed configuration with explicit exit timing.

Key changes from V2:
- Model outputs exit_days (1-10) for maximum hold duration
- Simulation runs as "trade episodes" with forced exits at deadline
- Training matches inference: exit at TP price OR deadline
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class TemperatureSchedule:
    """
    Temperature annealing from soft (differentiable) to hard (binary-like).

    Strategy: Start with high temperature for smooth gradients,
    anneal toward low temperature to match inference behavior.
    """

    initial_temp: float = 0.01  # Start: soft sigmoid
    final_temp: float = 0.0001  # End: near-binary
    warmup_epochs: int = 10  # Keep warm for initial exploration
    anneal_epochs: int = 150  # Linear anneal period
    anneal_type: str = "linear"  # "linear" or "cosine"

    def get_temperature(self, epoch: int, total_epochs: int = 200) -> float:
        """Get temperature for current epoch."""
        if epoch < self.warmup_epochs:
            return self.initial_temp

        progress = min(1.0, (epoch - self.warmup_epochs) / max(1, self.anneal_epochs))

        if self.anneal_type == "cosine":
            # Cosine annealing (slower at start and end)
            progress = 0.5 * (1 - math.cos(math.pi * progress))

        return self.initial_temp + (self.final_temp - self.initial_temp) * progress


@dataclass
class SimulationConfig:
    """Configuration for V3 trade episode simulation."""

    maker_fee: float = 0.0008  # 8 bps maker fee
    initial_cash: float = 1.0
    initial_inventory: float = 0.0
    equity_max_leverage: float = 2.0  # Stocks can use 2x margin
    crypto_max_leverage: float = 1.0  # Crypto no leverage
    leverage_fee_rate: float = 0.065  # 6.5% annual leverage cost

    # V3 Timed: Exit timing parameters
    max_hold_days: int = 20  # Maximum position hold time (model outputs 1-20)
    min_hold_days: int = 1   # Minimum hold (always at least 1 day)
    forced_exit_slippage: float = 0.001  # 10bps slippage on forced market exits
    use_trading_days: bool = True  # True for stocks (252/yr), False for crypto (365/yr)


@dataclass
class PolicyConfigV3:
    """Model architecture configuration for V3 transformer with exit timing."""

    input_dim: int = 20  # Number of input features
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    num_kv_heads: int = 4  # MQA: fewer KV heads than query heads
    dropout: float = 0.1
    max_len: int = 512
    equity_max_leverage: float = 2.0
    crypto_max_leverage: float = 1.0
    use_cross_attention: bool = True
    logits_softcap: float = 15.0  # Prevent extreme outputs
    price_offset_pct: float = 0.05  # 5% price range (more room for spreads)
    min_price_gap_pct: float = 0.005  # 0.5% minimum spread (covers 0.16% round-trip fees)

    # V3 Timed: Exit timing range
    min_exit_days: float = 1.0   # Minimum exit days
    max_exit_days: float = 20.0  # Maximum exit days

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def num_outputs(self) -> int:
        """V3 has 5 outputs: buy_price, sell_price, trade_amount, confidence, exit_days."""
        return 5


@dataclass
class DailyDatasetConfigV3:
    """Dataset configuration for V3 training with lookahead."""

    symbols: Tuple[str, ...] = (
        "SPY",
        "QQQ",
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "TSLA",
        "META",
        "BTCUSD",
        "ETHUSD",
    )
    data_root: Path = field(default_factory=lambda: Path("trainingdata/train"))
    forecast_cache_dir: Path = field(default_factory=lambda: Path("strategytraining/forecast_cache"))
    sequence_length: int = 256
    lookahead_days: int = 20  # V3: Need future data for trade episode simulation (20d max hold)
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
    # Grouping for cross-attention
    grouping_strategy: str = "correlation"  # "static" or "correlation"
    correlation_min_corr: float = 0.6
    correlation_max_group_size: int = 12
    correlation_window_days: int = 252
    correlation_min_overlap: int = 60
    split_crypto_groups: bool = True

    @property
    def total_length(self) -> int:
        """Total days needed per sample: history + lookahead."""
        return self.sequence_length + self.lookahead_days


@dataclass
class DailyTrainingConfigV3:
    """V3 training config with explicit exit timing."""

    # Training parameters
    epochs: int = 200
    batch_size: int = 32
    sequence_length: int = 256
    lookahead_days: int = 20  # V3: Future data for simulation (20d max hold)
    learning_rate: float = 0.0003  # Base LR (scaled per optimizer)
    weight_decay: float = 0.0  # No weight decay by default
    grad_clip: float = 1.0

    # Simulation parameters
    maker_fee: float = 0.0008
    return_weight: float = 0.08
    equity_max_leverage: float = 2.0
    crypto_max_leverage: float = 1.0
    leverage_fee_rate: float = 0.065
    steps_per_year: int = 252  # Trading days per year for stocks
    max_hold_days: int = 20    # V3: Maximum hold duration (20d)
    min_hold_days: int = 1     # V3: Minimum hold duration
    forced_exit_slippage: float = 0.001  # V3: Slippage on deadline exits

    # Model architecture
    transformer_dim: int = 256
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_kv_heads: int = 4  # MQA
    transformer_dropout: float = 0.1
    logits_softcap: float = 15.0
    price_offset_pct: float = 0.05  # 5% price range
    min_price_gap_pct: float = 0.005  # 0.5% minimum spread (covers fees)
    min_exit_days: float = 1.0   # V3: Exit days range
    max_exit_days: float = 20.0  # V3: Up to 20 days hold

    # Temperature annealing
    initial_temperature: float = 0.01
    final_temperature: float = 0.0001
    temp_warmup_epochs: int = 10
    temp_anneal_epochs: int = 150
    temp_anneal_type: str = "linear"

    # Dual optimizer settings (from nanochat)
    optimizer_name: str = "dual"  # "dual" (Muon+AdamW), "adamw", "muon"
    matrix_lr: float = 0.02  # Muon LR for matrix params
    embed_lr: float = 0.2  # AdamW LR for embeddings
    head_lr: float = 0.004  # AdamW LR for output head
    muon_momentum: float = 0.95
    adamw_betas: Tuple[float, float] = (0.8, 0.95)
    adamw_eps: float = 1e-10

    # Scheduler
    warmup_steps: int = 100
    use_cosine_schedule: bool = True

    # EMA
    ema_decay: float = 0.0  # 0 = disabled

    # V3 Loss weights
    forced_exit_penalty: float = 0.1  # Penalize trades that hit deadline
    risk_penalty: float = 0.05        # Penalize return variance
    hold_time_penalty: float = 0.02   # Small penalty for holding too long

    # Training settings
    dry_train_steps: Optional[int] = None
    device: Optional[str] = None
    run_name: str = "neuraldailyv3timed"
    wandb_project: str = "neuraldailyv3timed"
    wandb_entity: Optional[str] = None
    log_dir: str = "tensorboard_logs/neuraldailyv3timed"
    checkpoint_root: str = "neuraldailyv3timed/checkpoints"
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
    exposure_penalty: float = 0.0

    # Data augmentation
    permutation_rate: float = 0.5
    price_scale_range: Tuple[float, float] = (0.9, 1.1)
    price_scale_probability: float = 0.2

    # Dataset config (nested)
    dataset: Optional[DailyDatasetConfigV3] = None

    def get_temperature_schedule(self) -> TemperatureSchedule:
        """Create temperature schedule from config."""
        return TemperatureSchedule(
            initial_temp=self.initial_temperature,
            final_temp=self.final_temperature,
            warmup_epochs=self.temp_warmup_epochs,
            anneal_epochs=self.temp_anneal_epochs,
            anneal_type=self.temp_anneal_type,
        )

    def get_simulation_config(self) -> SimulationConfig:
        """Create simulation config from training config."""
        return SimulationConfig(
            maker_fee=self.maker_fee,
            equity_max_leverage=self.equity_max_leverage,
            crypto_max_leverage=self.crypto_max_leverage,
            leverage_fee_rate=self.leverage_fee_rate,
            max_hold_days=self.max_hold_days,
            min_hold_days=self.min_hold_days,
            forced_exit_slippage=self.forced_exit_slippage,
        )

    def get_policy_config(self, input_dim: int) -> PolicyConfigV3:
        """Create policy config from training config."""
        return PolicyConfigV3(
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
            price_offset_pct=self.price_offset_pct,
            min_price_gap_pct=self.min_price_gap_pct,
            min_exit_days=self.min_exit_days,
            max_exit_days=self.max_exit_days,
        )
