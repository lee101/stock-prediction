from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple

from differentiable_loss_utils import DEFAULT_MAKER_FEE_RATE


@dataclass
class ForecastConfig:
    """Configuration for Chronos2 OHLC forecasts."""

    symbol: str = "BTCUSD"
    data_root: Path = Path("trainingdatahourly") / "crypto"
    context_hours: int = 24 * 14
    prediction_horizon_hours: int = 1
    quantile_levels: Tuple[float, ...] = (0.1, 0.5, 0.9)
    batch_size: int = 128
    cache_dir: Path = Path("binanceneural") / "forecast_cache"


@dataclass
class DatasetConfig:
    """Dataset preparation parameters for Binance hourly sequences."""

    symbol: str = "BTCUSD"
    data_root: Path = Path("trainingdatahourly") / "crypto"
    forecast_cache_root: Path = Path("binanceneural") / "forecast_cache"
    forecast_horizons: Tuple[int, ...] = (1, 24)
    sequence_length: int = 72
    val_fraction: float = 0.15
    min_history_hours: int = 24 * 30
    max_feature_lookback_hours: int = 24 * 7
    feature_columns: Optional[Sequence[str]] = None
    refresh_hours: int = 0
    validation_days: int = 70
    cache_only: bool = False


@dataclass
class PolicyConfig:
    """Transformer policy head configuration."""

    input_dim: int
    hidden_dim: int = 256
    depth: int = 3
    dropout: float = 0.1
    model_arch: str = "classic"  # "classic" or "nano"
    price_offset_pct: float = 0.0003
    min_price_gap_pct: float = 0.0003
    trade_amount_scale: float = 100.0
    num_heads: int = 8
    num_kv_heads: Optional[int] = None
    num_layers: int = 4
    max_len: int = 2048
    use_midpoint_offsets: bool = True
    mlp_ratio: float = 4.0
    logits_softcap: float = 12.0
    rope_base: float = 10000.0
    use_qk_norm: bool = True
    use_causal_attention: bool = True
    rms_norm_eps: float = 1e-5
    attention_window: Optional[int] = None
    use_residual_scalars: bool = False
    residual_scale_init: float = 1.0
    skip_scale_init: float = 0.0
    use_value_embedding: bool = False
    value_embedding_every: int = 2
    value_embedding_scale: float = 1.0


@dataclass
class TrainingConfig:
    """Training configuration for Binance hourly policy."""

    epochs: int = 60
    batch_size: int = 16
    sequence_length: int = 72
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    maker_fee: float = 0.0
    return_weight: float = 0.08
    periods_per_year: Optional[float] = None
    price_offset_pct: float = 0.0003
    min_price_gap_pct: float = 0.0003
    trade_amount_scale: float = 100.0
    initial_cash: float = 1.0
    sortino_target_sign: float = 1.0
    transformer_dim: int = 256
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_dropout: float = 0.1
    optimizer_name: str = "adamw"
    model_arch: str = "classic"
    num_kv_heads: Optional[int] = None
    mlp_ratio: float = 4.0
    logits_softcap: float = 12.0
    rope_base: float = 10000.0
    use_qk_norm: bool = True
    use_causal_attention: bool = True
    rms_norm_eps: float = 1e-5
    attention_window: Optional[int] = None
    use_residual_scalars: bool = False
    residual_scale_init: float = 1.0
    skip_scale_init: float = 0.0
    use_value_embedding: bool = False
    value_embedding_every: int = 2
    value_embedding_scale: float = 1.0
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_momentum_start: Optional[float] = None
    muon_momentum_warmup_steps: int = 300
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    warmup_steps: int = 100
    weight_decay_schedule: str = "none"
    weight_decay_end: float = 0.0
    ema_decay: float = 0.0
    dry_train_steps: Optional[int] = None
    device: Optional[str] = None
    run_name: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    log_dir: Path = Path("tensorboard_logs") / "binanceneural"
    checkpoint_root: Path = Path("binanceneural") / "checkpoints"
    top_k_checkpoints: int = 50
    preload_checkpoint_path: Optional[Path] = None
    force_retrain: bool = False
    seed: int = 1337
    num_workers: int = 0
    use_compile: bool = True
    use_amp: bool = False
    amp_dtype: str = "bfloat16"
    use_tf32: bool = True
    use_flash_attention: bool = True
    forecast_config: ForecastConfig = field(default_factory=ForecastConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


__all__ = [
    "DatasetConfig",
    "ForecastConfig",
    "PolicyConfig",
    "TrainingConfig",
]
