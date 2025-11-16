from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple

from differentiable_loss_utils import DEFAULT_MAKER_FEE_RATE


@dataclass
class ForecastConfig:
    """Configuration for precomputing Chronos2 hourly forecasts."""

    symbol: str = "LINKUSD"
    data_root: Path = Path("trainingdatahourly") / "crypto"
    context_hours: int = 24 * 14
    prediction_horizon_hours: int = 1
    quantile_levels: Tuple[float, ...] = (0.1, 0.5, 0.9)
    batch_size: int = 128
    cache_dir: Path = Path("hourlycryptotraining") / "forecast_cache_hourly"


@dataclass
class DatasetConfig:
    """Dataset preparation parameters for LINKUSD hourly sequences."""

    symbol: str = "LINKUSD"
    data_root: Path = Path("trainingdatahourly") / "crypto"
    forecast_cache_dir: Path = Path("hourlycryptotraining") / "forecast_cache_hourly"
    sequence_length: int = 72
    val_fraction: float = 0.15
    min_history_hours: int = 24 * 30
    max_feature_lookback_hours: int = 24 * 7
    feature_columns: Optional[Sequence[str]] = None
    refresh_hours: int = 72
    validation_days: int = 70


@dataclass
class TrainingConfig:
    """Hyperparameters for the hourly crypto neural policy."""

    epochs: int = 75
    batch_size: int = 16
    sequence_length: int = 72
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    maker_fee: float = DEFAULT_MAKER_FEE_RATE
    return_weight: float = 0.08
    price_offset_pct: float = 0.03
    min_price_gap_pct: float = 0.0003
    max_trade_qty: float = 3.0
    initial_cash: float = 1.0
    sortino_target_sign: float = 1.0
    transformer_dim: int = 256
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_dropout: float = 0.1
    optimizer_name: str = "muon"
    warmup_steps: int = 100
    ema_decay: float = 0.0
    dry_train_steps: Optional[int] = None
    device: Optional[str] = None
    run_name: Optional[str] = None
    wandb_project: Optional[str] = "hourly-crypto"
    wandb_entity: Optional[str] = None
    log_dir: Path = Path("tensorboard_logs") / "hourlycrypto"
    checkpoint_root: Path = Path("hourlycryptotraining") / "checkpoints"
    top_k_checkpoints: int = 100
    preload_checkpoint_path: Optional[Path] = None
    force_retrain: bool = False
    seed: int = 1337
    num_workers: int = 0
    price_smoothing_hours: int = 4
    use_compile: bool = True
    use_amp: bool = False
    amp_dtype: str = "bfloat16"
    use_tf32: bool = True
    forecast_config: ForecastConfig = field(default_factory=ForecastConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
