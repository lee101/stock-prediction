from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple

from differentiable_loss_utils import DEFAULT_MAKER_FEE_RATE
from strategytraining.symbol_sources import load_trade_stock_symbols


_FALLBACK_SYMBOLS: Tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "NVDA",
    "SPY",
    "QQQ",
    "BTCUSD",
    "ETHUSD",
    "LINKUSD",
)


def _default_symbol_tuple() -> Tuple[str, ...]:
    try:
        symbols = load_trade_stock_symbols("trade_stock_e2e.py")
    except Exception:
        return _FALLBACK_SYMBOLS
    cleaned = tuple(symbols) if symbols else _FALLBACK_SYMBOLS
    return cleaned


@dataclass
class DailyDatasetConfig:
    """Dataset preparation parameters for multi-asset daily training."""

    symbols: Tuple[str, ...] = field(default_factory=_default_symbol_tuple)
    data_root: Path = Path("trainingdata") / "train"
    forecast_cache_dir: Path = Path("strategytraining") / "forecast_cache"
    sequence_length: int = 256
    val_fraction: float = 0.2
    min_history_days: int = 300
    feature_columns: Optional[Sequence[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    validation_days: int = 90


@dataclass
class DailyTrainingConfig:
    """Transformer policy hyperparameters for the daily neural model."""

    epochs: int = 200
    batch_size: int = 32
    sequence_length: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    maker_fee: float = DEFAULT_MAKER_FEE_RATE
    return_weight: float = 0.08
    transformer_dim: int = 256
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_dropout: float = 0.1
    price_offset_pct: float = 0.025
    max_trade_qty: float = 3.0
    min_price_gap_pct: float = 0.0005
    initial_cash: float = 1.0
    optimizer_name: str = "muon"
    warmup_steps: int = 100
    ema_decay: float = 0.0
    dry_train_steps: Optional[int] = None
    device: Optional[str] = None
    run_name: Optional[str] = None
    wandb_project: Optional[str] = "neuraldailytraining"
    wandb_entity: Optional[str] = None
    log_dir: Path = Path("tensorboard_logs") / "neuraldaily"
    checkpoint_root: Path = Path("neuraldailytraining") / "checkpoints"
    top_k_checkpoints: int = 50
    preload_checkpoint_path: Optional[Path] = None
    force_retrain: bool = False
    seed: int = 1337
    num_workers: int = 0
    use_compile: bool = True
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    use_tf32: bool = True
    risk_threshold: float = 1.0
    exposure_penalty: float = 0.0
    equity_max_leverage: float = 2.0
    crypto_max_leverage: float = 1.0
    leverage_fee_rate: float = 0.065
    steps_per_year: int = 252
    dataset: DailyDatasetConfig = field(default_factory=DailyDatasetConfig)

    def __post_init__(self) -> None:
        self.sequence_length = max(8, int(self.sequence_length))
        self.dataset.sequence_length = self.sequence_length
