from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class FastForecaster2Config:
    """Configuration for FastForecaster2 training and evaluation."""

    data_dir: Path = Path("trainingdatahourly/stocks")
    output_dir: Path = Path("fastforecaster2") / "artifacts"
    symbols: Tuple[str, ...] | None = None
    max_symbols: int = 24

    lookback: int = 256
    horizon: int = 24
    train_stride: int = 1
    eval_stride: int = 4
    val_fraction: float = 0.15
    test_fraction: float = 0.10
    min_rows_per_symbol: int = 1024
    max_train_windows_per_symbol: int | None = 80_000
    max_eval_windows_per_symbol: int | None = 10_000

    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    grad_accum_steps: int = 1
    warmup_steps: int = 200
    log_interval: int = 50
    early_stopping_patience: int = 8
    return_loss_weight: float = 0.20
    direction_loss_weight: float = 0.02
    direction_margin_scale: float = 16.0
    horizon_weight_power: float = 0.35
    use_ema_eval: bool = True
    ema_decay: float = 0.999

    hidden_dim: int = 384
    num_layers: int = 8
    num_heads: int = 8
    ff_multiplier: int = 4
    dropout: float = 0.05
    qk_norm: bool = True
    qk_norm_eps: float = 1e-6

    precision: str = "bf16"
    torch_compile: bool = True
    compile_mode: str = "max-autotune"
    use_fused_optimizer: bool = True
    use_cpp_kernels: bool = False
    build_cpp_extension: bool = False

    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 1337
    device: str | None = None

    feature_columns: Tuple[str, ...] = ("open", "high", "low", "close", "volume", "vwap")
    target_column: str = "close"
    include_time_features: bool = True

    chronos_embeddings_path: Path | None = None
    chronos_embeddings_blend: float = 0.0

    use_market_sim_eval: bool = True
    market_sim_fee: float = 0.0008
    market_sim_initial_cash: float = 10_000.0
    market_sim_max_hold_hours: int | None = None
    market_sim_buy_threshold: float = 0.001
    market_sim_sell_threshold: float = 0.001
    market_sim_entry_score_threshold: float = 0.0
    market_sim_max_trade_intensity: float = 35.0
    market_sim_min_trade_intensity: float = 2.0
    market_sim_top_k: int = 3
    market_sim_signal_ema_alpha: float = 0.25
    market_sim_switch_score_gap: float = 0.0
    market_sim_entry_buffer_bps: float = 1.0
    market_sim_exit_buffer_bps: float = 1.0
    market_sim_take_profit_scale: float = 0.65
    market_sim_stop_loss_scale: float = 0.65
    market_sim_vol_lookback: int = 32
    market_sim_vol_target: float = 0.025

    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_tags: Tuple[str, ...] = ()

    _checkpoint_dir: Path = field(init=False, repr=False)
    _metrics_dir: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        if self.chronos_embeddings_path is not None:
            self.chronos_embeddings_path = Path(self.chronos_embeddings_path)

        self._checkpoint_dir = self.output_dir / "checkpoints"
        self._metrics_dir = self.output_dir / "metrics"

        if self.symbols is not None:
            unique_symbols = sorted({symbol.strip().upper() for symbol in self.symbols if symbol.strip()})
            self.symbols = tuple(unique_symbols)

        if isinstance(self.wandb_tags, list):
            self.wandb_tags = tuple(self.wandb_tags)
        if self.wandb_tags:
            self.wandb_tags = tuple(sorted({tag.strip() for tag in self.wandb_tags if tag.strip()}))

        if self.lookback <= 0:
            raise ValueError("lookback must be > 0")
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if self.train_stride <= 0 or self.eval_stride <= 0:
            raise ValueError("train_stride and eval_stride must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.min_learning_rate <= 0:
            raise ValueError("min_learning_rate must be > 0")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate")
        if not (0.0 <= self.weight_decay <= 1.0):
            raise ValueError("weight_decay must be in [0, 1]")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be > 0")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1")
        if self.return_loss_weight < 0:
            raise ValueError("return_loss_weight must be >= 0")
        if self.direction_loss_weight < 0:
            raise ValueError("direction_loss_weight must be >= 0")
        if self.direction_margin_scale <= 0:
            raise ValueError("direction_margin_scale must be > 0")
        if self.horizon_weight_power < 0:
            raise ValueError("horizon_weight_power must be >= 0")
        if not (0.0 < self.ema_decay < 1.0):
            raise ValueError("ema_decay must be in (0, 1)")
        if self.max_symbols < 0:
            raise ValueError("max_symbols must be >= 0")
        if self.min_rows_per_symbol <= (self.lookback + self.horizon):
            raise ValueError("min_rows_per_symbol must exceed lookback + horizon")
        if not (0.0 < self.val_fraction < 0.45):
            raise ValueError("val_fraction must be in (0, 0.45)")
        if not (0.0 < self.test_fraction < 0.45):
            raise ValueError("test_fraction must be in (0, 0.45)")
        if (self.val_fraction + self.test_fraction) >= 0.9:
            raise ValueError("val_fraction + test_fraction must be < 0.9")

        precision = self.precision.lower()
        if precision not in {"bf16", "fp16", "fp32"}:
            raise ValueError("precision must be one of {'bf16', 'fp16', 'fp32'}")
        self.precision = precision
        if self.qk_norm_eps <= 0:
            raise ValueError("qk_norm_eps must be > 0")
        if not (0.0 <= self.chronos_embeddings_blend <= 1.0):
            raise ValueError("chronos_embeddings_blend must be in [0, 1]")
        if self.market_sim_fee < 0:
            raise ValueError("market_sim_fee must be >= 0")
        if self.market_sim_initial_cash <= 0:
            raise ValueError("market_sim_initial_cash must be > 0")
        if self.market_sim_max_hold_hours is not None and self.market_sim_max_hold_hours <= 0:
            raise ValueError("market_sim_max_hold_hours must be > 0 when provided")
        if self.market_sim_buy_threshold < 0:
            raise ValueError("market_sim_buy_threshold must be >= 0")
        if self.market_sim_sell_threshold < 0:
            raise ValueError("market_sim_sell_threshold must be >= 0")
        if self.market_sim_entry_score_threshold < 0:
            raise ValueError("market_sim_entry_score_threshold must be >= 0")
        if not (0.0 < self.market_sim_max_trade_intensity <= 100.0):
            raise ValueError("market_sim_max_trade_intensity must be in (0, 100]")
        if not (0.0 <= self.market_sim_min_trade_intensity <= self.market_sim_max_trade_intensity):
            raise ValueError("market_sim_min_trade_intensity must be in [0, market_sim_max_trade_intensity]")
        if self.market_sim_top_k <= 0:
            raise ValueError("market_sim_top_k must be > 0")
        if not (0.0 < self.market_sim_signal_ema_alpha <= 1.0):
            raise ValueError("market_sim_signal_ema_alpha must be in (0, 1]")
        if self.market_sim_switch_score_gap < 0:
            raise ValueError("market_sim_switch_score_gap must be >= 0")
        if self.market_sim_entry_buffer_bps < 0:
            raise ValueError("market_sim_entry_buffer_bps must be >= 0")
        if self.market_sim_exit_buffer_bps < 0:
            raise ValueError("market_sim_exit_buffer_bps must be >= 0")
        if self.market_sim_take_profit_scale <= 0:
            raise ValueError("market_sim_take_profit_scale must be > 0")
        if self.market_sim_stop_loss_scale <= 0:
            raise ValueError("market_sim_stop_loss_scale must be > 0")
        if self.market_sim_vol_lookback < 2:
            raise ValueError("market_sim_vol_lookback must be >= 2")
        if self.market_sim_vol_target <= 0:
            raise ValueError("market_sim_vol_target must be > 0")

    @property
    def checkpoint_dir(self) -> Path:
        return self._checkpoint_dir

    @property
    def metrics_dir(self) -> Path:
        return self._metrics_dir

    @property
    def metrics_file(self) -> Path:
        return self.metrics_dir / "summary.json"

    @property
    def simulator_metrics_file(self) -> Path:
        return self.metrics_dir / "simulator_summary.json"

    @property
    def simulator_equity_file(self) -> Path:
        return self.metrics_dir / "simulator_equity.csv"

    @property
    def simulator_actions_file(self) -> Path:
        return self.metrics_dir / "simulator_actions.csv"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "best.pt"

    @property
    def last_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "last.pt"

    def ensure_output_dirs(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def resolved_device(self) -> str:
        if self.device:
            return self.device

        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def as_dict(self) -> Dict[str, object]:
        return {
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "symbols": list(self.symbols) if self.symbols else None,
            "max_symbols": self.max_symbols,
            "lookback": self.lookback,
            "horizon": self.horizon,
            "train_stride": self.train_stride,
            "eval_stride": self.eval_stride,
            "val_fraction": self.val_fraction,
            "test_fraction": self.test_fraction,
            "min_rows_per_symbol": self.min_rows_per_symbol,
            "max_train_windows_per_symbol": self.max_train_windows_per_symbol,
            "max_eval_windows_per_symbol": self.max_eval_windows_per_symbol,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "weight_decay": self.weight_decay,
            "grad_clip_norm": self.grad_clip_norm,
            "grad_accum_steps": self.grad_accum_steps,
            "warmup_steps": self.warmup_steps,
            "log_interval": self.log_interval,
            "early_stopping_patience": self.early_stopping_patience,
            "return_loss_weight": self.return_loss_weight,
            "direction_loss_weight": self.direction_loss_weight,
            "direction_margin_scale": self.direction_margin_scale,
            "horizon_weight_power": self.horizon_weight_power,
            "use_ema_eval": self.use_ema_eval,
            "ema_decay": self.ema_decay,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ff_multiplier": self.ff_multiplier,
            "dropout": self.dropout,
            "qk_norm": self.qk_norm,
            "qk_norm_eps": self.qk_norm_eps,
            "precision": self.precision,
            "torch_compile": self.torch_compile,
            "compile_mode": self.compile_mode,
            "use_fused_optimizer": self.use_fused_optimizer,
            "use_cpp_kernels": self.use_cpp_kernels,
            "build_cpp_extension": self.build_cpp_extension,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "seed": self.seed,
            "device": self.resolved_device(),
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "include_time_features": self.include_time_features,
            "chronos_embeddings_path": str(self.chronos_embeddings_path) if self.chronos_embeddings_path else None,
            "chronos_embeddings_blend": self.chronos_embeddings_blend,
            "use_market_sim_eval": self.use_market_sim_eval,
            "market_sim_fee": self.market_sim_fee,
            "market_sim_initial_cash": self.market_sim_initial_cash,
            "market_sim_max_hold_hours": self.market_sim_max_hold_hours,
            "market_sim_buy_threshold": self.market_sim_buy_threshold,
            "market_sim_sell_threshold": self.market_sim_sell_threshold,
            "market_sim_entry_score_threshold": self.market_sim_entry_score_threshold,
            "market_sim_max_trade_intensity": self.market_sim_max_trade_intensity,
            "market_sim_min_trade_intensity": self.market_sim_min_trade_intensity,
            "market_sim_top_k": self.market_sim_top_k,
            "market_sim_signal_ema_alpha": self.market_sim_signal_ema_alpha,
            "market_sim_switch_score_gap": self.market_sim_switch_score_gap,
            "market_sim_entry_buffer_bps": self.market_sim_entry_buffer_bps,
            "market_sim_exit_buffer_bps": self.market_sim_exit_buffer_bps,
            "market_sim_take_profit_scale": self.market_sim_take_profit_scale,
            "market_sim_stop_loss_scale": self.market_sim_stop_loss_scale,
            "market_sim_vol_lookback": self.market_sim_vol_lookback,
            "market_sim_vol_target": self.market_sim_vol_target,
            "wandb_project": self.wandb_project,
            "wandb_run_name": self.wandb_run_name,
            "wandb_entity": self.wandb_entity,
            "wandb_group": self.wandb_group,
            "wandb_tags": list(self.wandb_tags),
        }
