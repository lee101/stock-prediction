from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(slots=True)
class DataConfig:
    """Configuration for loading OHLC data used during training and evaluation."""

    root: Path = Path("trainingdata")
    glob: str = "*.csv"
    max_assets: int | None = None
    cache_dir: Path | None = None
    normalize: Literal["standard", "log", "none"] = "log"
    # Exclude symbols explicitly if they should never appear in train/eval splits.
    exclude_symbols: tuple[str, ...] = field(default_factory=tuple)
    min_timesteps: int = 512
    include_cash: bool = False


@dataclass(slots=True)
class EnvironmentConfig:
    """Differentiable market environment hyper-parameters."""

    transaction_cost: float = 1e-3
    risk_aversion: float = 0.1
    variance_penalty_mode: Literal["pnl", "weights"] = "pnl"
    smooth_abs_eps: float = 1e-6
    wealth_objective: Literal["log", "sharpe"] = "log"
    sharpe_ema_alpha: float = 0.01
    epsilon_stability: float = 1e-8
    drawdown_lambda: float = 0.0


@dataclass(slots=True)
class TrainingConfig:
    """Training hyper-parameters for the GRPO loop."""

    lookback: int = 128
    rollout_groups: int = 4
    batch_windows: int = 64
    microbatch_windows: int | None = None
    epochs: int = 2000
    eval_interval: int = 100
    device: Literal["auto", "cpu", "cuda"] = "auto"
    dtype: Literal["auto", "bfloat16", "float32"] = "auto"
    grad_clip: float = 1.0
    entropy_coef: float = 1e-3
    kl_coef: float = 0.1
    lr_muon: float = 2e-2
    lr_adamw: float = 3e-4
    weight_decay: float = 1e-2
    use_muon: bool = True
    use_compile: bool = True
    seed: int = 0
    torch_compile_mode: str = "reduce-overhead"
    gradient_checkpointing: bool = False
    bf16_autocast: bool = True
    save_dir: Path = Path("differentiable_market") / "runs"
    max_eval_windows: int | None = None
    resume: bool = False
    include_cash: bool = False
    init_checkpoint: Path | None = None
    best_k_checkpoints: int = 3
    use_wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_tags: tuple[str, ...] = ()
    wandb_group: str | None = None
    wandb_notes: str | None = None
    wandb_mode: str = "auto"
    wandb_run_name: str | None = None
    wandb_settings: dict[str, Any] = field(default_factory=dict)
    wandb_log_metrics: bool = False
    wandb_metric_log_level: str = "DEBUG"
    tensorboard_root: Path | None = Path("tensorboard_logs")
    tensorboard_subdir: str | None = None
    soft_drawdown_lambda: float = 0.0
    risk_budget_lambda: float = 0.0
    risk_budget_target: tuple[float, ...] = ()
    trade_memory_lambda: float = 0.0
    trade_memory_ema_decay: float = 0.95
    use_taylor_features: bool = False
    taylor_order: int = 4
    taylor_scale: float = 32.0
    use_wavelet_features: bool = False
    wavelet_levels: int = 1
    wavelet_padding_mode: Literal["reflect", "replicate", "constant"] = "reflect"


@dataclass(slots=True)
class EvaluationConfig:
    """Configuration for evaluation / backtesting."""

    window_length: int = 256
    stride: int = 64
    metric: Literal["return", "sharpe"] = "sharpe"
    report_dir: Path = Path("differentiable_market") / "evals"
    store_trades: bool = True
    bootstrap_samples: int = 0
