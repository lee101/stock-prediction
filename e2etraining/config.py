from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class E2EDataConfig:
    data_root: Path = Path("trainingdata")
    universe_file: Path = Path("available_stocks_with_data.json")
    include_symbols: tuple[str, ...] = field(default_factory=tuple)
    exclude_symbols: tuple[str, ...] = field(default_factory=tuple)
    max_assets: int = 64
    min_timesteps: int = 1024
    train_ratio: float = 0.8
    cache_dir: Path | None = Path(".cache") / "e2etraining"


@dataclass(slots=True)
class E2EModelConfig:
    model_id: str = "amazon/chronos-2"
    context_length: int = 128
    prediction_length: int = 1
    cross_learning: bool = True
    include_cash: bool = True
    policy_hidden_dim: int = 64
    lora_enabled: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = ("q", "k", "v", "o")


@dataclass(slots=True)
class E2ETrainConfig:
    run_name: str | None = None
    save_root: Path = Path("e2etraining") / "runs"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    seed: int = 0
    steps: int = 200
    eval_every: int = 20
    rollout_length: int = 16
    batch_size: int = 4
    learning_rate: float = 1e-4
    backbone_learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    transaction_cost_bps: float = 5.0
    forecast_loss_weight: float = 0.25
    sortino_weight: float = 0.10
    drawdown_weight: float = 0.05
    grad_clip: float = 1.0
    log_every: int = 5
