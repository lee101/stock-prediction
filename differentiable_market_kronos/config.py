from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(slots=True)
class KronosFeatureConfig:
    model_path: str = "NeoQuasar/Kronos-base"
    tokenizer_path: str = "NeoQuasar/Kronos-Tokenizer-base"
    context_length: int = 512
    horizons: Tuple[int, ...] = (1, 12, 48)
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)
    include_path_stats: bool = True
    device: str = "auto"
    sample_count: int = 16
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0
    clip: float = 2.0
    bf16: bool = True


@dataclass(slots=True)
class KronosConfig:
    model_id: str = "NeoQuasar/Kronos-base"
    tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base"
    max_context: int = 512
    device: str = "cuda"
    sample_count: int = 16
    temperature: float = 1.0
    top_p: float = 0.9
    include_volume: bool = True


@dataclass(slots=True)
class EnvConfig:
    lookback: int = 512
    pred_horizon: int = 48
    initial_cash: float = 1_000_000.0
    max_position: float = 1.0
    transaction_cost_bps: float = 1.0
    slippage_bps: float = 0.5
    reward: str = "pnl"
    hold_penalty: float = 0.0
    seed: int = 42


@dataclass(slots=True)
class TrainConfig:
    total_timesteps: int = 2_000_000
    n_envs: int = 8
    rollout_steps: int = 2048
    batch_size: int = 4096
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    bf16: bool = True
    log_dir: str = "runs/differentiable_market_kronos"
    run_name: str = "ppo_kronos_base"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    save_freq_steps: int = 100_000


@dataclass(slots=True)
class DataConfig:
    path: str = "data/ohlcv.csv"
    timestamp_col: str = "timestamp"
    price_col: str = "close"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    volume_col: str = "volume"
    amount_col: str = "amount"
    freq: Optional[str] = None


@dataclass(slots=True)
class ExperimentConfig:
    kronos: KronosConfig = field(default_factory=KronosConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
