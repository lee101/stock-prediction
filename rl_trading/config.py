from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class EnvConfig:
    symbols: List[str] = field(default_factory=lambda: ["BTCUSD", "ETHUSD", "SOLUSD"])
    data_root: Path = Path("trainingdatahourly/crypto")
    initial_cash: float = 10000.0
    fee_rate: float = 0.0
    max_hold_bars: int = 6
    episode_length: int = 168
    validation_days: float = 70.0
    num_envs: int = 256


@dataclass
class TrainConfig:
    total_timesteps: int = 10_000_000
    batch_size: int = 4096
    bptt_horizon: int = 64
    minibatch_size: int = 2048
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    device: str = "cuda"
    seed: int = 42
    hidden_dim: int = 256
    checkpoint_dir: Path = Path("rl_trading/checkpoints")
