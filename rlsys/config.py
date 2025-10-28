"""Configuration dataclasses for the RL trading system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple


@dataclass(slots=True)
class DataConfig:
    """Configuration describing how price data should be prepared."""

    window_size: int = 64
    feature_columns: Sequence[str] = field(
        default_factory=lambda: ("open", "high", "low", "close", "volume")
    )
    target_column: str = "close"
    normalize_returns: bool = True
    include_technical_indicators: bool = True
    ema_periods: Tuple[int, int] = (12, 26)


@dataclass(slots=True)
class MarketConfig:
    """Hyper-parameters for the market execution simulator."""

    initial_capital: float = 1_000_000.0
    max_leverage: float = 5.0
    transaction_cost: float = 0.0005
    slippage: float = 0.0002
    market_impact: float = 0.0001
    risk_aversion: float = 0.01
    min_cash: float = 100.0
    max_position_change: float = 1.0
    max_drawdown_threshold: Optional[float] = None

    def clamp_action(self, action: float) -> float:
        return max(min(action, self.max_leverage), -self.max_leverage)

    def __post_init__(self) -> None:
        if self.max_drawdown_threshold is not None and self.max_drawdown_threshold <= 0:
            raise ValueError("max_drawdown_threshold must be positive if provided")


@dataclass(slots=True)
class PolicyConfig:
    """Model architecture parameters for the actor-critic policy."""

    hidden_sizes: Tuple[int, ...] = (256, 256)
    activation: str = "silu"
    dropout: float = 0.05
    shared_backbone: bool = True
    value_head_layers: int = 1
    policy_head_layers: int = 1
    log_std_bounds: Tuple[float, float] = (-20.0, 2.0)


@dataclass(slots=True)
class TrainingConfig:
    """Trainer hyper-parameters."""

    total_timesteps: int = 200_000
    rollout_steps: int = 1_024
    num_epochs: int = 10
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_value: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    use_amp: bool = True
    target_kl: Optional[float] = 0.015
    device: str | None = None
    seed: Optional[int] = 42
    lr_schedule: str = "none"
    normalize_observations: bool = True

    def __post_init__(self) -> None:
        if self.minibatch_size > self.rollout_steps:
            raise ValueError("minibatch_size cannot exceed rollout_steps")
        if self.rollout_steps % self.minibatch_size != 0:
            raise ValueError("rollout_steps must be divisible by minibatch_size")
        if self.total_timesteps < self.rollout_steps:
            raise ValueError("total_timesteps must be >= rollout_steps")
        if self.lr_schedule not in {"none", "linear", "cosine"}:
            raise ValueError("lr_schedule must be 'none', 'linear', or 'cosine'")


@dataclass(slots=True)
class LLMConfig:
    """Configuration for the LLM-based policy guidance module."""

    enabled: bool = False
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    max_new_tokens: int = 128
    temperature: float = 0.1
    top_p: float = 0.9
    strategy_summary_template: str = (
        "You are an expert quantitative researcher. Summarize risks and "
        "opportunities for the next training epoch based on the following metrics:\n{metrics}"
    )


@dataclass(slots=True)
class SystemConfig:
    """Top-level configuration bundling all subsystems together."""

    data: DataConfig = field(default_factory=DataConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def copy_with(
        self,
        *,
        data: Optional[DataConfig] = None,
        market: Optional[MarketConfig] = None,
        policy: Optional[PolicyConfig] = None,
        training: Optional[TrainingConfig] = None,
        llm: Optional[LLMConfig] = None,
    ) -> "SystemConfig":
        return SystemConfig(
            data=data or self.data,
            market=market or self.market,
            policy=policy or self.policy,
            training=training or self.training,
            llm=llm or self.llm,
        )
