from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import yaml


def _as_path(value: Any) -> Path:
    if value is None:
        raise ValueError("Path value cannot be None")
    return value if isinstance(value, Path) else Path(str(value))


def _normalise_symbol(sym: str) -> str:
    cleaned = sym.strip().upper()
    if not cleaned:
        raise ValueError("Empty symbol encountered while parsing configuration.")
    return cleaned


@dataclass(slots=True)
class DataConfig:
    """Configuration describing how to load and pre-process market data."""

    data_dir: Path = Path("trainingdata")
    symbols: Sequence[str] = field(
        default_factory=lambda: ("AAPL", "AMZN", "MSFT", "NVDA", "GOOGL")
    )
    window_size: int = 64
    min_history: int = 512
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    feature_columns: Optional[Sequence[str]] = None
    include_toto: bool = True
    normalise_volume: bool = True
    forward_fill_features: bool = True

    def __post_init__(self) -> None:
        self.data_dir = _as_path(self.data_dir)
        if self.feature_columns is not None:
            self.feature_columns = tuple(str(col) for col in self.feature_columns)
        self.symbols = tuple(_normalise_symbol(sym) for sym in self.symbols)

    def validated_symbols(self) -> List[str]:
        return list(self.symbols)

    def ensure_min_history(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2.")
        if self.min_history <= self.window_size:
            raise ValueError("min_history must exceed window_size to provide a lookback buffer.")


@dataclass(slots=True)
class EnvConfig:
    """Parameters governing the trading environment dynamics."""

    initial_balance: float = 100_000.0
    leverage_limit: Optional[float] = None
    borrowing_cost_annual: Optional[float] = None
    trading_days_per_year: Optional[int] = None
    transaction_cost_bps: float = 5.0
    spread_bps: float = 0.5
    max_intraday_leverage: float = 3.0
    max_overnight_leverage: float = 2.0
    trade_timing: str = "open"
    risk_scale: float = 1.0
    device: Optional[str] = None
    reward_scale: float = 1.0

    def __post_init__(self) -> None:
        self.trade_timing = self.normalised_trade_timing()
        if self.initial_balance <= 0.0:
            raise ValueError("initial_balance must be positive.")
        if self.risk_scale < 0.0 or self.risk_scale > 1.0:
            raise ValueError("risk_scale must be in [0, 1].")
        if self.transaction_cost_bps < 0.0 or self.spread_bps < 0.0:
            raise ValueError("transaction/spread costs must be non-negative.")

    def normalised_trade_timing(self) -> str:
        trade_timing = (self.trade_timing or "open").strip().lower()
        if trade_timing not in {"open", "close"}:
            raise ValueError("trade_timing must be either 'open' or 'close'.")
        return trade_timing


@dataclass(slots=True)
class VecConfig:
    """Parallel environment execution details."""

    backend: str = "Multiprocessing"
    num_envs: int = 64
    num_workers: int = 8
    batch_size: int = 4
    seed: int = 42
    device: str = "cuda"
    async_reset: bool = True

    def __post_init__(self) -> None:
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.backend = str(self.backend)
        self.device = str(self.device)

    def as_kwargs(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "num_envs": self.num_envs,
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }


@dataclass(slots=True)
class ModelConfig:
    """Neural architecture hyperparameters."""

    hidden_size: int = 256
    actor_layers: Sequence[int] = field(default_factory=lambda: (256, 256))
    critic_layers: Sequence[int] = field(default_factory=lambda: (256, 256))
    dropout_p: float = 0.0
    layer_norm: bool = True
    rnn_hidden_size: int = 256
    use_lstm: bool = True
    activation: str = "swish"


@dataclass(slots=True)
class TrainConfig:
    """PufferLib PPO training configuration."""

    total_timesteps: int = 20_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.8
    max_grad_norm: float = 1.0
    batch_size: int = 262_144
    minibatch_size: int = 65_536
    update_epochs: int = 3
    bptt_horizon: int = 512
    use_rnn: bool = True
    optimizer: str = "adam"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-5
    compile: bool = False
    compile_mode: str = "max-autotune"
    torch_deterministic: bool = False

    def __post_init__(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive.")
        if self.update_epochs <= 0:
            raise ValueError("update_epochs must be positive.")
        if self.bptt_horizon <= 0:
            raise ValueError("bptt_horizon must be positive.")
        self.optimizer = self.optimizer.lower()

    def apply_overrides(self, base: Mapping[str, Any], *, device: str) -> Dict[str, Any]:
        cfg = dict(base)
        cfg.update(
            {
                "total_timesteps": self.total_timesteps,
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_coef": self.clip_coef,
                "ent_coef": self.ent_coef,
                "vf_coef": self.vf_coef,
                "max_grad_norm": self.max_grad_norm,
                "batch_size": self.batch_size,
                "minibatch_size": self.minibatch_size,
                "update_epochs": self.update_epochs,
                "bptt_horizon": self.bptt_horizon,
                "use_rnn": self.use_rnn,
                "optimizer": self.optimizer,
                "adam_beta1": self.adam_beta1,
                "adam_beta2": self.adam_beta2,
                "adam_eps": self.adam_eps,
                "compile": self.compile,
                "compile_mode": self.compile_mode,
                "torch_deterministic": self.torch_deterministic,
                "device": device,
            }
        )
        cfg.setdefault("cpu_offload", False)
        cfg.setdefault("seed", 42)
        cfg.setdefault("anneal_lr", False)
        cfg.setdefault("precision", "32-true")
        cfg.setdefault("amp", True)
        cfg.setdefault("max_minibatch_size", self.minibatch_size)
        cfg.setdefault("vf_clip_coef", 0.1)
        cfg.setdefault("vtrace_rho_clip", 1.0)
        cfg.setdefault("vtrace_c_clip", 1.0)
        cfg.setdefault("prio_alpha", 0.0)
        cfg.setdefault("prio_beta0", 1.0)
        cfg.setdefault("checkpoint_interval", 10)
        cfg.setdefault("data_dir", "pufferlibtraining2/models")
        cfg.setdefault("no_model_upload", True)
        return cfg


@dataclass(slots=True)
class LoggingConfig:
    """Tracking and checkpointing options."""

    tensorboard_dir: Path = Path("pufferlibtraining2/logs")
    checkpoint_dir: Path = Path("pufferlibtraining2/models")
    summary_path: Path = Path("pufferlibtraining2/pipeline_summary.json")
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: Sequence[str] = field(default_factory=tuple)
    flush_interval: int = 10

    def __post_init__(self) -> None:
        self.tensorboard_dir = _as_path(self.tensorboard_dir)
        self.checkpoint_dir = _as_path(self.checkpoint_dir)
        self.summary_path = _as_path(self.summary_path)
        if self.wandb_run_name is not None:
            self.wandb_run_name = str(self.wandb_run_name)
        if self.wandb_project is not None:
            self.wandb_project = str(self.wandb_project)
        if self.wandb_entity is not None:
            self.wandb_entity = str(self.wandb_entity)
        if self.wandb_tags:
            self.wandb_tags = tuple(str(tag) for tag in self.wandb_tags)

    def ensure_directories(self) -> None:
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.summary_path.parent:
            self.summary_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class TrainingPlan:
    """Aggregate configuration for the complete training pipeline."""

    data: DataConfig = field(default_factory=DataConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    vec: VecConfig = field(default_factory=VecConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TrainingPlan":
        plan = cls()
        for field in dataclasses.fields(cls):
            if field.name not in mapping:
                continue
            section = mapping[field.name]
            if section is None:
                continue
            if not isinstance(section, Mapping):
                raise TypeError(f"Expected mapping for section '{field.name}', got {type(section)}")
            current = getattr(plan, field.name)
            updated = dataclasses.replace(current, **section)
            setattr(plan, field.name, updated)
        plan.data.ensure_min_history()
        plan.logging.ensure_directories()
        return plan

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            payload[field.name] = dataclasses.asdict(value)
        return payload


def _deep_update(base: MutableMapping[str, Any], overlay: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, Mapping):
            node = base.setdefault(key, {})
            if not isinstance(node, MutableMapping):
                raise TypeError(f"Cannot merge mapping into non-mapping at key '{key}'")
            _deep_update(node, value)
        else:
            base[key] = value
    return base


def load_plan(
    config_path: Optional[str | Path] = None,
    *,
    overrides: Optional[Mapping[str, Any]] = None,
) -> TrainingPlan:
    """
    Load a training configuration from YAML. When ``config_path`` is ``None`` the
    defaults defined in :class:`TrainingPlan` are used. ``overrides`` can supply a
    nested mapping to patch the loaded configuration (useful for tests).
    """

    payload: Dict[str, Any] = {}
    if config_path is not None:
        path = _as_path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        raw = yaml.safe_load(path.read_text())
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise TypeError(f"Root of configuration must be a mapping, received {type(raw)}")
        payload = dict(raw)

    if overrides:
        payload = _deep_update(payload, overrides)  # type: ignore[arg-type]

    plan = TrainingPlan.from_mapping(payload)
    plan.logging.ensure_directories()
    return plan


__all__ = [
    "DataConfig",
    "EnvConfig",
    "VecConfig",
    "ModelConfig",
    "TrainConfig",
    "LoggingConfig",
    "TrainingPlan",
    "load_plan",
]
