from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


def _coerce_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


TRAINER_BACKENDS = ("torch", "jax_classic")
TrainerBackend = Literal["torch", "jax_classic"]


def _normalize_trainer_backend(value: object) -> str:
    backend = str(value or "torch").strip().lower().replace("-", "_")
    aliases = {
        "default": "torch",
        "pytorch": "torch",
        "torch_trainer": "torch",
        "jax": "jax_classic",
        "jax_classic_trainer": "jax_classic",
    }
    return aliases.get(backend, backend)


@dataclass
class ForecastConfig:
    """Configuration for Chronos2 OHLC forecasts."""

    symbol: str = "BTCUSD"
    data_root: Path = Path("trainingdatahourly") / "crypto"
    context_hours: int = 24 * 14
    prediction_horizon_hours: int = 1
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9)
    batch_size: int = 128
    cache_dir: Path = Path("binanceneural") / "forecast_cache"
    use_time_covariates: bool | None = None
    force_multivariate: bool | None = None
    force_cross_learning: bool | None = None
    narrative_backend: str = "off"
    narrative_model: str | None = None
    narrative_summary_cache_dir: Path | None = None
    narrative_context_hours: int = 24 * 7


@dataclass
class DatasetConfig:
    """Dataset preparation parameters for Binance hourly sequences."""

    symbol: str = "BTCUSD"
    data_root: Path = Path("trainingdatahourly") / "crypto"
    forecast_cache_root: Path = Path("binanceneural") / "forecast_cache"
    forecast_horizons: tuple[int, ...] = (1, 4, 12, 24)
    sequence_length: int = 72
    val_fraction: float = 0.15
    min_history_hours: int = 24 * 30
    max_feature_lookback_hours: int = 24 * 7
    feature_columns: Sequence[str] | None = None
    refresh_hours: int = 0
    validation_days: int = 70
    max_train_days: int = 0   # 0 = use all; >0 = cap training to most-recent N days (before val split)
    cache_only: bool = False
    bar_shift_range: int = 0  # random ±N bar temporal jitter augmentation (train only)


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
    num_kv_heads: int | None = None
    num_layers: int = 4
    max_len: int = 2048
    use_midpoint_offsets: bool = True
    mlp_ratio: float = 4.0
    logits_softcap: float = 12.0
    rope_base: float = 10000.0
    use_qk_norm: bool = True
    use_causal_attention: bool = True
    rms_norm_eps: float = 1e-5
    attention_window: int | None = None
    attention_backend: str = "auto"  # auto, sdpa, flex, flash4
    flex_block_size: int = 128
    use_residual_scalars: bool = False
    residual_scale_init: float = 1.0
    skip_scale_init: float = 0.0
    use_value_embedding: bool = False
    value_embedding_every: int = 2
    value_embedding_scale: float = 1.0
    # Memory tokens: learnable global slots that attend to full sequence
    num_memory_tokens: int = 0  # 0=disabled, e.g. 4-16 for global memory
    # Sparse MoE FFN: replace dense FFN with N fine-grained experts (0 = disabled)
    moe_num_experts: int = 0  # e.g. 8 — same param budget, learned routing
    # Dilated attention: different head groups attend at different strides
    dilated_strides: str = ""  # e.g. "1,4,24" - one stride per head group
    use_flex_attention: bool = True  # use FlexAttention when available (PyTorch 2.4+, CUDA)
    num_outputs: int = 4
    max_hold_hours: float = 24.0

    def __post_init__(self) -> None:
        self.model_arch = str(self.model_arch or "classic")
        backend = str(self.attention_backend or "auto").strip().lower()
        backend_aliases = {
            "default": "auto",
            "none": "auto",
            "flash": "flash4",
            "flash-attn-4": "flash4",
            "flash_attn_4": "flash4",
            "flash_attn4": "flash4",
        }
        backend = backend_aliases.get(backend, backend)
        if backend not in {"auto", "sdpa", "flex", "flash4"}:
            backend = "auto"
        self.attention_backend = backend

        try:
            self.flex_block_size = max(1, int(self.flex_block_size))
        except (TypeError, ValueError):
            self.flex_block_size = 128

        use_flex = _coerce_bool(self.use_flex_attention, True)
        if backend == "flex":
            self.use_flex_attention = True
        elif backend in {"sdpa", "flash4"}:
            self.use_flex_attention = False
        else:
            self.use_flex_attention = use_flex


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
    max_leverage: float = 1.0
    margin_annual_rate: float = 0.0
    return_weight: float = 0.08
    smoothness_penalty: float = 0.0
    periods_per_year: float | None = None
    price_offset_pct: float = 0.0003
    min_price_gap_pct: float = 0.0003
    trade_amount_scale: float = 100.0
    use_midpoint_offsets: bool = True
    fill_temperature: float = 0.01
    decision_lag_bars: int = 0
    decision_lag_range: str = ""  # e.g. "0,1,2" -- average loss across lags during training
    market_order_entry: bool = False
    use_rebalance_sim: bool = False
    rebalance_allow_short: bool = False
    rebalance_regime_mode: bool = False
    rebalance_entropy_weight: float = 0.0
    rebalance_smoothness_weight: float = 0.0
    fill_buffer_pct: float = 0.0005
    spread_penalty: float = 0.0
    spread_target: float = 0.0005
    fill_buffer_warmup_epochs: int = 0
    initial_cash: float = 1.0
    sortino_target_sign: float = 1.0
    loss_type: str = "sortino"  # sortino, sharpe, calmar, log_wealth, sortino_dd, multiwindow, multiwindow_dd
    dd_penalty: float = 1.0
    multiwindow_fractions: str = "0.33,0.5,0.75,1.0"
    multiwindow_aggregation: str = "minimax"  # minimax, mean, softmin
    transformer_dim: int = 256
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_dropout: float = 0.1
    optimizer_name: str = "adamw"
    trainer_backend: TrainerBackend | str = "torch"
    model_arch: str = "classic"
    num_kv_heads: int | None = None
    mlp_ratio: float = 4.0
    logits_softcap: float = 12.0
    rope_base: float = 10000.0
    use_qk_norm: bool = True
    use_causal_attention: bool = True
    rms_norm_eps: float = 1e-5
    attention_window: int | None = None
    attention_backend: str = "auto"
    flex_block_size: int = 128
    use_residual_scalars: bool = False
    residual_scale_init: float = 1.0
    skip_scale_init: float = 0.0
    use_value_embedding: bool = False
    value_embedding_every: int = 2
    value_embedding_scale: float = 1.0
    num_memory_tokens: int = 0
    dilated_strides: str = ""
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_momentum_start: float | None = None
    muon_momentum_warmup_steps: int = 300
    muon_momentum_end: float = 0.85
    cooldown_fraction: float = 0.0
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    cautious_weight_decay: bool = False
    embed_lr_mult: float = 1.0
    head_lr_mult: float = 1.0
    embed_weight_decay: float | None = None
    head_weight_decay: float | None = None
    warmup_steps: int = 100
    lr_schedule: str = "none"  # "none", "cosine", "linear_warmdown"
    lr_warmdown_ratio: float = 0.5  # fraction of training for warmdown
    lr_min_ratio: float = 0.0  # final LR as fraction of base LR
    weight_decay_schedule: str = "none"
    weight_decay_end: float = 0.0
    ema_decay: float = 0.0
    validation_use_binary_fills: bool = True
    validation_lag_aggregation: str = "minimax"  # minimax or mean across decision_lag_range
    dry_train_steps: int | None = None
    device: str | None = None
    run_name: str | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_tags: str = ""
    wandb_notes: str | None = None
    wandb_mode: str = "auto"
    wandb_log_metrics: bool = False
    log_dir: Path = Path("tensorboard_logs") / "binanceneural"
    checkpoint_root: Path = Path("binanceneural") / "checkpoints"
    top_k_checkpoints: int = 10
    checkpoint_metric: str = "robust_score"  # val_score, val_sortino, val_return, robust_score, robust_sortino
    checkpoint_gap_penalty: float = 0.25
    preload_checkpoint_path: Path | None = None
    force_retrain: bool = False
    seed: int = 1337
    num_workers: int = 0
    num_outputs: int = 4
    max_hold_hours: float = 24.0
    feature_noise_std: float = 0.0
    bar_shift_range: int = 0  # propagated to DatasetConfig at training time
    moe_num_experts: int = 0  # propagated to PolicyConfig
    use_compile: bool = True
    use_amp: bool = False
    amp_dtype: str = "bfloat16"
    split_amp: bool = False
    use_vectorized_sim: bool = False
    use_compiled_sim_loss: bool = True
    use_tf32: bool = True
    use_flash_attention: bool = True
    accumulation_steps: int = 1
    use_flex_attention: bool = True
    forecast_config: ForecastConfig = field(default_factory=ForecastConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    def __post_init__(self) -> None:
        backend = _normalize_trainer_backend(self.trainer_backend)
        if backend not in TRAINER_BACKENDS:
            allowed = ", ".join(TRAINER_BACKENDS)
            raise ValueError(f"trainer_backend must be one of {{{allowed}}}, got {self.trainer_backend!r}")
        self.trainer_backend = backend


__all__ = [
    "DatasetConfig",
    "ForecastConfig",
    "PolicyConfig",
    "TRAINER_BACKENDS",
    "TrainingConfig",
    "TrainerBackend",
]
