"""Configuration for TRL-based trading-plan training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


MODEL_PRESETS: dict[str, str] = {
    "qwen2_05b_instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2_15b_instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2_3b_instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2_7b_instruct": "Qwen/Qwen2.5-7B-Instruct",
}

RECOMMENDED_MODEL_PRESET = "qwen2_05b_instruct"
SUPPORTED_REWARD_TYPES = frozenset({"sortino_only", "sortino_drawdown", "sortino_smoothness"})
SUPPORTED_VLLM_MODES = frozenset({"colocate", "server"})


@dataclass(slots=True)
class TRLTradingConfig:
    """Small, explicit config for GRPO-based trading-plan training."""

    model_preset: str = RECOMMENDED_MODEL_PRESET
    trainer_type: str = "grpo"
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.85
    bf16: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    group_size: int = 8
    learning_rate: float = 5e-6
    kl_coef: float = 0.05
    num_train_epochs: int = 3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_prompt_length: int = 1024
    max_completion_length: int = 512
    n_symbols: int = 10
    lookback_hours: int = 24
    eval_horizon_hours: int = 24
    stride: int = 4
    prompt_variant: str = "with_chronos2"
    reward_type: str = "sortino_drawdown"
    sft_warmstart: bool = True
    sft_epochs: int = 1
    eval_every_steps: int = 50
    early_stop_patience: int = 3
    top_k_checkpoints: int = 5
    seed: int = 42
    time_budget: int = 0
    data_dir: str = "trainingdatahourly/crypto"
    forecast_cache_dir: str = "binanceneural/forecast_cache"
    sft_data: str = "rl-trainingbinance/trading_plans_train.jsonl"
    output_dir: str = "trltraining/checkpoints/default"
    report_to: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""

    @property
    def model_name(self) -> str:
        try:
            return MODEL_PRESETS[self.model_preset]
        except KeyError as exc:
            supported = ", ".join(sorted(MODEL_PRESETS))
            raise ValueError(f"unsupported model_preset {self.model_preset!r}; expected one of: {supported}") from exc

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    def validate(self) -> None:
        if self.trainer_type != "grpo":
            raise ValueError(f"unsupported trainer_type {self.trainer_type!r}; only 'grpo' is supported currently")
        if self.reward_type not in SUPPORTED_REWARD_TYPES:
            supported = ", ".join(sorted(SUPPORTED_REWARD_TYPES))
            raise ValueError(f"unsupported reward_type {self.reward_type!r}; expected one of: {supported}")
        if self.use_vllm and self.vllm_mode not in SUPPORTED_VLLM_MODES:
            supported = ", ".join(sorted(SUPPORTED_VLLM_MODES))
            raise ValueError(f"unsupported vllm_mode {self.vllm_mode!r}; expected one of: {supported}")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.vllm_tensor_parallel_size <= 0:
            raise ValueError("vllm_tensor_parallel_size must be positive")
        if not 0.1 <= self.vllm_gpu_memory_utilization <= 0.98:
            raise ValueError("vllm_gpu_memory_utilization must be within [0.1, 0.98]")
        if self.max_prompt_length <= 0 or self.max_completion_length <= 0:
            raise ValueError("max prompt/completion lengths must be positive")
        if self.n_symbols <= 0:
            raise ValueError("n_symbols must be positive")
