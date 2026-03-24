"""Experiment pool for Qwen GRPO autoresearch."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QwenTrialConfig:
    model_size: str = "0.6B"
    lora_r: int = 16
    lora_alpha: int = 32
    group_size: int = 8
    lr: float = 5e-6
    kl_coef: float = 0.05
    max_completion_length: int = 512
    n_symbols: int = 10
    reward_type: str = "sortino_only"
    prompt_variant: str = "detailed"
    sft_warmstart: bool = True
    eval_horizon_hours: int = 24
    seed: int = 42
    description: str = ""


EXPERIMENTS: list[dict] = [
    # --- Scaling law baselines (same config, different sizes) ---
    {"description": "qwen_06b_baseline", "model_size": "0.6B"},
    {"description": "qwen_18b_baseline", "model_size": "1.8B"},
    {"description": "qwen_3b_baseline", "model_size": "3B"},
    {"description": "qwen_7b_baseline", "model_size": "7B"},

    # --- SFT warmstart ablation ---
    {"description": "qwen_06b_cold", "model_size": "0.6B", "sft_warmstart": False},
    {"description": "qwen_18b_cold", "model_size": "1.8B", "sft_warmstart": False},

    # --- LoRA rank sweep (0.6B) ---
    {"description": "qwen_06b_r8", "model_size": "0.6B", "lora_r": 8},
    {"description": "qwen_06b_r32", "model_size": "0.6B", "lora_r": 32},
    {"description": "qwen_06b_r64", "model_size": "0.6B", "lora_r": 64},

    # --- Group size sweep ---
    {"description": "qwen_06b_g4", "model_size": "0.6B", "group_size": 4},
    {"description": "qwen_06b_g16", "model_size": "0.6B", "group_size": 16},

    # --- LR sweep ---
    {"description": "qwen_06b_lr1e6", "model_size": "0.6B", "lr": 1e-6},
    {"description": "qwen_06b_lr1e5", "model_size": "0.6B", "lr": 1e-5},
    {"description": "qwen_06b_lr5e5", "model_size": "0.6B", "lr": 5e-5},

    # --- KL coefficient sweep ---
    {"description": "qwen_06b_kl001", "model_size": "0.6B", "kl_coef": 0.01},
    {"description": "qwen_06b_kl01", "model_size": "0.6B", "kl_coef": 0.1},

    # --- Reward function variants ---
    {"description": "qwen_06b_sortino_dd", "model_size": "0.6B", "reward_type": "sortino_drawdown"},
    {"description": "qwen_06b_sortino_sm", "model_size": "0.6B", "reward_type": "sortino_smoothness"},

    # --- Symbol count ---
    {"description": "qwen_06b_5sym", "model_size": "0.6B", "n_symbols": 5},
    {"description": "qwen_06b_20sym", "model_size": "0.6B", "n_symbols": 20},
    {"description": "qwen_06b_30sym", "model_size": "0.6B", "n_symbols": 30},

    # --- Prompt variants ---
    {"description": "qwen_06b_minimal", "model_size": "0.6B", "prompt_variant": "minimal"},
    {"description": "qwen_06b_chronos", "model_size": "0.6B", "prompt_variant": "with_chronos2"},

    # --- Completion length ---
    {"description": "qwen_06b_len256", "model_size": "0.6B", "max_completion_length": 256},
    {"description": "qwen_06b_len1024", "model_size": "0.6B", "max_completion_length": 1024},

    # --- Eval horizon ---
    {"description": "qwen_06b_6h", "model_size": "0.6B", "eval_horizon_hours": 6},
    {"description": "qwen_06b_12h", "model_size": "0.6B", "eval_horizon_hours": 12},

    # --- Seed variants ---
    {"description": "qwen_06b_s7", "model_size": "0.6B", "seed": 7},
    {"description": "qwen_06b_s123", "model_size": "0.6B", "seed": 123},

    # --- Best 0.6B replicated at larger sizes (populated dynamically) ---
]


def build_trial_config(overrides: dict) -> QwenTrialConfig:
    """Build a QwenTrialConfig from default + overrides dict."""
    cfg = QwenTrialConfig()
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
