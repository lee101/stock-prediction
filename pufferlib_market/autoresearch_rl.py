"""
Auto-research loop for PufferLib RL trading.

Runs timeboxed training experiments (default 5 min each), evaluates on
held-out validation data, and tracks results in a leaderboard CSV.

The primary ranking signal can now come from multi-window holdout robustness
and optional 30-day market validation, not only the raw C-env validation
return. That keeps the search loop closer to the deployed replay target.

Usage:
  python -u -m pufferlib_market.autoresearch_rl \
    --train-data pufferlib_market/data/crypto6_train.bin \
    --val-data pufferlib_market/data/crypto6_val.bin \
    --time-budget 300 --max-trials 50
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from src.robust_trading_metrics import compute_replay_composite_score, summarize_scenario_results
from pufferlib_market.early_stopper import combined_score as _combined_score, PolynomialEarlyStopper, BestKnownTracker, HoldCashDetector

try:
    import wandb as _wandb_module
except ImportError:
    _wandb_module = None

REPO = Path(__file__).resolve().parent.parent


def _read_mktd_header(path: str | Path) -> tuple[int, int]:
    """Read num_symbols and num_timesteps from an MKTD .bin header."""
    import struct
    with open(str(path), "rb") as f:
        header = f.read(64)
    _, _, num_symbols, num_timesteps, _, _ = struct.unpack("<4sIIIII", header[:24])
    return int(num_symbols), int(num_timesteps)


@dataclass
class TrialConfig:
    """Hyperparameters to sweep."""
    hidden_size: int = 1024
    lr: float = 3e-4
    anneal_lr: bool = True
    ent_coef: float = 0.05
    ent_coef_end: float = 0.02
    anneal_ent: bool = False
    clip_eps: float = 0.2
    clip_eps_end: float = 0.05
    anneal_clip: bool = False
    clip_vloss: bool = False
    weight_decay: float = 0.0
    obs_norm: bool = False
    lr_schedule: str = "none"
    lr_warmup_frac: float = 0.02
    lr_min_ratio: float = 0.05
    gamma: float = 0.99
    gae_lambda: float = 0.95
    advantage_norm: str = "global"
    group_relative_size: int = 8
    group_relative_mix: float = 0.0
    group_relative_clip: float = 2.0
    num_envs: int = 128
    rollout_len: int = 256
    ppo_epochs: int = 4
    reward_scale: float = 10.0
    reward_clip: float = 5.0
    cash_penalty: float = 0.01
    fill_slippage_bps: float = 0.0
    fee_rate: float = 0.001
    trade_penalty: float = 0.0
    drawdown_penalty: float = 0.0
    downside_penalty: float = 0.0
    smooth_downside_penalty: float = 0.0
    smooth_downside_temperature: float = 0.02
    smoothness_penalty: float = 0.0
    arch: str = "mlp"
    optimizer: str = "adamw"  # "adamw" or "muon"
    muon_norm_update: bool = False  # NorMuon: scale update norm to match param norm
    max_steps: int = 720
    periods_per_year: float = 8760.0
    seed: int = 42
    description: str = ""
    max_leverage: float = 1.0
    short_borrow_apr: float = 0.0
    requires_gpu: str = ""  # e.g. "a100", "h100", "" = any GPU (dispatcher metadata only)
    # Training performance settings: BF16 + CUDA graph PPO give ~30-50% speedup on modern GPUs
    minibatch_size: int = 2048
    use_bf16: bool = True   # BF16 safe for PPO; ~20-30% speedup on RTX 5090/A40/H100
    cuda_graph_ppo: bool = True  # Static shapes work for PPO; ~10-20% extra speedup
    no_cuda_graph: bool = False  # Disable ALL CUDA graphs + torch.compile (for shared-GPU compat)
    time_budget_override: int = 0  # Override global time_budget for this config (0 = use global)
    vf_coef: float = 0.5   # Value function loss coefficient (default 0.5)
    max_grad_norm: float = 0.5  # Gradient clipping norm (default 0.5)


# Define experiment configurations to test
EXPERIMENTS: list[dict] = [
    # Baseline: vanilla PPO with anneal-LR
    {"description": "baseline_anneal_lr"},

    # Obs norm (was critical in earlier tests)
    {"description": "obs_norm", "obs_norm": True},

    # Cosine LR schedule
    {"description": "cosine_lr", "lr_schedule": "cosine", "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05},

    # Entropy annealing
    {"description": "ent_anneal", "anneal_ent": True, "ent_coef": 0.08, "ent_coef_end": 0.02},

    # Clip annealing
    {"description": "clip_anneal", "anneal_clip": True, "clip_eps": 0.2, "clip_eps_end": 0.05},

    # Value clipping
    {"description": "clip_vloss", "clip_vloss": True},

    # Weight decay
    {"description": "wd_005", "weight_decay": 0.005},
    {"description": "wd_01", "weight_decay": 0.01},
    {"description": "wd_05", "weight_decay": 0.05},

    # Higher weight decay as regularization
    {"description": "wd_1", "weight_decay": 0.1},

    # Train WITH slippage to learn robust strategies
    {"description": "slip_5bps", "fill_slippage_bps": 5.0},
    {"description": "slip_10bps", "fill_slippage_bps": 10.0},

    # Higher fees to force more robust edge
    {"description": "fee_2x", "fee_rate": 0.002},

    # Trade penalty to reduce churn
    {"description": "trade_pen_01", "trade_penalty": 0.01},
    {"description": "trade_pen_05", "trade_penalty": 0.05},

    # Downside penalty for Sortino
    {"description": "downside_pen", "downside_penalty": 0.5},
    {"description": "smooth_ds", "smooth_downside_penalty": 0.5},

    # Combined regularization
    {"description": "reg_combo_1", "weight_decay": 0.01, "fill_slippage_bps": 8.0, "trade_penalty": 0.01},
    {"description": "reg_combo_2", "weight_decay": 0.05, "fill_slippage_bps": 8.0, "obs_norm": True},
    {"description": "reg_combo_3", "obs_norm": True, "anneal_ent": True, "ent_coef": 0.08, "ent_coef_end": 0.02,
     "lr_schedule": "cosine", "weight_decay": 0.005, "fill_slippage_bps": 5.0},

    # Robust daily variants centered on the best 3-window mixed23 family
    {"description": "robust_reg_wd02", "weight_decay": 0.02, "fill_slippage_bps": 8.0, "obs_norm": True},
    {"description": "robust_reg_tp005", "weight_decay": 0.05, "fill_slippage_bps": 8.0, "obs_norm": True, "trade_penalty": 0.005},
    {"description": "robust_reg_tp01", "weight_decay": 0.05, "fill_slippage_bps": 8.0, "obs_norm": True, "trade_penalty": 0.01},
    {"description": "robust_reg_tp005_sds02", "weight_decay": 0.05, "fill_slippage_bps": 8.0, "obs_norm": True,
     "trade_penalty": 0.005, "smooth_downside_penalty": 0.2},
    {"description": "robust_reg_tp005_sds02_t01", "weight_decay": 0.05, "fill_slippage_bps": 8.0, "obs_norm": True,
     "trade_penalty": 0.005, "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.01},
    {"description": "robust_reg_tp005_sds02_t05", "weight_decay": 0.05, "fill_slippage_bps": 8.0, "obs_norm": True,
     "trade_penalty": 0.005, "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.05},
    {"description": "robust_reg_tp005_dd002", "weight_decay": 0.05, "fill_slippage_bps": 8.0, "obs_norm": True,
     "trade_penalty": 0.005, "drawdown_penalty": 0.02},
    {"description": "robust_reg_tp005_sm001", "weight_decay": 0.05, "fill_slippage_bps": 8.0, "obs_norm": True,
     "trade_penalty": 0.005, "smoothness_penalty": 0.01},
    {"description": "robust_reg_tp005_ent", "weight_decay": 0.05, "fill_slippage_bps": 8.0, "obs_norm": True,
     "trade_penalty": 0.005, "anneal_ent": True, "ent_coef": 0.08, "ent_coef_end": 0.02},
    {"description": "robust_reg_h512_tp005", "hidden_size": 512, "weight_decay": 0.05, "fill_slippage_bps": 8.0,
     "obs_norm": True, "trade_penalty": 0.005},

    # Multi-seed validation of champion (robust_reg_tp005_ent)
    {"description": "robust_reg_tp005_ent_seed42", "weight_decay": 0.05, "fill_slippage_bps": 8.0, "obs_norm": True,
     "trade_penalty": 0.005, "anneal_ent": True, "ent_coef": 0.08, "ent_coef_end": 0.02, "seed": 42},
    {"description": "robust_reg_tp005_ent_seed7", "weight_decay": 0.05, "fill_slippage_bps": 8.0, "obs_norm": True,
     "trade_penalty": 0.005, "anneal_ent": True, "ent_coef": 0.08, "ent_coef_end": 0.02, "seed": 7},
    {"description": "robust_reg_tp005_ent_seed123", "weight_decay": 0.05, "fill_slippage_bps": 8.0, "obs_norm": True,
     "trade_penalty": 0.005, "anneal_ent": True, "ent_coef": 0.08, "ent_coef_end": 0.02, "seed": 123},

    # h1536: larger model with champion regularization
    {"description": "h1536_robust_ent", "hidden_size": 1536, "weight_decay": 0.05, "fill_slippage_bps": 8.0,
     "obs_norm": True, "trade_penalty": 0.005, "anneal_ent": True, "ent_coef": 0.08, "ent_coef_end": 0.02},

    # h2048 with champion regularization
    {"description": "h2048_robust_ent", "hidden_size": 2048, "weight_decay": 0.05, "fill_slippage_bps": 8.0,
     "obs_norm": True, "trade_penalty": 0.005, "anneal_ent": True, "ent_coef": 0.08, "ent_coef_end": 0.02},

    # Kitchen sink
    {"description": "kitchen_sink", "obs_norm": True, "anneal_ent": True, "anneal_clip": True,
     "clip_vloss": True, "lr_schedule": "cosine", "weight_decay": 0.01,
     "fill_slippage_bps": 8.0, "trade_penalty": 0.01, "downside_penalty": 0.2},

    # Smaller model (faster, may generalize better)
    {"description": "h512", "hidden_size": 512},
    {"description": "h256", "hidden_size": 256},
    {"description": "h512_wd01", "hidden_size": 512, "weight_decay": 0.01},

    # Lower entropy (more exploitation)
    {"description": "ent_001", "ent_coef": 0.01},
    {"description": "ent_01", "ent_coef": 0.1},

    # Lower LR
    {"description": "lr_1e4", "lr": 1e-4},

    # Higher gamma
    {"description": "gamma_999", "gamma": 0.999},

    # GSPO/GRPO-inspired sequence/group-relative advantage shaping.
    {"description": "per_env_adv", "obs_norm": True, "weight_decay": 0.05,
     "fill_slippage_bps": 8.0, "advantage_norm": "per_env"},
    {"description": "per_env_adv_smooth", "obs_norm": True, "weight_decay": 0.05,
     "fill_slippage_bps": 8.0, "trade_penalty": 0.005, "smooth_downside_penalty": 0.2,
     "smoothness_penalty": 0.01, "advantage_norm": "per_env"},
    {"description": "gspo_like", "obs_norm": True, "weight_decay": 0.05,
     "fill_slippage_bps": 8.0, "advantage_norm": "group_relative",
     "group_relative_size": 8, "group_relative_mix": 0.25, "group_relative_clip": 1.5},
    {"description": "gspo_like_mix15", "obs_norm": True, "weight_decay": 0.05,
     "fill_slippage_bps": 8.0, "advantage_norm": "group_relative",
     "group_relative_size": 16, "group_relative_mix": 0.15, "group_relative_clip": 1.0},
    {"description": "gspo_like_mix40", "obs_norm": True, "weight_decay": 0.05,
     "fill_slippage_bps": 8.0, "advantage_norm": "group_relative",
     "group_relative_size": 16, "group_relative_mix": 0.4, "group_relative_clip": 1.5},
    {"description": "gspo_like_smooth", "obs_norm": True, "weight_decay": 0.05,
     "fill_slippage_bps": 8.0, "trade_penalty": 0.005, "smooth_downside_penalty": 0.2,
     "smoothness_penalty": 0.01, "advantage_norm": "group_relative",
     "group_relative_size": 8, "group_relative_mix": 0.25, "group_relative_clip": 1.5},
    {"description": "gspo_like_smooth_mix15", "obs_norm": True, "weight_decay": 0.05,
     "fill_slippage_bps": 8.0, "trade_penalty": 0.005, "smooth_downside_penalty": 0.2,
     "smoothness_penalty": 0.01, "advantage_norm": "group_relative",
     "group_relative_size": 16, "group_relative_mix": 0.15, "group_relative_clip": 1.0},
    {"description": "gspo_like_drawdown_mix15", "obs_norm": True, "weight_decay": 0.05,
     "fill_slippage_bps": 8.0, "trade_penalty": 0.005, "drawdown_penalty": 0.02,
     "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.01,
     "smoothness_penalty": 0.005, "advantage_norm": "group_relative",
     "group_relative_size": 16, "group_relative_mix": 0.15, "group_relative_clip": 1.0},
    {"description": "gspo_like_drawdown_mix15_slip12", "obs_norm": True, "weight_decay": 0.05,
     "fill_slippage_bps": 12.0, "trade_penalty": 0.005, "drawdown_penalty": 0.02,
     "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.01,
     "smoothness_penalty": 0.005, "advantage_norm": "group_relative",
     "group_relative_size": 16, "group_relative_mix": 0.15, "group_relative_clip": 1.0},
    {"description": "gspo_like_drawdown_mix15_tp01", "obs_norm": True, "weight_decay": 0.05,
     "fill_slippage_bps": 8.0, "trade_penalty": 0.01, "drawdown_penalty": 0.02,
     "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.01,
     "smoothness_penalty": 0.005, "advantage_norm": "group_relative",
     "group_relative_size": 16, "group_relative_mix": 0.15, "group_relative_clip": 1.0},
    {"description": "gspo_like_drawdown_mix15_dd03", "obs_norm": True, "weight_decay": 0.05,
     "fill_slippage_bps": 8.0, "trade_penalty": 0.005, "drawdown_penalty": 0.03,
     "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.01,
     "smoothness_penalty": 0.005, "advantage_norm": "group_relative",
     "group_relative_size": 16, "group_relative_mix": 0.15, "group_relative_clip": 1.0},
    {"description": "gspo_like_drawdown_mix15_sds03", "obs_norm": True, "weight_decay": 0.05,
     "fill_slippage_bps": 8.0, "trade_penalty": 0.005, "drawdown_penalty": 0.02,
     "smooth_downside_penalty": 0.3, "smooth_downside_temperature": 0.01,
     "smoothness_penalty": 0.005, "advantage_norm": "group_relative",
     "group_relative_size": 16, "group_relative_mix": 0.15, "group_relative_clip": 1.0},
    {"description": "gspo_like_drawdown_mix15_h512", "hidden_size": 512, "obs_norm": True, "weight_decay": 0.05,
     "fill_slippage_bps": 8.0, "trade_penalty": 0.005, "drawdown_penalty": 0.02,
     "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.01,
     "smoothness_penalty": 0.005, "advantage_norm": "group_relative",
     "group_relative_size": 16, "group_relative_mix": 0.15, "group_relative_clip": 1.0},
    {"description": "gspo_like_drawdown_mix15_tp01_dd03", "obs_norm": True, "weight_decay": 0.05,
     "fill_slippage_bps": 8.0, "trade_penalty": 0.01, "drawdown_penalty": 0.03,
     "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.01,
     "smoothness_penalty": 0.005, "advantage_norm": "group_relative",
     "group_relative_size": 16, "group_relative_mix": 0.15, "group_relative_clip": 1.0},
    {"description": "gspo_like_drawdown_mix15_tp01_dd03_slip10", "obs_norm": True, "weight_decay": 0.05,
     "fill_slippage_bps": 10.0, "trade_penalty": 0.01, "drawdown_penalty": 0.03,
     "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.01,
     "smoothness_penalty": 0.005, "advantage_norm": "group_relative",
     "group_relative_size": 16, "group_relative_mix": 0.15, "group_relative_clip": 1.0},

    # Shorter episodes (more episodes per training budget)
    {"description": "ep_360h", "max_steps": 360},

    # Different seed
    {"description": "seed_123", "seed": 123},
    {"description": "seed_7", "seed": 7},

    # ResidualMLP architecture
    {"description": "resmlp", "arch": "resmlp"},
    {"description": "resmlp_wd", "arch": "resmlp", "weight_decay": 0.01},

    # More envs (more diverse experience per update)
    {"description": "envs_256", "num_envs": 256},

    # Random mutations of best config
    {"description": "random_1"},
    {"description": "random_2"},
    {"description": "random_3"},

    # -----------------------------------------------------------------------
    # Sortino-focused configs (optimise for risk-adjusted returns, not raw PnL)
    # -----------------------------------------------------------------------

    # High trade penalty → less churn → smoother equity curve
    {"description": "trade_pen_high",
     "trade_penalty": 0.10},

    # Very high trade penalty variant
    {"description": "trade_pen_vhigh",
     "trade_penalty": 0.20},

    # Match production crypto slippage closely (3bps)
    {"description": "slip_3bps",
     "fill_slippage_bps": 3.0},

    # Combined conservative: high trade penalty + low slippage
    {"description": "combined_smooth",
     "trade_penalty": 0.08, "fill_slippage_bps": 3.0, "ent_coef": 0.02},

    # reg_combo_3 was top-1 Sortino on crypto10 (val_sortino=2.82) — replicate
    # with tighter trade penalty for even smoother PnL
    {"description": "sortino_top1_tp",
     "weight_decay": 0.005, "fill_slippage_bps": 5.0, "obs_norm": True,
     "anneal_ent": True, "ent_coef": 0.08, "ent_coef_end": 0.02,
     "lr_schedule": "cosine", "trade_penalty": 0.05},

    # Smooth-downside penalty focused on minimising downside volatility
    {"description": "sortino_sds_pen",
     "weight_decay": 0.05, "fill_slippage_bps": 8.0, "obs_norm": True,
     "smooth_downside_penalty": 0.5, "smooth_downside_temperature": 0.02,
     "trade_penalty": 0.02},

    # Drawdown penalty + slippage to stay out of losing streaks
    {"description": "sortino_dd_slip",
     "fill_slippage_bps": 5.0, "obs_norm": True, "weight_decay": 0.02,
     "drawdown_penalty": 0.05, "trade_penalty": 0.02},

    # Low entropy (more exploitation) + trade penalty (already trade_pen_05
    # is #1 Sortino on autoresearch_daily; push entropy lower too)
    {"description": "sortino_low_ent_tp",
     "ent_coef": 0.01, "trade_penalty": 0.10},

    # reg_combo_3 exact clone but with higher trade penalty
    {"description": "sortino_rc3_tp08",
     "weight_decay": 0.005, "fill_slippage_bps": 5.0, "obs_norm": True,
     "anneal_ent": True, "ent_coef": 0.08, "ent_coef_end": 0.02,
     "lr_schedule": "cosine", "trade_penalty": 0.08},

    # Best of two worlds: cosine LR (good Sortino) + slippage training
    {"description": "sortino_cosine_slip",
     "lr_schedule": "cosine", "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05,
     "fill_slippage_bps": 5.0, "trade_penalty": 0.05},

    # Smoothness + downside penalty ensemble
    {"description": "sortino_smooth_combo",
     "weight_decay": 0.03, "fill_slippage_bps": 5.0, "obs_norm": True,
     "smooth_downside_penalty": 0.3, "smoothness_penalty": 0.01,
     "trade_penalty": 0.05},

    # --- Leverage experiments ---
    {"description": "leverage_15x",
     "max_leverage": 1.5, "short_borrow_apr": 0.0001712,
     "fill_slippage_bps": 5.0, "anneal_lr": True,
     "ent_coef": 0.05, "trade_penalty": 0.05},

    {"description": "leverage_2x",
     "max_leverage": 2.0, "short_borrow_apr": 0.0001712,
     "fill_slippage_bps": 5.0, "anneal_lr": True,
     "ent_coef": 0.05, "trade_penalty": 0.05},

    {"description": "leverage_2x_tp01",
     "max_leverage": 2.0, "short_borrow_apr": 0.0001712,
     "fill_slippage_bps": 8.0, "anneal_lr": True,
     "ent_coef": 0.05, "trade_penalty": 0.01},

    {"description": "leverage_2x_no_slip",
     "max_leverage": 2.0, "short_borrow_apr": 0.0001712,
     "fill_slippage_bps": 0.0, "anneal_lr": True,
     "ent_coef": 0.05},

    # --- Large architecture (requires A100+) ---
    {"description": "h2048_anneal",
     "hidden_size": 2048, "anneal_lr": True,
     "ent_coef": 0.05, "fill_slippage_bps": 5.0,
     "trade_penalty": 0.05, "requires_gpu": "a100"},

    {"description": "h2048_ent_anneal",
     "hidden_size": 2048, "anneal_lr": True, "anneal_ent": True,
     "ent_coef": 0.08, "ent_coef_end": 0.02, "fill_slippage_bps": 5.0,
     "trade_penalty": 0.05, "requires_gpu": "a100"},

    {"description": "h2048_resmlp_anneal",
     "hidden_size": 2048, "arch": "resmlp", "anneal_lr": True,
     "ent_coef": 0.05, "fill_slippage_bps": 5.0,
     "requires_gpu": "a100"},

    {"description": "h4096_anneal",
     "hidden_size": 4096, "anneal_lr": True,
     "ent_coef": 0.05, "fill_slippage_bps": 5.0,
     "requires_gpu": "h100"},

    # Architecture experiments — transformer / GRU / depth-recurrence / relu_sq
    # arch values silently fall back to mlp in train.py until those archs land
    {"description": "transformer_h256",
     "arch": "transformer", "hidden_size": 256},
    {"description": "transformer_h256_tp05",
     "arch": "transformer", "hidden_size": 256, "trade_penalty": 0.05},
    {"description": "gru_h256",
     "arch": "gru", "hidden_size": 256},
    {"description": "gru_h512",
     "arch": "gru", "hidden_size": 512},
    {"description": "depth_recur_h512",
     "arch": "depth_recurrence", "hidden_size": 512},
    {"description": "depth_recur_h1024",
     "arch": "depth_recurrence", "hidden_size": 1024},
    # mlp_relu_sq: relu² activation sharpens gradients on positive activations
    {"description": "relu_sq_h1024",
     "arch": "mlp_relu_sq", "hidden_size": 1024},
    {"description": "relu_sq_tp05",
     "arch": "mlp_relu_sq", "hidden_size": 1024, "trade_penalty": 0.05},

    # Large model variants
    {"description": "h2048_anneal_tp05",
     "hidden_size": 2048, "anneal_lr": True,
     "ent_coef": 0.05, "fill_slippage_bps": 5.0,
     "trade_penalty": 0.05, "requires_gpu": "a100"},

    # Combinations of best daily factors
    {"description": "combo_best_daily",
     "hidden_size": 1024, "trade_penalty": 0.05,
     "fill_slippage_bps": 5.0, "lr_schedule": "cosine",
     "obs_norm": True, "anneal_lr": True},
    {"description": "combo_best_hourly",
     "hidden_size": 1024, "fill_slippage_bps": 5.0,
     "obs_norm": True, "ent_coef": 0.05, "anneal_lr": True,
     "trade_penalty": 0.01},

    # Calmar-proxy: drawdown + downside penalties approximate annual_return/max_dd
    {"description": "calmar_focus",
     "trade_penalty": 0.03,
     "drawdown_penalty": 0.1, "smooth_downside_penalty": 0.2},
    {"description": "calmar_strong",
     "trade_penalty": 0.05,
     "drawdown_penalty": 0.2, "smooth_downside_penalty": 0.3},

    # Cosine LR × trade-penalty and slippage crosses
    {"description": "cosine_lr_tp05",
     "lr_schedule": "cosine", "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05,
     "trade_penalty": 0.05, "anneal_lr": True},
    {"description": "cosine_lr_slip5",
     "lr_schedule": "cosine", "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05,
     "fill_slippage_bps": 5.0, "anneal_lr": True},

    # Entropy annealing × trade-penalty cross
    {"description": "ent_anneal_tp05",
     "anneal_ent": True, "ent_coef": 0.08, "ent_coef_end": 0.02,
     "trade_penalty": 0.05},
    {"description": "ent01_tp03",
     "ent_coef": 0.1, "trade_penalty": 0.03},

    # Variance testing: best configs at alternative seeds
    {"description": "trade_pen_05_s123",
     "trade_penalty": 0.05, "seed": 123},
    {"description": "trade_pen_05_s7",
     "trade_penalty": 0.05, "seed": 7},
    {"description": "slip5_s123",
     "fill_slippage_bps": 5.0, "seed": 123},

    # GAE lambda sweep
    {"description": "gae_lambda_09",
     "gae_lambda": 0.9, "trade_penalty": 0.05},
    {"description": "gae_lambda_099",
     "gae_lambda": 0.99, "trade_penalty": 0.05},

    # Reward scale sweep
    {"description": "reward_scale_5",
     "reward_scale": 5.0, "trade_penalty": 0.05},
    {"description": "reward_scale_20",
     "reward_scale": 20.0, "trade_penalty": 0.05},

    # Longer rollout / more PPO reuse
    {"description": "rollout_512_tp05",
     "rollout_len": 512, "trade_penalty": 0.05},
    {"description": "ppo_epochs_8",
     "ppo_epochs": 8, "trade_penalty": 0.05},

    # Combined champion: trade_pen + obs_norm + cosine + slip + anneal + wd
    {"description": "robust_champion",
     "hidden_size": 1024, "trade_penalty": 0.05, "fill_slippage_bps": 5.0,
     "obs_norm": True, "anneal_lr": True, "lr_schedule": "cosine",
     "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05,
     "weight_decay": 0.005, "ent_coef": 0.05},

    # Muon optimizer experiments — Newton-Schulz orthogonalization, 2-3x more stable than Adam
    {"description": "muon_baseline",
     "optimizer": "muon", "lr": 0.02, "trade_penalty": 0.0},
    {"description": "muon_tp05",
     "optimizer": "muon", "lr": 0.02, "trade_penalty": 0.05},
    {"description": "muon_tp05_slip5",
     "optimizer": "muon", "lr": 0.02, "trade_penalty": 0.05, "fill_slippage_bps": 5.0},
    {"description": "muon_lr001",
     "optimizer": "muon", "lr": 0.01, "trade_penalty": 0.05},
    {"description": "muon_relu_sq",
     "optimizer": "muon", "lr": 0.02, "arch": "mlp_relu_sq", "trade_penalty": 0.05},

    # NorMuon — scales update Frobenius norm to match param norm (from modded-nanogpt speedrun)
    {"description": "normuon_tp05",
     "optimizer": "muon", "lr": 0.02, "trade_penalty": 0.05,
     "muon_norm_update": True},
    {"description": "normuon_tp05_slip5",
     "optimizer": "muon", "lr": 0.02, "trade_penalty": 0.05, "fill_slippage_bps": 5.0,
     "muon_norm_update": True},
]

# Alias used by sweep scripts and verification commands.
TRIAL_CONFIGS = EXPERIMENTS

# ---------------------------------------------------------------------------
# Crypto34 hourly focused experiments -- 34-symbol Binance hourly bars.
#
# Based on champion findings: h1024, mlp, obs_norm, anneal_lr, entropy
# annealing 0.05->0.02. Systematic sweep of trade_penalty, fill_slippage,
# weight_decay across 6 seeds.
#
# 6 base configs x 6 seeds = 36 experiments.
# ---------------------------------------------------------------------------

_C34H_BASE = {
    "hidden_size": 1024,
    "arch": "mlp",
    "obs_norm": True,
    "anneal_lr": True,
    "anneal_ent": True,
    "ent_coef": 0.05,
    "ent_coef_end": 0.02,
}
_C34H_SEEDS = [7, 19, 33, 42, 80, 99]

_C34H_VARIANTS = [
    ("c34h_tp01_slip5_wd01", {"trade_penalty": 0.01, "fill_slippage_bps": 5.0, "weight_decay": 0.01}),
    ("c34h_tp01_slip5_wd05", {"trade_penalty": 0.01, "fill_slippage_bps": 5.0, "weight_decay": 0.05}),
    ("c34h_tp03_slip5_wd01", {"trade_penalty": 0.03, "fill_slippage_bps": 5.0, "weight_decay": 0.01}),
    ("c34h_tp03_slip5_wd05", {"trade_penalty": 0.03, "fill_slippage_bps": 5.0, "weight_decay": 0.05}),
    ("c34h_tp05_slip8_wd01", {"trade_penalty": 0.05, "fill_slippage_bps": 8.0, "weight_decay": 0.01}),
    ("c34h_tp05_slip8_wd05", {"trade_penalty": 0.05, "fill_slippage_bps": 8.0, "weight_decay": 0.05}),
]

CRYPTO34_HOURLY_EXPERIMENTS: list[dict] = []
for _prefix, _overrides in _C34H_VARIANTS:
    for _seed in _C34H_SEEDS:
        CRYPTO34_HOURLY_EXPERIMENTS.append({
            **_C34H_BASE,
            **_overrides,
            "seed": _seed,
            "description": f"{_prefix}_s{_seed}",
        })


# ---------------------------------------------------------------------------
# Stock-specific experiment configurations for Alpaca US equity daily trading.
#
# Key differences vs crypto:
#   - Alpaca fee ~10bps per trade (fee_rate=0.001); include realistic slippage too.
#   - Daily bars: periods_per_year=252. max_steps set at run time via --max-steps-override.
#   - Long-only makes sense for the bull-market regime; no --long-only flag in train.py
#     so we use heavy short_borrow_apr to deter shorts, and high trade_penalty to
#     discourage excessive churn on daily bars.
#   - anneal_lr is critical — keeps it for all configs.
# ---------------------------------------------------------------------------

STOCK_EXPERIMENTS: list[dict] = [
    # Baseline: same defaults but fee=10bps (Alpaca maker/taker)
    {"description": "stock_baseline"},

    # --- Trade penalty sweep (reduce churn on daily bars) ---
    {"description": "stock_trade_pen_01", "trade_penalty": 0.01},
    {"description": "stock_trade_pen_02", "trade_penalty": 0.02},
    {"description": "stock_trade_pen_03", "trade_penalty": 0.03},
    {"description": "stock_trade_pen_05", "trade_penalty": 0.05},
    {"description": "stock_trade_pen_08", "trade_penalty": 0.08},
    {"description": "stock_trade_pen_10", "trade_penalty": 0.10},

    # --- Long/short access (no borrow cost — full short access) ---
    {"description": "stock_longshort",
     "short_borrow_apr": 0.0, "trade_penalty": 0.02},

    # --- Entropy coefficient variants ---
    {"description": "stock_ent_03", "ent_coef": 0.03},
    {"description": "stock_ent_05", "ent_coef": 0.05},   # matches baseline
    {"description": "stock_ent_08", "ent_coef": 0.08},

    # --- Hidden size variants ---
    {"description": "stock_h512", "hidden_size": 512},
    {"description": "stock_h1024", "hidden_size": 1024},   # matches baseline

    # --- Slippage variants (train with friction to force wider edges) ---
    {"description": "stock_slip_5bps",  "fill_slippage_bps": 5.0},
    {"description": "stock_slip_10bps", "fill_slippage_bps": 10.0},
    {"description": "stock_slip_15bps", "fill_slippage_bps": 15.0},

    # --- LR schedule variants ---
    {"description": "stock_cosine_lr",
     "lr_schedule": "cosine", "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05},
    {"description": "stock_no_anneal", "anneal_lr": False},

    # --- Gamma (discount) variants ---
    {"description": "stock_high_gamma_999", "gamma": 0.999},
    {"description": "stock_gamma_995",      "gamma": 0.995},

    # --- Risk/penalty shaping ---
    {"description": "stock_drawdown_pen",
     "drawdown_penalty": 0.05, "trade_penalty": 0.03},
    {"description": "stock_smooth_pen",
     "smooth_downside_penalty": 0.5, "smooth_downside_temperature": 0.02,
     "trade_penalty": 0.02},

    # --- Reward scale variants ---
    {"description": "stock_reward_scale_5",  "reward_scale": 5.0,  "trade_penalty": 0.03},
    {"description": "stock_reward_scale_20", "reward_scale": 20.0, "trade_penalty": 0.03},

    # --- Observation normalisation (often helps with heterogeneous stock features) ---
    {"description": "stock_obs_norm",       "obs_norm": True},
    {"description": "stock_obs_norm_tp05",  "obs_norm": True, "trade_penalty": 0.05},

    # --- Weight decay for generalisation ---
    {"description": "stock_wd_01",   "weight_decay": 0.01},
    {"description": "stock_wd_05",   "weight_decay": 0.05},

    # --- Combined strong regularisation ---
    {"description": "stock_reg_combo",
     "obs_norm": True, "weight_decay": 0.05, "fill_slippage_bps": 10.0,
     "trade_penalty": 0.05},

    # --- Robust champion (transplanted from crypto best-daily findings) ---
    {"description": "stock_robust_champion",
     "hidden_size": 1024, "trade_penalty": 0.05, "fill_slippage_bps": 5.0,
     "obs_norm": True, "anneal_lr": True, "lr_schedule": "cosine",
     "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05,
     "weight_decay": 0.005, "ent_coef": 0.05},

    # --- Sortino-focused: low entropy + high trade penalty ---
    {"description": "stock_sortino_low_ent_tp",
     "ent_coef": 0.01, "trade_penalty": 0.10},

    # --- cosine + slippage cross ---
    {"description": "stock_cosine_slip",
     "lr_schedule": "cosine", "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05,
     "fill_slippage_bps": 10.0, "trade_penalty": 0.05},

    # --- Alternative seeds (variance check on best config) ---
    {"description": "stock_trade_pen_05_s123",
     "trade_penalty": 0.05, "seed": 123},
    {"description": "stock_trade_pen_05_s7",
     "trade_penalty": 0.05, "seed": 7},

    # --- Smaller model + strong reg (may generalise with fewer daily bars) ---
    {"description": "stock_h512_reg",
     "hidden_size": 512, "obs_norm": True, "weight_decay": 0.05,
     "fill_slippage_bps": 10.0, "trade_penalty": 0.05},

    # -----------------------------------------------------------------------
    # H100-scale configs: 256 parallel envs, minibatch 4096, BF16, CUDA graph
    # -----------------------------------------------------------------------

    # FIRST: lr=1e-4 + anneal H100 configs — validated best for stocks11 data (2015/2012)
    # Arch comparison 2026-03-22: lr1e4_anneal_s777 → robust=-40.3 (2012), -54.9 (2015)
    # beats all lr=3e-4 configs AND h2048 configs on stocks11 data
    # H100 at ~450k sps × 90s = 40M steps — converges well above 33M threshold
    {"description": "lr1e4_anneal_h100",
     "lr": 1e-4, "anneal_lr": True, "ent_coef": 0.05, "seed": 777,
     "num_envs": 256, "minibatch_size": 4096, "cuda_graph_ppo": True, "use_bf16": True,
     "requires_gpu": "h100"},
    {"description": "lr1e4_anneal_h100_s42",
     "lr": 1e-4, "anneal_lr": True, "ent_coef": 0.05, "seed": 42,
     "num_envs": 256, "minibatch_size": 4096, "cuda_graph_ppo": True, "use_bf16": True,
     "requires_gpu": "h100"},
    {"description": "lr1e4_anneal_h100_s9621",
     "lr": 1e-4, "anneal_lr": True, "ent_coef": 0.05, "seed": 9621,
     "num_envs": 256, "minibatch_size": 4096, "cuda_graph_ppo": True, "use_bf16": True,
     "requires_gpu": "h100"},
    {"description": "lr1e4_noanneal_h100",
     "lr": 1e-4, "anneal_lr": False, "ent_coef": 0.05, "seed": 9621,
     "num_envs": 256, "minibatch_size": 4096, "cuda_graph_ppo": True, "use_bf16": True,
     "requires_gpu": "h100"},
    {"description": "lr1e4_anneal_wd01_h100",
     "lr": 1e-4, "anneal_lr": True, "ent_coef": 0.05, "weight_decay": 0.01, "seed": 777,
     "num_envs": 256, "minibatch_size": 4096, "cuda_graph_ppo": True, "use_bf16": True,
     "requires_gpu": "h100"},

    # Replicate best known config (h1024 + anneal_lr) at H100 scale
    {"description": "h1024_h100",
     "hidden_size": 1024, "anneal_lr": True, "ent_coef": 0.05,
     "num_envs": 256, "minibatch_size": 4096, "cuda_graph_ppo": True, "use_bf16": True,
     "requires_gpu": "h100"},

    # h2048 at H100 scale — 256 envs gets ~148k sps on RTX5090, ~520k on H100 → 47M steps @ 90s
    # trade_penalty=0.03 is proven winner (tp03_s777 +9.79%). fill_slippage=5bps for friction.
    {"description": "h2048_h100",
     "hidden_size": 2048, "anneal_lr": True, "ent_coef": 0.05,
     "fill_slippage_bps": 5.0, "trade_penalty": 0.03,
     "num_envs": 256, "minibatch_size": 4096, "cuda_graph_ppo": True, "use_bf16": True,
     "requires_gpu": "h100"},

    # h2048 H100 variant with tp03 seed 777 (known convergence seed)
    {"description": "h2048_h100_tp03_s777",
     "hidden_size": 2048, "anneal_lr": True, "ent_coef": 0.05, "seed": 777,
     "fill_slippage_bps": 5.0, "trade_penalty": 0.03,
     "num_envs": 256, "minibatch_size": 4096, "cuda_graph_ppo": True, "use_bf16": True,
     "requires_gpu": "h100"},

    # h2048 H100 with weight_decay for regularization (h2048 is 4x larger, needs more reg)
    {"description": "h2048_h100_wd01",
     "hidden_size": 2048, "anneal_lr": True, "ent_coef": 0.05,
     "fill_slippage_bps": 5.0, "trade_penalty": 0.03, "weight_decay": 0.01,
     "num_envs": 256, "minibatch_size": 4096, "cuda_graph_ppo": True, "use_bf16": True,
     "requires_gpu": "h100"},

    # ResidualMLP at H100 scale
    {"description": "resmlp_h100",
     "arch": "resmlp", "hidden_size": 1024, "anneal_lr": True, "ent_coef": 0.05,
     "num_envs": 256, "minibatch_size": 4096, "cuda_graph_ppo": True, "use_bf16": True,
     "requires_gpu": "h100"},

    # Transformer can now afford wider hidden at H100 speed
    {"description": "transformer_h100",
     "arch": "transformer", "hidden_size": 512, "anneal_lr": True,
     "trade_penalty": 0.05,
     "num_envs": 256, "minibatch_size": 4096, "cuda_graph_ppo": True, "use_bf16": True,
     "requires_gpu": "h100"},

    # Muon optimizer needs faster steps to converge — H100 enables that
    {"description": "muon_h100",
     "optimizer": "muon", "lr": 0.02, "hidden_size": 1024,
     "trade_penalty": 0.05,
     "num_envs": 256, "minibatch_size": 4096, "cuda_graph_ppo": True, "use_bf16": True,
     "requires_gpu": "h100"},

    # Replicate OOS best (ent_coef=0.05) at H100 scale
    {"description": "ent_005_h100",
     "ent_coef": 0.05, "hidden_size": 1024, "anneal_lr": True,
     "num_envs": 256, "minibatch_size": 4096, "cuda_graph_ppo": True, "use_bf16": True,
     "requires_gpu": "h100"},

    # Slippage friction to force wider edges — H100 scale
    {"description": "slip_5bps_h100",
     "fill_slippage_bps": 5.0, "hidden_size": 1024, "anneal_lr": True,
     "num_envs": 256, "minibatch_size": 4096, "cuda_graph_ppo": True, "use_bf16": True,
     "requires_gpu": "h100"},

    # --- Muon H100 variants ---
    {"description": "muon_wd_0",
     "optimizer": "muon", "lr": 0.02, "weight_decay": 0.0,
     "hidden_size": 1024, "trade_penalty": 0.05,
     "num_envs": 256, "minibatch_size": 4096, "cuda_graph_ppo": True, "use_bf16": True,
     "requires_gpu": "h100"},

    {"description": "muon_wd_005",
     "optimizer": "muon", "lr": 0.02, "weight_decay": 0.005,
     "hidden_size": 1024, "trade_penalty": 0.05,
     "num_envs": 256, "minibatch_size": 4096, "cuda_graph_ppo": True, "use_bf16": True,
     "requires_gpu": "h100"},

    {"description": "muon_ent_005",
     "optimizer": "muon", "lr": 0.02, "ent_coef": 0.05,
     "hidden_size": 1024, "trade_penalty": 0.05,
     "num_envs": 256, "minibatch_size": 4096, "cuda_graph_ppo": True, "use_bf16": True,
     "requires_gpu": "h100"},

    # -----------------------------------------------------------------------
    # A40/RTX6000-Ada configs: 128 parallel envs, minibatch 2048, BF16, CUDA graph
    # A40: CC 8.6 (Ampere), 48GB VRAM, $0.69/hr — same price as RTX 4090 but 48GB
    # RTX 6000 Ada: CC 8.9 (Ada Lovelace), 48GB VRAM, $0.79/hr
    # -----------------------------------------------------------------------

    # Replicate best known config (h1024 + anneal_lr) at A40 scale
    {"description": "h1024_a40",
     "hidden_size": 1024, "anneal_lr": True, "ent_coef": 0.05,
     "num_envs": 128, "minibatch_size": 2048, "cuda_graph_ppo": True, "use_bf16": True,
     "requires_gpu": "a40"},

    # h2048 now feasible with A40's 48GB VRAM
    {"description": "h2048_a40",
     "hidden_size": 2048, "anneal_lr": True, "ent_coef": 0.05,
     "fill_slippage_bps": 5.0, "trade_penalty": 0.05,
     "num_envs": 128, "minibatch_size": 2048, "cuda_graph_ppo": True, "use_bf16": True,
     "requires_gpu": "a40"},

    # ResidualMLP at A40 scale
    {"description": "resmlp_a40",
     "arch": "resmlp", "hidden_size": 1024, "anneal_lr": True, "ent_coef": 0.05,
     "num_envs": 128, "minibatch_size": 2048, "cuda_graph_ppo": True, "use_bf16": True,
     "requires_gpu": "a40"},

    # Slippage friction to force wider edges — A40 scale
    {"description": "slip_5bps_a40",
     "fill_slippage_bps": 5.0, "hidden_size": 1024, "anneal_lr": True,
     "num_envs": 128, "minibatch_size": 2048, "cuda_graph_ppo": True, "use_bf16": True,
     "requires_gpu": "a40"},

    # Replicate OOS best (ent_coef=0.05) at A40 scale
    {"description": "ent_005_a40",
     "ent_coef": 0.05, "hidden_size": 1024, "anneal_lr": True,
     "num_envs": 128, "minibatch_size": 2048, "cuda_graph_ppo": True, "use_bf16": True,
     "requires_gpu": "a40"},

    # -----------------------------------------------------------------------
    # trade_pen_03 variants — KEY CONFIG for extended training (2020-2025).
    # local calibration 2026-03-22: trade_pen_03 scored +3.1 (seed 777) and
    # -7.8 (seed 999) on hard 201-day val vs -102 with old training data.
    # These variants explore the neighbourhood around the sweet spot.
    # -----------------------------------------------------------------------
    # Multi-seed tp03 sweep: seed=777 was the ONLY positive on hard val (holdout=+3.10).
    # Run without --seed override so each config uses its own explicit seed.
    # Confirmed: trade_penalty=0.03 forces sit-out behaviour in bear markets.
    {"description": "tp03_s777",  "trade_penalty": 0.03, "seed": 777},   # KNOWN WINNER
    {"description": "tp03_s7",    "trade_penalty": 0.03, "seed": 7},
    {"description": "tp03_s42",   "trade_penalty": 0.03, "seed": 42},
    {"description": "tp03_s123",  "trade_penalty": 0.03, "seed": 123},
    {"description": "tp03_s888",  "trade_penalty": 0.03, "seed": 888},
    {"description": "tp03_s1111", "trade_penalty": 0.03, "seed": 1111},
    {"description": "tp03_s2272", "trade_penalty": 0.03, "seed": 2272},
    {"description": "tp03_s3141", "trade_penalty": 0.03, "seed": 3141},
    {"description": "tp03_s4242", "trade_penalty": 0.03, "seed": 4242},
    {"description": "tp03_s5678", "trade_penalty": 0.03, "seed": 5678},
    {"description": "tp03_s7777", "trade_penalty": 0.03, "seed": 7777},
    {"description": "tp03_s9999", "trade_penalty": 0.03, "seed": 9999},
    # tp03 + wd=0.01: 2nd best modifier; try with multiple seeds
    {"description": "tp03_wd01_s777",  "trade_penalty": 0.03, "weight_decay": 0.01, "seed": 777},
    {"description": "tp03_wd01_s42",   "trade_penalty": 0.03, "weight_decay": 0.01, "seed": 42},
    {"description": "tp03_wd01_s2272", "trade_penalty": 0.03, "weight_decay": 0.01, "seed": 2272},
    # tp03 + h2048: 3rd best modifier; larger net may converge better
    {"description": "tp03_h2048_s777",  "trade_penalty": 0.03, "hidden_size": 2048, "seed": 777},
    {"description": "tp03_h2048_s42",   "trade_penalty": 0.03, "hidden_size": 2048, "seed": 42},
    {"description": "tp03_h2048_s2272", "trade_penalty": 0.03, "hidden_size": 2048, "seed": 2272},
    # Remaining modifier variants (no explicit seed = uses default 42)
    {"description": "tp03_slip5",  "trade_penalty": 0.03, "fill_slippage_bps": 5.0},
    {"description": "tp03_slip10", "trade_penalty": 0.03, "fill_slippage_bps": 10.0},
    {"description": "tp03_wd01",   "trade_penalty": 0.03, "weight_decay": 0.01},
    {"description": "tp03_wd05",   "trade_penalty": 0.03, "weight_decay": 0.05},
    {"description": "tp03_obs",    "trade_penalty": 0.03, "obs_norm": True},
    {"description": "tp03_ent03",  "trade_penalty": 0.03, "ent_coef": 0.03},
    {"description": "tp03_annent", "trade_penalty": 0.03, "anneal_ent": True},
    {"description": "tp03_h512",   "trade_penalty": 0.03, "hidden_size": 512},
    {"description": "tp03_h2048",  "trade_penalty": 0.03, "hidden_size": 2048},
    {"description": "tp03_cosine", "trade_penalty": 0.03,
     "lr_schedule": "cosine", "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05},
    {"description": "tp03_full_reg",
     "trade_penalty": 0.03, "obs_norm": True, "weight_decay": 0.05,
     "fill_slippage_bps": 5.0, "ent_coef": 0.05},

    # Best combinations (from 2026-03-22 tp03 variants sweep — note: those used --seed 1337
    # override which masked explicit seeds. Retry these without global seed override.)
    {"description": "tp03_s2272_wd01",
     "trade_penalty": 0.03, "seed": 2272, "weight_decay": 0.01},
    {"description": "tp03_h2048_wd01",
     "trade_penalty": 0.03, "hidden_size": 2048, "weight_decay": 0.01},
    {"description": "tp03_s2272_h2048",
     "trade_penalty": 0.03, "seed": 2272, "hidden_size": 2048},

    # -----------------------------------------------------------------------
    # lr=1e-4 configs — KEY for stocks11 data (2015-2025, longer/more-volatile history)
    # Finding 2026-03-22: lr=3e-4 (anneal) converges for stocks12_2019 but collapses to
    # hold-cash on stocks11_2015. lr=1e-4 (no-anneal) converges on stocks11_2015.
    # random_mut_9621 (lr=1e-4, no-anneal) gave robust=-41.2 vs stocks12's -128.7 on
    # the same config — 3x better robustness. These named configs ensure H100 runs on
    # stocks11 data hit converging configs early (before random mutations at trial ~92).
    # -----------------------------------------------------------------------
    {"description": "lr1e4_s777",     "lr": 1e-4, "anneal_lr": False, "seed": 777},
    {"description": "lr1e4_s42",      "lr": 1e-4, "anneal_lr": False, "seed": 42},
    {"description": "lr1e4_s9621",    "lr": 1e-4, "anneal_lr": False, "seed": 9621},  # seed from winning trial
    {"description": "lr1e4_s1137",    "lr": 1e-4, "anneal_lr": False, "seed": 1137},
    {"description": "lr1e4_wd01_s777", "lr": 1e-4, "anneal_lr": False, "weight_decay": 0.01, "seed": 777},
    {"description": "lr1e4_wd005_s777","lr": 1e-4, "anneal_lr": False, "weight_decay": 0.005, "seed": 777},
    {"description": "lr1e4_slip5_s777","lr": 1e-4, "anneal_lr": False, "fill_slippage_bps": 5.0, "seed": 777},
    {"description": "lr1e4_anneal_s777","lr": 1e-4, "anneal_lr": True, "seed": 777},
    # h2048 + lr=1e-4: test if larger net helps on stocks11's longer/noisier data
    # num_envs=256 keeps sps high despite larger net (148k on RTX5090, ~520k on H100)
    {"description": "lr1e4_h2048_s777", "lr": 1e-4, "anneal_lr": False, "hidden_size": 2048,
     "num_envs": 256, "minibatch_size": 4096, "seed": 777},
    {"description": "lr1e4_h2048_s42",  "lr": 1e-4, "anneal_lr": False, "hidden_size": 2048,
     "num_envs": 256, "minibatch_size": 4096, "seed": 42},

    # -----------------------------------------------------------------------
    # Focused lr=1e-4 + anneal_lr=True exploration (added 2026-03-22)
    # Context: arch comparison confirmed lr1e4_anneal_s777 → robust=-40.3 on stocks11_2012
    # (BEST known). Now systematically exploring around this winner:
    #   (A) More seeds — is seed=777 lucky or is anneal+lr1e4 consistently good?
    #   (B) Trade penalty — reduce over-trading (10bps fee already, adding penalty may generalize)
    #   (C) Fill slippage training — forces wider edges → should generalize better
    #   (D) Entropy tuning — 0.05 default; try 0.03 (less noise) and 0.08 (more explore)
    #   (E) Weight decay — standard regularizer
    #   (F) Combo tests — best features combined
    #   (G) lr=2e-4 midpoint — between 1e-4 and 3e-4
    # All use h1024 (confirmed >> h2048), stocks11_2012 data (confirmed >> 2015/2019)
    # -----------------------------------------------------------------------
    # (A) Seed sweep — same config as lr1e4_anneal_s777, different seeds
    {"description": "lr1e4_anneal_s42",    "lr": 1e-4, "anneal_lr": True, "seed": 42},
    {"description": "lr1e4_anneal_s9621",  "lr": 1e-4, "anneal_lr": True, "seed": 9621},
    {"description": "lr1e4_anneal_s1137",  "lr": 1e-4, "anneal_lr": True, "seed": 1137},
    {"description": "lr1e4_anneal_s2718",  "lr": 1e-4, "anneal_lr": True, "seed": 2718},
    {"description": "lr1e4_anneal_s31415", "lr": 1e-4, "anneal_lr": True, "seed": 31415},
    {"description": "lr1e4_anneal_s1234",  "lr": 1e-4, "anneal_lr": True, "seed": 1234},
    {"description": "lr1e4_anneal_s5678",  "lr": 1e-4, "anneal_lr": True, "seed": 5678},
    {"description": "lr1e4_anneal_s314",   "lr": 1e-4, "anneal_lr": True, "seed": 314},
    {"description": "lr1e4_anneal_s271",   "lr": 1e-4, "anneal_lr": True, "seed": 271},
    {"description": "lr1e4_anneal_s42_ent05", "lr": 1e-4, "anneal_lr": True, "seed": 42, "ent_coef": 0.05},
    # (B) Trade penalty (all anneal=True, seed=777)
    {"description": "lr1e4_anneal_tp01",   "lr": 1e-4, "anneal_lr": True, "seed": 777, "trade_penalty": 0.01},
    {"description": "lr1e4_anneal_tp02",   "lr": 1e-4, "anneal_lr": True, "seed": 777, "trade_penalty": 0.02},
    {"description": "lr1e4_anneal_tp03",   "lr": 1e-4, "anneal_lr": True, "seed": 777, "trade_penalty": 0.03},
    # (C) Fill slippage during training (all anneal=True, seed=777)
    {"description": "lr1e4_anneal_slip5",  "lr": 1e-4, "anneal_lr": True, "seed": 777, "fill_slippage_bps": 5.0},
    {"description": "lr1e4_anneal_slip8",  "lr": 1e-4, "anneal_lr": True, "seed": 777, "fill_slippage_bps": 8.0},
    {"description": "lr1e4_anneal_slip12", "lr": 1e-4, "anneal_lr": True, "seed": 777, "fill_slippage_bps": 12.0},
    # (D) Entropy tuning (anneal=True, seed=777)
    {"description": "lr1e4_anneal_ent03",  "lr": 1e-4, "anneal_lr": True, "seed": 777, "ent_coef": 0.03},
    {"description": "lr1e4_anneal_ent08",  "lr": 1e-4, "anneal_lr": True, "seed": 777, "ent_coef": 0.08},
    # (E) Weight decay (anneal=True, seed=777)
    {"description": "lr1e4_anneal_wd001",  "lr": 1e-4, "anneal_lr": True, "seed": 777, "weight_decay": 0.001},
    {"description": "lr1e4_anneal_wd005",  "lr": 1e-4, "anneal_lr": True, "seed": 777, "weight_decay": 0.005},
    {"description": "lr1e4_anneal_wd01",   "lr": 1e-4, "anneal_lr": True, "seed": 777, "weight_decay": 0.01},
    # (F) Combo tests
    {"description": "lr1e4_anneal_tp01_slip5",  "lr": 1e-4, "anneal_lr": True, "seed": 777,
     "trade_penalty": 0.01, "fill_slippage_bps": 5.0},
    {"description": "lr1e4_anneal_tp01_slip8",  "lr": 1e-4, "anneal_lr": True, "seed": 777,
     "trade_penalty": 0.01, "fill_slippage_bps": 8.0},
    {"description": "lr1e4_anneal_wd005_slip5", "lr": 1e-4, "anneal_lr": True, "seed": 777,
     "weight_decay": 0.005, "fill_slippage_bps": 5.0},
    {"description": "lr1e4_anneal_wd001_s42",   "lr": 1e-4, "anneal_lr": True, "seed": 42,
     "weight_decay": 0.001},
    {"description": "lr1e4_anneal_tp01_s42",    "lr": 1e-4, "anneal_lr": True, "seed": 42,
     "trade_penalty": 0.01},
    # (G) lr=2e-4 (midpoint — may converge faster while still stable)
    {"description": "lr2e4_anneal_s777",   "lr": 2e-4, "anneal_lr": True, "seed": 777},
    {"description": "lr2e4_anneal_s42",    "lr": 2e-4, "anneal_lr": True, "seed": 42},
    {"description": "lr2e4_anneal_s9621",  "lr": 2e-4, "anneal_lr": True, "seed": 9621},
    # -----------------------------------------------------------------------
    # Architecture exploration (added 2026-03-22)
    # Context: mlp h1024 is current best. Testing:
    #   (H) transformer — cross-symbol attention (each symbol = token).
    #       Learns correlations like "when NVDA up → buy MSFT".
    #       Could be powerful for stocks where cross-symbol signals exist.
    #   (I) resmlp — residual MLP, deeper effective capacity without h2048 cost.
    #   (J) gru — gated temporal policy, learns short-term trend gating.
    # All use lr=1e-4 + anneal (confirmed critical for stocks11_2012).
    # seed=777 (best known) + seed=42 (comparison).
    # -----------------------------------------------------------------------
    # (H) Transformer (cross-symbol attention)
    {"description": "transformer_lr1e4_s777",    "lr": 1e-4, "anneal_lr": True, "seed": 777, "arch": "transformer"},
    {"description": "transformer_lr1e4_s42",     "lr": 1e-4, "anneal_lr": True, "seed": 42,  "arch": "transformer"},
    {"description": "transformer_lr1e4_s9621",   "lr": 1e-4, "anneal_lr": True, "seed": 9621,"arch": "transformer"},
    {"description": "transformer_lr1e4_h512_s777","lr": 1e-4, "anneal_lr": True, "seed": 777, "arch": "transformer", "hidden_size": 512},
    # (I) ResidualMLP
    {"description": "resmlp_lr1e4_s777",         "lr": 1e-4, "anneal_lr": True, "seed": 777, "arch": "resmlp"},
    {"description": "resmlp_lr1e4_s42",          "lr": 1e-4, "anneal_lr": True, "seed": 42,  "arch": "resmlp"},
    {"description": "resmlp_lr1e4_h2048_s777",   "lr": 1e-4, "anneal_lr": True, "seed": 777, "arch": "resmlp",
     "hidden_size": 2048, "num_envs": 256, "minibatch_size": 4096},
    # (J) GRU
    {"description": "gru_lr1e4_s777",            "lr": 1e-4, "anneal_lr": True, "seed": 777, "arch": "gru"},
    {"description": "gru_lr1e4_s42",             "lr": 1e-4, "anneal_lr": True, "seed": 42,  "arch": "gru"},
    # -----------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # (K) s1137 hyperparam cross — seed=1137 is best seed on stocks11_2012 (robust=-21.4)
    # Cross with best hyperparams to find the optimal combination.
    # s5678 is 2nd best (robust=-55.5), worth testing too.
    # 2026-03-22: added after main seed sweep confirmed s1137 dominance.
    # ---------------------------------------------------------------------------
    # s1137 × trade penalty
    {"description": "s1137_tp01", "lr": 1e-4, "anneal_lr": True, "seed": 1137, "trade_penalty": 0.01},
    {"description": "s1137_tp02", "lr": 1e-4, "anneal_lr": True, "seed": 1137, "trade_penalty": 0.02},
    {"description": "s1137_tp03", "lr": 1e-4, "anneal_lr": True, "seed": 1137, "trade_penalty": 0.03},
    # s1137 × slippage (forces agent to find wider edges)
    {"description": "s1137_slip5",  "lr": 1e-4, "anneal_lr": True, "seed": 1137, "fill_slippage_bps": 5.0},
    {"description": "s1137_slip8",  "lr": 1e-4, "anneal_lr": True, "seed": 1137, "fill_slippage_bps": 8.0},
    {"description": "s1137_slip12", "lr": 1e-4, "anneal_lr": True, "seed": 1137, "fill_slippage_bps": 12.0},
    # s1137 × entropy
    {"description": "s1137_ent03",  "lr": 1e-4, "anneal_lr": True, "seed": 1137, "ent_coef": 0.03},
    {"description": "s1137_ent07",  "lr": 1e-4, "anneal_lr": True, "seed": 1137, "ent_coef": 0.07},
    # s1137 × weight decay
    {"description": "s1137_wd003",  "lr": 1e-4, "anneal_lr": True, "seed": 1137, "weight_decay": 0.003},
    {"description": "s1137_wd01",   "lr": 1e-4, "anneal_lr": True, "seed": 1137, "weight_decay": 0.01},
    # s1137 × combo (trade penalty + slippage — best of both friction sources)
    {"description": "s1137_tp02_slip8",  "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "trade_penalty": 0.02, "fill_slippage_bps": 8.0},
    {"description": "s1137_tp01_ent03",  "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "trade_penalty": 0.01, "ent_coef": 0.03},
    {"description": "s1137_tp01_wd003",  "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "trade_penalty": 0.01, "weight_decay": 0.003},
    # s5678 × best hyperparams (2nd best seed)
    {"description": "s5678_tp01",   "lr": 1e-4, "anneal_lr": True, "seed": 5678, "trade_penalty": 0.01},
    {"description": "s5678_tp02",   "lr": 1e-4, "anneal_lr": True, "seed": 5678, "trade_penalty": 0.02},
    {"description": "s5678_slip8",  "lr": 1e-4, "anneal_lr": True, "seed": 5678, "fill_slippage_bps": 8.0},
    {"description": "s5678_ent03",  "lr": 1e-4, "anneal_lr": True, "seed": 5678, "ent_coef": 0.03},
    # Additional seeds to explore (beyond the initial 9-seed sweep)
    {"description": "lr1e4_anneal_s8675", "lr": 1e-4, "anneal_lr": True, "seed": 8675},
    {"description": "lr1e4_anneal_s2345", "lr": 1e-4, "anneal_lr": True, "seed": 2345},
    {"description": "lr1e4_anneal_s999",  "lr": 1e-4, "anneal_lr": True, "seed": 999},
    {"description": "lr1e4_anneal_s13",   "lr": 1e-4, "anneal_lr": True, "seed": 13},
    # s1137 × architecture
    {"description": "s1137_transformer", "lr": 1e-4, "anneal_lr": True, "seed": 1137, "arch": "transformer"},
    {"description": "s1137_resmlp",      "lr": 1e-4, "anneal_lr": True, "seed": 1137, "arch": "resmlp"},
    {"description": "s1137_gru",         "lr": 1e-4, "anneal_lr": True, "seed": 1137, "arch": "gru"},
    # -----------------------------------------------------------------------

    # (L) s1137 × additional axes — critical gaps from first sweep pass
    # 2026-03-22: tp/slip/ent/wd confirmed to hurt s1137. Testing LR, capacity, and regularization.
    # ---------------------------------------------------------------------------
    # lr=2e-4 is between 1e-4 (good) and 3e-4 (collapses) — may improve in-sample learning
    {"description": "s1137_lr2e4",     "lr": 2e-4, "anneal_lr": True, "seed": 1137},
    # h2048 failed badly with s777(-163)/s42(-133) but s1137 may handle larger capacity differently
    {"description": "s1137_h2048",     "lr": 1e-4, "anneal_lr": True, "seed": 1137, "hidden_size": 2048,
     "num_envs": 256, "minibatch_size": 4096},
    # obs_norm=True: normalizing observations may stabilize gradients for daily data
    {"description": "s1137_obs_norm",  "lr": 1e-4, "anneal_lr": True, "seed": 1137, "obs_norm": True},
    # gamma=0.995: longer planning horizon may help daily bars where trends last weeks
    {"description": "s1137_gamma995",  "lr": 1e-4, "anneal_lr": True, "seed": 1137, "gamma": 0.995},
    # anneal_ent: decaying entropy may help exploitation late in training
    {"description": "s1137_anneal_ent","lr": 1e-4, "anneal_lr": True, "seed": 1137, "anneal_ent": True},
    # lr=2e-4 × s5678 (2nd best seed at -55.5 — does higher LR help it?)
    {"description": "s5678_lr2e4",    "lr": 2e-4, "anneal_lr": True, "seed": 5678},

    # (M) smooth_downside_penalty experiments — KEY for reducing negative windows
    # robust_score formula: -50 × negative_return_rate dominates.
    # random_mut_2272 (stocks12 champion) used smooth_downside_temperature=0.01 → 0% negative.
    # Hypothesis: smooth_downside_penalty penalizes training-time drawdowns → more conservative
    # strategy → fewer negative holdout windows.
    # ---------------------------------------------------------------------------
    # Exact random_mut_2272 formula applied to s1137 base
    {"description": "s1137_sdp01_t001",  "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "smooth_downside_penalty": 0.1, "smooth_downside_temperature": 0.01},
    # Lighter version — don't disrupt s1137's good strategy too much
    {"description": "s1137_sdp005_t002", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "smooth_downside_penalty": 0.05, "smooth_downside_temperature": 0.02},
    # Temperature sweep (how sharp the downside penalty) with fixed penalty=0.1
    {"description": "s1137_sdp01_t002",  "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "smooth_downside_penalty": 0.1, "smooth_downside_temperature": 0.02},
    {"description": "s1137_sdp02_t001",  "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.01},
    # s5678 also gets the smooth downside treatment (2nd best seed)
    {"description": "s5678_sdp01_t001",  "lr": 1e-4, "anneal_lr": True, "seed": 5678,
     "smooth_downside_penalty": 0.1, "smooth_downside_temperature": 0.01},

    # --- SDP-02 seed sweep (indices 622-631): sdp=0.2, sdt=0.01 + varied seeds
    # Tests whether the smooth_downside_penalty=0.2, temperature=0.01 config
    # (the random_mut_2272 formula) allows MORE seeds to escape the -64.87 degenerate
    # minimum compared to seed-only mode (no sdp). If success rate is >20% here
    # vs ~11% seed-only, H100 should use this config for the seed sweep.
    {"description": "sdp02_s1464", "lr": 1e-4, "anneal_lr": True, "seed": 1464,
     "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.01},
    {"description": "sdp02_s2718", "lr": 1e-4, "anneal_lr": True, "seed": 2718,
     "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.01},
    {"description": "sdp02_s31415", "lr": 1e-4, "anneal_lr": True, "seed": 31415,
     "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.01},
    {"description": "sdp02_s1234", "lr": 1e-4, "anneal_lr": True, "seed": 1234,
     "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.01},
    {"description": "sdp02_s314", "lr": 1e-4, "anneal_lr": True, "seed": 314,
     "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.01},
    {"description": "sdp02_s271", "lr": 1e-4, "anneal_lr": True, "seed": 271,
     "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.01},
    {"description": "sdp02_s999", "lr": 1e-4, "anneal_lr": True, "seed": 999,
     "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.01},
    {"description": "sdp02_s555", "lr": 1e-4, "anneal_lr": True, "seed": 555,
     "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.01},
    {"description": "sdp02_s7777", "lr": 1e-4, "anneal_lr": True, "seed": 7777,
     "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.01},
    {"description": "sdp02_s2024", "lr": 1e-4, "anneal_lr": True, "seed": 2024,
     "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.01},
    # Known-bad seeds without sdp: all at -64.87. Testing if sdp=0.2 rescues them.
    {"description": "sdp02_s7860", "lr": 1e-4, "anneal_lr": True, "seed": 7860,
     "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.01},
    {"description": "sdp02_s4533", "lr": 1e-4, "anneal_lr": True, "seed": 4533,
     "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.01},
    {"description": "sdp02_s4438", "lr": 1e-4, "anneal_lr": True, "seed": 4438,
     "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.01},
    {"description": "sdp02_s6828", "lr": 1e-4, "anneal_lr": True, "seed": 6828,
     "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.01},
    {"description": "sdp02_s5678", "lr": 1e-4, "anneal_lr": True, "seed": 5678,
     "smooth_downside_penalty": 0.2, "smooth_downside_temperature": 0.01},

    # --- O-block: PPO infrastructure params + gamma sweep (indices 195-202) ---
    # s1137 is baseline (h=1024, lr=1e-4, ent=0.05, seed=1137, rollout=256, envs=128, mb=2048)
    # Testing whether PPO update dynamics affect the s1137 performance.
    {"description": "s1137_rollout128", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "rollout_len": 128},
    {"description": "s1137_rollout512", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "rollout_len": 512},
    {"description": "s1137_envs256", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "num_envs": 256, "minibatch_size": 4096},
    {"description": "s1137_mb4096", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "minibatch_size": 4096},
    {"description": "s1137_gamma98", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "gamma": 0.98},
    {"description": "s1137_ppo_epochs2", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "ppo_epochs": 2},
    {"description": "s1137_ppo_epochs8", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "ppo_epochs": 8},
    {"description": "s1137_lr_schedule_cos", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "lr_schedule": "cosine"},
    # h=512 with proven lr=1e-4 + s1137: test if smaller capacity also escapes degenerate min
    {"description": "s1137_h512", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "hidden_size": 512},

    # --- N-block: stocks12-champion formula on stocks11_2012 (indices 203-210) ---
    # random_mut_4424 (stocks12 leaderboard) got robust=-4.02 with 0% neg using:
    # h=256, lr=3e-4, slip=12bps, dp=0.01, anneal_lr=True (no sdp, no wd)
    # Testing whether this formula transfers to stocks11_2012 (4840 days vs shorter stocks12).
    # Key question: does slip=12bps+dp=0.01 prevent the lr=3e-4 hold-cash collapse?
    {"description": "h256_lr3e4_slip12_dp01_s1137", "hidden_size": 256, "lr": 3e-4,
     "fill_slippage_bps": 12.0, "drawdown_penalty": 0.01, "seed": 1137, "anneal_lr": True},
    {"description": "h256_lr3e4_slip12_dp01_s5678", "hidden_size": 256, "lr": 3e-4,
     "fill_slippage_bps": 12.0, "drawdown_penalty": 0.01, "seed": 5678, "anneal_lr": True},
    {"description": "h256_lr1e4_slip12_dp01_s1137", "hidden_size": 256, "lr": 1e-4,
     "fill_slippage_bps": 12.0, "drawdown_penalty": 0.01, "seed": 1137, "anneal_lr": True},
    {"description": "h256_lr1e4_s1137", "hidden_size": 256, "lr": 1e-4,
     "seed": 1137, "anneal_lr": True},
    {"description": "h512_lr3e4_slip12_dp01_s1137", "hidden_size": 512, "lr": 3e-4,
     "fill_slippage_bps": 12.0, "drawdown_penalty": 0.01, "seed": 1137, "anneal_lr": True},
    {"description": "h256_lr3e4_slip12_dp0_s1137", "hidden_size": 256, "lr": 3e-4,
     "fill_slippage_bps": 12.0, "seed": 1137, "anneal_lr": True},
    {"description": "h256_lr3e4_slip0_dp01_s1137", "hidden_size": 256, "lr": 3e-4,
     "drawdown_penalty": 0.01, "seed": 1137, "anneal_lr": True},
    {"description": "h256_lr3e4_s1137", "hidden_size": 256, "lr": 3e-4,
     "seed": 1137, "anneal_lr": True},


    # -----------------------------------------------------------------------
    # P-block: unexplored PPO infrastructure + value/gradient tuning (added 2026-03-22)
    #
    # Context: s1137 (h=1024, lr=1e-4, anneal_lr=True, ent=0.05) gives robust=-21.38.
    # All prior sweeps: reward shaping, architecture, data friction, seeds.
    # NOT yet tested: vf_coef, max_grad_norm, clip_eps, reward_clip, cash_penalty
    # sweep, gae_lambda, minibatch size below default (256/512), mid-LR variants.
    #
    # Key design choices:
    #   - All use s1137 (best seed) + lr=1e-4 + anneal_lr=True as base
    #   - mb256/mb512: smaller batches = noisier but more frequent updates
    #   - vf_coef: 0.25 reduces value pressure (more PG), 1.0 increases
    #   - max_grad_norm: 0.3 tighter, 1.0/2.0 looser (helps sparse rewards)
    #   - clip_eps: 0.1 conservative updates, 0.3 aggressive
    #   - reward_clip: 2.0 tighter, 10.0 looser (default 5.0)
    #   - cash_penalty: 0 removes hold-cash penalty, 0.005 halves it
    #   - gae_lambda: 0.9 shorter credit, 0.99 longer (daily trends last weeks)
    #   - num_envs=32: fewer envs, more on-policy, may help escape degenerate attractor
    #   - lr=5e-5: very slow LR, ultra-stable convergence
    # -----------------------------------------------------------------------

    # --- P1: Value function coefficient sweep ---
    {"description": "p_vfcoef_025", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "vf_coef": 0.25},
    {"description": "p_vfcoef_10", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "vf_coef": 1.0},
    {"description": "p_vfcoef_075", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "vf_coef": 0.75},

    # --- P2: Gradient clipping norm sweep ---
    {"description": "p_gradnorm_03", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "max_grad_norm": 0.3},
    {"description": "p_gradnorm_10", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "max_grad_norm": 1.0},
    {"description": "p_gradnorm_20", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "max_grad_norm": 2.0},

    # --- P3: PPO clip epsilon sweep ---
    {"description": "p_clipeps_01", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "clip_eps": 0.1},
    {"description": "p_clipeps_03", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "clip_eps": 0.3},
    {"description": "p_clipeps_015", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "clip_eps": 0.15},

    # --- P4: Reward clip sweep ---
    {"description": "p_rwclip_20", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "reward_clip": 2.0},
    {"description": "p_rwclip_10", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "reward_clip": 10.0},

    # --- P5: Cash penalty sweep ---
    {"description": "p_cashpen_0", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "cash_penalty": 0.0},
    {"description": "p_cashpen_005", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "cash_penalty": 0.005},
    {"description": "p_cashpen_02", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "cash_penalty": 0.02},

    # --- P6: GAE lambda sweep ---
    {"description": "p_gae_09", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "gae_lambda": 0.9},
    {"description": "p_gae_099", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "gae_lambda": 0.99},

    # --- P7: Small minibatch (below default 2048) ---
    {"description": "p_mb512", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "minibatch_size": 512},
    {"description": "p_mb1024", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "minibatch_size": 1024},

    # --- P8: Very low LR (ultra-stable convergence) ---
    {"description": "p_lr5e5_s1137", "lr": 5e-5, "anneal_lr": True, "seed": 1137},
    {"description": "p_lr5e5_s5678", "lr": 5e-5, "anneal_lr": True, "seed": 5678},

    # --- P9: num_envs=32 (fewer, more on-policy) ---
    {"description": "p_envs32", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "num_envs": 32},
    {"description": "p_envs32_s5678", "lr": 1e-4, "anneal_lr": True, "seed": 5678,
     "num_envs": 32},

    # --- P10: Combos of promising P-block dims ---
    {"description": "p_clip01_gn10", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "clip_eps": 0.1, "max_grad_norm": 1.0},
    {"description": "p_vf025_clip01", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "vf_coef": 0.25, "clip_eps": 0.1},
    {"description": "p_nocp_gae99", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "cash_penalty": 0.0, "gae_lambda": 0.99},
    {"description": "p_mb512_gn10", "lr": 1e-4, "anneal_lr": True, "seed": 1137,
     "minibatch_size": 512, "max_grad_norm": 1.0},

    # --- P11: P-block dims with s5678 (2nd best seed) ---
    {"description": "p_vfcoef_025_s5678", "lr": 1e-4, "anneal_lr": True, "seed": 5678,
     "vf_coef": 0.25},
    {"description": "p_gradnorm_10_s5678", "lr": 1e-4, "anneal_lr": True, "seed": 5678,
     "max_grad_norm": 1.0},
    {"description": "p_clipeps_01_s5678", "lr": 1e-4, "anneal_lr": True, "seed": 5678,
     "clip_eps": 0.1},

    # -----------------------------------------------------------------------
    # Q-block: h100_combo_wd01 winner sweep (added 2026-03-23)
    #
    # BEST KNOWN: h100_combo_wd01 at holdout_robust_score=-16.41
    # Config: lr=3e-4, wd=0.01, tp=0.05, slip=5.0bps, anneal_lr=True (default)
    # Trained 90s / 13.8M steps on RTX 5090. Short training = better OOS.
    # Key insight: 37M steps overfits; 13M steps generalizes better.
    #
    # Strategy:
    #  (A) Seed sweep — is wd=0.01+tp=0.05+slip=5 consistently good?
    #  (B) Slippage cross — more friction forces wider edges
    #  (C) WD cross — wd=0.005/0.02 brackets wd=0.01
    #  (D) LR cross — does lr=1e-4 help when combined with wd=0.01?
    #  (E) TP cross — tp=0.03/0.08 brackets tp=0.05
    #  (F) Combo: wd=0.01 + slip=10 + tp=0.05 (more total friction)
    #  (G) Combo: wd=0.01 + anneal_ent (entropy decay for later exploitation)
    #  (H) Combo: wd=0.01 + obs_norm (normalise heterogeneous stock features)
    #  (I) lr=3e-4 + wd=0.01 + tp=0.05 + slip=5 + h256 (smaller=more regularised)
    #  (J) lr=3e-4 + wd=0.01 + tp=0.05 + slip=5 + cosine LR
    # -----------------------------------------------------------------------

    # (A) Seed sweep of exact winning formula
    {"description": "q_wd01_tp05_slip5_s42",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0, "seed": 42},
    {"description": "q_wd01_tp05_slip5_s123",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0, "seed": 123},
    {"description": "q_wd01_tp05_slip5_s777",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0, "seed": 777},
    {"description": "q_wd01_tp05_slip5_s1137",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0, "seed": 1137},
    {"description": "q_wd01_tp05_slip5_s5678",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0, "seed": 5678},
    {"description": "q_wd01_tp05_slip5_s9621",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0, "seed": 9621},

    # (B) Slippage cross
    {"description": "q_wd01_tp05_slip8",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 8.0},
    {"description": "q_wd01_tp05_slip10",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 10.0},
    {"description": "q_wd01_tp05_slip12",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 12.0},
    {"description": "q_wd01_tp05_slip15",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 15.0},

    # (C) WD cross
    {"description": "q_wd005_tp05_slip5",
     "weight_decay": 0.005, "trade_penalty": 0.05, "fill_slippage_bps": 5.0},
    {"description": "q_wd02_tp05_slip5",
     "weight_decay": 0.02, "trade_penalty": 0.05, "fill_slippage_bps": 5.0},
    {"description": "q_wd05_tp05_slip5",
     "weight_decay": 0.05, "trade_penalty": 0.05, "fill_slippage_bps": 5.0},

    # (D) LR=1e-4 + anneal combined with wd=0.01 winner formula
    {"description": "q_lr1e4_wd01_tp05_slip5_s777",
     "lr": 1e-4, "anneal_lr": True, "weight_decay": 0.01,
     "trade_penalty": 0.05, "fill_slippage_bps": 5.0, "seed": 777},
    {"description": "q_lr1e4_wd01_tp05_slip5_s1137",
     "lr": 1e-4, "anneal_lr": True, "weight_decay": 0.01,
     "trade_penalty": 0.05, "fill_slippage_bps": 5.0, "seed": 1137},
    {"description": "q_lr1e4_wd01_tp05_slip5_s42",
     "lr": 1e-4, "anneal_lr": True, "weight_decay": 0.01,
     "trade_penalty": 0.05, "fill_slippage_bps": 5.0, "seed": 42},
    {"description": "q_lr2e4_wd01_tp05_slip5_s777",
     "lr": 2e-4, "anneal_lr": True, "weight_decay": 0.01,
     "trade_penalty": 0.05, "fill_slippage_bps": 5.0, "seed": 777},

    # (E) Trade penalty cross
    {"description": "q_wd01_tp03_slip5",
     "weight_decay": 0.01, "trade_penalty": 0.03, "fill_slippage_bps": 5.0},
    {"description": "q_wd01_tp08_slip5",
     "weight_decay": 0.01, "trade_penalty": 0.08, "fill_slippage_bps": 5.0},
    {"description": "q_wd01_tp10_slip5",
     "weight_decay": 0.01, "trade_penalty": 0.10, "fill_slippage_bps": 5.0},

    # (F) More total friction
    {"description": "q_wd01_tp05_slip10",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 10.0},
    {"description": "q_wd02_tp05_slip10",
     "weight_decay": 0.02, "trade_penalty": 0.05, "fill_slippage_bps": 10.0},
    {"description": "q_wd01_tp08_slip10",
     "weight_decay": 0.01, "trade_penalty": 0.08, "fill_slippage_bps": 10.0},

    # (G) Entropy annealing added
    {"description": "q_wd01_tp05_slip5_annent",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0,
     "anneal_ent": True, "ent_coef": 0.08, "ent_coef_end": 0.02},
    {"description": "q_wd01_tp05_slip5_ent03",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0,
     "ent_coef": 0.03},
    {"description": "q_wd01_tp05_slip5_ent08",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0,
     "ent_coef": 0.08},

    # (H) Obs norm
    {"description": "q_wd01_tp05_slip5_obsn",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0,
     "obs_norm": True},
    {"description": "q_wd01_tp05_slip10_obsn",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 10.0,
     "obs_norm": True},

    # (I) h256 — smaller net = more regularised (random_mut_4424 was h256+slip12, 0% neg windows)
    {"description": "q_h256_wd01_tp05_slip5",
     "hidden_size": 256, "weight_decay": 0.01, "trade_penalty": 0.05,
     "fill_slippage_bps": 5.0},
    {"description": "q_h256_wd01_tp05_slip10",
     "hidden_size": 256, "weight_decay": 0.01, "trade_penalty": 0.05,
     "fill_slippage_bps": 10.0},
    {"description": "q_h512_wd01_tp05_slip5",
     "hidden_size": 512, "weight_decay": 0.01, "trade_penalty": 0.05,
     "fill_slippage_bps": 5.0},

    # (J) Cosine LR with winning wd/tp/slip formula
    {"description": "q_cosine_wd01_tp05_slip5",
     "lr_schedule": "cosine", "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05,
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0},
    {"description": "q_cosine_wd01_tp05_slip10",
     "lr_schedule": "cosine", "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05,
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 10.0},

    # -----------------------------------------------------------------------
    # R-block: reward shaping novelties + multi-objective (added 2026-03-23)
    #
    # Try per-env advantage norm (GRPO/GSPO-style) combined with q-block winner
    # -----------------------------------------------------------------------
    {"description": "r_perenv_wd01_tp05_slip5",
     "advantage_norm": "per_env", "weight_decay": 0.01,
     "trade_penalty": 0.05, "fill_slippage_bps": 5.0},
    {"description": "r_grpo_wd01_tp05_slip5",
     "advantage_norm": "group_relative", "group_relative_size": 8,
     "group_relative_mix": 0.25, "group_relative_clip": 1.5,
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0},

    # Calmar-proxy with wd=0.01 combo
    {"description": "r_calmar_wd01_slip5",
     "weight_decay": 0.01, "drawdown_penalty": 0.1,
     "smooth_downside_penalty": 0.2, "trade_penalty": 0.05,
     "fill_slippage_bps": 5.0},

    # -----------------------------------------------------------------------
    # S-block: q_wd01_tp05_slip5 winner seed sweep + Q+P combos (added 2026-03-23)
    #
    # q_wd01_tp05_slip5_s123 scored -4.05 (NEW BEST, beating -16.41).
    # Strategy:
    #  (A) More seeds for the winning formula — find stable seeds for ensemble
    #  (B) Q winner + P-block tuning: smaller clip_eps, lower vf_coef
    #  (C) Q winner + P-block tuning: tighter grad norm (0.5), cosine anneal
    #  (D) Q winner + slightly more trade friction (tp=0.07) or wd=0.015
    #  (E) Q winner s123 + varied slippage to understand friction sensitivity
    # -----------------------------------------------------------------------

    # (A) Extra seed sweep — is -4.05 a fluke or robust?
    {"description": "s_wd01_tp05_slip5_s200",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0, "seed": 200},
    {"description": "s_wd01_tp05_slip5_s314",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0, "seed": 314},
    {"description": "s_wd01_tp05_slip5_s2024",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0, "seed": 2024},
    {"description": "s_wd01_tp05_slip5_s3141",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0, "seed": 3141},
    {"description": "s_wd01_tp05_slip5_s4242",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0, "seed": 4242},
    {"description": "s_wd01_tp05_slip5_s7777",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0, "seed": 7777},
    {"description": "s_wd01_tp05_slip5_s8192",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0, "seed": 8192},
    {"description": "s_wd01_tp05_slip5_s9999",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0, "seed": 9999},

    # (B) Q winner + small clip_eps (conservative updates = less overfit)
    {"description": "s_clip01_wd01_tp05_slip5_s123",
     "clip_eps": 0.1, "weight_decay": 0.01, "trade_penalty": 0.05,
     "fill_slippage_bps": 5.0, "seed": 123},
    {"description": "s_vf025_wd01_tp05_slip5_s123",
     "vf_coef": 0.25, "weight_decay": 0.01, "trade_penalty": 0.05,
     "fill_slippage_bps": 5.0, "seed": 123},

    # (C) Q winner + tighter grad norm / cosine LR
    {"description": "s_gn03_wd01_tp05_slip5_s123",
     "max_grad_norm": 0.3, "weight_decay": 0.01, "trade_penalty": 0.05,
     "fill_slippage_bps": 5.0, "seed": 123},
    {"description": "s_cosine_wd01_tp05_slip5_s123",
     "lr_schedule": "cosine", "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05,
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0, "seed": 123},

    # (D) Q winner + varied friction levels
    {"description": "s_wd015_tp05_slip5_s123",
     "weight_decay": 0.015, "trade_penalty": 0.05, "fill_slippage_bps": 5.0, "seed": 123},
    {"description": "s_wd01_tp07_slip5_s123",
     "weight_decay": 0.01, "trade_penalty": 0.07, "fill_slippage_bps": 5.0, "seed": 123},
    {"description": "s_wd01_tp05_slip7_s123",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 7.0, "seed": 123},

    # (D2) WD cross with seed=123 — Q-block WD cross used seed=42 (bad seed); need proper test
    {"description": "s_wd005_tp05_slip5_s123",
     "weight_decay": 0.005, "trade_penalty": 0.05, "fill_slippage_bps": 5.0, "seed": 123},
    {"description": "s_wd02_tp05_slip5_s123",
     "weight_decay": 0.02, "trade_penalty": 0.05, "fill_slippage_bps": 5.0, "seed": 123},

    # (D3) h256 + drawdown_penalty + no slippage with seed=123 (inspired by h256_lr3e4_slip0_dp01_s1137 at -37.05)
    {"description": "s_h256_dp01_slip0_s123",
     "hidden_size": 256, "drawdown_penalty": 0.01, "fill_slippage_bps": 0.0, "seed": 123},

    # (E) Q winner s123 with varied slippage (confirm 5bps is optimal)
    {"description": "s_wd01_tp05_slip3_s123",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 3.0, "seed": 123},
    {"description": "s_wd01_tp05_slip0_s123",
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 0.0, "seed": 123},

    # (F) s123 with PPO infra tweaks (clip, GAE, entropy)
    {"description": "s_gae99_wd01_tp05_slip5_s123",
     "gae_lambda": 0.99, "weight_decay": 0.01, "trade_penalty": 0.05,
     "fill_slippage_bps": 5.0, "seed": 123},
    {"description": "s_ppo4_wd01_tp05_slip5_s123",
     "ppo_epochs": 4, "weight_decay": 0.01, "trade_penalty": 0.05,
     "fill_slippage_bps": 5.0, "seed": 123},
    {"description": "s_envs256_wd01_tp05_slip5_s123",
     "num_envs": 256, "weight_decay": 0.01, "trade_penalty": 0.05,
     "fill_slippage_bps": 5.0, "seed": 123},

    # -----------------------------------------------------------------------
    # T-block: h100_robust_champion formula on stocks11_2012 (added 2026-03-23)
    #
    # h100_robust_champion scored +26.33 on stocks20 with:
    #   lr=3e-4, wd=0.005, slip=5bps, tp=0.05, obs_norm=True, lr_schedule=cosine,
    #   anneal_lr=True, ent=0.05, h=1024. val_return=-0.17 (negative!) but OOS +47%.
    #
    # Test if this formula transfers to stocks11_2012 (different symbols, longer history).
    # Strategy:
    #  (A) Exact formula with seed sweep
    #  (B) s123 (our best seed) + champion formula
    #  (C) Champion + wd=0.01 (our best wd)
    #  (D) Champion formula without obs_norm (is that the key?)
    # -----------------------------------------------------------------------

    # (A) Champion formula seed sweep
    {"description": "t_champ_s42",
     "lr": 3e-4, "weight_decay": 0.005, "fill_slippage_bps": 5.0,
     "trade_penalty": 0.05, "obs_norm": True, "lr_schedule": "cosine",
     "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05, "anneal_lr": True, "seed": 42},
    {"description": "t_champ_s123",
     "lr": 3e-4, "weight_decay": 0.005, "fill_slippage_bps": 5.0,
     "trade_penalty": 0.05, "obs_norm": True, "lr_schedule": "cosine",
     "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05, "anneal_lr": True, "seed": 123},
    {"description": "t_champ_s777",
     "lr": 3e-4, "weight_decay": 0.005, "fill_slippage_bps": 5.0,
     "trade_penalty": 0.05, "obs_norm": True, "lr_schedule": "cosine",
     "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05, "anneal_lr": True, "seed": 777},
    {"description": "t_champ_s1137",
     "lr": 3e-4, "weight_decay": 0.005, "fill_slippage_bps": 5.0,
     "trade_penalty": 0.05, "obs_norm": True, "lr_schedule": "cosine",
     "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05, "anneal_lr": True, "seed": 1137},
    {"description": "t_champ_s5678",
     "lr": 3e-4, "weight_decay": 0.005, "fill_slippage_bps": 5.0,
     "trade_penalty": 0.05, "obs_norm": True, "lr_schedule": "cosine",
     "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05, "anneal_lr": True, "seed": 5678},

    # (B) s123 + champion formula variants
    {"description": "t_wd01_obsn_cosine_s123",
     "lr": 3e-4, "weight_decay": 0.01, "fill_slippage_bps": 5.0,
     "trade_penalty": 0.05, "obs_norm": True, "lr_schedule": "cosine",
     "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05, "anneal_lr": True, "seed": 123},
    {"description": "t_wd005_obsn_s123",
     "lr": 3e-4, "weight_decay": 0.005, "fill_slippage_bps": 5.0,
     "trade_penalty": 0.05, "obs_norm": True, "anneal_lr": True, "seed": 123},

    # (C) Champion formula without obs_norm (isolate its contribution)
    {"description": "t_champ_noobsn_s123",
     "lr": 3e-4, "weight_decay": 0.005, "fill_slippage_bps": 5.0,
     "trade_penalty": 0.05, "obs_norm": False, "lr_schedule": "cosine",
     "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05, "anneal_lr": True, "seed": 123},
    {"description": "t_champ_noobsn_s42",
     "lr": 3e-4, "weight_decay": 0.005, "fill_slippage_bps": 5.0,
     "trade_penalty": 0.05, "obs_norm": False, "lr_schedule": "cosine",
     "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05, "anneal_lr": True, "seed": 42},

    # -----------------------------------------------------------------------
    # U-block: GRPO + winning formula seed sweep (added 2026-03-23)
    #
    # r_grpo_wd01_tp05_slip5 scored -5.81 (val_ret=0.1693, val_sort=1.59) —
    # almost as good as s123 (-4.05) at 90s budget. GRPO advantage norm with
    # group_size=8, mix=0.25, clip=1.5 is a strong combination.
    # Strategy: sweep 20 seeds of GRPO+winning formula to find best GRPO seed.
    # Plus: GRPO+s123 with different mix ratios and group sizes.
    # -----------------------------------------------------------------------
    # (A) GRPO seed sweep with winning formula
    *[{"description": f"u_grpo_s{s}",
       "advantage_norm": "group_relative", "group_relative_size": 8,
       "group_relative_mix": 0.25, "group_relative_clip": 1.5,
       "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0,
       "seed": s}
      for s in [123, 5678, 314, 200, 400, 500, 600, 700, 800, 900,
                1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 9999]],

    # (B) GRPO mix ratio variations with s123 (best seed so far)
    {"description": "u_grpo_mix0_s123",
     "advantage_norm": "group_relative", "group_relative_size": 8,
     "group_relative_mix": 0.0, "group_relative_clip": 1.5,
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0,
     "seed": 123},
    {"description": "u_grpo_mix15_s123",
     "advantage_norm": "group_relative", "group_relative_size": 8,
     "group_relative_mix": 0.15, "group_relative_clip": 1.5,
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0,
     "seed": 123},
    {"description": "u_grpo_mix50_s123",
     "advantage_norm": "group_relative", "group_relative_size": 8,
     "group_relative_mix": 0.5, "group_relative_clip": 1.5,
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0,
     "seed": 123},

    # (C) GRPO group size variations with s123
    {"description": "u_grpo_gs4_s123",
     "advantage_norm": "group_relative", "group_relative_size": 4,
     "group_relative_mix": 0.25, "group_relative_clip": 1.5,
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0,
     "seed": 123},
    {"description": "u_grpo_gs16_s123",
     "advantage_norm": "group_relative", "group_relative_size": 16,
     "group_relative_mix": 0.25, "group_relative_clip": 1.5,
     "weight_decay": 0.01, "trade_penalty": 0.05, "fill_slippage_bps": 5.0,
     "seed": 123},

    # --- random_mut_2201 seed sweep (2026-03-23: BEST model +11.74% med, 1/50 neg) ---
    # rmu4424_style: h=256, ent=0.05, slip=12, dp=0.01, sp=0.0 — best-ranked in H100 autoresearch
    # (autoresearch score=-4.02, 50-win: 3.49% med, 9/50 neg — decent backup)
    {"description": "rmu4424_style",
     "hidden_size": 256, "ent_coef": 0.05, "fill_slippage_bps": 12.0,
     "drawdown_penalty": 0.01, "smoothness_penalty": 0.0, "weight_decay": 0.0,
     "smooth_downside_temperature": 0.01, "anneal_lr": True},
    # Config: h=256, ent=0.08, slip=12bps, drawdown_pen=0.01, smoothness_pen=0.005
    # All variants use anneal_lr=True, wd=0.0, lr=3e-4 (defaults)
    {"description": "v_rmu2201_style",
     "hidden_size": 256, "ent_coef": 0.08, "fill_slippage_bps": 12.0,
     "drawdown_penalty": 0.01, "smoothness_penalty": 0.005, "weight_decay": 0.0,
     "smooth_downside_temperature": 0.01, "anneal_lr": True},
    # per_env advantage_norm: 33% deployment rate vs 0% for global (discovered 2026-03-23)
    # random_mut_8597 (seed=1168) used per_env and achieved 50-win 9.38% med, 5/50 neg
    # Use as --init-best-config to bias mutations toward per_env space
    {"description": "v_rmu2201_per_env_style",
     "hidden_size": 256, "ent_coef": 0.08, "fill_slippage_bps": 12.0,
     "drawdown_penalty": 0.01, "smoothness_penalty": 0.005, "weight_decay": 0.0,
     "smooth_downside_temperature": 0.01, "anneal_lr": True,
     "advantage_norm": "per_env"},
    {"description": "v_rmu2201_per_env_slip8",
     "hidden_size": 256, "ent_coef": 0.08, "fill_slippage_bps": 8.0,
     "drawdown_penalty": 0.01, "smoothness_penalty": 0.005, "weight_decay": 0.0,
     "smooth_downside_temperature": 0.01, "anneal_lr": True,
     "advantage_norm": "per_env"},
    {"description": "v_rmu2201_per_env_ent06",
     "hidden_size": 256, "ent_coef": 0.06, "fill_slippage_bps": 12.0,
     "drawdown_penalty": 0.01, "smoothness_penalty": 0.005, "weight_decay": 0.0,
     "smooth_downside_temperature": 0.01, "anneal_lr": True,
     "advantage_norm": "per_env"},
    {"description": "v_rmu2201_s123",
     "hidden_size": 256, "ent_coef": 0.08, "fill_slippage_bps": 12.0,
     "drawdown_penalty": 0.01, "smoothness_penalty": 0.005, "weight_decay": 0.0,
     "smooth_downside_temperature": 0.01, "anneal_lr": True, "seed": 123},
    {"description": "v_rmu2201_s42",
     "hidden_size": 256, "ent_coef": 0.08, "fill_slippage_bps": 12.0,
     "drawdown_penalty": 0.01, "smoothness_penalty": 0.005, "weight_decay": 0.0,
     "smooth_downside_temperature": 0.01, "anneal_lr": True, "seed": 42},
    {"description": "v_rmu2201_s777",
     "hidden_size": 256, "ent_coef": 0.08, "fill_slippage_bps": 12.0,
     "drawdown_penalty": 0.01, "smoothness_penalty": 0.005, "weight_decay": 0.0,
     "smooth_downside_temperature": 0.01, "anneal_lr": True, "seed": 777},
    {"description": "v_rmu2201_s2272",
     "hidden_size": 256, "ent_coef": 0.08, "fill_slippage_bps": 12.0,
     "drawdown_penalty": 0.01, "smoothness_penalty": 0.005, "weight_decay": 0.0,
     "smooth_downside_temperature": 0.01, "anneal_lr": True, "seed": 2272},
    {"description": "v_rmu2201_s9999",
     "hidden_size": 256, "ent_coef": 0.08, "fill_slippage_bps": 12.0,
     "drawdown_penalty": 0.01, "smoothness_penalty": 0.005, "weight_decay": 0.0,
     "smooth_downside_temperature": 0.01, "anneal_lr": True, "seed": 9999},
    {"description": "v_rmu2201_sp01",
     "hidden_size": 256, "ent_coef": 0.08, "fill_slippage_bps": 12.0,
     "drawdown_penalty": 0.01, "smoothness_penalty": 0.01, "weight_decay": 0.0,
     "smooth_downside_temperature": 0.01, "anneal_lr": True},
    {"description": "v_rmu2201_sp02",
     "hidden_size": 256, "ent_coef": 0.08, "fill_slippage_bps": 12.0,
     "drawdown_penalty": 0.01, "smoothness_penalty": 0.02, "weight_decay": 0.0,
     "smooth_downside_temperature": 0.01, "anneal_lr": True},
    {"description": "v_rmu2201_ent06",
     "hidden_size": 256, "ent_coef": 0.06, "fill_slippage_bps": 12.0,
     "drawdown_penalty": 0.01, "smoothness_penalty": 0.005, "weight_decay": 0.0,
     "smooth_downside_temperature": 0.01, "anneal_lr": True},
    {"description": "v_rmu2201_slip15",
     "hidden_size": 256, "ent_coef": 0.08, "fill_slippage_bps": 15.0,
     "drawdown_penalty": 0.01, "smoothness_penalty": 0.005, "weight_decay": 0.0,
     "smooth_downside_temperature": 0.01, "anneal_lr": True},

    # Random mutations — slots so H100 1000-trial runs get ~800+ random trials.
    # best_config is pre-seeded with winning formula when stocks_mode+seed_only.
    # Each slot calls mutate_config(best_config) at runtime.
    *[{"description": f"random_{i}"} for i in range(1, 1001)],
]

# Dense tp03 seed sweeps — moved out of main pool (all negative at 300s on extended val).
# Run with: --descriptions tp03_seed_1,...,tp03_seed_50
STOCK_TP03_SEED_EXPERIMENTS = [
    *[{"description": f"tp03_seed_{i}", "trade_penalty": 0.03, "seed": i} for i in range(1, 51)],
    *[{"description": f"tp03_wd01_seed_{i}", "trade_penalty": 0.03, "weight_decay": 0.01, "seed": i}
      for i in range(1, 26)],
]

# ---------------------------------------------------------------------------
# H100 experiment configurations for stocks20 dataset.
#
# Local RTX 5090 scaling sweep findings (2026-03-22, 90s trials, 67 configs):
#
#   Symbol count vs holdout_robust_score (higher=better):
#     stocks12: best=+21.04 (slip_10bps), 1/21 configs > 0
#     stocks20: best=+19.19 (ent_05), 2/23 configs > 0
#     stocks15: best=-4.03  (trade_pen_08), 0/23 configs > 0
#
#   Top local configurations:
#     stocks12 slip_10bps:   holdout=+21.04, val=1.37, 0% neg windows (90s run)
#     stocks20 ent_05:       holdout=+19.19, val=-0.14, 0% neg windows
#     stocks20 trade_pen_10: holdout=+2.23,  val=+0.03, 0% neg windows
#
#   H100 strategy: use stocks20 (more opportunity), focus on configs that showed
#   0% negative holdout windows. Longer training (200s vs 90s) should improve val
#   returns since 90s only gets ~4M steps — H100 ~3.5x faster = ~14M steps.
#
#   Also test h2048 on stocks20 (too slow for 90s RTX 5090 but viable on H100).
# ---------------------------------------------------------------------------

H100_STOCK_EXPERIMENTS: list[dict] = [
    # --- Proven winners: clones/variations of random_mut_2272 adapted to stocks20 ---
    # random_mut_2272 is best-in-class on stocks12 (all 20 holdout windows profitable,
    # p10=+7%, median=+8% on eval_fast on stocks12 val). Adapting to stocks20.
    # Core: ent_coef=0.03, wd=0.005, slip=12bps, no obs_norm, anneal_lr, h1024.
    {"description": "h100_mut2272_style",
     "ent_coef": 0.03, "weight_decay": 0.005, "fill_slippage_bps": 12.0},
    {"description": "h100_mut2272_slip5",
     "ent_coef": 0.03, "weight_decay": 0.005, "fill_slippage_bps": 5.0},
    {"description": "h100_mut2272_slip8",
     "ent_coef": 0.03, "weight_decay": 0.005, "fill_slippage_bps": 8.0},
    {"description": "h100_mut2272_ent05",
     "ent_coef": 0.05, "weight_decay": 0.005, "fill_slippage_bps": 12.0},
    {"description": "h100_mut2272_wd01",
     "ent_coef": 0.03, "weight_decay": 0.01, "fill_slippage_bps": 12.0},

    # --- Replicate best local findings on stocks20 ---
    # slip_10bps: best on stocks12 (holdout=+21.04). Test on stocks20.
    {"description": "h100_slip_10bps",
     "fill_slippage_bps": 10.0},
    {"description": "h100_slip_5bps",
     "fill_slippage_bps": 5.0},
    {"description": "h100_slip_15bps",
     "fill_slippage_bps": 15.0},

    # ent_05: best on stocks20 (holdout=+19.19). Replicate at full budget.
    {"description": "h100_ent_05",
     "ent_coef": 0.05},
    {"description": "h100_ent_03",
     "ent_coef": 0.03},
    {"description": "h100_ent_08",
     "ent_coef": 0.08},

    # trade_pen: stock_trade_pen_05 showed score=-3.5, 5%neg, +27.8%med on stocks12 (v2 sweep)
    # — best single-trial result, beating random_mut_2272. Multiple seeds + variations critical.
    {"description": "h100_trade_pen_05",
     "trade_penalty": 0.05},
    {"description": "h100_trade_pen_05_s123",
     "trade_penalty": 0.05, "seed": 123},
    {"description": "h100_trade_pen_05_s7",
     "trade_penalty": 0.05, "seed": 7},
    {"description": "h100_trade_pen_05_s42",
     "trade_penalty": 0.05, "seed": 42},
    {"description": "h100_trade_pen_05_ent03",
     "trade_penalty": 0.05, "ent_coef": 0.03},
    {"description": "h100_trade_pen_05_ent08",
     "trade_penalty": 0.05, "ent_coef": 0.08},
    {"description": "h100_trade_pen_05_wd005",
     "trade_penalty": 0.05, "weight_decay": 0.005},
    {"description": "h100_trade_pen_05_anneal_ent",
     "trade_penalty": 0.05, "anneal_ent": True, "ent_coef": 0.08, "ent_coef_end": 0.02},
    # trade_pen_10: 2nd best on stocks20 (holdout=+2.23). Cross with slippage.
    {"description": "h100_trade_pen_10",
     "trade_penalty": 0.10},
    {"description": "h100_trade_pen_08_slip10",
     "trade_penalty": 0.08, "fill_slippage_bps": 10.0},
    {"description": "h100_trade_pen_10_slip5",
     "trade_penalty": 0.10, "fill_slippage_bps": 5.0},
    {"description": "h100_trade_pen_05_slip10",
     "trade_penalty": 0.05, "fill_slippage_bps": 10.0},

    # --- H100-only: h2048 configs (too slow for RTX 5090 90s budget) ---
    {"description": "h100_h2048_ent05",
     "hidden_size": 2048, "ent_coef": 0.05,
     "requires_gpu": "h100"},
    {"description": "h100_h2048_slip10",
     "hidden_size": 2048, "fill_slippage_bps": 10.0,
     "requires_gpu": "h100"},
    {"description": "h100_h2048_tp10",
     "hidden_size": 2048, "trade_penalty": 0.10,
     "requires_gpu": "h100"},
    {"description": "h100_h2048_tp08_slip10",
     "hidden_size": 2048, "trade_penalty": 0.08, "fill_slippage_bps": 10.0,
     "requires_gpu": "h100"},

    # --- Combinations of best local features ---
    {"description": "h100_ent05_slip10",
     "ent_coef": 0.05, "fill_slippage_bps": 10.0},
    {"description": "h100_ent05_tp10",
     "ent_coef": 0.05, "trade_penalty": 0.10},
    {"description": "h100_ent03_slip10",
     "ent_coef": 0.03, "fill_slippage_bps": 10.0},
    {"description": "h100_ent03_tp10",
     "ent_coef": 0.03, "trade_penalty": 0.10},
    {"description": "h100_obs_norm_slip10",
     "obs_norm": True, "fill_slippage_bps": 10.0},
    {"description": "h100_wd005_slip10",
     "weight_decay": 0.005, "fill_slippage_bps": 10.0},
    {"description": "h100_wd005_ent05",
     "weight_decay": 0.005, "ent_coef": 0.05},
    {"description": "h100_cosine_ent05",
     "lr_schedule": "cosine", "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05,
     "ent_coef": 0.05},
    {"description": "h100_cosine_slip10",
     "lr_schedule": "cosine", "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05,
     "fill_slippage_bps": 10.0},
    {"description": "h100_anneal_ent_slip10",
     "anneal_ent": True, "ent_coef": 0.08, "ent_coef_end": 0.02,
     "fill_slippage_bps": 10.0},

    # --- random_mut_4424 variants (0% negative in orig sweep; h=256 is surprisingly good) ---
    # random_mut_4424: h=256, slip=12, ent=0.05, wd=0.0, obs_norm=False, anneal_lr=True
    # Smaller network = more regularized, fewer params to overfit
    {"description": "h100_rmu4424_style",
     "hidden_size": 256, "fill_slippage_bps": 12.0, "ent_coef": 0.05,
     "weight_decay": 0.0, "anneal_lr": True},
    {"description": "h100_rmu4424_wd005",
     "hidden_size": 256, "fill_slippage_bps": 12.0, "ent_coef": 0.05,
     "weight_decay": 0.005, "anneal_lr": True},
    {"description": "h100_rmu4424_slip8",
     "hidden_size": 256, "fill_slippage_bps": 8.0, "ent_coef": 0.05,
     "weight_decay": 0.0, "anneal_lr": True},
    {"description": "h100_h256_mut2272",
     "hidden_size": 256, "fill_slippage_bps": 12.0, "ent_coef": 0.03,
     "weight_decay": 0.005, "anneal_lr": True},

    # --- random_mut_1228 variants (0% negative; obs_norm=True, high ent, no slippage) ---
    # random_mut_1228: lr=5e-4, ent=0.08, wd=0.001, slip=0, obs_norm=True, anneal_lr=True
    {"description": "h100_rmu1228_style",
     "lr": 5e-4, "ent_coef": 0.08, "weight_decay": 0.001,
     "fill_slippage_bps": 0.0, "obs_norm": True, "anneal_lr": True},
    {"description": "h100_rmu1228_slip5",
     "lr": 5e-4, "ent_coef": 0.08, "weight_decay": 0.001,
     "fill_slippage_bps": 5.0, "obs_norm": True, "anneal_lr": True},
    {"description": "h100_rmu1228_wd005",
     "lr": 5e-4, "ent_coef": 0.08, "weight_decay": 0.005,
     "fill_slippage_bps": 0.0, "obs_norm": True, "anneal_lr": True},

    # --- stock_drawdown_pen variants (0% neg, +22.9% med, sortino=7.25 on stocks12 v2 sweep) ---
    # drawdown_penalty=0.05 + trade_penalty=0.03 found as new SOTA (v2 sweep trial 20)
    # Score=+24.9 > random_mut_2272 score=-5.15; worst window +3.3%, max_dd_worst=9.2%
    {"description": "h100_drawpen_style",
     "drawdown_penalty": 0.05, "trade_penalty": 0.03},
    {"description": "h100_drawpen_s123",
     "drawdown_penalty": 0.05, "trade_penalty": 0.03, "seed": 123},
    {"description": "h100_drawpen_s7",
     "drawdown_penalty": 0.05, "trade_penalty": 0.03, "seed": 7},
    {"description": "h100_drawpen_s42",
     "drawdown_penalty": 0.05, "trade_penalty": 0.03, "seed": 42},
    {"description": "h100_drawpen_s999",
     "drawdown_penalty": 0.05, "trade_penalty": 0.03, "seed": 999},
    {"description": "h100_drawpen_s2272",
     "drawdown_penalty": 0.05, "trade_penalty": 0.03, "seed": 2272},
    {"description": "h100_drawpen_tp05",
     "drawdown_penalty": 0.05, "trade_penalty": 0.05},
    {"description": "h100_drawpen_tp02",
     "drawdown_penalty": 0.05, "trade_penalty": 0.02},
    {"description": "h100_drawpen_dd02",
     "drawdown_penalty": 0.02, "trade_penalty": 0.03},
    {"description": "h100_drawpen_dd10",
     "drawdown_penalty": 0.10, "trade_penalty": 0.03},
    {"description": "h100_drawpen_ent03",
     "drawdown_penalty": 0.05, "trade_penalty": 0.03, "ent_coef": 0.03},
    {"description": "h100_drawpen_wd005",
     "drawdown_penalty": 0.05, "trade_penalty": 0.03, "weight_decay": 0.005},
    {"description": "h100_drawpen_slip5",
     "drawdown_penalty": 0.05, "trade_penalty": 0.03, "fill_slippage_bps": 5.0},

    # --- random_mut_2201 seed sweep (2026-03-23 BEST model: +11.74% med, 1/50 neg) ---
    # random_mut_2201: h=256, ent=0.08, slip=12bps, dp=0.01, sp=0.005, wd=0.0, anneal_lr=True
    # This config is UNIQUELY good in deterministic eval (stochastic eval was wrong/misleading)
    # Sweep many seeds — escape rate unknown, need many trials to find similar models
    {"description": "rmu2201_style",
     "hidden_size": 256, "ent_coef": 0.08, "fill_slippage_bps": 12.0,
     "drawdown_penalty": 0.01, "smoothness_penalty": 0.005, "weight_decay": 0.0,
     "smooth_downside_temperature": 0.01, "anneal_lr": True},
    {"description": "rmu2201_s123",
     "hidden_size": 256, "ent_coef": 0.08, "fill_slippage_bps": 12.0,
     "drawdown_penalty": 0.01, "smoothness_penalty": 0.005, "weight_decay": 0.0,
     "smooth_downside_temperature": 0.01, "anneal_lr": True, "seed": 123},
    {"description": "rmu2201_s42",
     "hidden_size": 256, "ent_coef": 0.08, "fill_slippage_bps": 12.0,
     "drawdown_penalty": 0.01, "smoothness_penalty": 0.005, "weight_decay": 0.0,
     "smooth_downside_temperature": 0.01, "anneal_lr": True, "seed": 42},
    {"description": "rmu2201_s777",
     "hidden_size": 256, "ent_coef": 0.08, "fill_slippage_bps": 12.0,
     "drawdown_penalty": 0.01, "smoothness_penalty": 0.005, "weight_decay": 0.0,
     "smooth_downside_temperature": 0.01, "anneal_lr": True, "seed": 777},
    {"description": "rmu2201_s2272",
     "hidden_size": 256, "ent_coef": 0.08, "fill_slippage_bps": 12.0,
     "drawdown_penalty": 0.01, "smoothness_penalty": 0.005, "weight_decay": 0.0,
     "smooth_downside_temperature": 0.01, "anneal_lr": True, "seed": 2272},
    {"description": "rmu2201_s9999",
     "hidden_size": 256, "ent_coef": 0.08, "fill_slippage_bps": 12.0,
     "drawdown_penalty": 0.01, "smoothness_penalty": 0.005, "weight_decay": 0.0,
     "smooth_downside_temperature": 0.01, "anneal_lr": True, "seed": 9999},
    # Variants: try smoothness_penalty = 0.01 (2x) and 0.02 (4x)
    {"description": "rmu2201_sp01",
     "hidden_size": 256, "ent_coef": 0.08, "fill_slippage_bps": 12.0,
     "drawdown_penalty": 0.01, "smoothness_penalty": 0.01, "weight_decay": 0.0,
     "smooth_downside_temperature": 0.01, "anneal_lr": True},
    {"description": "rmu2201_sp02",
     "hidden_size": 256, "ent_coef": 0.08, "fill_slippage_bps": 12.0,
     "drawdown_penalty": 0.01, "smoothness_penalty": 0.02, "weight_decay": 0.0,
     "smooth_downside_temperature": 0.01, "anneal_lr": True},
    # Variant: ent=0.06 (between 0.05 and 0.08)
    {"description": "rmu2201_ent06",
     "hidden_size": 256, "ent_coef": 0.06, "fill_slippage_bps": 12.0,
     "drawdown_penalty": 0.01, "smoothness_penalty": 0.005, "weight_decay": 0.0,
     "smooth_downside_temperature": 0.01, "anneal_lr": True},
    # Variant: slip=15bps (even more friction)
    {"description": "rmu2201_slip15",
     "hidden_size": 256, "ent_coef": 0.08, "fill_slippage_bps": 15.0,
     "drawdown_penalty": 0.01, "smoothness_penalty": 0.005, "weight_decay": 0.0,
     "smooth_downside_temperature": 0.01, "anneal_lr": True},

    # --- Cross seeds for best configs ---
    {"description": "h100_slip_10bps_s123",
     "fill_slippage_bps": 10.0, "seed": 123},
    {"description": "h100_ent_05_s123",
     "ent_coef": 0.05, "seed": 123},
    {"description": "h100_slip_10bps_s7",
     "fill_slippage_bps": 10.0, "seed": 7},
    {"description": "h100_mut2272_s4424",
     "fill_slippage_bps": 12.0, "ent_coef": 0.03,
     "weight_decay": 0.005, "anneal_lr": True, "seed": 4424},
    {"description": "h100_rmu4424_s2272",
     "hidden_size": 256, "fill_slippage_bps": 12.0, "ent_coef": 0.05,
     "weight_decay": 0.0, "anneal_lr": True, "seed": 2272},

    # --- Random mutations to explore neighborhood ---
    {"description": "random_1"},
    {"description": "random_2"},
    {"description": "random_3"},
    {"description": "random_4"},
    {"description": "random_5"},
    {"description": "random_6"},
    {"description": "random_7"},
    {"description": "random_8"},
    {"description": "random_9"},
    {"description": "random_10"},
    {"description": "random_11"},
    {"description": "random_12"},
    {"description": "random_13"},
    {"description": "random_14"},
    {"description": "random_15"},
    {"description": "random_16"},
    {"description": "random_17"},
    {"description": "random_18"},
    {"description": "random_19"},
    {"description": "random_20"},
    {"description": "random_21"},
    {"description": "random_22"},
    {"description": "random_23"},
    {"description": "random_24"},
    {"description": "random_25"},
    {"description": "random_26"},
    {"description": "random_27"},
    {"description": "random_28"},
    {"description": "random_29"},
    {"description": "random_30"},
    {"description": "random_31"},
    {"description": "random_32"},
    {"description": "random_33"},
    {"description": "random_34"},
    {"description": "random_35"},
    {"description": "random_36"},
    {"description": "random_37"},
    {"description": "random_38"},
    {"description": "random_39"},
    {"description": "random_40"},
    {"description": "random_41"},
    {"description": "random_42"},
    {"description": "random_43"},
    {"description": "random_44"},
    {"description": "random_45"},
    {"description": "random_46"},
    {"description": "random_47"},
    {"description": "random_48"},
    {"description": "random_49"},
    {"description": "random_50"},
    {"description": "random_51"},
    {"description": "random_52"},
    {"description": "random_53"},
    {"description": "random_54"},
    {"description": "random_55"},
    {"description": "random_56"},
    {"description": "random_57"},
    {"description": "random_58"},
    {"description": "random_59"},
    {"description": "random_60"},
    {"description": "random_61"},
    {"description": "random_62"},
    {"description": "random_63"},
    {"description": "random_64"},
    {"description": "random_65"},
    {"description": "random_66"},
    {"description": "random_67"},
    {"description": "random_68"},
    {"description": "random_69"},
    {"description": "random_70"},
    {"description": "random_71"},
    {"description": "random_72"},
    {"description": "random_73"},
    {"description": "random_74"},
    {"description": "random_75"},
    {"description": "random_76"},
    {"description": "random_77"},
    {"description": "random_78"},
    {"description": "random_79"},
    {"description": "random_80"},
    {"description": "random_81"},
    {"description": "random_82"},
    {"description": "random_83"},
    {"description": "random_84"},
    {"description": "random_85"},
    {"description": "random_86"},
    {"description": "random_87"},
    {"description": "random_88"},
    {"description": "random_89"},
    {"description": "random_90"},
    {"description": "random_91"},
    {"description": "random_92"},
    {"description": "random_93"},
    {"description": "random_94"},
    {"description": "random_95"},
    {"description": "random_96"},
    {"description": "random_97"},
    {"description": "random_98"},
    {"description": "random_99"},
    {"description": "random_100"},
]

# Default data paths used when --stocks is given and no explicit --train-data is provided.
# Updated 2026-03-22: stocks20 is now the default — broader market coverage (20 diverse
# symbols) produced 2 configs with positive holdout on the RTX 5090 scaling sweep.
_STOCK_DEFAULT_TRAIN = "pufferlib_market/data/stocks20_daily_train.bin"
_STOCK_DEFAULT_VAL   = "pufferlib_market/data/stocks20_daily_val.bin"


def build_config(overrides: dict) -> TrialConfig:
    """Create a TrialConfig with overrides applied."""
    cfg = TrialConfig(**{k: v for k, v in overrides.items() if k in TrialConfig.__dataclass_fields__})
    if "description" in overrides:
        cfg.description = overrides["description"]
    return cfg


def mutate_config(base: TrialConfig, *, stocks_mode: bool = False, seed_only: bool = False,
                  per_env_focused: bool = False) -> TrialConfig:
    """Randomly mutate a config for exploration.

    stocks_mode=True: restricts to lr in [1e-4, 3e-4] and slip in [0, 5bps] (safe for stocks11_2012).
      - lr=3e-4 + wd=0.01 + tp=0.05 + slip=5bps + anneal_lr is the best known formula (-4.05).
      - slip>5bps (8/10/12/15bps) all cause hold-cash on stocks11_2012.
    seed_only=True: only mutates the seed (for pure seed sweeps around a known-good config).
    per_env_focused=True: tight mutation space around proven per_env stocks12 config.
      - Locks anneal_lr=True always (False causes degenerate hold-cash on stocks12).
      - Locks advantage_norm in [per_env, group_relative] (global collapses to NVDA-only).
      - Locks hidden_size in [256, 512] (proven range; 1024 rarely escapes per_env).
      - Only varies: ent_coef, fill_slippage_bps, drawdown_penalty, smoothness_penalty, gamma.
    """
    d = asdict(base)
    if not seed_only:
        if per_env_focused:
            # Tight per_env mutation space based on rmu8597 (seed=1168, stocks12 daily)
            mutable_params = {
                "hidden_size": [256, 512],
                "ent_coef": [0.05, 0.08, 0.1],
                "fill_slippage_bps": [8.0, 12.0],
                "drawdown_penalty": [0.0, 0.01, 0.02],
                "smoothness_penalty": [0.0, 0.005, 0.01],
                "gamma": [0.99, 0.995],
                "weight_decay": [0.0, 0.001],
                "advantage_norm": ["per_env", "group_relative"],
                "anneal_lr": [True],  # never mutate away from True
            }
        else:
            # Pick 2-3 params to mutate
            mutable_params = {
                # stocks_mode: h=256 catastrophically bad (-121 to -146) on stocks11_2012. Only 512/1024 viable.
                "hidden_size": [512, 1024] if stocks_mode else [256, 512, 1024],
                # stocks_mode: lr=3e-4 + wd=0.01 + tp=0.05 + slip=5bps is BEST (-4.05).
                # lr=1e-4 P-block did not beat q-block winner. Allow both 1e-4 and 3e-4.
                # Exclude 2e-4/5e-4: no evidence of benefit yet.
                "lr": [1e-4, 3e-4] if stocks_mode else [1e-4, 2e-4, 3e-4, 5e-4],
                "ent_coef": [0.01, 0.03, 0.05, 0.08, 0.1],
                "weight_decay": [0.0, 0.001, 0.005, 0.01, 0.05],
                # stocks_mode: slip>5bps destroys training (slip10/12/15 all hold-cash). Keep 0 or 5.
                "fill_slippage_bps": [0.0, 5.0] if stocks_mode else [0.0, 5.0, 8.0, 12.0],
                "gamma": [0.98, 0.99, 0.995],
                "advantage_norm": ["global", "per_env", "group_relative"],
                "group_relative_mix": [0.0, 0.15, 0.25, 0.4],
                "reward_scale": [5.0, 10.0, 20.0],
                "cash_penalty": [0.0, 0.005, 0.01, 0.02],
                "trade_penalty": [0.0, 0.01, 0.02, 0.03, 0.05],
                "drawdown_penalty": [0.0, 0.01, 0.02, 0.05],
                "smooth_downside_penalty": [0.0, 0.1, 0.2, 0.5],
                "smooth_downside_temperature": [0.01, 0.02, 0.05],
                "smoothness_penalty": [0.0, 0.005, 0.01, 0.02],
                "obs_norm": [True, False],
                # stocks_mode: anneal_lr=True is critical; False collapses. Don't mutate away from it.
                "anneal_lr": [True] if stocks_mode else [True, False],
                "anneal_ent": [True, False],
            }
        keys = random.sample(list(mutable_params.keys()), min(3, len(mutable_params)))
        for k in keys:
            d[k] = random.choice(mutable_params[k])
    # h2048 needs 256 envs + 4096 minibatch for adequate step throughput on H100
    if d.get("hidden_size") == 2048:
        d["num_envs"] = 256
        d["minibatch_size"] = 4096
    d["description"] = f"random_mut_{random.randint(0, 9999)}"
    d["seed"] = random.randint(1, 9999)
    return TrialConfig(**{k: v for k, v in d.items() if k in TrialConfig.__dataclass_fields__})


def _safe_float(value: object) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trim_error(text: str, *, limit: int = 400) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _run_capture(
    cmd: list[str],
    *,
    cwd: Path,
    timeout_s: int = 0,
) -> subprocess.CompletedProcess[str]:
    kwargs: dict[str, object] = {
        "capture_output": True,
        "text": True,
        "cwd": str(cwd),
    }
    if timeout_s > 0:
        kwargs["timeout"] = int(timeout_s)
    return subprocess.run(cmd, **kwargs)


def summarize_holdout_payload(payload: dict[str, object]) -> dict[str, float]:
    """Convert evaluate_holdout JSON into leaderboard-friendly metrics."""
    windows = payload.get("windows")
    if not isinstance(windows, list) or not windows:
        return {}

    scenario_rows: list[dict[str, float]] = []
    for row in windows:
        if not isinstance(row, dict):
            continue
        scenario_rows.append(
            {
                "return_pct": 100.0 * float(row.get("total_return", 0.0) or 0.0),
                "annualized_return_pct": 100.0 * float(row.get("annualized_return", 0.0) or 0.0),
                "sortino": float(row.get("sortino", 0.0) or 0.0),
                "max_drawdown_pct": 100.0 * float(row.get("max_drawdown", 0.0) or 0.0),
                "pnl_smoothness": 0.0,
                "trade_count": float(row.get("num_trades", 0.0) or 0.0),
            }
        )

    if not scenario_rows:
        return {}

    robust = summarize_scenario_results(scenario_rows)
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}

    return {
        "holdout_robust_score": float(robust["robust_score"]),
        "holdout_return_mean_pct": float(robust["return_mean_pct"]),
        "holdout_return_p25_pct": float(robust["return_p25_pct"]),
        "holdout_return_worst_pct": float(robust["return_worst_pct"]),
        "holdout_sortino_p25": float(robust["sortino_p25"]),
        "holdout_max_drawdown_worst_pct": float(robust["max_drawdown_worst_pct"]),
        "holdout_negative_return_rate": float(robust["negative_return_rate"]),
        "holdout_median_return_pct": 100.0 * float(summary.get("median_total_return", 0.0) or 0.0),
        "holdout_p10_return_pct": 100.0 * float(summary.get("p10_total_return", 0.0) or 0.0),
        "holdout_median_sortino": float(summary.get("median_sortino", 0.0) or 0.0),
        "holdout_p90_max_drawdown_pct": 100.0 * float(summary.get("p90_max_drawdown", 0.0) or 0.0),
    }


def summarize_market_validation_payload(payload: object) -> dict[str, float]:
    """Convert market_validation JSON into leaderboard-friendly metrics."""
    if isinstance(payload, list):
        row = payload[0] if payload else None
    else:
        row = payload
    if not isinstance(row, dict):
        return {}
    return {
        "market_return_pct": float(row.get("return_pct", 0.0) or 0.0),
        "market_sortino": float(row.get("sortino", 0.0) or 0.0),
        "market_max_drawdown_pct": float(row.get("max_drawdown_pct", 0.0) or 0.0),
        "market_trade_count": float(row.get("trade_count", 0.0) or 0.0),
        "market_goodness_score": float(row.get("goodness_score", 0.0) or 0.0),
    }


def summarize_replay_eval_payload(payload: object) -> dict[str, float]:
    """Convert replay_eval JSON into leaderboard-friendly metrics."""
    if not isinstance(payload, dict):
        return {}

    summary: dict[str, float] = {}
    for section, prefix in (
        ("daily", "replay_daily"),
        ("hourly_replay", "replay_hourly"),
        ("hourly_policy", "replay_hourly_policy"),
    ):
        row = payload.get(section)
        if not isinstance(row, dict):
            continue
        summary[f"{prefix}_return_pct"] = 100.0 * float(row.get("total_return", 0.0) or 0.0)
        summary[f"{prefix}_sortino"] = float(row.get("sortino", 0.0) or 0.0)
        summary[f"{prefix}_max_drawdown_pct"] = 100.0 * float(row.get("max_drawdown", 0.0) or 0.0)
        if "num_trades" in row:
            summary[f"{prefix}_trade_count"] = float(row.get("num_trades", 0.0) or 0.0)
        if "num_orders" in row:
            summary[f"{prefix}_order_count"] = float(row.get("num_orders", 0.0) or 0.0)
        if "pnl_smoothness" in row:
            summary[f"{prefix}_pnl_smoothness"] = float(row.get("pnl_smoothness", 0.0) or 0.0)
        if "ulcer_index" in row:
            summary[f"{prefix}_ulcer_index"] = float(row.get("ulcer_index", 0.0) or 0.0)
        if "goodness_score" in row:
            summary[f"{prefix}_goodness_score"] = float(row.get("goodness_score", 0.0) or 0.0)

    robust = payload.get("robust_start_summary")
    if isinstance(robust, dict):
        for section, prefix in (
            ("daily", "replay_daily_robust"),
            ("hourly_replay", "replay_hourly_robust"),
            ("hourly_policy", "replay_hourly_policy_robust"),
        ):
            row = robust.get(section)
            if not isinstance(row, dict):
                continue
            if "median_total_return" in row:
                summary[f"{prefix}_median_return_pct"] = 100.0 * float(row.get("median_total_return", 0.0) or 0.0)
            if "worst_total_return" in row:
                summary[f"{prefix}_worst_return_pct"] = 100.0 * float(row.get("worst_total_return", 0.0) or 0.0)
            if "worst_sortino" in row:
                summary[f"{prefix}_worst_sortino"] = float(row.get("worst_sortino", 0.0) or 0.0)
            if "worst_max_drawdown" in row:
                summary[f"{prefix}_worst_max_drawdown_pct"] = 100.0 * float(row.get("worst_max_drawdown", 0.0) or 0.0)

    summary.update(
        compute_replay_composite_score(
            daily_return_pct=summary.get("replay_daily_return_pct"),
            daily_sortino=summary.get("replay_daily_sortino"),
            daily_max_drawdown_pct=summary.get("replay_daily_max_drawdown_pct"),
            daily_pnl_smoothness=summary.get("replay_daily_pnl_smoothness", 0.0),
            daily_trade_count=summary.get("replay_daily_trade_count", 0.0),
            hourly_return_pct=summary.get("replay_hourly_return_pct"),
            hourly_sortino=summary.get("replay_hourly_sortino"),
            hourly_max_drawdown_pct=summary.get("replay_hourly_max_drawdown_pct"),
            hourly_pnl_smoothness=summary.get("replay_hourly_pnl_smoothness", 0.0),
            hourly_trade_count=summary.get("replay_hourly_trade_count", 0.0),
            hourly_policy_return_pct=summary.get("replay_hourly_policy_return_pct"),
            hourly_policy_sortino=summary.get("replay_hourly_policy_sortino"),
            hourly_policy_max_drawdown_pct=summary.get("replay_hourly_policy_max_drawdown_pct"),
            hourly_policy_pnl_smoothness=summary.get("replay_hourly_policy_pnl_smoothness", 0.0),
            hourly_policy_trade_count=summary.get("replay_hourly_policy_trade_count", 0.0),
        )
    )
    return summary


def select_rank_score(
    metrics: dict[str, object],
    *,
    rank_metric: str = "auto",
) -> tuple[str, float | None]:
    """Choose the leaderboard ranking signal with sensible fallbacks."""
    candidates = {
        "smooth_score": _safe_float(metrics.get("smooth_score")),
        "replay_combo_score": _safe_float(metrics.get("replay_combo_score")),
        "market_goodness_score": _safe_float(metrics.get("market_goodness_score")),
        "holdout_robust_score": _safe_float(metrics.get("holdout_robust_score")),
        "replay_hourly_policy_robust_worst_return_pct": _safe_float(metrics.get("replay_hourly_policy_robust_worst_return_pct")),
        "replay_hourly_return_pct": _safe_float(metrics.get("replay_hourly_return_pct")),
        "replay_hourly_policy_return_pct": _safe_float(metrics.get("replay_hourly_policy_return_pct")),
        "replay_hourly_robust_worst_return_pct": _safe_float(metrics.get("replay_hourly_robust_worst_return_pct")),
        "val_return": _safe_float(metrics.get("val_return")),
    }
    if rank_metric == "auto":
        for name in (
            "smooth_score",
            "replay_combo_score",
            "market_goodness_score",
            "holdout_robust_score",
            "replay_hourly_policy_robust_worst_return_pct",
            "replay_hourly_policy_return_pct",
            "replay_hourly_robust_worst_return_pct",
            "replay_hourly_return_pct",
            "val_return",
        ):
            score = candidates.get(name)
            if score is not None:
                return name, score
        return "none", None
    return rank_metric, candidates.get(rank_metric)


def _leaderboard_sort_value(row: dict[str, str]) -> float:
    rank_score = _safe_float(row.get("rank_score"))
    if rank_score is not None:
        return rank_score
    val_return = _safe_float(row.get("val_return"))
    if val_return is not None:
        return val_return
    return -float("inf")


def _select_from_pool(
    pool: list[dict],
    *,
    start_from: int = 0,
    descriptions: str = "",
) -> list[dict]:
    """Generic helper: filter a pool list by start_from offset and optional description subset."""
    experiments = pool[max(0, int(start_from)) :]
    requested = [part.strip() for part in str(descriptions).split(",") if part.strip()]
    if not requested:
        return experiments

    requested_set = set(requested)
    selected = [exp for exp in experiments if str(exp.get("description", "")) in requested_set]
    found = {str(exp.get("description", "")) for exp in selected}
    missing = [name for name in requested if name not in found]
    if missing:
        raise ValueError(f"Unknown experiment description(s): {', '.join(missing)}")
    return selected


def select_experiments(
    *,
    start_from: int = 0,
    descriptions: str = "",
) -> list[dict]:
    pool = EXPERIMENTS + CRYPTO34_HOURLY_EXPERIMENTS
    return _select_from_pool(pool, start_from=start_from, descriptions=descriptions)


def run_trial(
    config: TrialConfig,
    train_data: str,
    val_data: str,
    time_budget: int,
    checkpoint_dir: str,
    *,
    wandb_project: str | None = None,
    wandb_group: str | None = None,
    holdout_data: str | None = None,
    holdout_eval_steps: int = 0,
    holdout_n_windows: int = 0,
    holdout_seed: int = 1337,
    holdout_end_within_steps: int = 0,
    holdout_fee_rate: float = -1.0,
    holdout_fill_buffer_bps: float = 5.0,
    holdout_max_leverage: float = 1.0,
    holdout_short_borrow_apr: float = 0.0,
    eval_timeout_s: int = 0,
    holdout_timeout_s: int = 0,
    market_validation_asset_class: str = "",
    market_validation_days: int = 30,
    market_validation_cash: float = 10_000.0,
    market_validation_decision_cadence: str = "hourly",
    market_validation_symbols: str | None = None,
    market_validation_timeout_s: int = 0,
    replay_eval_data: str | None = None,
    replay_eval_hourly_root: str = "",
    replay_eval_start_date: str = "",
    replay_eval_end_date: str = "",
    replay_eval_run_hourly_policy: bool = False,
    replay_eval_robust_start_states: str = "",
    replay_eval_fill_buffer_bps: float = 5.0,
    replay_eval_hourly_periods_per_year: float = 8760.0,
    replay_eval_timeout_s: int = 0,
    rank_metric: str = "auto",
    max_timesteps_per_sample: int = 1000,
    best_trial_rank_score: float = -float("inf"),
    best_trial_val_return: float = -float("inf"),
    best_trial_combined_score: float = -float("inf"),
    early_reject_threshold: float = 0.8,
    use_poly_prune: bool = True,
    multi_period_eval_windows: tuple[int, ...] | None = None,
    multi_period_n_windows_per_size: int = 8,
    multi_period_fill_slippage_bps: float = 8.0,
) -> dict:
    """Run a single training trial with time budget, then evaluate on val."""
    # Cap total_timesteps based on dataset size to prevent overfitting
    try:
        num_symbols, num_timesteps = _read_mktd_header(train_data)
        num_samples = num_symbols * num_timesteps
        total_timesteps = min(999_999_999, num_samples * max_timesteps_per_sample)
        print(f"  Data: {num_symbols} syms x {num_timesteps} steps = {num_samples:,} samples, "
              f"cap={total_timesteps:,} steps ({max_timesteps_per_sample}x)")
    except Exception:
        total_timesteps = 999_999_999

    # Build training command
    cmd = [
        sys.executable, "-u", "-m", "pufferlib_market.train",
        "--data-path", train_data,
        "--total-timesteps", str(total_timesteps),
        "--max-steps", str(config.max_steps),
        "--hidden-size", str(config.hidden_size),
        "--lr", str(config.lr),
        "--ent-coef", str(config.ent_coef),
        "--gamma", str(config.gamma),
        "--gae-lambda", str(config.gae_lambda),
        "--advantage-norm", str(config.advantage_norm),
        "--group-relative-size", str(config.group_relative_size),
        "--group-relative-mix", str(config.group_relative_mix),
        "--group-relative-clip", str(config.group_relative_clip),
        "--clip-eps", str(config.clip_eps),
        "--num-envs", str(config.num_envs),
        "--rollout-len", str(config.rollout_len),
        "--ppo-epochs", str(config.ppo_epochs),
        "--seed", str(config.seed),
        "--reward-scale", str(config.reward_scale),
        "--reward-clip", str(config.reward_clip),
        "--cash-penalty", str(config.cash_penalty),
        "--fee-rate", str(config.fee_rate),
        "--fill-slippage-bps", str(config.fill_slippage_bps),
        "--trade-penalty", str(config.trade_penalty),
        "--drawdown-penalty", str(config.drawdown_penalty),
        "--downside-penalty", str(config.downside_penalty),
        "--smooth-downside-penalty", str(config.smooth_downside_penalty),
        "--smooth-downside-temperature", str(config.smooth_downside_temperature),
        "--smoothness-penalty", str(config.smoothness_penalty),
        "--weight-decay", str(config.weight_decay),
        "--checkpoint-dir", checkpoint_dir,
        "--arch", config.arch,
        "--periods-per-year", str(config.periods_per_year),
        "--max-leverage", str(config.max_leverage),
        "--short-borrow-apr", str(config.short_borrow_apr),
    ]
    if config.anneal_lr:
        cmd.append("--anneal-lr")
    if config.obs_norm:
        cmd.append("--obs-norm")
    if config.anneal_ent:
        cmd.extend(["--anneal-ent", "--ent-coef-end", str(config.ent_coef_end)])
    if config.anneal_clip:
        cmd.extend(["--anneal-clip", "--clip-eps-end", str(config.clip_eps_end)])
    if config.clip_vloss:
        cmd.append("--clip-vloss")
    if config.lr_schedule != "none":
        cmd.extend([
            "--lr-schedule", config.lr_schedule,
            "--lr-warmup-frac", str(config.lr_warmup_frac),
            "--lr-min-ratio", str(config.lr_min_ratio),
        ])
    if config.optimizer != "adamw":
        cmd.extend(["--optimizer", config.optimizer])
    if getattr(config, "muon_norm_update", False):
        cmd.append("--muon-norm-update")
    if config.minibatch_size != 2048:
        cmd.extend(["--minibatch-size", str(config.minibatch_size)])
    if config.vf_coef != 0.5:
        cmd.extend(["--vf-coef", str(config.vf_coef)])
    if config.max_grad_norm != 0.5:
        cmd.extend(["--max-grad-norm", str(config.max_grad_norm)])
    if config.use_bf16:
        cmd.append("--use-bf16")
    if config.no_cuda_graph:
        cmd.append("--no-cuda-graph")
    elif config.cuda_graph_ppo:
        cmd.append("--cuda-graph-ppo")
    if wandb_project:
        cmd.extend([
            "--wandb-project", wandb_project,
            "--wandb-run-name", config.description,
        ])
        if wandb_group:
            cmd.extend(["--wandb-group", wandb_group])

    # Allow per-config time budget override (e.g. h2048 needs more steps than h1024)
    if config.time_budget_override > 0:
        time_budget = config.time_budget_override

    # Run training with time budget
    print(f"\n  Training for {time_budget}s...")
    t0 = time.time()
    early_rejected = False

    def _quick_val_eval(ckpt: Path) -> tuple[float | None, float | None, float | None]:
        """Run a fast C-env eval on a checkpoint against val data.

        Returns (val_return, val_sortino, val_wr).
        """
        qcmd = [
            sys.executable, "-u", "-m", "pufferlib_market.evaluate",
            "--checkpoint", str(ckpt),
            "--data-path", val_data,
            "--deterministic",
            "--hidden-size", str(config.hidden_size),
            "--max-steps", str(config.max_steps),
            "--num-episodes", "30",
            "--seed", "42",
            "--fill-slippage-bps", "8",
            "--periods-per-year", str(config.periods_per_year),
        ]
        if config.arch != "mlp":
            qcmd.extend(["--arch", config.arch])
        q_return: float | None = None
        q_sortino: float | None = None
        q_wr: float | None = None
        try:
            qr = _run_capture(qcmd, cwd=REPO, timeout_s=60)
            for qline in qr.stdout.split("\n"):
                if "Return:" in qline and "mean=" in qline:
                    try:
                        q_return = float(qline.split("mean=")[1].split()[0])
                    except Exception:
                        pass
                elif "Sortino:" in qline and "mean=" in qline:
                    try:
                        q_sortino = float(qline.split("mean=")[1].split()[0])
                    except Exception:
                        pass
                elif "Win rate:" in qline and "mean=" in qline:
                    try:
                        q_wr = float(qline.split("mean=")[1].split()[0])
                    except Exception:
                        pass
        except Exception:
            pass
        return q_return, q_sortino, q_wr

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(REPO), preexec_fn=os.setsid,
        )
        stdout_lines = []
        _checks_done: set[float] = set()
        _poly_stopper = PolynomialEarlyStopper()
        _hold_cash_detector = HoldCashDetector(patience=6)
        # (progress_threshold, poly_tolerance): 25% only collects; 50%/75% may prune
        _check_schedule = [(0.25, None), (0.50, 0.70), (0.75, 0.80)]
        try:
            while time.time() - t0 < time_budget:
                if proc.poll() is not None:
                    break
                try:
                    line = proc.stdout.readline()
                    if line:
                        decoded = line.decode("utf-8", errors="replace").strip()
                        stdout_lines.append(decoded)
                        if _hold_cash_detector.update(decoded):
                            print(
                                f"  HOLD-CASH KILL: trades=0 for "
                                f"{_hold_cash_detector.consecutive_zero_trades} consecutive "
                                f"log lines — policy stuck in no-trade attractor"
                            )
                            early_rejected = True
                            break
                except Exception:
                    pass

                if time_budget >= 60:
                    progress = (time.time() - t0) / max(time_budget, 1)
                    for _threshold, _poly_tol in _check_schedule:
                        if progress < _threshold or _threshold in _checks_done:
                            continue
                        _checks_done.add(_threshold)
                        pts = sorted(Path(checkpoint_dir).glob("*.pt"), key=lambda p: p.stat().st_mtime)
                        if not pts:
                            continue
                        mid_ret, mid_sort, mid_wr = _quick_val_eval(pts[-1])
                        mid_comb = _combined_score(mid_ret, mid_sort, mid_wr)
                        pct = f"{int(_threshold * 100)}%"
                        if mid_comb is not None:
                            _poly_stopper.add_observation(_threshold, mid_comb)
                        if _poly_tol is None:
                            # collect-only check (25%)
                            if not use_poly_prune and mid_ret is not None and best_trial_val_return > -float("inf"):
                                threshold_val = best_trial_val_return * early_reject_threshold
                                if mid_ret < threshold_val:
                                    print(f"  EARLY REJECT at {pct}: val_ret={mid_ret:+.4f} "
                                          f"< {threshold_val:+.4f} (best_val*{early_reject_threshold})")
                                    early_rejected = True
                                    break
                                else:
                                    print(f"  {pct} check: val_ret={mid_ret:+.4f} comb={mid_comb} >= {threshold_val:+.4f}, continuing")
                            else:
                                comb_str = f"{mid_comb:+.4f}" if mid_comb is not None else "None"
                                print(f"  {pct} check: val_ret={mid_ret} sortino={mid_sort} comb={comb_str} (collecting)")
                        elif use_poly_prune:
                            prune, proj = _poly_stopper.should_prune(best_trial_combined_score, tolerance=_poly_tol)
                            proj_str = f"{proj:+.4f}" if proj is not None else "None"
                            if prune:
                                print(f"  POLY PRUNE at {pct}: projected={proj_str} "
                                      f"< best_combined*{_poly_tol}={best_trial_combined_score * _poly_tol:+.4f}")
                                early_rejected = True
                                break
                            else:
                                print(f"  {pct} poly check: projected={proj_str} >= threshold, continuing")
                        else:
                            if mid_ret is not None and best_trial_val_return > -float("inf"):
                                threshold_val = best_trial_val_return * early_reject_threshold
                                if mid_ret < threshold_val:
                                    print(f"  EARLY REJECT at {pct}: val_ret={mid_ret:+.4f} "
                                          f"< {threshold_val:+.4f} (best_val*{early_reject_threshold})")
                                    early_rejected = True
                                    break
                                else:
                                    print(f"  {pct} check: val_ret={mid_ret:+.4f} >= {threshold_val:+.4f}, continuing")
                    if early_rejected:
                        break

            # Kill if still running
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=10)
            try:
                if proc.stdout is not None:
                    remainder = proc.stdout.read()
                    if remainder:
                        stdout_lines.extend(
                            line.strip()
                            for line in remainder.decode("utf-8", errors="replace").splitlines()
                            if line.strip()
                        )
            except Exception:
                pass
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        elapsed = time.time() - t0
        return_code = proc.poll()

        # Parse training stats from last logged line
        train_return = None
        train_sortino = None
        train_wr = None
        total_steps = 0
        for line in reversed(stdout_lines):
            if "ret=" in line and train_return is None:
                try:
                    for part in line.split():
                        if part.startswith("ret="):
                            train_return = float(part.split("=")[1])
                        elif part.startswith("sortino="):
                            train_sortino = float(part.split("=")[1])
                        elif part.startswith("wr="):
                            train_wr = float(part.split("=")[1])
                        elif part.startswith("step="):
                            total_steps = int(part.split("=")[1].replace(",", ""))
                except Exception:
                    pass
                if train_return is not None:
                    break

    except Exception as e:
        return {"error": str(e), "train_return": None}

    early_tag = " [EARLY REJECTED]" if early_rejected else ""
    print(f"  Training done: {elapsed:.0f}s, {total_steps:,} steps, "
          f"ret={train_return}, sortino={train_sortino}, wr={train_wr}{early_tag}")

    # Check if checkpoint exists
    ckpt_path = Path(checkpoint_dir) / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = Path(checkpoint_dir) / "final.pt"
    if not ckpt_path.exists():
        pts = list(Path(checkpoint_dir).glob("*.pt"))
        if pts:
            ckpt_path = max(pts, key=lambda p: p.stat().st_mtime)
        else:
            error_text = "no checkpoint"
            if early_rejected:
                error_text = "early rejected, no checkpoint"
            elif return_code not in (None, 0):
                error_tail = _trim_error("\n".join(stdout_lines[-12:]))
                if error_tail:
                    error_text = f"train failed (exit {return_code}): {error_tail}"
            return {
                "error": error_text,
                "train_return": train_return,
                "train_steps": total_steps,
                "early_rejected": early_rejected,
            }

    # Evaluate on validation data
    print(f"  Evaluating on validation data...")
    eval_cmd = [
        sys.executable, "-u", "-m", "pufferlib_market.evaluate",
        "--checkpoint", str(ckpt_path),
        "--data-path", val_data,
        "--deterministic",
        "--hidden-size", str(config.hidden_size),
        "--max-steps", str(config.max_steps),
        "--num-episodes", "100",
        "--seed", "42",
        "--fill-slippage-bps", "8",  # always eval with realistic slippage
        "--periods-per-year", str(config.periods_per_year),
    ]
    if config.arch != "mlp":
        eval_cmd.extend(["--arch", config.arch])
    val_return = None
    val_wr = None
    val_sortino = None
    val_profitable_pct = None
    eval_error = ""
    try:
        result = _run_capture(eval_cmd, cwd=REPO, timeout_s=eval_timeout_s)
        eval_output = result.stdout + result.stderr
        for line in eval_output.split("\n"):
            if "Return:" in line and "mean=" in line:
                try:
                    val_return = float(line.split("mean=")[1].split()[0])
                except Exception:
                    pass
            if "Win rate:" in line and "mean=" in line:
                try:
                    val_wr = float(line.split("mean=")[1].split()[0])
                except Exception:
                    pass
            if "Sortino:" in line and "mean=" in line:
                try:
                    val_sortino = float(line.split("mean=")[1].split()[0])
                except Exception:
                    pass
            if ">0:" in line:
                try:
                    pct_str = line.split("(")[1].split("%")[0]
                    val_profitable_pct = float(pct_str)
                except Exception:
                    pass
        if result.returncode != 0:
            eval_error = _trim_error(result.stderr or result.stdout or f"eval exit {result.returncode}")
    except subprocess.TimeoutExpired:
        eval_error = "eval timeout"
    except Exception as e:
        eval_error = f"eval error: {e}"

    print(f"  Val: ret={val_return}, sortino={val_sortino}, "
          f"wr={val_wr}, profitable={val_profitable_pct}%")

    holdout_metrics: dict[str, float] = {}
    holdout_error = ""
    effective_holdout_data = holdout_data or val_data
    if holdout_n_windows > 0 and holdout_eval_steps > 0:
        print(
            f"  Holdout: windows={holdout_n_windows}, steps={holdout_eval_steps}, "
            f"data={effective_holdout_data}"
        )
        holdout_json_path = Path(checkpoint_dir) / "holdout_summary.json"
        effective_holdout_fee = config.fee_rate if holdout_fee_rate < 0.0 else holdout_fee_rate
        holdout_cmd = [
            sys.executable, "-u", "-m", "pufferlib_market.evaluate_holdout",
            "--checkpoint", str(ckpt_path),
            "--data-path", effective_holdout_data,
            "--eval-hours", str(holdout_eval_steps),
            "--n-windows", str(holdout_n_windows),
            "--seed", str(holdout_seed),
            "--fee-rate", str(effective_holdout_fee),
            "--fill-buffer-bps", str(holdout_fill_buffer_bps),
            "--max-leverage", str(holdout_max_leverage),
            "--short-borrow-apr", str(holdout_short_borrow_apr),
            "--periods-per-year", str(config.periods_per_year),
            "--deterministic",
            "--no-early-stop",
            "--out", str(holdout_json_path),
        ]
        if holdout_end_within_steps > 0:
            holdout_cmd.extend(["--end-within-hours", str(holdout_end_within_steps)])
        try:
            holdout_result = _run_capture(holdout_cmd, cwd=REPO, timeout_s=holdout_timeout_s)
            if holdout_result.returncode != 0:
                holdout_error = _trim_error(
                    holdout_result.stderr or holdout_result.stdout or f"holdout exit {holdout_result.returncode}"
                )
            elif not holdout_json_path.exists():
                holdout_error = "holdout output missing"
            else:
                holdout_payload = json.loads(holdout_json_path.read_text())
                holdout_metrics = summarize_holdout_payload(holdout_payload)
        except subprocess.TimeoutExpired:
            holdout_error = "holdout timeout"
        except Exception as e:
            holdout_error = f"holdout error: {e}"

        if holdout_metrics:
            print(
                "  Holdout summary: "
                f"robust={holdout_metrics.get('holdout_robust_score')}, "
                f"p25_ret={holdout_metrics.get('holdout_return_p25_pct')}%, "
                f"worst_ret={holdout_metrics.get('holdout_return_worst_pct')}%"
            )

    market_metrics: dict[str, float] = {}
    market_validation_error = ""
    if market_validation_asset_class:
        print(
            f"  Market validation: asset_class={market_validation_asset_class}, "
            f"days={market_validation_days}, "
            f"decision_cadence={market_validation_decision_cadence}"
        )
        market_json_path = Path(checkpoint_dir) / "market_validation.json"
        market_cmd = [
            sys.executable, "-u", "-m", "unified_orchestrator.market_validation",
            "--asset-class", market_validation_asset_class,
            "--days", str(market_validation_days),
            "--cash", str(market_validation_cash),
            "--decision-cadence", str(market_validation_decision_cadence),
            "--checkpoint", str(ckpt_path),
            "--write-json", str(market_json_path),
        ]
        if market_validation_symbols:
            symbols = [sym.strip().upper() for sym in market_validation_symbols.split(",") if sym.strip()]
            if symbols:
                market_cmd.extend(["--symbols", *symbols])
        try:
            market_result = _run_capture(market_cmd, cwd=REPO, timeout_s=market_validation_timeout_s)
            if market_result.returncode != 0:
                market_validation_error = _trim_error(
                    market_result.stderr or market_result.stdout or f"market validation exit {market_result.returncode}"
                )
            elif not market_json_path.exists():
                market_validation_error = "market validation output missing"
            else:
                market_payload = json.loads(market_json_path.read_text())
                market_metrics = summarize_market_validation_payload(market_payload)
        except subprocess.TimeoutExpired:
            market_validation_error = "market validation timeout"
        except Exception as e:
            market_validation_error = f"market validation error: {e}"

        if market_metrics:
            print(
                "  Market validation summary: "
                f"return={market_metrics.get('market_return_pct')}%, "
                f"sortino={market_metrics.get('market_sortino')}, "
                f"goodness={market_metrics.get('market_goodness_score')}"
            )

    replay_metrics: dict[str, float] = {}
    replay_eval_error = ""
    effective_replay_data = replay_eval_data or holdout_data or val_data
    if replay_eval_hourly_root and replay_eval_start_date and replay_eval_end_date:
        print(
            "  Replay eval: "
            f"data={effective_replay_data}, "
            f"hourly_root={replay_eval_hourly_root}, "
            f"dates={replay_eval_start_date}..{replay_eval_end_date}"
        )
        replay_max_steps = int(config.max_steps)
        try:
            _, replay_timesteps = _read_mktd_header(str(effective_replay_data))
            replay_max_steps = min(replay_max_steps, max(1, int(replay_timesteps) - 1))
        except Exception:
            replay_max_steps = int(config.max_steps)

        replay_json_path = Path(checkpoint_dir) / "replay_eval.json"
        replay_cmd = [
            sys.executable, "-u", "-m", "pufferlib_market.replay_eval",
            "--checkpoint", str(ckpt_path),
            "--daily-data-path", str(effective_replay_data),
            "--hourly-data-root", str(replay_eval_hourly_root),
            "--start-date", str(replay_eval_start_date),
            "--end-date", str(replay_eval_end_date),
            "--max-steps", str(replay_max_steps),
            "--fee-rate", str(config.fee_rate),
            "--fill-buffer-bps", str(replay_eval_fill_buffer_bps),
            "--max-leverage", str(holdout_max_leverage),
            "--short-borrow-apr", str(holdout_short_borrow_apr),
            "--daily-periods-per-year", str(config.periods_per_year),
            "--hourly-periods-per-year", str(replay_eval_hourly_periods_per_year),
            "--arch", str(config.arch),
            "--hidden-size", str(config.hidden_size),
            "--deterministic",
            "--output-json", str(replay_json_path),
        ]
        if replay_eval_robust_start_states:
            replay_cmd.extend(["--robust-start-states", str(replay_eval_robust_start_states)])
        if replay_eval_run_hourly_policy:
            replay_cmd.append("--run-hourly-policy")
        try:
            replay_result = _run_capture(replay_cmd, cwd=REPO, timeout_s=replay_eval_timeout_s)
            if replay_result.returncode != 0:
                replay_eval_error = _trim_error(
                    replay_result.stderr or replay_result.stdout or f"replay eval exit {replay_result.returncode}"
                )
            elif not replay_json_path.exists():
                replay_eval_error = "replay eval output missing"
            else:
                replay_payload = json.loads(replay_json_path.read_text())
                replay_metrics = summarize_replay_eval_payload(replay_payload)
        except subprocess.TimeoutExpired:
            replay_eval_error = "replay eval timeout"
        except Exception as e:
            replay_eval_error = f"replay eval error: {e}"

        if replay_metrics:
            print(
                "  Replay eval summary: "
                f"hourly_replay_return={replay_metrics.get('replay_hourly_return_pct')}%, "
                f"hourly_replay_sortino={replay_metrics.get('replay_hourly_sortino')}"
            )

    smooth_metrics: dict[str, float] = {}
    smooth_error = ""
    if multi_period_eval_windows:
        print(
            f"  Multi-period eval: windows={multi_period_eval_windows}, "
            f"n_per_size={multi_period_n_windows_per_size}, data={effective_holdout_data}"
        )
        try:
            from pufferlib_market.evaluate_fast import multi_period_eval as _mpe
            effective_holdout_fee = config.fee_rate if holdout_fee_rate < 0.0 else holdout_fee_rate
            mp_result = _mpe(
                str(ckpt_path),
                str(effective_holdout_data),
                window_sizes=multi_period_eval_windows,
                n_windows_per_size=multi_period_n_windows_per_size,
                fee_rate=effective_holdout_fee,
                fill_slippage_bps=multi_period_fill_slippage_bps,
                periods_per_year=config.periods_per_year,
                max_leverage=holdout_max_leverage,
                short_borrow_apr=holdout_short_borrow_apr,
                deterministic=True,
                arch=config.arch,
                hidden_size=config.hidden_size,
            )
            smooth_metrics["smooth_score"] = float(mp_result["smoothness_score"])
            for ws in multi_period_eval_windows:
                ps = mp_result["per_size"].get(ws, {})
                smooth_metrics[f"smooth_{ws}d_p10_sortino"] = float(ps.get("p10_sortino", 0.0))
            print(
                f"  Multi-period smooth_score={smooth_metrics.get('smooth_score'):+.4f} "
                + " ".join(
                    f"{ws}d={smooth_metrics.get(f'smooth_{ws}d_p10_sortino', 0.0):+.3f}"
                    for ws in multi_period_eval_windows
                )
            )
        except Exception as e:
            smooth_error = f"multi_period_eval error: {e}"
            print(f"  Multi-period eval FAILED: {smooth_error}")

    result_payload: dict[str, object] = {
        "train_return": train_return,
        "train_sortino": train_sortino,
        "train_wr": train_wr,
        "train_steps": total_steps,
        "val_return": val_return,
        "val_sortino": val_sortino,
        "val_wr": val_wr,
        "val_profitable_pct": val_profitable_pct,
        "elapsed_s": elapsed,
        "error": eval_error,
        "holdout_error": holdout_error,
        "market_validation_error": market_validation_error,
        "replay_eval_error": replay_eval_error,
        "early_rejected": early_rejected,
        "smooth_error": smooth_error,
        "poly_projected_final": _poly_stopper.projected_final(),
    }
    result_payload.update(holdout_metrics)
    result_payload.update(market_metrics)
    result_payload.update(replay_metrics)
    result_payload.update(smooth_metrics)
    selected_metric, rank_score = select_rank_score(result_payload, rank_metric=rank_metric)
    result_payload["rank_metric"] = selected_metric
    result_payload["rank_score"] = rank_score

    if wandb_project and _wandb_module is not None:
        try:
            summary_run = _wandb_module.init(
                project=wandb_project,
                name=config.description,
                group=wandb_group or None,
                config=asdict(config),
                reinit=True,
            )
            summary_updates: dict = {
                "trial/elapsed_s": elapsed,
                "trial/train_steps": total_steps,
                "trial/rank_score": rank_score,
                "trial/rank_metric": selected_metric,
            }
            if val_return is not None:
                summary_updates["best_val_return"] = val_return
            if val_sortino is not None:
                summary_updates["val/sortino"] = val_sortino
            if val_wr is not None:
                summary_updates["val/win_rate"] = val_wr
            if val_profitable_pct is not None:
                summary_updates["val/profitable_pct"] = val_profitable_pct
            if train_return is not None:
                summary_updates["train/final_return"] = train_return
            if train_sortino is not None:
                summary_updates["train/final_sortino"] = train_sortino
            for k, v in holdout_metrics.items():
                summary_updates[f"holdout/{k}"] = v
            for k, v in market_metrics.items():
                summary_updates[f"market/{k}"] = v
            summary_run.summary.update(summary_updates)
            summary_run.finish()
        except Exception:
            pass  # never crash autoresearch due to wandb errors

    return result_payload


def main():
    parser = argparse.ArgumentParser(description="Auto-research RL trading configs")
    listing_only = "--list-experiments" in sys.argv
    stocks_mode = "--stocks" in sys.argv or "--h100-mode" in sys.argv or "--stocks12" in sys.argv
    # In stocks mode the data paths default to stocks12/stocks20 daily bins so they are
    # optional even if not listing.
    data_required = not listing_only and not stocks_mode
    parser.add_argument("--seed-only", action="store_true",
                        help="Random mutations only change seed (not other hyperparams). "
                             "Used for pure seed sweeps around a known-good config. "
                             "Best with --stocks and --start-from 172.")
    parser.add_argument("--per-env-focused", action="store_true",
                        help="Restrict random mutations to safe per_env stocks12 combinations: "
                             "locks anneal_lr=True, advantage_norm in [per_env, group_relative], "
                             "h in [256, 512]. Use with --init-best-config v_rmu2201_per_env_style "
                             "to focus search around proven per_env config. "
                             "Increases per_env escape rate by avoiding known-bad mutations.")
    parser.add_argument("--stocks", action="store_true",
                        help="Use stock-specific configs and Alpaca daily defaults "
                             "(fee_rate=0.001, periods_per_year=252, "
                             "data defaults to stocks20_daily_{train,val}.bin)")
    parser.add_argument("--h100-mode", action="store_true",
                        help="Use H100-optimized experiment pool (H100_STOCK_EXPERIMENTS) focused on "
                             "stocks20 with configs derived from local RTX 5090 scaling sweep. "
                             "Implies --stocks. Data defaults to stocks20_daily_{train,val}.bin. "
                             "Also sets time_budget=200s if not overridden.")
    parser.add_argument("--stocks12", action="store_true",
                        help="Convenience flag: use stocks12 data by default and run the combined pool "
                             "(STOCK_EXPERIMENTS + H100_STOCK_EXPERIMENTS, excluding requires_gpu='h100' "
                             "configs unless --h100-mode is also set). "
                             "Sets periods_per_year=252, fee_rate=0.001, holdout_eval_steps=90. "
                             "Implies --stocks. Data defaults to stocks12_daily_{train,val}.bin.")
    parser.add_argument("--a40-mode", action="store_true",
                        help="A40/RTX6000-Ada optimized: 128 envs, bf16, cuda-graph-ppo, "
                             "time_budget=250s. Overrides per-config settings for all trials.")
    parser.add_argument("--scale-pairs", type=int, default=0,
                        help="Use stocksN_daily_{train,val}.bin instead of stocks20. "
                             "Falls back to stocks20 if the file does not exist.")
    parser.add_argument("--train-data", required=data_required, default=None)
    parser.add_argument("--val-data", required=data_required, default=None)
    parser.add_argument("--time-budget", type=int, default=300,
                        help="Training time budget per trial in seconds")
    parser.add_argument("--max-trials", type=int, default=50)
    parser.add_argument("--leaderboard", default="pufferlib_market/autoresearch_leaderboard.csv")
    parser.add_argument("--checkpoint-root", default="pufferlib_market/checkpoints/autoresearch")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Skip first N experiments")
    parser.add_argument("--descriptions", default="",
                        help="Optional comma-separated subset of experiment descriptions to run")
    parser.add_argument("--periods-per-year", type=float, default=8760.0,
                        help="8760 for hourly, 365 for daily")
    parser.add_argument("--max-steps-override", type=int, default=0,
                        help="Override max_steps for all experiments (e.g. 90 for daily)")
    parser.add_argument("--fee-rate-override", type=float, default=-1.0,
                        help="Override fee_rate for all experiments (e.g. 0.0 for FDUSD zero-fee)")
    parser.add_argument("--holdout-data", default=None,
                        help="Optional MKTD data for robust holdout evaluation (defaults to --val-data)")
    parser.add_argument("--holdout-eval-steps", type=int, default=0,
                        help="Window size for holdout evaluation; 0 uses each trial's max_steps")
    parser.add_argument("--holdout-n-windows", type=int, default=20,
                        help="Number of random holdout windows; 0 disables holdout robustness scoring")
    parser.add_argument("--holdout-seed", type=int, default=1337)
    parser.add_argument("--holdout-end-within-steps", type=int, default=0,
                        help="Restrict holdout windows to end within the latest N steps")
    parser.add_argument("--holdout-fee-rate", type=float, default=-1.0,
                        help="Holdout fee rate override; negative inherits each trial config")
    parser.add_argument("--holdout-fill-buffer-bps", type=float, default=5.0,
                        help="Require holdout daily bars to trade through each limit by this many bps")
    parser.add_argument("--holdout-max-leverage", type=float, default=1.0)
    parser.add_argument("--holdout-short-borrow-apr", type=float, default=0.0)
    parser.add_argument("--rank-metric",
                        choices=[
                            "auto",
                            "val_return",
                            "holdout_robust_score",
                            "smooth_score",
                            "replay_combo_score",
                            "market_goodness_score",
                            "replay_hourly_policy_robust_worst_return_pct",
                            "replay_hourly_return_pct",
                            "replay_hourly_policy_return_pct",
                            "replay_hourly_robust_worst_return_pct",
                        ],
                        default="auto")
    parser.add_argument("--multi-period-eval", action="store_true",
                        help="Evaluate across multiple window sizes (5,15,30,60,90 trading days) and rank by smoothness_score")
    parser.add_argument("--multi-period-windows", default="5,15,30,60,90",
                        help="Comma-separated window sizes for multi-period eval (default: 5,15,30,60,90)")
    parser.add_argument("--multi-period-n-per-size", type=int, default=8,
                        help="Number of random windows per window size in multi-period eval (default: 8)")
    parser.add_argument("--multi-period-slippage-bps", type=float, default=8.0,
                        help="Fill slippage bps for multi-period eval (default: 8.0)")
    parser.add_argument("--market-validation-asset-class", choices=["", "crypto", "stock"], default="",
                        help="Run unified_orchestrator.market_validation for each checkpoint when set")
    parser.add_argument("--market-validation-days", type=int, default=30)
    parser.add_argument("--market-validation-cash", type=float, default=10_000.0)
    parser.add_argument(
        "--market-validation-decision-cadence",
        choices=["hourly", "daily"],
        default="hourly",
        help="Decision frequency for unified_orchestrator.market_validation",
    )
    parser.add_argument("--market-validation-symbols", default=None,
                        help="Comma-separated symbols override for market validation")
    parser.add_argument("--eval-timeout-seconds", type=int, default=0,
                        help="Optional timeout for the base validation subprocess; 0 disables it")
    parser.add_argument("--holdout-timeout-seconds", type=int, default=0,
                        help="Optional timeout for holdout evaluation; 0 disables it")
    parser.add_argument("--market-validation-timeout-seconds", type=int, default=0,
                        help="Optional timeout for market validation; 0 disables it")
    parser.add_argument("--replay-eval-data", default=None,
                        help="Optional daily MKTD path for replay_eval (defaults to --holdout-data or --val-data)")
    parser.add_argument("--replay-eval-hourly-root", default="",
                        help="Run pufferlib_market.replay_eval when set to an hourly data root")
    parser.add_argument("--replay-eval-start-date", default="",
                        help="Inclusive UTC start date used for the replay_eval daily MKTD export")
    parser.add_argument("--replay-eval-end-date", default="",
                        help="Inclusive UTC end date used for the replay_eval daily MKTD export")
    parser.add_argument("--replay-eval-run-hourly-policy", action="store_true",
                        help="Also run the hourly-policy stress mode inside replay_eval")
    parser.add_argument("--replay-eval-robust-start-states", default="",
                        help="Optional comma-separated replay start states like 'flat,long:BTCUSD:0.25'")
    parser.add_argument("--replay-eval-fill-buffer-bps", type=float, default=5.0,
                        help="Require replay_eval daily bars to trade through each limit by this many bps")
    parser.add_argument("--replay-eval-hourly-periods-per-year", type=float, default=8760.0)
    parser.add_argument("--replay-eval-timeout-seconds", type=int, default=0,
                        help="Optional timeout for replay_eval; 0 disables it")
    parser.add_argument("--max-timesteps-per-sample", type=int, default=1000,
                        help="Cap total_timesteps at num_samples * this value (prevents overfitting)")
    parser.add_argument("--early-reject-threshold", type=float, default=0.8,
                        help="Kill trial if val_ret < best_so_far * this at 25%%/50%% budget")
    parser.add_argument("--list-experiments", action="store_true",
                        help="Print all experiment names and exit")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed for all trials (default: use each trial config's seed field)")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name; when set each trial is logged as a separate run in this project")
    parser.add_argument("--poly-prune", action="store_true", default=True,
                        help="Use polynomial curve fitting for early stopping (default: True). "
                             "Pass --no-poly-prune to revert to fixed-threshold early rejection.")
    parser.add_argument("--no-poly-prune", dest="poly_prune", action="store_false")
    parser.add_argument("--init-best-config", type=str, default="",
                        help="Pre-seed best_config from a named experiment (e.g. 'v_rmu2201_style'). "
                             "Subsequent random mutations start from this config instead of TrialConfig() defaults.")
    parser.add_argument("--lock-best-config", action="store_true", default=False,
                        help="Never update best_config from trial results. All mutations always come from the "
                             "init config (set via --init-best-config or defaults). Useful for broad random "
                             "search around a known-good config without drift into bad regions.")
    parser.add_argument("--local", action="store_true",
                        help="Preset for local RTX GPU: sets --time-budget 60 if not overridden")
    parser.add_argument("--a40", action="store_true",
                        help="Preset for A40 GPU: sets --time-budget 180 and --a40-mode if not overridden")
    args = parser.parse_args()

    # --local: RTX GPU preset — short time budget for quick iteration
    if args.local and args.time_budget == 300:
        args.time_budget = 60

    # --a40: convenience alias that implies --a40-mode with a sensible budget
    if args.a40:
        args.a40_mode = True
        if args.time_budget == 300:
            args.time_budget = 180

    # --h100-mode implies --stocks and selects the H100-optimised experiment pool.
    # Also shorten time budget to 200s (H100 ~2x faster than A100 baseline of 300s).
    if args.h100_mode:
        args.stocks = True
        if args.time_budget == 300:
            args.time_budget = 200

    # --stocks12 implies --stocks and defaults to stocks12 data.
    if args.stocks12:
        args.stocks = True

    # --a40-mode: A40 is ~1.5x faster than A100 per dollar; use 250s budget
    if args.a40_mode and args.time_budget == 300:
        args.time_budget = 250

    # --stocks / --h100-mode / --stocks12: apply stock-mode defaults before anything else so that
    # user overrides (explicit --train-data etc.) can still win.
    if args.stocks:
        if args.train_data is None:
            if args.stocks12 and not args.h100_mode:
                args.train_data = "pufferlib_market/data/stocks12_daily_train.bin"
            else:
                args.train_data = _STOCK_DEFAULT_TRAIN
        if args.val_data is None:
            if args.stocks12 and not args.h100_mode:
                args.val_data = "pufferlib_market/data/stocks12_daily_val.bin"
            else:
                args.val_data = _STOCK_DEFAULT_VAL
        # periods_per_year default is 8760 (hourly); override to 252 (daily)
        # only when the user has NOT already supplied their own value.
        if args.periods_per_year == 8760.0:
            args.periods_per_year = 252.0
        # fee_rate override: 10bps Alpaca fee when not already overridden
        if args.fee_rate_override < 0.0:
            args.fee_rate_override = 0.001
        # Sensible daily max_steps when not already overridden
        if args.max_steps_override == 0:
            args.max_steps_override = 252
        # Holdout eval window: stocks20_daily_val has 158 timesteps; cap at 90
        # so 20 random windows can fit comfortably.
        if args.holdout_eval_steps == 0:
            args.holdout_eval_steps = 90
        # Default leaderboard and checkpoint root for stocks
        if args.leaderboard == "pufferlib_market/autoresearch_leaderboard.csv":
            if args.h100_mode:
                args.leaderboard = "autoresearch_stock_h100_leaderboard.csv"
            elif args.stocks12:
                args.leaderboard = "autoresearch_stock12_daily_leaderboard.csv"
            else:
                args.leaderboard = "autoresearch_stock_daily_leaderboard.csv"
        if args.checkpoint_root == "pufferlib_market/checkpoints/autoresearch":
            if args.h100_mode:
                args.checkpoint_root = "pufferlib_market/checkpoints/autoresearch_stock_h100"
            elif args.stocks12:
                args.checkpoint_root = "pufferlib_market/checkpoints/autoresearch_stock12"
            else:
                args.checkpoint_root = "pufferlib_market/checkpoints/autoresearch_stock"

        # --scale-pairs N: resolve stocksN data paths, fallback to stocks12
        if args.scale_pairs > 0:
            n = args.scale_pairs
            candidate_train = f"pufferlib_market/data/stocks{n}_daily_train.bin"
            candidate_val   = f"pufferlib_market/data/stocks{n}_daily_val.bin"
            train_abs = REPO / candidate_train
            val_abs   = REPO / candidate_val
            if train_abs.exists() and val_abs.exists():
                if args.train_data is None or args.train_data == _STOCK_DEFAULT_TRAIN:
                    args.train_data = candidate_train
                if args.val_data is None or args.val_data == _STOCK_DEFAULT_VAL:
                    args.val_data = candidate_val
                print(f"[scale-pairs] Using stocks{n} data: {candidate_train}")
            else:
                print(
                    f"[scale-pairs] WARNING: stocks{n}_daily_{{train,val}}.bin not found "
                    f"at {train_abs} — falling back to stocks12"
                )

    if args.list_experiments:
        if args.h100_mode:
            experiment_pool = H100_STOCK_EXPERIMENTS
        elif args.stocks12:
            experiment_pool = STOCK_EXPERIMENTS + [
                e for e in H100_STOCK_EXPERIMENTS
                if e.get("requires_gpu") != "h100"
            ]
        elif args.stocks:
            experiment_pool = STOCK_EXPERIMENTS
        else:
            experiment_pool = EXPERIMENTS + CRYPTO34_HOURLY_EXPERIMENTS
        for cfg_dict in experiment_pool:
            desc = cfg_dict.get("description", "")
            gpu = cfg_dict.get("requires_gpu", "")
            gpu_note = f" [requires_gpu={gpu}]" if gpu else ""
            print(f"{desc}{gpu_note}")
        sys.exit(0)

    leaderboard_path = Path(args.leaderboard)
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_root = Path(args.checkpoint_root)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # W&B group ties all trials in this autoresearch run together on the dashboard
    wandb_group = f"autoresearch_{int(time.time())}" if args.wandb_project else None

    # Initialize or load leaderboard
    fieldnames = [
        "trial", "description", "gpu_type", "rank_metric", "rank_score", "val_return", "val_sortino", "val_wr",
        "val_profitable_pct", "train_return", "train_sortino", "train_wr",
        "train_steps", "elapsed_s", "error", "holdout_error", "market_validation_error", "replay_eval_error",
        "holdout_robust_score", "holdout_return_mean_pct", "holdout_return_p25_pct",
        "holdout_return_worst_pct", "holdout_sortino_p25", "holdout_max_drawdown_worst_pct",
        "holdout_negative_return_rate", "holdout_median_return_pct", "holdout_p10_return_pct",
        "holdout_median_sortino", "holdout_p90_max_drawdown_pct",
        "market_return_pct", "market_sortino", "market_max_drawdown_pct",
        "market_trade_count", "market_goodness_score",
        "replay_daily_return_pct", "replay_daily_sortino", "replay_daily_max_drawdown_pct",
        "replay_daily_trade_count", "replay_daily_pnl_smoothness", "replay_daily_ulcer_index",
        "replay_daily_goodness_score",
        "replay_hourly_return_pct", "replay_hourly_sortino", "replay_hourly_max_drawdown_pct",
        "replay_hourly_trade_count", "replay_hourly_order_count", "replay_hourly_pnl_smoothness",
        "replay_hourly_ulcer_index", "replay_hourly_goodness_score",
        "replay_hourly_policy_return_pct", "replay_hourly_policy_sortino",
        "replay_hourly_policy_max_drawdown_pct", "replay_hourly_policy_trade_count",
        "replay_hourly_policy_order_count", "replay_hourly_policy_pnl_smoothness",
        "replay_hourly_policy_ulcer_index", "replay_hourly_policy_goodness_score",
        "replay_daily_robust_median_return_pct", "replay_daily_robust_worst_return_pct",
        "replay_daily_robust_worst_sortino", "replay_daily_robust_worst_max_drawdown_pct",
        "replay_hourly_robust_median_return_pct", "replay_hourly_robust_worst_return_pct",
        "replay_hourly_robust_worst_sortino", "replay_hourly_robust_worst_max_drawdown_pct",
        "replay_hourly_policy_robust_median_return_pct", "replay_hourly_policy_robust_worst_return_pct",
        "replay_hourly_policy_robust_worst_sortino", "replay_hourly_policy_robust_worst_max_drawdown_pct",
        "replay_combo_score", "replay_combo_return_mean_pct", "replay_combo_return_worst_pct",
        "replay_combo_sortino_p25", "replay_combo_max_drawdown_worst_pct",
        "replay_combo_negative_return_rate", "replay_combo_scenario_count",
        "smooth_score", "smooth_error",
        "smooth_5d_p10_sortino", "smooth_15d_p10_sortino", "smooth_30d_p10_sortino",
        "smooth_60d_p10_sortino", "smooth_90d_p10_sortino",
        "hidden_size", "lr", "ent_coef", "weight_decay", "fill_slippage_bps",
        "obs_norm", "anneal_lr", "anneal_ent", "anneal_clip", "lr_schedule",
        "arch", "fee_rate", "trade_penalty", "drawdown_penalty", "downside_penalty",
        "smooth_downside_penalty", "smooth_downside_temperature", "smoothness_penalty", "gamma",
    ]

    existing_trials = set()
    if leaderboard_path.exists():
        with open(leaderboard_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_trials.add(row.get("description", ""))

    if args.stocks:
        if args.h100_mode:
            pool = H100_STOCK_EXPERIMENTS
            mode_label = "h100 stocks mode"
        elif args.stocks12:
            pool = STOCK_EXPERIMENTS + [
                e for e in H100_STOCK_EXPERIMENTS
                if e.get("requires_gpu") != "h100"
            ]
            mode_label = "stocks12 mode"
        else:
            pool = STOCK_EXPERIMENTS
            mode_label = "stocks mode"
        experiments = _select_from_pool(
            pool,
            start_from=args.start_from,
            descriptions=args.descriptions,
        )
        print(f"[{mode_label}] {len(experiments)} configs, "
              f"train={args.train_data}, val={args.val_data}, "
              f"fee_override={args.fee_rate_override}, "
              f"periods_per_year={args.periods_per_year}, "
              f"max_steps={args.max_steps_override}")
    else:
        experiments = select_experiments(start_from=args.start_from, descriptions=args.descriptions)

    _best_known_path = Path(args.checkpoint_root).parent / "best_known_metrics.json"
    _best_tracker = BestKnownTracker(_best_known_path)

    def _infer_track(train_data_path: str, val_data_path: str) -> str:
        combined = (train_data_path + val_data_path).lower()
        if "stock" in combined:
            return "stocks_daily"
        if "fdusd" in combined or "usdt" in combined:
            return "binance_crypto"
        if "mixed" in combined:
            return "mixed"
        return "hourly_crypto"

    _track = _infer_track(args.train_data or "", args.val_data or "")

    # Add random mutations
    best_rank_score = -float("inf")
    best_val_return = -float("inf")  # tracked separately — same scale as _quick_val_eval
    best_combined_score = _best_tracker.get_best(_track)

    # In stocks_mode + seed_only: pre-seed best_config with the confirmed winning formula
    # (lr=3e-4, wd=0.01, tp=0.05, slip=5bps, h=1024, anneal_lr=True) so that all random
    # seed_only mutations use this formula from trial 1, even with --start-from 301.
    # Without this, seed_only mutations of the default TrialConfig() have wd=0/tp=0/slip=0
    # which ALL collapse to hold-cash on stocks11_2012.
    _seed_only = getattr(args, "seed_only", False)
    _init_desc = getattr(args, "init_best_config", "")
    if _init_desc:
        # Load a named config as the starting best_config for mutations.
        _init_pool = experiments if experiments else (pool if "pool" in dir() else [])
        _found = next((e for e in _init_pool if e.get("description") == _init_desc), None)
        if _found is None:
            # Also search the full STOCK_EXPERIMENTS and H100_STOCK_EXPERIMENTS pools
            for _p in [STOCK_EXPERIMENTS, H100_STOCK_EXPERIMENTS]:
                _found = next((e for e in _p if e.get("description") == _init_desc), None)
                if _found:
                    break
        if _found:
            best_config = build_config(_found)
            print(f"  [init-best-config] Pre-seeding best_config from '{_init_desc}': {_found}")
        else:
            print(f"  [init-best-config] WARNING: '{_init_desc}' not found, using TrialConfig() defaults")
            best_config = TrialConfig()
    elif getattr(args, "stocks", False) and _seed_only:
        best_config = TrialConfig(
            weight_decay=0.01,
            trade_penalty=0.05,
            fill_slippage_bps=5.0,
            lr=3e-4,
            anneal_lr=True,
            hidden_size=1024,
            ent_coef=0.05,
        )
    else:
        best_config = TrialConfig()

    trial_num = len(existing_trials)

    for i, exp_overrides in enumerate(experiments):
        if trial_num >= args.max_trials:
            print(f"\nReached max trials ({args.max_trials})")
            break

        desc = exp_overrides.get("description", f"trial_{trial_num}")
        if desc in existing_trials and not desc.startswith("random"):
            print(f"\n[{trial_num}] SKIP {desc} (already done)")
            continue

        # Handle random mutations
        if desc.startswith("random_"):
            seed_only = getattr(args, "seed_only", False)
            per_env_focused = getattr(args, "per_env_focused", False)
            config = mutate_config(best_config, stocks_mode=args.stocks, seed_only=seed_only,
                                   per_env_focused=per_env_focused)
            desc = config.description
        else:
            config = build_config(exp_overrides)

        # Apply global overrides from CLI
        if args.periods_per_year != 8760.0:
            config.periods_per_year = args.periods_per_year
        if args.max_steps_override > 0:
            config.max_steps = args.max_steps_override
        if args.fee_rate_override >= 0.0:
            config.fee_rate = args.fee_rate_override
        if args.seed is not None:
            config.seed = args.seed

        # --h100-mode: force H100 hardware settings on every trial
        if args.h100_mode:
            config.num_envs = 256
            config.minibatch_size = 4096
            config.use_bf16 = True
            config.cuda_graph_ppo = True

        # --a40-mode: force A40/RTX6000-Ada hardware settings on every trial
        if args.a40_mode:
            config.num_envs = 128
            config.minibatch_size = 2048
            config.use_bf16 = True
            config.cuda_graph_ppo = True

        if args.h100_mode or config.requires_gpu == "h100":
            gpu_type = "h100"
        elif args.a40_mode or config.requires_gpu == "a40":
            gpu_type = "a40"
        else:
            gpu_type = "a40"  # default to A40 (cost-efficient 48GB)

        holdout_eval_steps = int(args.holdout_eval_steps) if int(args.holdout_eval_steps) > 0 else int(config.max_steps)

        print(f"\n{'='*60}")
        print(f"[{trial_num}] {desc}")
        print(f"{'='*60}")

        # Key params
        key_params = {k: v for k, v in asdict(config).items()
                      if v != asdict(TrialConfig()).get(k) and k != "description"}
        if key_params:
            print(f"  Overrides: {key_params}")

        ckpt_dir = str(ckpt_root / desc)
        os.makedirs(ckpt_dir, exist_ok=True)

        result = run_trial(
            config,
            args.train_data,
            args.val_data,
            args.time_budget,
            ckpt_dir,
            wandb_project=args.wandb_project,
            wandb_group=wandb_group,
            holdout_data=args.holdout_data,
            holdout_eval_steps=holdout_eval_steps,
            holdout_n_windows=args.holdout_n_windows,
            holdout_seed=args.holdout_seed,
            holdout_end_within_steps=args.holdout_end_within_steps,
            holdout_fee_rate=args.holdout_fee_rate,
            holdout_fill_buffer_bps=args.holdout_fill_buffer_bps,
            holdout_max_leverage=args.holdout_max_leverage,
            holdout_short_borrow_apr=args.holdout_short_borrow_apr,
            eval_timeout_s=args.eval_timeout_seconds,
            holdout_timeout_s=args.holdout_timeout_seconds,
            market_validation_asset_class=args.market_validation_asset_class,
            market_validation_days=args.market_validation_days,
            market_validation_cash=args.market_validation_cash,
            market_validation_decision_cadence=args.market_validation_decision_cadence,
            market_validation_symbols=args.market_validation_symbols,
            market_validation_timeout_s=args.market_validation_timeout_seconds,
            replay_eval_data=args.replay_eval_data,
            replay_eval_hourly_root=args.replay_eval_hourly_root,
            replay_eval_start_date=args.replay_eval_start_date,
            replay_eval_end_date=args.replay_eval_end_date,
            replay_eval_run_hourly_policy=args.replay_eval_run_hourly_policy,
            replay_eval_robust_start_states=args.replay_eval_robust_start_states,
            replay_eval_fill_buffer_bps=args.replay_eval_fill_buffer_bps,
            replay_eval_hourly_periods_per_year=args.replay_eval_hourly_periods_per_year,
            replay_eval_timeout_s=args.replay_eval_timeout_seconds,
            rank_metric=args.rank_metric,
            max_timesteps_per_sample=args.max_timesteps_per_sample,
            best_trial_rank_score=best_rank_score,
            best_trial_val_return=best_val_return,
            best_trial_combined_score=best_combined_score,
            early_reject_threshold=args.early_reject_threshold,
            use_poly_prune=args.poly_prune,
            multi_period_eval_windows=(
                tuple(int(x.strip()) for x in args.multi_period_windows.split(",") if x.strip())
                if args.multi_period_eval else None
            ),
            multi_period_n_windows_per_size=args.multi_period_n_per_size,
            multi_period_fill_slippage_bps=args.multi_period_slippage_bps,
        )

        # Update leaderboard
        row = {
            "trial": trial_num,
            "description": desc,
            "gpu_type": gpu_type,
            "rank_metric": result.get("rank_metric"),
            "rank_score": result.get("rank_score"),
            "val_return": result.get("val_return"),
            "val_sortino": result.get("val_sortino"),
            "val_wr": result.get("val_wr"),
            "val_profitable_pct": result.get("val_profitable_pct"),
            "train_return": result.get("train_return"),
            "train_sortino": result.get("train_sortino"),
            "train_wr": result.get("train_wr"),
            "train_steps": result.get("train_steps"),
            "elapsed_s": result.get("elapsed_s"),
            "error": result.get("error", ""),
            "holdout_error": result.get("holdout_error", ""),
            "market_validation_error": result.get("market_validation_error", ""),
            "replay_eval_error": result.get("replay_eval_error", ""),
            "holdout_robust_score": result.get("holdout_robust_score"),
            "holdout_return_mean_pct": result.get("holdout_return_mean_pct"),
            "holdout_return_p25_pct": result.get("holdout_return_p25_pct"),
            "holdout_return_worst_pct": result.get("holdout_return_worst_pct"),
            "holdout_sortino_p25": result.get("holdout_sortino_p25"),
            "holdout_max_drawdown_worst_pct": result.get("holdout_max_drawdown_worst_pct"),
            "holdout_negative_return_rate": result.get("holdout_negative_return_rate"),
            "holdout_median_return_pct": result.get("holdout_median_return_pct"),
            "holdout_p10_return_pct": result.get("holdout_p10_return_pct"),
            "holdout_median_sortino": result.get("holdout_median_sortino"),
            "holdout_p90_max_drawdown_pct": result.get("holdout_p90_max_drawdown_pct"),
            "market_return_pct": result.get("market_return_pct"),
            "market_sortino": result.get("market_sortino"),
            "market_max_drawdown_pct": result.get("market_max_drawdown_pct"),
            "market_trade_count": result.get("market_trade_count"),
            "market_goodness_score": result.get("market_goodness_score"),
            "replay_daily_return_pct": result.get("replay_daily_return_pct"),
            "replay_daily_sortino": result.get("replay_daily_sortino"),
            "replay_daily_max_drawdown_pct": result.get("replay_daily_max_drawdown_pct"),
            "replay_daily_trade_count": result.get("replay_daily_trade_count"),
            "replay_daily_pnl_smoothness": result.get("replay_daily_pnl_smoothness"),
            "replay_daily_ulcer_index": result.get("replay_daily_ulcer_index"),
            "replay_daily_goodness_score": result.get("replay_daily_goodness_score"),
            "replay_hourly_return_pct": result.get("replay_hourly_return_pct"),
            "replay_hourly_sortino": result.get("replay_hourly_sortino"),
            "replay_hourly_max_drawdown_pct": result.get("replay_hourly_max_drawdown_pct"),
            "replay_hourly_trade_count": result.get("replay_hourly_trade_count"),
            "replay_hourly_order_count": result.get("replay_hourly_order_count"),
            "replay_hourly_pnl_smoothness": result.get("replay_hourly_pnl_smoothness"),
            "replay_hourly_ulcer_index": result.get("replay_hourly_ulcer_index"),
            "replay_hourly_goodness_score": result.get("replay_hourly_goodness_score"),
            "replay_hourly_policy_return_pct": result.get("replay_hourly_policy_return_pct"),
            "replay_hourly_policy_sortino": result.get("replay_hourly_policy_sortino"),
            "replay_hourly_policy_max_drawdown_pct": result.get("replay_hourly_policy_max_drawdown_pct"),
            "replay_hourly_policy_trade_count": result.get("replay_hourly_policy_trade_count"),
            "replay_hourly_policy_order_count": result.get("replay_hourly_policy_order_count"),
            "replay_hourly_policy_pnl_smoothness": result.get("replay_hourly_policy_pnl_smoothness"),
            "replay_hourly_policy_ulcer_index": result.get("replay_hourly_policy_ulcer_index"),
            "replay_hourly_policy_goodness_score": result.get("replay_hourly_policy_goodness_score"),
            "replay_daily_robust_median_return_pct": result.get("replay_daily_robust_median_return_pct"),
            "replay_daily_robust_worst_return_pct": result.get("replay_daily_robust_worst_return_pct"),
            "replay_daily_robust_worst_sortino": result.get("replay_daily_robust_worst_sortino"),
            "replay_daily_robust_worst_max_drawdown_pct": result.get("replay_daily_robust_worst_max_drawdown_pct"),
            "replay_hourly_robust_median_return_pct": result.get("replay_hourly_robust_median_return_pct"),
            "replay_hourly_robust_worst_return_pct": result.get("replay_hourly_robust_worst_return_pct"),
            "replay_hourly_robust_worst_sortino": result.get("replay_hourly_robust_worst_sortino"),
            "replay_hourly_robust_worst_max_drawdown_pct": result.get("replay_hourly_robust_worst_max_drawdown_pct"),
            "replay_hourly_policy_robust_median_return_pct": result.get("replay_hourly_policy_robust_median_return_pct"),
            "replay_hourly_policy_robust_worst_return_pct": result.get("replay_hourly_policy_robust_worst_return_pct"),
            "replay_hourly_policy_robust_worst_sortino": result.get("replay_hourly_policy_robust_worst_sortino"),
            "replay_hourly_policy_robust_worst_max_drawdown_pct": result.get("replay_hourly_policy_robust_worst_max_drawdown_pct"),
            "replay_combo_score": result.get("replay_combo_score"),
            "replay_combo_return_mean_pct": result.get("replay_combo_return_mean_pct"),
            "replay_combo_return_worst_pct": result.get("replay_combo_return_worst_pct"),
            "replay_combo_sortino_p25": result.get("replay_combo_sortino_p25"),
            "replay_combo_max_drawdown_worst_pct": result.get("replay_combo_max_drawdown_worst_pct"),
            "replay_combo_negative_return_rate": result.get("replay_combo_negative_return_rate"),
            "replay_combo_scenario_count": result.get("replay_combo_scenario_count"),
            "smooth_score": result.get("smooth_score"),
            "smooth_error": result.get("smooth_error", ""),
            "smooth_5d_p10_sortino": result.get("smooth_5d_p10_sortino"),
            "smooth_15d_p10_sortino": result.get("smooth_15d_p10_sortino"),
            "smooth_30d_p10_sortino": result.get("smooth_30d_p10_sortino"),
            "smooth_60d_p10_sortino": result.get("smooth_60d_p10_sortino"),
            "smooth_90d_p10_sortino": result.get("smooth_90d_p10_sortino"),
            "hidden_size": config.hidden_size,
            "lr": config.lr,
            "ent_coef": config.ent_coef,
            "weight_decay": config.weight_decay,
            "fill_slippage_bps": config.fill_slippage_bps,
            "obs_norm": config.obs_norm,
            "anneal_lr": config.anneal_lr,
            "anneal_ent": config.anneal_ent,
            "anneal_clip": config.anneal_clip,
            "lr_schedule": config.lr_schedule,
            "arch": config.arch,
            "fee_rate": config.fee_rate,
            "trade_penalty": config.trade_penalty,
            "drawdown_penalty": config.drawdown_penalty,
            "downside_penalty": config.downside_penalty,
            "smooth_downside_penalty": config.smooth_downside_penalty,
            "smooth_downside_temperature": config.smooth_downside_temperature,
            "smoothness_penalty": config.smoothness_penalty,
            "gamma": config.gamma,
        }

        write_header = not leaderboard_path.exists()
        with open(leaderboard_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        # Track best
        rank_score = _safe_float(result.get("rank_score"))
        _lock = getattr(args, "lock_best_config", False)
        if rank_score is not None and rank_score > best_rank_score:
            best_rank_score = rank_score
            if not _lock:
                best_config = config
            metric_name = str(result.get("rank_metric", "rank_score"))
            print(f"  *** NEW BEST {metric_name}={rank_score:.4f} ***")
        # Track best val_return separately (same scale as early-reject _quick_val_eval)
        val_ret = _safe_float(result.get("val_return"))
        if val_ret is not None and val_ret > best_val_return:
            best_val_return = val_ret
        trial_combined = _combined_score(
            result.get("val_return"), result.get("val_sortino"), result.get("val_wr")
        )
        if trial_combined is not None:
            if _best_tracker.update(_track, trial_combined, config.description):
                best_combined_score = trial_combined

        trial_num += 1
        existing_trials.add(desc)

    # Print final leaderboard
    print(f"\n{'='*60}")
    print("LEADERBOARD (sorted by rank_score)")
    print(f"{'='*60}")
    if leaderboard_path.exists():
        with open(leaderboard_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        rows_with_rank = [r for r in rows if _leaderboard_sort_value(r) > -float("inf")]
        rows_with_rank.sort(key=_leaderboard_sort_value, reverse=True)
        for r in rows_with_rank[:15]:
            rank_metric = r.get("rank_metric") or "val_return"
            rank_score = _safe_float(r.get("rank_score"))
            val_ret = _safe_float(r.get("val_return"))
            holdout_score = _safe_float(r.get("holdout_robust_score"))
            market_score = _safe_float(r.get("market_goodness_score"))
            rank_text = "n/a" if rank_score is None else f"{rank_score:+.4f}"
            val_text = "n/a" if val_ret is None else f"{val_ret:+.4f}"
            holdout_text = "n/a" if holdout_score is None else f"{holdout_score:+.2f}"
            market_text = "n/a" if market_score is None else f"{market_score:+.2f}"
            print(
                f"  {r['description']:30s} rank[{rank_metric}]={rank_text} "
                f"val_ret={val_text} holdout={holdout_text} market={market_text} "
                f"steps={r['train_steps']:>10s}"
            )


if __name__ == "__main__":
    main()
