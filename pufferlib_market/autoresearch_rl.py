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

from src.robust_trading_metrics import summarize_scenario_results

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
    max_steps: int = 720
    periods_per_year: float = 8760.0
    seed: int = 42
    description: str = ""
    max_leverage: float = 1.0
    short_borrow_apr: float = 0.0
    requires_gpu: str = ""  # e.g. "a100", "h100", "" = any GPU (dispatcher metadata only)
    # H100-scale training settings (passed through to train.py)
    minibatch_size: int = 2048
    use_bf16: bool = False
    cuda_graph_ppo: bool = False


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
]

# Alias used by sweep scripts and verification commands.
TRIAL_CONFIGS = EXPERIMENTS


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

    # Replicate best known config (h1024 + anneal_lr) at H100 scale
    {"description": "h1024_h100",
     "hidden_size": 1024, "anneal_lr": True, "ent_coef": 0.05,
     "num_envs": 256, "minibatch_size": 4096, "cuda_graph_ppo": True, "use_bf16": True,
     "requires_gpu": "h100"},

    # h2048 was under-converged at 50M steps — H100's 4x more data/step may fix it
    {"description": "h2048_h100",
     "hidden_size": 2048, "anneal_lr": True, "ent_coef": 0.05,
     "fill_slippage_bps": 5.0, "trade_penalty": 0.05,
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

    # Random mutations to explore the neighbourhood (30 slots per sweep pass)
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
]

# Default data paths used when --stocks is given and no explicit --train-data is provided.
_STOCK_DEFAULT_TRAIN = "pufferlib_market/data/stocks12_daily_train.bin"
_STOCK_DEFAULT_VAL   = "pufferlib_market/data/stocks12_daily_val.bin"


def build_config(overrides: dict) -> TrialConfig:
    """Create a TrialConfig with overrides applied."""
    cfg = TrialConfig(**{k: v for k, v in overrides.items() if k in TrialConfig.__dataclass_fields__})
    if "description" in overrides:
        cfg.description = overrides["description"]
    return cfg


def mutate_config(base: TrialConfig) -> TrialConfig:
    """Randomly mutate a config for exploration."""
    d = asdict(base)
    # Pick 2-3 params to mutate
    mutable_params = {
        "hidden_size": [256, 512, 1024],
        "lr": [1e-4, 2e-4, 3e-4, 5e-4],
        "ent_coef": [0.01, 0.03, 0.05, 0.08, 0.1],
        "weight_decay": [0.0, 0.001, 0.005, 0.01, 0.05],
        "fill_slippage_bps": [0.0, 5.0, 8.0, 12.0],
        "gamma": [0.98, 0.99, 0.995],
        "advantage_norm": ["global", "per_env", "group_relative"],
        "group_relative_mix": [0.0, 0.15, 0.25, 0.4],
        "reward_scale": [5.0, 10.0, 20.0],
        "cash_penalty": [0.0, 0.005, 0.01, 0.02],
        "trade_penalty": [0.0, 0.01, 0.02, 0.05],
        "drawdown_penalty": [0.0, 0.01, 0.02, 0.05],
        "smooth_downside_temperature": [0.01, 0.02, 0.05],
        "smoothness_penalty": [0.0, 0.005, 0.01, 0.02],
        "obs_norm": [True, False],
        "anneal_lr": [True, False],
    }
    keys = random.sample(list(mutable_params.keys()), min(3, len(mutable_params)))
    for k in keys:
        d[k] = random.choice(mutable_params[k])
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
    return summary


def select_rank_score(
    metrics: dict[str, object],
    *,
    rank_metric: str = "auto",
) -> tuple[str, float | None]:
    """Choose the leaderboard ranking signal with sensible fallbacks."""
    candidates = {
        "market_goodness_score": _safe_float(metrics.get("market_goodness_score")),
        "holdout_robust_score": _safe_float(metrics.get("holdout_robust_score")),
        "replay_hourly_return_pct": _safe_float(metrics.get("replay_hourly_return_pct")),
        "replay_hourly_policy_return_pct": _safe_float(metrics.get("replay_hourly_policy_return_pct")),
        "val_return": _safe_float(metrics.get("val_return")),
    }
    if rank_metric == "auto":
        for name in (
            "market_goodness_score",
            "holdout_robust_score",
            "replay_hourly_return_pct",
            "val_return",
        ):
            score = candidates[name]
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
    return _select_from_pool(EXPERIMENTS, start_from=start_from, descriptions=descriptions)


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
    replay_eval_fill_buffer_bps: float = 5.0,
    replay_eval_hourly_periods_per_year: float = 8760.0,
    replay_eval_timeout_s: int = 0,
    rank_metric: str = "auto",
    max_timesteps_per_sample: int = 1000,
    best_trial_rank_score: float = -float("inf"),
    early_reject_threshold: float = 0.8,
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
    if config.minibatch_size != 2048:
        cmd.extend(["--minibatch-size", str(config.minibatch_size)])
    if config.use_bf16:
        cmd.append("--use-bf16")
    if config.cuda_graph_ppo:
        cmd.append("--cuda-graph-ppo")
    if wandb_project:
        cmd.extend([
            "--wandb-project", wandb_project,
            "--wandb-run-name", config.description,
        ])
        if wandb_group:
            cmd.extend(["--wandb-group", wandb_group])

    # Run training with time budget
    print(f"\n  Training for {time_budget}s...")
    t0 = time.time()
    early_rejected = False

    def _quick_val_eval(ckpt: Path) -> float | None:
        """Run a fast C-env eval on a checkpoint against val data."""
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
        try:
            qr = _run_capture(qcmd, cwd=REPO, timeout_s=60)
            for qline in qr.stdout.split("\n"):
                if "Return:" in qline and "mean=" in qline:
                    return float(qline.split("mean=")[1].split()[0])
        except Exception:
            pass
        return None

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(REPO), preexec_fn=os.setsid,
        )
        stdout_lines = []
        _check_25 = False
        _check_50 = False
        try:
            while time.time() - t0 < time_budget:
                if proc.poll() is not None:
                    break
                try:
                    line = proc.stdout.readline()
                    if line:
                        stdout_lines.append(line.decode("utf-8", errors="replace").strip())
                except Exception:
                    pass

                # Early rejection at 25% and 50% of time budget
                if best_trial_rank_score > -float("inf") and time_budget >= 60:
                    progress = (time.time() - t0) / max(time_budget, 1)
                    if progress >= 0.25 and not _check_25:
                        _check_25 = True
                        pts = sorted(Path(checkpoint_dir).glob("*.pt"), key=lambda p: p.stat().st_mtime)
                        if pts:
                            mid_ckpt = pts[-1]
                            mid_val = _quick_val_eval(mid_ckpt)
                            if mid_val is not None:
                                threshold = best_trial_rank_score * early_reject_threshold
                                if mid_val < threshold:
                                    print(f"  EARLY REJECT at 25%: val_ret={mid_val:+.4f} "
                                          f"< {threshold:+.4f} (best*{early_reject_threshold})")
                                    early_rejected = True
                                    break
                                else:
                                    print(f"  25% check: val_ret={mid_val:+.4f} >= {threshold:+.4f}, continuing")
                    if progress >= 0.50 and not _check_50:
                        _check_50 = True
                        pts = sorted(Path(checkpoint_dir).glob("*.pt"), key=lambda p: p.stat().st_mtime)
                        if pts:
                            mid_ckpt = pts[-1]
                            mid_val = _quick_val_eval(mid_ckpt)
                            if mid_val is not None:
                                threshold = best_trial_rank_score * early_reject_threshold
                                if mid_val < threshold:
                                    print(f"  EARLY REJECT at 50%: val_ret={mid_val:+.4f} "
                                          f"< {threshold:+.4f} (best*{early_reject_threshold})")
                                    early_rejected = True
                                    break
                                else:
                                    print(f"  50% check: val_ret={mid_val:+.4f} >= {threshold:+.4f}, continuing")

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
        replay_json_path = Path(checkpoint_dir) / "replay_eval.json"
        replay_cmd = [
            sys.executable, "-u", "-m", "pufferlib_market.replay_eval",
            "--checkpoint", str(ckpt_path),
            "--daily-data-path", str(effective_replay_data),
            "--hourly-data-root", str(replay_eval_hourly_root),
            "--start-date", str(replay_eval_start_date),
            "--end-date", str(replay_eval_end_date),
            "--max-steps", str(config.max_steps),
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
    }
    result_payload.update(holdout_metrics)
    result_payload.update(market_metrics)
    result_payload.update(replay_metrics)
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
    stocks_mode = "--stocks" in sys.argv
    # In stocks mode the data paths default to stocks12 daily bins so they are
    # optional even if not listing.
    data_required = not listing_only and not stocks_mode
    parser.add_argument("--stocks", action="store_true",
                        help="Use stock-specific configs and Alpaca daily defaults "
                             "(fee_rate=0.001, periods_per_year=252, "
                             "data defaults to stocks12_daily_{train,val}.bin)")
    parser.add_argument("--h100-mode", action="store_true",
                        help="H100 scale-up: num_envs=256, minibatch=4096, bf16, cuda-graph-ppo, "
                             "time_budget=200s. Overrides per-config settings for all trials.")
    parser.add_argument("--scale-pairs", type=int, default=0,
                        help="Use stocksN_daily_{train,val}.bin instead of stocks12. "
                             "Falls back to stocks12 if the file does not exist.")
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
                            "market_goodness_score",
                            "replay_hourly_return_pct",
                            "replay_hourly_policy_return_pct",
                        ],
                        default="auto")
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
    args = parser.parse_args()

    # --h100-mode: shorten time budget to match A100 wall-time (H100 is ~2x faster)
    if args.h100_mode and args.time_budget == 300:
        args.time_budget = 200

    # --stocks: apply stock-mode defaults before anything else so that
    # user overrides (explicit --train-data etc.) can still win.
    if args.stocks:
        if args.train_data is None:
            args.train_data = _STOCK_DEFAULT_TRAIN
        if args.val_data is None:
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
        # Holdout eval window must fit inside val data (stocks12_daily_val has 194 timesteps;
        # window needs steps+1 rows, so cap at 90 to leave plenty of room for 20 windows).
        if args.holdout_eval_steps == 0:
            args.holdout_eval_steps = 90
        # Default leaderboard and checkpoint root for stocks
        if args.leaderboard == "pufferlib_market/autoresearch_leaderboard.csv":
            args.leaderboard = "autoresearch_stock_daily_leaderboard.csv"
        if args.checkpoint_root == "pufferlib_market/checkpoints/autoresearch":
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
        experiment_pool = STOCK_EXPERIMENTS if args.stocks else EXPERIMENTS
        for cfg_dict in experiment_pool:
            desc = cfg_dict.get("description", "")
            gpu = cfg_dict.get("requires_gpu", "")
            gpu_note = f" [requires_gpu={gpu}]" if gpu else ""
            print(f"{desc}{gpu_note}")
        sys.exit(0)

    leaderboard_path = Path(args.leaderboard)
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
        "replay_daily_trade_count",
        "replay_hourly_return_pct", "replay_hourly_sortino", "replay_hourly_max_drawdown_pct",
        "replay_hourly_trade_count", "replay_hourly_order_count",
        "replay_hourly_policy_return_pct", "replay_hourly_policy_sortino",
        "replay_hourly_policy_max_drawdown_pct", "replay_hourly_policy_trade_count",
        "replay_hourly_policy_order_count",
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
        experiments = _select_from_pool(
            STOCK_EXPERIMENTS,
            start_from=args.start_from,
            descriptions=args.descriptions,
        )
        print(f"[stocks mode] {len(experiments)} stock configs, "
              f"train={args.train_data}, val={args.val_data}, "
              f"fee_override={args.fee_rate_override}, "
              f"periods_per_year={args.periods_per_year}, "
              f"max_steps={args.max_steps_override}")
    else:
        experiments = select_experiments(start_from=args.start_from, descriptions=args.descriptions)

    # Add random mutations
    best_rank_score = -float("inf")
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
            config = mutate_config(best_config)
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

        gpu_type = "h100" if (args.h100_mode or config.requires_gpu == "h100") else "a100"

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
            replay_eval_fill_buffer_bps=args.replay_eval_fill_buffer_bps,
            replay_eval_hourly_periods_per_year=args.replay_eval_hourly_periods_per_year,
            replay_eval_timeout_s=args.replay_eval_timeout_seconds,
            rank_metric=args.rank_metric,
            max_timesteps_per_sample=args.max_timesteps_per_sample,
            best_trial_rank_score=best_rank_score,
            early_reject_threshold=args.early_reject_threshold,
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
            "replay_hourly_return_pct": result.get("replay_hourly_return_pct"),
            "replay_hourly_sortino": result.get("replay_hourly_sortino"),
            "replay_hourly_max_drawdown_pct": result.get("replay_hourly_max_drawdown_pct"),
            "replay_hourly_trade_count": result.get("replay_hourly_trade_count"),
            "replay_hourly_order_count": result.get("replay_hourly_order_count"),
            "replay_hourly_policy_return_pct": result.get("replay_hourly_policy_return_pct"),
            "replay_hourly_policy_sortino": result.get("replay_hourly_policy_sortino"),
            "replay_hourly_policy_max_drawdown_pct": result.get("replay_hourly_policy_max_drawdown_pct"),
            "replay_hourly_policy_trade_count": result.get("replay_hourly_policy_trade_count"),
            "replay_hourly_policy_order_count": result.get("replay_hourly_policy_order_count"),
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
        if rank_score is not None and rank_score > best_rank_score:
            best_rank_score = rank_score
            best_config = config
            metric_name = str(result.get("rank_metric", "rank_score"))
            print(f"  *** NEW BEST {metric_name}={rank_score:.4f} ***")

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
