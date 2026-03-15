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

REPO = Path(__file__).resolve().parent.parent


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
    num_envs: int = 128
    rollout_len: int = 256
    ppo_epochs: int = 4
    reward_scale: float = 10.0
    reward_clip: float = 5.0
    cash_penalty: float = 0.01
    fill_slippage_bps: float = 0.0
    fee_rate: float = 0.001
    trade_penalty: float = 0.0
    downside_penalty: float = 0.0
    smooth_downside_penalty: float = 0.0
    arch: str = "mlp"
    max_steps: int = 720
    periods_per_year: float = 8760.0
    seed: int = 42
    description: str = ""


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
]


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
        "reward_scale": [5.0, 10.0, 20.0],
        "cash_penalty": [0.0, 0.005, 0.01, 0.02],
        "trade_penalty": [0.0, 0.01, 0.02, 0.05],
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


def select_rank_score(
    metrics: dict[str, object],
    *,
    rank_metric: str = "auto",
) -> tuple[str, float | None]:
    """Choose the leaderboard ranking signal with sensible fallbacks."""
    candidates = {
        "market_goodness_score": _safe_float(metrics.get("market_goodness_score")),
        "holdout_robust_score": _safe_float(metrics.get("holdout_robust_score")),
        "val_return": _safe_float(metrics.get("val_return")),
    }
    if rank_metric == "auto":
        for name in ("market_goodness_score", "holdout_robust_score", "val_return"):
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


def run_trial(
    config: TrialConfig,
    train_data: str,
    val_data: str,
    time_budget: int,
    checkpoint_dir: str,
    *,
    holdout_data: str | None = None,
    holdout_eval_steps: int = 0,
    holdout_n_windows: int = 0,
    holdout_seed: int = 1337,
    holdout_end_within_steps: int = 0,
    holdout_fee_rate: float = -1.0,
    holdout_max_leverage: float = 1.0,
    holdout_short_borrow_apr: float = 0.0,
    eval_timeout_s: int = 0,
    holdout_timeout_s: int = 0,
    market_validation_asset_class: str = "",
    market_validation_days: int = 30,
    market_validation_cash: float = 10_000.0,
    market_validation_symbols: str | None = None,
    market_validation_timeout_s: int = 0,
    rank_metric: str = "auto",
) -> dict:
    """Run a single training trial with time budget, then evaluate on val."""
    # Build training command
    cmd = [
        sys.executable, "-u", "-m", "pufferlib_market.train",
        "--data-path", train_data,
        "--total-timesteps", "999999999",  # will be killed by timeout
        "--max-steps", str(config.max_steps),
        "--hidden-size", str(config.hidden_size),
        "--lr", str(config.lr),
        "--ent-coef", str(config.ent_coef),
        "--gamma", str(config.gamma),
        "--gae-lambda", str(config.gae_lambda),
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
        "--downside-penalty", str(config.downside_penalty),
        "--smooth-downside-penalty", str(config.smooth_downside_penalty),
        "--weight-decay", str(config.weight_decay),
        "--checkpoint-dir", checkpoint_dir,
        "--arch", config.arch,
        "--periods-per-year", str(config.periods_per_year),
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

    # Run training with time budget
    print(f"\n  Training for {time_budget}s...")
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(REPO), preexec_fn=os.setsid,
        )
        stdout_lines = []
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
            # Kill if still running
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        elapsed = time.time() - t0

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

    print(f"  Training done: {elapsed:.0f}s, {total_steps:,} steps, "
          f"ret={train_return}, sortino={train_sortino}, wr={train_wr}")

    # Check if checkpoint exists
    ckpt_path = Path(checkpoint_dir) / "best.pt"
    if not ckpt_path.exists():
        # Try final.pt
        ckpt_path = Path(checkpoint_dir) / "final.pt"
    if not ckpt_path.exists():
        # Find any .pt file
        pts = list(Path(checkpoint_dir).glob("*.pt"))
        if pts:
            ckpt_path = max(pts, key=lambda p: p.stat().st_mtime)
        else:
            return {
                "error": "no checkpoint",
                "train_return": train_return,
                "train_steps": total_steps,
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
    if config.arch == "resmlp":
        eval_cmd.extend(["--arch", "resmlp"])
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
            f"days={market_validation_days}"
        )
        market_json_path = Path(checkpoint_dir) / "market_validation.json"
        market_cmd = [
            sys.executable, "-u", "-m", "unified_orchestrator.market_validation",
            "--asset-class", market_validation_asset_class,
            "--days", str(market_validation_days),
            "--cash", str(market_validation_cash),
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
    }
    result_payload.update(holdout_metrics)
    result_payload.update(market_metrics)
    selected_metric, rank_score = select_rank_score(result_payload, rank_metric=rank_metric)
    result_payload["rank_metric"] = selected_metric
    result_payload["rank_score"] = rank_score
    return result_payload


def main():
    parser = argparse.ArgumentParser(description="Auto-research RL trading configs")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--time-budget", type=int, default=300,
                        help="Training time budget per trial in seconds")
    parser.add_argument("--max-trials", type=int, default=50)
    parser.add_argument("--leaderboard", default="pufferlib_market/autoresearch_leaderboard.csv")
    parser.add_argument("--checkpoint-root", default="pufferlib_market/checkpoints/autoresearch")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Skip first N experiments")
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
    parser.add_argument("--holdout-max-leverage", type=float, default=1.0)
    parser.add_argument("--holdout-short-borrow-apr", type=float, default=0.0)
    parser.add_argument("--rank-metric",
                        choices=["auto", "val_return", "holdout_robust_score", "market_goodness_score"],
                        default="auto")
    parser.add_argument("--market-validation-asset-class", choices=["", "crypto", "stock"], default="",
                        help="Run unified_orchestrator.market_validation for each checkpoint when set")
    parser.add_argument("--market-validation-days", type=int, default=30)
    parser.add_argument("--market-validation-cash", type=float, default=10_000.0)
    parser.add_argument("--market-validation-symbols", default=None,
                        help="Comma-separated symbols override for market validation")
    parser.add_argument("--eval-timeout-seconds", type=int, default=0,
                        help="Optional timeout for the base validation subprocess; 0 disables it")
    parser.add_argument("--holdout-timeout-seconds", type=int, default=0,
                        help="Optional timeout for holdout evaluation; 0 disables it")
    parser.add_argument("--market-validation-timeout-seconds", type=int, default=0,
                        help="Optional timeout for market validation; 0 disables it")
    args = parser.parse_args()

    leaderboard_path = Path(args.leaderboard)
    ckpt_root = Path(args.checkpoint_root)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # Initialize or load leaderboard
    fieldnames = [
        "trial", "description", "rank_metric", "rank_score", "val_return", "val_sortino", "val_wr",
        "val_profitable_pct", "train_return", "train_sortino", "train_wr",
        "train_steps", "elapsed_s", "error", "holdout_error", "market_validation_error",
        "holdout_robust_score", "holdout_return_mean_pct", "holdout_return_p25_pct",
        "holdout_return_worst_pct", "holdout_sortino_p25", "holdout_max_drawdown_worst_pct",
        "holdout_negative_return_rate", "holdout_median_return_pct", "holdout_p10_return_pct",
        "holdout_median_sortino", "holdout_p90_max_drawdown_pct",
        "market_return_pct", "market_sortino", "market_max_drawdown_pct",
        "market_trade_count", "market_goodness_score",
        "hidden_size", "lr", "ent_coef", "weight_decay", "fill_slippage_bps",
        "obs_norm", "anneal_lr", "anneal_ent", "anneal_clip", "lr_schedule",
        "arch", "fee_rate", "trade_penalty", "gamma",
    ]

    existing_trials = set()
    if leaderboard_path.exists():
        with open(leaderboard_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_trials.add(row.get("description", ""))

    experiments = EXPERIMENTS[args.start_from:]

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
            holdout_data=args.holdout_data,
            holdout_eval_steps=holdout_eval_steps,
            holdout_n_windows=args.holdout_n_windows,
            holdout_seed=args.holdout_seed,
            holdout_end_within_steps=args.holdout_end_within_steps,
            holdout_fee_rate=args.holdout_fee_rate,
            holdout_max_leverage=args.holdout_max_leverage,
            holdout_short_borrow_apr=args.holdout_short_borrow_apr,
            eval_timeout_s=args.eval_timeout_seconds,
            holdout_timeout_s=args.holdout_timeout_seconds,
            market_validation_asset_class=args.market_validation_asset_class,
            market_validation_days=args.market_validation_days,
            market_validation_cash=args.market_validation_cash,
            market_validation_symbols=args.market_validation_symbols,
            market_validation_timeout_s=args.market_validation_timeout_seconds,
            rank_metric=args.rank_metric,
        )

        # Update leaderboard
        row = {
            "trial": trial_num,
            "description": desc,
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
