"""
Auto-research loop for PufferLib RL trading.

Runs timeboxed training experiments (default 5 min each), evaluates on
held-out validation data, and tracks results in a leaderboard CSV.

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
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

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


def run_trial(
    config: TrialConfig,
    train_data: str,
    val_data: str,
    time_budget: int,
    checkpoint_dir: str,
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
    ]
    if config.arch == "resmlp":
        eval_cmd.extend(["--arch", "resmlp"])

    try:
        result = subprocess.run(
            eval_cmd, capture_output=True, text=True, timeout=120, cwd=str(REPO),
        )
        eval_output = result.stdout + result.stderr

        # Parse eval results
        val_return = None
        val_wr = None
        val_sortino = None
        val_profitable_pct = None
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

        print(f"  Val: ret={val_return}, sortino={val_sortino}, "
              f"wr={val_wr}, profitable={val_profitable_pct}%")

        return {
            "train_return": train_return,
            "train_sortino": train_sortino,
            "train_wr": train_wr,
            "train_steps": total_steps,
            "val_return": val_return,
            "val_sortino": val_sortino,
            "val_wr": val_wr,
            "val_profitable_pct": val_profitable_pct,
            "elapsed_s": elapsed,
        }

    except subprocess.TimeoutExpired:
        return {
            "error": "eval timeout",
            "train_return": train_return,
            "train_steps": total_steps,
        }
    except Exception as e:
        return {
            "error": f"eval error: {e}",
            "train_return": train_return,
            "train_steps": total_steps,
        }


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
    args = parser.parse_args()

    leaderboard_path = Path(args.leaderboard)
    ckpt_root = Path(args.checkpoint_root)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # Initialize or load leaderboard
    fieldnames = [
        "trial", "description", "val_return", "val_sortino", "val_wr",
        "val_profitable_pct", "train_return", "train_sortino", "train_wr",
        "train_steps", "elapsed_s", "error",
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
    best_val_return = -float("inf")
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

        result = run_trial(config, args.train_data, args.val_data,
                           args.time_budget, ckpt_dir)

        # Update leaderboard
        row = {
            "trial": trial_num,
            "description": desc,
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
        val_ret = result.get("val_return")
        if val_ret is not None and val_ret > best_val_return:
            best_val_return = val_ret
            best_config = config
            print(f"  *** NEW BEST val_return={val_ret:.4f} ***")

        trial_num += 1
        existing_trials.add(desc)

    # Print final leaderboard
    print(f"\n{'='*60}")
    print("LEADERBOARD (sorted by val_return)")
    print(f"{'='*60}")
    if leaderboard_path.exists():
        with open(leaderboard_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        rows_with_val = [r for r in rows if r.get("val_return") and r["val_return"] != "None"]
        rows_with_val.sort(key=lambda r: float(r["val_return"]), reverse=True)
        for r in rows_with_val[:15]:
            print(f"  {r['description']:30s} val_ret={float(r['val_return']):+.4f} "
                  f"val_sortino={r['val_sortino']:>8s} val_wr={r['val_wr']:>6s} "
                  f"train_ret={r['train_return']:>10s} steps={r['train_steps']:>10s}")


if __name__ == "__main__":
    main()
