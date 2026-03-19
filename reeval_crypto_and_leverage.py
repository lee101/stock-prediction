#!/usr/bin/env python3
"""Re-evaluate crypto5 daily checkpoints with periods_per_year=365 and run leverage sanity check.

Part 1: Re-eval existing crypto5 daily checkpoints
  - Discovers best.pt under mass_daily/, autoresearch_daily/, autoresearch_daily_combos/
  - Filters by obs_size == 90 (crypto5)
  - Evaluates on crypto5_daily_val.bin with 120d period, periods_per_year=365

Part 2: Leverage sanity check
  - Trains 6 configs: tp=0.15 x leverage(1x,2x,3x) x seeds(42,314) for 300s each
  - Evaluates all 6 on 60d/90d/120d and prints comparison table

Usage:
    # Full run (re-eval + leverage training)
    python reeval_crypto_and_leverage.py

    # Re-eval only (no training)
    python reeval_crypto_and_leverage.py --reeval-only

    # Leverage only
    python reeval_crypto_and_leverage.py --leverage-only
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

# Disable early exit so we get full-period results
import src.market_sim_early_exit as _mse


def _no_early_exit(*args, **kwargs):
    return _mse.EarlyExitDecision(
        should_stop=False,
        progress_fraction=0.0,
        total_return=0.0,
        max_drawdown=0.0,
    )


_mse.evaluate_drawdown_vs_profit_early_exit = _no_early_exit

from pufferlib_market.hourly_replay import read_mktd, simulate_daily_policy, MktdData
from pufferlib_market.metrics import annualize_total_return
from pufferlib_market.evaluate_multiperiod import load_policy, make_policy_fn
from pufferlib_market.evaluate_tail import _slice_tail

REPO = Path(__file__).resolve().parent
CKPT_BASE = REPO / "pufferlib_market" / "checkpoints"
DATA_PATH = REPO / "pufferlib_market" / "data" / "crypto5_daily_val.bin"
TRAIN_DATA = REPO / "pufferlib_market" / "data" / "crypto5_daily_train.bin"

CRYPTO5_OBS_SIZE = 90  # 5 symbols * 16 features + 5 + 5
FEE_RATE = 0.001
FILL_BUFFER_BPS = 8.0
MAX_LEVERAGE = 1.0
PERIODS_PER_YEAR = 365.0


def discover_checkpoints() -> dict[str, Path]:
    """Find all best.pt files under the three checkpoint directories."""
    dirs = [
        CKPT_BASE / "mass_daily",
        CKPT_BASE / "autoresearch_daily",
        CKPT_BASE / "autoresearch_daily_combos",
    ]
    found = {}
    for d in dirs:
        if not d.exists():
            continue
        for best_pt in sorted(d.rglob("best.pt")):
            rel = best_pt.relative_to(CKPT_BASE)
            name = str(rel.parent)
            found[name] = best_pt
    return found


def get_obs_size_from_checkpoint(path: Path) -> int:
    """Infer obs_size from the first encoder layer's input dimension."""
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        # MLP arch: encoder.0.weight has shape [hidden, obs_size]
        if "encoder.0.weight" in state_dict:
            return int(state_dict["encoder.0.weight"].shape[1])
        # ResidualMLP arch: input_proj.weight has shape [hidden, obs_size]
        if "input_proj.weight" in state_dict:
            return int(state_dict["input_proj.weight"].shape[1])
    except Exception as e:
        print(f"  WARNING: could not read {path}: {e}")
    return -1


def _sim_and_summarize(
    policy_fn,
    data: MktdData,
    max_steps: int,
    *,
    max_leverage: float = MAX_LEVERAGE,
    include_avg_hold: bool = False,
) -> dict:
    """Run simulate_daily_policy on a tail slice and return a summary dict."""
    if data.num_timesteps < max_steps + 1:
        return {"error": f"data too short ({data.num_timesteps} < {max_steps + 1})"}

    tail = _slice_tail(data, steps=max_steps)
    sim = simulate_daily_policy(
        tail,
        policy_fn,
        max_steps=max_steps,
        fee_rate=FEE_RATE,
        fill_buffer_bps=FILL_BUFFER_BPS,
        max_leverage=max_leverage,
        periods_per_year=PERIODS_PER_YEAR,
    )
    ann = annualize_total_return(
        float(sim.total_return),
        periods=float(max_steps),
        periods_per_year=PERIODS_PER_YEAR,
    )
    result = {
        "return_pct": sim.total_return * 100,
        "annualized_pct": ann * 100,
        "sortino": sim.sortino,
        "max_dd_pct": sim.max_drawdown * 100,
        "trades": sim.num_trades,
        "win_rate": sim.win_rate,
    }
    if include_avg_hold:
        result["avg_hold"] = sim.avg_hold_steps
    return result


def _load_policy_fn(ckpt_path: Path, nsym: int, device: torch.device):
    """Load checkpoint and return (policy_fn, None) or (None, error_dict)."""
    try:
        policy, _, _ = load_policy(str(ckpt_path), nsym, device=device)
    except Exception as e:
        return None, {"error": str(e)}
    policy_fn = make_policy_fn(
        policy,
        num_symbols=nsym,
        deterministic=True,
        device=device,
    )
    return policy_fn, None


def eval_checkpoint(
    ckpt_path: Path,
    data: MktdData,
    max_steps: int,
    *,
    max_leverage: float = MAX_LEVERAGE,
    device: str = "cpu",
) -> dict:
    """Evaluate a single checkpoint on the given data slice."""
    dev = torch.device(device)
    policy_fn, err = _load_policy_fn(ckpt_path, data.num_symbols, dev)
    if err is not None:
        return err

    try:
        return _sim_and_summarize(
            policy_fn, data, max_steps,
            max_leverage=max_leverage, include_avg_hold=True,
        )
    except Exception as e:
        return {"error": str(e)}


# ─── Part 1: Re-eval ────────────────────────────────────────────────


def run_reeval(device: str = "cpu") -> list[dict]:
    """Discover and re-evaluate all crypto5 daily checkpoints."""
    if not DATA_PATH.exists():
        print(f"ERROR: Validation data not found at {DATA_PATH}")
        return []

    data = read_mktd(DATA_PATH)
    print(f"Loaded {DATA_PATH.name}: {data.num_timesteps} days, {data.num_symbols} symbols")
    print(f"Settings: {FILL_BUFFER_BPS}bps fill, {FEE_RATE*100:.1f}% fee, "
          f"periods_per_year={PERIODS_PER_YEAR}, deterministic\n")

    all_ckpts = discover_checkpoints()
    print(f"Found {len(all_ckpts)} checkpoints total.")

    # Filter to crypto5 (obs_size=90)
    crypto5_ckpts = {}
    for name, path in sorted(all_ckpts.items()):
        obs_sz = get_obs_size_from_checkpoint(path)
        if obs_sz == CRYPTO5_OBS_SIZE:
            crypto5_ckpts[name] = path

    print(f"Filtered to {len(crypto5_ckpts)} crypto5 checkpoints (obs_size={CRYPTO5_OBS_SIZE}).\n")

    if not crypto5_ckpts:
        print("No crypto5 checkpoints found.")
        return []

    max_steps = 120  # 120d evaluation
    results = []

    header = (
        f"{'Name':<40} {'Ret%':>8} {'Ann%':>8} {'Sortino':>8} "
        f"{'MaxDD%':>8} {'Trades':>7} {'WR':>6} {'Hold':>6}"
    )
    print(header)
    print("-" * len(header))

    for name, ckpt_path in sorted(crypto5_ckpts.items()):
        r = eval_checkpoint(ckpt_path, data, max_steps, device=device)
        row = {"name": name, "checkpoint": str(ckpt_path), **r}
        results.append(row)

        if "error" in r:
            print(f"{name:<40} ERROR: {r['error']}")
        else:
            print(
                f"{name:<40} "
                f"{r['return_pct']:>+7.1f}% "
                f"{r['annualized_pct']:>+7.1f}% "
                f"{r['sortino']:>8.2f} "
                f"{r['max_dd_pct']:>7.1f}% "
                f"{r['trades']:>7} "
                f"{r['win_rate']:>5.1%} "
                f"{r['avg_hold']:>6.1f}"
            )

    # Sort by return and print top 10
    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda x: x.get("return_pct", -999), reverse=True)
    print(f"\n{'='*60}")
    print(f"TOP 10 BY RETURN (120d, periods_per_year={PERIODS_PER_YEAR})")
    print(f"{'='*60}")
    for i, r in enumerate(valid[:10]):
        print(
            f"  {i+1:2d}. {r['name']:<35} "
            f"ret={r['return_pct']:>+.1f}% "
            f"ann={r['annualized_pct']:>+.1f}% "
            f"sortino={r['sortino']:.2f} "
            f"dd={r['max_dd_pct']:.1f}%"
        )

    # Save CSV
    csv_path = REPO / "crypto5_daily_reeval_365.csv"
    if valid:
        fields = list(valid[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in valid:
                writer.writerow(r)
        print(f"\nSaved re-eval results to {csv_path}")

    return results


# ─── Part 2: Leverage Sanity Check ──────────────────────────────────


@dataclass
class LeverageTrialConfig:
    leverage: float
    seed: int
    trade_penalty: float = 0.15
    time_budget_s: int = 300


LEVERAGE_CONFIGS = [
    LeverageTrialConfig(leverage=1.0, seed=42),
    LeverageTrialConfig(leverage=1.0, seed=314),
    LeverageTrialConfig(leverage=2.0, seed=42),
    LeverageTrialConfig(leverage=2.0, seed=314),
    LeverageTrialConfig(leverage=3.0, seed=42),
    LeverageTrialConfig(leverage=3.0, seed=314),
]

EVAL_PERIODS = {"60d": 60, "90d": 90, "120d": 120}


def train_leverage_config(
    config: LeverageTrialConfig,
    checkpoint_dir: str,
) -> dict:
    """Train a single leverage config and return training stats."""
    cmd = [
        sys.executable, "-u", "-m", "pufferlib_market.train",
        "--data-path", str(TRAIN_DATA),
        "--total-timesteps", "999999999",
        "--max-steps", "720",
        "--hidden-size", "1024",
        "--lr", "3e-4",
        "--ent-coef", "0.05",
        "--gamma", "0.99",
        "--gae-lambda", "0.95",
        "--num-envs", "128",
        "--rollout-len", "256",
        "--ppo-epochs", "4",
        "--seed", str(config.seed),
        "--reward-scale", "10.0",
        "--reward-clip", "5.0",
        "--cash-penalty", "0.01",
        "--fee-rate", "0.001",
        "--trade-penalty", str(config.trade_penalty),
        "--anneal-lr",
        "--max-leverage", str(config.leverage),
        "--checkpoint-dir", checkpoint_dir,
        "--periods-per-year", str(PERIODS_PER_YEAR),
    ]

    print(f"  Training lev{config.leverage}x_s{config.seed} for {config.time_budget_s}s...")
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(REPO),
            preexec_fn=os.setsid,
        )
        stdout_lines = []
        while time.time() - t0 < config.time_budget_s:
            if proc.poll() is not None:
                break
            try:
                line = proc.stdout.readline()
                if line:
                    decoded = line.decode("utf-8", errors="replace").strip()
                    stdout_lines.append(decoded)
            except Exception:
                pass
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=5)
        elapsed = time.time() - t0

        train_return = None
        total_steps = 0
        for line in reversed(stdout_lines):
            if "ret=" in line and train_return is None:
                try:
                    for part in line.split():
                        if part.startswith("ret="):
                            train_return = float(part.split("=")[1])
                        elif part.startswith("step="):
                            total_steps = int(part.split("=")[1].replace(",", ""))
                except Exception:
                    pass
                if train_return is not None:
                    break

        print(f"  Training done: {elapsed:.0f}s, {total_steps:,} steps, "
              f"train_ret={train_return}")
        return {
            "train_return": train_return,
            "train_steps": total_steps,
            "elapsed_s": elapsed,
        }
    except Exception as e:
        return {"error": str(e)}


def eval_leverage_checkpoint(
    ckpt_path: Path,
    data: MktdData,
    leverage: float,
    periods: dict[str, int],
    device: str = "cpu",
) -> dict[str, dict]:
    """Evaluate a leverage checkpoint across multiple periods."""
    dev = torch.device(device)
    policy_fn, err = _load_policy_fn(ckpt_path, data.num_symbols, dev)
    if err is not None:
        return {p: dict(err) for p in periods}

    results = {}
    for pname, steps in periods.items():
        try:
            results[pname] = _sim_and_summarize(
                policy_fn, data, steps, max_leverage=leverage,
            )
        except Exception as e:
            results[pname] = {"error": str(e)}
    return results


def run_leverage_sanity(
    time_budget: int = 300,
    device: str = "cpu",
) -> list[dict]:
    """Train and evaluate leverage configs to confirm 1x >> 2x >> 3x."""
    if not TRAIN_DATA.exists():
        print(f"ERROR: Training data not found at {TRAIN_DATA}")
        return []
    if not DATA_PATH.exists():
        print(f"ERROR: Validation data not found at {DATA_PATH}")
        return []

    data = read_mktd(DATA_PATH)
    print(f"\nLoaded val data: {data.num_timesteps} days, {data.num_symbols} symbols")

    ckpt_root = REPO / "pufferlib_market" / "checkpoints" / "crypto_leverage"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    all_results = []

    for base_config in LEVERAGE_CONFIGS:
        config = LeverageTrialConfig(
            leverage=base_config.leverage,
            seed=base_config.seed,
            trade_penalty=base_config.trade_penalty,
            time_budget_s=time_budget,
        )
        name = f"lev{config.leverage:.0f}x_tp0.15_s{config.seed}"
        ckpt_dir = str(ckpt_root / name)
        os.makedirs(ckpt_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        # Check if already trained
        best_pt = Path(ckpt_dir) / "best.pt"
        if not best_pt.exists():
            train_result = train_leverage_config(config, ckpt_dir)
            if "error" in train_result:
                print(f"  TRAIN ERROR: {train_result['error']}")
                all_results.append({"name": name, "error": train_result["error"]})
                continue
        else:
            print(f"  Using existing checkpoint: {best_pt}")
            train_result = {}

        # Find the checkpoint
        if not best_pt.exists():
            final_pt = Path(ckpt_dir) / "final.pt"
            if final_pt.exists():
                best_pt = final_pt
            else:
                pts = list(Path(ckpt_dir).glob("*.pt"))
                if pts:
                    best_pt = max(pts, key=lambda p: p.stat().st_mtime)
                else:
                    print(f"  NO CHECKPOINT FOUND in {ckpt_dir}")
                    all_results.append({"name": name, "error": "no checkpoint"})
                    continue

        print(f"  Evaluating {best_pt.name}...")
        eval_results = eval_leverage_checkpoint(
            best_pt, data, config.leverage, EVAL_PERIODS, device=device,
        )

        row = {
            "name": name,
            "leverage": config.leverage,
            "seed": config.seed,
            **train_result,
        }
        for pname, r in eval_results.items():
            if "error" in r:
                row[f"{pname}_error"] = r["error"]
            else:
                for k, v in r.items():
                    row[f"{pname}_{k}"] = v

        all_results.append(row)

        # Print per-period results
        for pname in EVAL_PERIODS:
            r = eval_results.get(pname, {})
            if "error" in r:
                print(f"    {pname}: ERROR {r['error']}")
            else:
                print(
                    f"    {pname}: ret={r['return_pct']:>+.1f}% "
                    f"ann={r['annualized_pct']:>+.1f}% "
                    f"sortino={r['sortino']:.2f} "
                    f"dd={r['max_dd_pct']:.1f}%"
                )

    # Print comparison table
    print(f"\n{'='*70}")
    print("LEVERAGE COMPARISON TABLE (crypto5 daily, periods_per_year=365)")
    print(f"{'='*70}")

    header = (
        f"{'Config':<25} "
        f"{'60d Ret%':>9} {'60d Sort':>9} {'60d DD%':>8} "
        f"{'90d Ret%':>9} {'90d Sort':>9} {'90d DD%':>8} "
        f"{'120d Ret%':>9} {'120d Sort':>9} {'120d DD%':>8}"
    )
    print(header)
    print("-" * len(header))

    for r in all_results:
        if "error" in r:
            print(f"{r['name']:<25} ERROR: {r['error']}")
            continue

        parts = [f"{r['name']:<25}"]
        for pname in ["60d", "90d", "120d"]:
            ret = r.get(f"{pname}_return_pct")
            sort = r.get(f"{pname}_sortino")
            dd = r.get(f"{pname}_max_dd_pct")
            if ret is not None:
                parts.append(f"{ret:>+8.1f}%")
                parts.append(f"{sort:>9.2f}")
                parts.append(f"{dd:>7.1f}%")
            else:
                parts.append(f"{'err':>9}")
                parts.append(f"{'':>9}")
                parts.append(f"{'':>8}")
        print(" ".join(parts))

    # Print per-leverage summary (averaging across seeds)
    print(f"\n{'='*60}")
    print("PER-LEVERAGE AVERAGES")
    print(f"{'='*60}")
    for lev in [1.0, 2.0, 3.0]:
        lev_rows = [r for r in all_results if r.get("leverage") == lev and "error" not in r]
        if not lev_rows:
            continue
        print(f"\n  {lev:.0f}x leverage ({len(lev_rows)} seeds):")
        for pname in ["60d", "90d", "120d"]:
            rets = [r[f"{pname}_return_pct"] for r in lev_rows if f"{pname}_return_pct" in r]
            sorts = [r[f"{pname}_sortino"] for r in lev_rows if f"{pname}_sortino" in r]
            dds = [r[f"{pname}_max_dd_pct"] for r in lev_rows if f"{pname}_max_dd_pct" in r]
            if rets:
                print(
                    f"    {pname}: ret={np.mean(rets):>+.1f}% "
                    f"sortino={np.mean(sorts):.2f} "
                    f"dd={np.mean(dds):.1f}%"
                )

    # Save CSV
    csv_path = REPO / "crypto5_leverage_sanity.csv"
    if all_results:
        fields = sorted(set().union(*(r.keys() for r in all_results)))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)
        print(f"\nSaved leverage results to {csv_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Re-eval crypto5 daily with periods_per_year=365 + leverage sanity check"
    )
    parser.add_argument("--reeval-only", action="store_true", help="Only run re-eval, skip leverage training")
    parser.add_argument("--leverage-only", action="store_true", help="Only run leverage sanity check")
    parser.add_argument("--time-budget", type=int, default=300, help="Training time per config (seconds)")
    parser.add_argument("--device", default="cpu", help="Device for inference (cpu or cuda)")
    args = parser.parse_args()

    if args.leverage_only:
        run_leverage_sanity(time_budget=args.time_budget, device=args.device)
        return

    if args.reeval_only:
        run_reeval(device=args.device)
        return

    # Full run: both parts
    print("=" * 70)
    print("PART 1: RE-EVAL ALL CRYPTO5 DAILY CHECKPOINTS (periods_per_year=365)")
    print("=" * 70)
    run_reeval(device=args.device)

    print("\n\n")
    print("=" * 70)
    print("PART 2: LEVERAGE SANITY CHECK (1x vs 2x vs 3x)")
    print("=" * 70)
    run_leverage_sanity(time_budget=args.time_budget, device=args.device)


if __name__ == "__main__":
    main()
