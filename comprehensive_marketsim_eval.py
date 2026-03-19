#!/usr/bin/env python3
"""Comprehensive market sim dashboard for all daily RL checkpoints.

Auto-discovers ALL daily RL checkpoints across multiple directories, matches
each to the correct validation dataset by obs_size, runs the pure-python
market simulator for multiple evaluation periods, and outputs a unified CSV
sorted by 120d Sortino.
"""

from __future__ import annotations

import argparse
import csv
import collections
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Monkey-patch early exit so we never stop mid-simulation
# ---------------------------------------------------------------------------
import src.market_sim_early_exit as _mse


def _no_early_exit(*args, **kwargs):
    return _mse.EarlyExitDecision(
        should_stop=False,
        progress_fraction=0.0,
        total_return=0.0,
        max_drawdown=0.0,
    )


_mse.evaluate_drawdown_vs_profit_early_exit = _no_early_exit

from pufferlib_market.hourly_replay import MktdData, read_mktd, simulate_daily_policy
from pufferlib_market.metrics import annualize_total_return
from pufferlib_market.evaluate_tail import (
    TradingPolicy,
    ResidualTradingPolicy,
    _infer_num_actions,
    _infer_arch,
    _infer_hidden_size,
    _infer_resmlp_blocks,
    _slice_tail,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIRS = [
    "pufferlib_market/checkpoints/mass_daily",
    "pufferlib_market/checkpoints/autoresearch_daily",
    "pufferlib_market/checkpoints/autoresearch_daily_combos",
    "pufferlib_market/checkpoints/autoresearch_crypto8_daily",
    "pufferlib_market/checkpoints/autoresearch_mixed23_daily",
    "pufferlib_market/checkpoints/autoresearch_mixed32_daily",
    "pufferlib_market/checkpoints/autoresearch_daily_v2",
    "pufferlib_market/checkpoints/autoresearch_fdusd_daily",
    "pufferlib_market/checkpoints/tp_fine",
    "pufferlib_market/checkpoints/mass_daily_v2",
    "pufferlib_market/checkpoints/long_daily",
    "pufferlib_market/checkpoints/daily_crypto5_baseline",
    "pufferlib_market/checkpoints/stocks12_daily_tp05",
    "pufferlib_market/checkpoints/stocks12_daily_tp05_longonly",
    "pufferlib_market/checkpoints/mixed32_daily_ent_anneal",
]

DATA_DIR = "pufferlib_market/data"

# obs_size -> val data filename
OBS_SIZE_TO_VAL_DATA = {
    56: "fdusd3_daily_val.bin",
    90: "crypto5_daily_val.bin",
    141: "crypto8_daily_val.bin",
    175: "crypto10_daily_val.bin",
    192: "crypto11_daily_val.bin",
    209: "stocks12_daily_val.bin",
    260: "crypto15_daily_val.bin",
    396: "mixed23_daily_val.bin",
    549: "mixed32_daily_val.bin",
}

# Market sim settings
FEE_RATE = 0.001
FILL_BUFFER_BPS = 8.0
MAX_LEVERAGE = 1.0
PERIODS_PER_YEAR = 365.0


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def discover_checkpoints(root: Path, checkpoint_dirs: list[str]) -> list[Path]:
    """Find all best.pt files under the given checkpoint directories."""
    found = []
    for rel_dir in checkpoint_dirs:
        d = root / rel_dir
        if not d.is_dir():
            continue
        for dirpath, _, filenames in os.walk(str(d)):
            if "best.pt" in filenames:
                found.append(Path(dirpath) / "best.pt")
    return sorted(set(found))


# ---------------------------------------------------------------------------
# Checkpoint loading & obs_size detection
# ---------------------------------------------------------------------------

@dataclass
class CheckpointInfo:
    path: Path
    state_dict: dict
    obs_size: int
    num_actions: int
    arch: str
    hidden_size: int
    resmlp_blocks: int


def load_checkpoint_info(path: Path, device: torch.device) -> CheckpointInfo:
    """Load a checkpoint and extract architecture metadata."""
    payload = torch.load(str(path), map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        state_dict = payload["model"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError(f"Unsupported checkpoint format in {path}")

    # Detect obs_size
    if "encoder.0.weight" in state_dict:
        obs_size = int(state_dict["encoder.0.weight"].shape[1])
    elif "input_proj.weight" in state_dict:
        obs_size = int(state_dict["input_proj.weight"].shape[1])
    else:
        raise ValueError(f"Cannot detect obs_size from {path}")

    arch = _infer_arch(state_dict)
    hidden_size = _infer_hidden_size(state_dict, arch=arch)
    num_actions = _infer_num_actions(state_dict, fallback=0)
    resmlp_blocks = _infer_resmlp_blocks(state_dict) if arch == "resmlp" else 0

    return CheckpointInfo(
        path=path,
        state_dict=state_dict,
        obs_size=obs_size,
        num_actions=num_actions,
        arch=arch,
        hidden_size=hidden_size,
        resmlp_blocks=resmlp_blocks,
    )


# ---------------------------------------------------------------------------
# Policy construction
# ---------------------------------------------------------------------------

def build_policy(info: CheckpointInfo, device: torch.device) -> nn.Module:
    """Build and load a policy network from checkpoint info."""
    if info.arch == "resmlp":
        policy = ResidualTradingPolicy(
            info.obs_size,
            info.num_actions,
            hidden=info.hidden_size,
            num_blocks=info.resmlp_blocks,
        ).to(device)
    elif info.arch == "mlp":
        policy = TradingPolicy(
            info.obs_size,
            info.num_actions,
            hidden=info.hidden_size,
        ).to(device)
    else:
        raise ValueError(f"Unsupported arch: {info.arch}")
    policy.load_state_dict(info.state_dict)
    policy.eval()
    return policy


def make_policy_fn(policy: nn.Module, device: torch.device) -> Callable[[np.ndarray], int]:
    """Create a deterministic policy function for simulate_daily_policy."""
    def _fn(obs: np.ndarray) -> int:
        obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).to(device=device).view(1, -1)
        with torch.no_grad():
            logits, _ = policy(obs_t)
        return int(torch.argmax(logits, dim=-1).item())
    return _fn


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    checkpoint: str
    universe: str
    period: int
    return_pct: float
    annualized_pct: float
    sortino: float
    max_dd_pct: float
    num_trades: int
    win_rate: float
    avg_hold: float


def evaluate_checkpoint_period(
    policy_fn: Callable[[np.ndarray], int],
    data: MktdData,
    period_days: int,
    ckpt_name: str,
    universe: str,
) -> Optional[EvalResult]:
    """Evaluate a single checkpoint on a single period.

    The caller is responsible for building the policy_fn once and reusing it
    across periods to avoid redundant model construction.
    """
    max_steps = period_days
    # Need max_steps + 1 timesteps (max_steps < num_timesteps)
    if data.num_timesteps < max_steps + 1:
        return None

    try:
        tail = _slice_tail(data, steps=max_steps)
    except ValueError:
        return None

    result = simulate_daily_policy(
        tail,
        policy_fn,
        max_steps=max_steps,
        fee_rate=FEE_RATE,
        fill_buffer_bps=FILL_BUFFER_BPS,
        max_leverage=MAX_LEVERAGE,
        periods_per_year=PERIODS_PER_YEAR,
    )

    annualized = annualize_total_return(
        float(result.total_return),
        periods=float(max_steps),
        periods_per_year=PERIODS_PER_YEAR,
    )

    return EvalResult(
        checkpoint=ckpt_name,
        universe=universe,
        period=period_days,
        return_pct=round(result.total_return * 100.0, 2),
        annualized_pct=round(annualized * 100.0, 2),
        sortino=round(result.sortino, 3),
        max_dd_pct=round(result.max_drawdown * 100.0, 2),
        num_trades=result.num_trades,
        win_rate=round(result.win_rate * 100.0, 1),
        avg_hold=round(result.avg_hold_steps, 1),
    )


def _short_checkpoint_name(path: Path) -> str:
    """Turn a long path into a readable short name like 'mass_daily/tp0.05_s42'."""
    parts = path.parts
    # Find 'checkpoints' in the path
    try:
        idx = list(parts).index("checkpoints")
        # Everything after checkpoints, minus 'best.pt'
        relevant = parts[idx + 1 : -1]
        return "/".join(relevant)
    except ValueError:
        return str(path.parent.name)


def _universe_name(data: MktdData) -> str:
    """Generate a universe description from data symbols."""
    syms = data.symbols
    n = len(syms)
    if n <= 5:
        return "+".join(syms)
    return f"{n}sym"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive market sim eval for all daily RL checkpoints"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root directory (default: cwd)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comprehensive_marketsim_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--periods",
        type=str,
        default="30,60,90,120,180",
        help="Comma-separated evaluation periods in days",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--sort-period",
        type=int,
        default=120,
        help="Period to sort the final table by Sortino (default: 120)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    device = torch.device(args.device)
    periods = [int(p.strip()) for p in args.periods.split(",")]
    sort_period = int(args.sort_period)

    print(f"Root: {root}")
    print(f"Device: {device}")
    print(f"Periods: {periods}")
    print(f"Sort by: {sort_period}d Sortino")
    print()

    # 1. Discover checkpoints
    all_checkpoints = discover_checkpoints(root, CHECKPOINT_DIRS)
    print(f"Discovered {len(all_checkpoints)} checkpoints")

    if len(all_checkpoints) == 0:
        print("No checkpoints found. Check CHECKPOINT_DIRS.")
        sys.exit(1)

    # 2. Load val data files (cache by obs_size)
    data_cache: dict[int, Optional[MktdData]] = {}
    data_dir = root / DATA_DIR

    for obs_size, fname in OBS_SIZE_TO_VAL_DATA.items():
        fpath = data_dir / fname
        if fpath.exists():
            try:
                data_cache[obs_size] = read_mktd(fpath)
                d = data_cache[obs_size]
                print(f"  Loaded {fname}: {d.num_symbols} symbols, {d.num_timesteps} days")
            except Exception as e:
                print(f"  ERROR loading {fname}: {e}")
                data_cache[obs_size] = None
        else:
            print(f"  Missing {fname} (obs_size={obs_size})")
            data_cache[obs_size] = None

    print()

    # 3. Evaluate all checkpoints
    results: list[EvalResult] = []
    skipped = 0
    errors = 0
    t_start = time.time()

    for i, ckpt_path in enumerate(all_checkpoints):
        short_name = _short_checkpoint_name(ckpt_path)
        progress = f"[{i+1}/{len(all_checkpoints)}]"

        try:
            info = load_checkpoint_info(ckpt_path, device)
        except Exception as e:
            print(f"{progress} SKIP {short_name}: load error: {e}")
            errors += 1
            continue

        data = data_cache.get(info.obs_size)
        if data is None:
            print(f"{progress} SKIP {short_name}: no val data for obs_size={info.obs_size}")
            skipped += 1
            continue

        # Check num_actions compatibility
        expected_actions = 1 + 2 * data.num_symbols
        if info.num_actions != expected_actions:
            print(f"{progress} SKIP {short_name}: num_actions={info.num_actions} != expected={expected_actions}")
            skipped += 1
            continue

        # Build policy once, reuse across all periods
        try:
            policy = build_policy(info, device)
            policy_fn = make_policy_fn(policy, device)
        except Exception as e:
            print(f"{progress} SKIP {short_name}: policy build error: {e}")
            errors += 1
            continue

        universe = _universe_name(data)
        period_results = []
        for period in periods:
            try:
                r = evaluate_checkpoint_period(policy_fn, data, period, short_name, universe)
                if r is not None:
                    period_results.append(r)
            except Exception as e:
                print(f"{progress} ERROR {short_name} period={period}: {e}")
                errors += 1

        results.extend(period_results)

        # Print compact progress
        if period_results:
            # Show the sort_period result if available, else last
            show = next((r for r in period_results if r.period == sort_period), period_results[-1])
            print(
                f"{progress} {short_name}: "
                f"{show.period}d ret={show.return_pct:+.1f}% "
                f"sortino={show.sortino:.2f} "
                f"maxDD={show.max_dd_pct:.1f}% "
                f"WR={show.win_rate:.0f}% "
                f"trades={show.num_trades}"
            )
        else:
            print(f"{progress} {short_name}: no valid periods (data too short?)")
            skipped += 1

    elapsed = time.time() - t_start
    print()
    print(f"Evaluation complete in {elapsed:.0f}s")
    print(f"  Results: {len(results)} rows from {len(all_checkpoints)} checkpoints")
    print(f"  Skipped: {skipped}, Errors: {errors}")

    if not results:
        print("No results to write.")
        sys.exit(1)

    # 4. Write CSV
    output_path = Path(args.output)
    fieldnames = [
        "checkpoint", "universe", "period", "return_pct", "annualized_pct",
        "sortino", "max_dd_pct", "num_trades", "win_rate", "avg_hold",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "checkpoint": r.checkpoint,
                "universe": r.universe,
                "period": r.period,
                "return_pct": r.return_pct,
                "annualized_pct": r.annualized_pct,
                "sortino": r.sortino,
                "max_dd_pct": r.max_dd_pct,
                "num_trades": r.num_trades,
                "win_rate": r.win_rate,
                "avg_hold": r.avg_hold,
            })
    print(f"\nCSV written to: {output_path}")

    # 5. Print formatted table sorted by sort_period Sortino
    print()
    print(f"{'='*120}")
    print(f"  LEADERBOARD (sorted by {sort_period}d Sortino)")
    print(f"{'='*120}")

    # Filter to the sort_period, then sort
    period_rows = [r for r in results if r.period == sort_period]
    if not period_rows:
        # Fall back to longest available period
        available_periods = sorted(set(r.period for r in results), reverse=True)
        if available_periods:
            fallback = available_periods[0]
            period_rows = [r for r in results if r.period == fallback]
            print(f"  (No {sort_period}d results; showing {fallback}d instead)")

    period_rows.sort(key=lambda r: r.sortino, reverse=True)

    # Header
    fmt = "{rank:>4}  {name:<50} {universe:<12} {period:>4}  {ret:>8}  {ann:>8}  {sortino:>8}  {dd:>7}  {trades:>6}  {wr:>5}  {hold:>5}"
    print(fmt.format(
        rank="Rank",
        name="Checkpoint",
        universe="Universe",
        period="Days",
        ret="Ret%",
        ann="Ann%",
        sortino="Sortino",
        dd="MaxDD%",
        trades="Trades",
        wr="WR%",
        hold="Hold",
    ))
    print("-" * 120)

    for rank, r in enumerate(period_rows, 1):
        print(fmt.format(
            rank=rank,
            name=r.checkpoint[:50],
            universe=r.universe[:12],
            period=r.period,
            ret=f"{r.return_pct:+.1f}",
            ann=f"{r.annualized_pct:+.1f}",
            sortino=f"{r.sortino:.3f}",
            dd=f"{r.max_dd_pct:.1f}",
            trades=r.num_trades,
            wr=f"{r.win_rate:.0f}",
            hold=f"{r.avg_hold:.1f}",
        ))

    print(f"\nTotal: {len(period_rows)} checkpoints evaluated at {sort_period}d")

    # Also print a multi-period summary for the top 10
    if len(period_rows) >= 1:
        print()
        print(f"{'='*120}")
        print(f"  TOP 10 - Multi-Period Summary")
        print(f"{'='*120}")

        top_names = [r.checkpoint for r in period_rows[:10]]
        results_by_name = collections.defaultdict(dict)
        for r in results:
            results_by_name[r.checkpoint][r.period] = r

        hdr_periods = sorted(periods)
        hdr = f"{'Checkpoint':<50}"
        for p in hdr_periods:
            hdr += f"  {p}d Ret%  {p}d Sort"
        print(hdr)
        print("-" * (50 + len(hdr_periods) * 20))

        for name in top_names:
            line = f"{name[:50]:<50}"
            for p in hdr_periods:
                r = results_by_name[name].get(p)
                if r:
                    line += f"  {r.return_pct:>+7.1f}  {r.sortino:>8.3f}"
                else:
                    line += f"  {'N/A':>7}  {'N/A':>8}"
            print(line)


if __name__ == "__main__":
    main()
