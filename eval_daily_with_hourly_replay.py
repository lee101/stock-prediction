#!/usr/bin/env python3
"""Evaluate top daily RL checkpoints with hourly replay for more accurate risk metrics.

Runs each checkpoint through:
  1. Daily market sim  -> daily Sortino, daily max drawdown, daily return
  2. Hourly replay     -> hourly Sortino, hourly max drawdown, hourly return

Compares the two to see if intraday price swings materially change risk metrics.

Usage:
    source .venv313/bin/activate
    python eval_daily_with_hourly_replay.py
    python eval_daily_with_hourly_replay.py --checkpoints mass_daily/tp0.15_s314/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- Monkey-patch early exit BEFORE importing anything that uses it ---
import src.market_sim_early_exit as _mse


def _no_early_exit(*args, **kwargs):
    return _mse.EarlyExitDecision(
        should_stop=False,
        progress_fraction=0.0,
        total_return=0.0,
        max_drawdown=0.0,
    )


_mse.evaluate_drawdown_vs_profit_early_exit = _no_early_exit
_mse.evaluate_metric_threshold_early_exit = _no_early_exit

import numpy as np
import pandas as pd
import torch

from pufferlib_market.evaluate_multiperiod import load_policy, make_policy_fn
from pufferlib_market.evaluate_tail import _slice_tail
from pufferlib_market.hourly_replay import (
    load_hourly_market,
    read_mktd,
    replay_hourly_frozen_daily_actions,
    simulate_daily_policy,
)
from pufferlib_market.metrics import annualize_total_return

# --- Defaults ---
REPO = Path(__file__).resolve().parent

# Data and checkpoints may live in the main repo when running from a worktree.
# Try worktree-local first, fall back to the main repo root.
_MAIN_REPO = Path("/nvme0n1-disk/code/stock-prediction")


def _resolve_path(*candidates: Path) -> Path:
    """Return the first path that exists, or the last one as fallback."""
    for p in candidates:
        if p.exists():
            return p
    return candidates[-1]


CKPT_BASE = _resolve_path(
    REPO / "pufferlib_market" / "checkpoints",
    _MAIN_REPO / "pufferlib_market" / "checkpoints",
)
DAILY_DATA = _resolve_path(
    REPO / "pufferlib_market" / "data" / "crypto5_daily_val.bin",
    _MAIN_REPO / "pufferlib_market" / "data" / "crypto5_daily_val.bin",
)
HOURLY_ROOT = _resolve_path(
    REPO / "trainingdatahourly",
    _MAIN_REPO / "trainingdatahourly",
)

# Date range for crypto5_daily_val.bin (286 timesteps = 2025-06-01 to 2026-03-13)
START_DATE = "2025-06-01"
END_DATE = "2026-03-13"

FEE_RATE = 0.001
FILL_BUFFER_BPS = 8.0
MAX_LEVERAGE = 1.0
DAILY_PERIODS_PER_YEAR = 365.0
HOURLY_PERIODS_PER_YEAR = 8760.0

DEFAULT_CHECKPOINTS = [
    "mass_daily/tp0.15_s314/best.pt",
    "mass_daily/tp0.10_s42/best.pt",
    "mass_daily/tp0.20_s123/best.pt",
    "mass_daily/tp0.05_s123/best.pt",
    "mass_daily/tp0.20_s2024/best.pt",
]


def _ckpt_short_name(ckpt_path: str) -> str:
    """Extract a short display name like 'mass_daily/tp0.15_s314' from a full path."""
    p = Path(ckpt_path)
    return p.parent.parent.name + "/" + p.parent.name


def evaluate_checkpoint_daily_and_hourly(
    policy_fn,
    data,
    market,
    *,
    ckpt_path: str,
    max_steps: int,
    start_date: str,
    end_date: str,
) -> dict:
    """Run daily sim and hourly replay for a single checkpoint.

    policy_fn is already constructed -- the caller loads the policy once per checkpoint.
    """
    # --- Daily sim ---
    daily_result = simulate_daily_policy(
        data,
        policy_fn,
        max_steps=max_steps,
        fee_rate=FEE_RATE,
        fill_buffer_bps=FILL_BUFFER_BPS,
        max_leverage=MAX_LEVERAGE,
        periods_per_year=DAILY_PERIODS_PER_YEAR,
    )

    daily_ann = annualize_total_return(
        daily_result.total_return,
        periods=float(max_steps),
        periods_per_year=DAILY_PERIODS_PER_YEAR,
    )

    # --- Hourly replay ---
    hourly_result = replay_hourly_frozen_daily_actions(
        data=data,
        actions=daily_result.actions,
        market=market,
        start_date=start_date,
        end_date=end_date,
        max_steps=max_steps,
        fee_rate=FEE_RATE,
        max_leverage=MAX_LEVERAGE,
        periods_per_year=HOURLY_PERIODS_PER_YEAR,
    )

    hourly_ann = annualize_total_return(
        hourly_result.total_return,
        periods=float(max_steps),
        periods_per_year=DAILY_PERIODS_PER_YEAR,
    )

    return {
        "checkpoint": ckpt_path,
        "max_steps": max_steps,
        "daily": {
            "total_return": daily_result.total_return,
            "annualized_return": daily_ann,
            "sortino": daily_result.sortino,
            "max_drawdown": daily_result.max_drawdown,
            "num_trades": daily_result.num_trades,
            "win_rate": daily_result.win_rate,
        },
        "hourly": {
            "total_return": hourly_result.total_return,
            "annualized_return": hourly_ann,
            "sortino": hourly_result.sortino,
            "max_drawdown": hourly_result.max_drawdown,
            "num_trades": hourly_result.num_trades,
            "num_orders": hourly_result.num_orders,
            "win_rate": hourly_result.win_rate,
        },
    }


def evaluate_checkpoint_multiperiod(
    ckpt_path: str,
    full_data,
    *,
    periods: dict[str, int | None],
    device: torch.device,
) -> list[dict]:
    """Evaluate one checkpoint across multiple tail periods."""
    nsym = full_data.num_symbols
    policy, _, _ = load_policy(ckpt_path, nsym, device=device)
    policy_fn = make_policy_fn(
        policy, num_symbols=nsym, deterministic=True, device=device
    )

    all_days = pd.date_range(START_DATE, END_DATE, freq="D", tz="UTC")
    assert len(all_days) == full_data.num_timesteps, (
        f"Date range length {len(all_days)} != data timesteps {full_data.num_timesteps}"
    )

    results = []
    for pname, pdays in periods.items():
        if pdays is None:
            # "full" = use all but 1 timestep
            max_steps = full_data.num_timesteps - 1
            data = full_data
            tail_start_date = START_DATE
            tail_end_date = END_DATE
        else:
            need = pdays + 1
            if full_data.num_timesteps < need:
                results.append({
                    "checkpoint": ckpt_path,
                    "period": pname,
                    "error": f"Data too short ({full_data.num_timesteps} < {need})",
                })
                continue
            max_steps = pdays
            data = _slice_tail(full_data, steps=pdays)

            # _slice_tail takes the last (steps+1) timesteps.
            tail_start_idx = full_data.num_timesteps - (max_steps + 1)
            tail_start_date = str(all_days[tail_start_idx].date())
            tail_end_date = str(all_days[-1].date())

        # Load hourly market for this sub-range
        market = load_hourly_market(
            full_data.symbols,
            str(HOURLY_ROOT),
            start=f"{tail_start_date} 00:00",
            end=f"{tail_end_date} 23:00",
        )

        r = evaluate_checkpoint_daily_and_hourly(
            policy_fn,
            data,
            market,
            ckpt_path=ckpt_path,
            max_steps=max_steps,
            start_date=tail_start_date,
            end_date=tail_end_date,
        )
        r["period"] = pname
        results.append(r)

    return results


def format_comparison_table(all_results: list[dict]) -> str:
    """Format results into a comparison table."""
    lines = []

    # Header
    hdr = (
        f"{'Checkpoint':<30} {'Period':<7} "
        f"{'D-Sort':>7} {'H-Sort':>7} {'Delta':>7}  "
        f"{'D-DD%':>6} {'H-DD%':>6} {'Delta':>6}  "
        f"{'D-Ret%':>8} {'H-Ret%':>8}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for r in all_results:
        if "error" in r:
            ckpt_short = _ckpt_short_name(r["checkpoint"])
            lines.append(f"{ckpt_short:<30} {r['period']:<7} {r['error']}")
            continue

        ckpt_short = _ckpt_short_name(r["checkpoint"])
        period = r.get("period", "full")
        d = r["daily"]
        h = r["hourly"]

        sort_delta = h["sortino"] - d["sortino"]
        dd_delta = (h["max_drawdown"] - d["max_drawdown"]) * 100

        lines.append(
            f"{ckpt_short:<30} {period:<7} "
            f"{d['sortino']:>7.2f} {h['sortino']:>7.2f} {sort_delta:>+7.2f}  "
            f"{d['max_drawdown']*100:>5.1f}% {h['max_drawdown']*100:>5.1f}% {dd_delta:>+5.1f}%  "
            f"{d['total_return']*100:>+7.1f}% {h['total_return']*100:>+7.1f}%"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare daily vs hourly risk metrics for daily RL checkpoints"
    )
    parser.add_argument(
        "--checkpoints",
        default=None,
        help="Comma-separated checkpoint paths relative to pufferlib_market/checkpoints/ "
        "(default: top 5 mass_daily)",
    )
    parser.add_argument(
        "--daily-data",
        default=str(DAILY_DATA),
        help="Path to daily MKTD .bin",
    )
    parser.add_argument(
        "--hourly-root",
        default=str(HOURLY_ROOT),
        help="Root directory for hourly CSV data",
    )
    parser.add_argument(
        "--periods",
        default="120d,full",
        help="Comma-separated evaluation periods (e.g. 120d,180d,full)",
    )
    parser.add_argument("--device", default="cpu", help="torch device")
    parser.add_argument("--json", action="store_true", help="Also output JSON")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Parse checkpoints
    if args.checkpoints:
        ckpt_names = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    else:
        ckpt_names = list(DEFAULT_CHECKPOINTS)

    ckpt_paths = []
    for name in ckpt_names:
        p = Path(name)
        if not p.is_absolute():
            p = CKPT_BASE / name
        if not p.exists():
            print(f"WARNING: checkpoint not found: {p}", file=sys.stderr)
            continue
        ckpt_paths.append(str(p))

    if not ckpt_paths:
        print("ERROR: no valid checkpoints found", file=sys.stderr)
        sys.exit(1)

    # Parse periods
    periods: dict[str, int | None] = {}
    for tok in args.periods.split(","):
        tok = tok.strip()
        if tok == "full":
            periods["full"] = None
        elif tok.endswith("d") and tok[:-1].isdigit():
            periods[tok] = int(tok[:-1])
        else:
            print(f"WARNING: unknown period '{tok}', skipping", file=sys.stderr)

    # Load data
    print(f"Loading daily data from {args.daily_data}...")
    full_data = read_mktd(args.daily_data)
    print(
        f"  Symbols: {full_data.symbols}, "
        f"Timesteps: {full_data.num_timesteps}, "
        f"Date range: {START_DATE} to {END_DATE}"
    )

    # Evaluate all checkpoints
    all_results = []
    for ckpt in ckpt_paths:
        ckpt_short = _ckpt_short_name(ckpt)
        print(f"\nEvaluating {ckpt_short}...")
        results = evaluate_checkpoint_multiperiod(
            ckpt, full_data, periods=periods, device=device
        )
        all_results.extend(results)

    # Print comparison table
    print("\n" + "=" * 80)
    print("DAILY vs HOURLY RISK METRICS COMPARISON")
    print("=" * 80 + "\n")
    print(format_comparison_table(all_results))

    # Summary statistics
    valid = [r for r in all_results if "error" not in r]
    if valid:
        sort_deltas = [r["hourly"]["sortino"] - r["daily"]["sortino"] for r in valid]
        dd_deltas = [
            (r["hourly"]["max_drawdown"] - r["daily"]["max_drawdown"]) * 100
            for r in valid
        ]
        ret_deltas = [
            (r["hourly"]["total_return"] - r["daily"]["total_return"]) * 100
            for r in valid
        ]

        print(f"\n{'--- Summary ---':^80}")
        print(f"  Sortino delta (hourly - daily): mean={np.mean(sort_deltas):+.2f}, "
              f"min={np.min(sort_deltas):+.2f}, max={np.max(sort_deltas):+.2f}")
        print(f"  Max DD delta (hourly - daily):  mean={np.mean(dd_deltas):+.1f}%, "
              f"min={np.min(dd_deltas):+.1f}%, max={np.max(dd_deltas):+.1f}%")
        print(f"  Return delta (hourly - daily):  mean={np.mean(ret_deltas):+.1f}%, "
              f"min={np.min(ret_deltas):+.1f}%, max={np.max(ret_deltas):+.1f}%")

    if args.json:
        print("\n--- JSON ---")
        print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
