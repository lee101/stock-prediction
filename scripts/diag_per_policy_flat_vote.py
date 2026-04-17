"""Per-policy flat-vote distribution for the 12-model prod ensemble.

Investigates whether persistent flat ensemble output is driven by:
 (a) member consensus — most policies individually argmax to action 0 (flat), OR
 (b) aggregation artifact — individual policies pick different non-flat actions,
     but softmax_avg concentrates mass on flat.

For each policy, runs the last `--window-days` of the val set and records:
  - flat_fraction: fraction of steps where argmax == 0
  - top1_symbol_idx mode: most common non-flat choice
  - mean flat-probability (softmax p[0]) across all steps
Then runs the ensemble (softmax_avg) and reports the same metrics to compare.

Output: JSON + markdown table to docs/diagnostic_per_policy_flat.{json,md}

Usage:
    python scripts/diag_per_policy_flat_vote.py \
        --val-data pufferlib_market/data/screened32_single_offset_val_full.bin \
        --window-days 50
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.evaluate_holdout import _mask_all_shorts, _slice_window, load_policy  # noqa: E402
from pufferlib_market.hourly_replay import read_mktd, simulate_daily_policy  # noqa: E402
from src.daily_stock_defaults import DEFAULT_CHECKPOINT, DEFAULT_EXTRA_CHECKPOINTS  # noqa: E402


def build_obs_trace(data, max_steps: int) -> list[np.ndarray]:
    """Run zero-action policy to collect observations at each step (no trades)."""
    obs_list: list[np.ndarray] = []

    def zero_policy(obs: np.ndarray) -> int:
        obs_list.append(obs.copy())
        return 0  # flat — no trades

    simulate_daily_policy(
        data,
        zero_policy,
        max_steps=max_steps,
        fee_rate=0.001,
        slippage_bps=5.0,
        fill_buffer_bps=5.0,
        max_leverage=1.0,
    )
    return obs_list


def measure_policy_on_obs(
    policy: torch.nn.Module,
    obs_list: list[np.ndarray],
    num_symbols: int,
    disable_shorts: bool = True,
) -> dict:
    flat_count = 0
    argmax_actions: list[int] = []
    flat_probs: list[float] = []

    with torch.inference_mode():
        for obs in obs_list:
            obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).view(1, -1)
            logits, _ = policy(obs_t)
            if disable_shorts:
                logits = _mask_all_shorts(logits, num_symbols=num_symbols)
            probs = F.softmax(logits, dim=-1)
            argmax = int(torch.argmax(logits, dim=-1).item())
            flat_p = float(probs[0, 0].item())
            argmax_actions.append(argmax)
            flat_probs.append(flat_p)
            if argmax == 0:
                flat_count += 1

    return {
        "n_steps": len(obs_list),
        "flat_count": flat_count,
        "flat_fraction": flat_count / max(1, len(obs_list)),
        "mean_flat_prob": float(np.mean(flat_probs)),
        "p90_flat_prob": float(np.percentile(flat_probs, 90)),
        "argmax_mode": Counter(argmax_actions).most_common(3),
    }


def measure_ensemble_on_obs(
    policies: list[torch.nn.Module],
    obs_list: list[np.ndarray],
    num_symbols: int,
    disable_shorts: bool = True,
) -> dict:
    flat_count = 0
    argmax_actions: list[int] = []
    flat_probs: list[float] = []

    with torch.inference_mode():
        for obs in obs_list:
            obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).view(1, -1)
            stacked = []
            for p in policies:
                logits, _ = p(obs_t)
                if disable_shorts:
                    logits = _mask_all_shorts(logits, num_symbols=num_symbols)
                stacked.append(logits)
            stacked_t = torch.stack(stacked, dim=0)
            per_policy_probs = F.softmax(stacked_t, dim=-1)
            avg_probs = per_policy_probs.mean(dim=0)
            argmax = int(torch.argmax(avg_probs, dim=-1).item())
            flat_p = float(avg_probs[0, 0].item())
            argmax_actions.append(argmax)
            flat_probs.append(flat_p)
            if argmax == 0:
                flat_count += 1

    return {
        "n_steps": len(obs_list),
        "flat_count": flat_count,
        "flat_fraction": flat_count / max(1, len(obs_list)),
        "mean_flat_prob": float(np.mean(flat_probs)),
        "p90_flat_prob": float(np.percentile(flat_probs, 90)),
        "argmax_mode": Counter(argmax_actions).most_common(3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-data", default="pufferlib_market/data/screened32_single_offset_val_full.bin")
    parser.add_argument("--window-days", type=int, default=50)
    parser.add_argument("--num-windows", type=int, default=5,
                        help="Evaluate this many windows drawn from the tail of the val set")
    parser.add_argument("--out-json", default="docs/diagnostic_per_policy_flat.json")
    parser.add_argument("--out-md", default="docs/diagnostic_per_policy_flat.md")
    args = parser.parse_args()

    data_path = Path(args.val_data).resolve()
    print(f"Reading val data: {data_path}")
    data = read_mktd(data_path)
    num_symbols = data.num_symbols
    print(f"  num_symbols={num_symbols}, num_timesteps={data.num_timesteps}")

    # Build obs from the last N*window_days of the val set (most recent = tariff-crash regime).
    total_steps = int(data.num_timesteps)
    starts = []
    for i in range(args.num_windows):
        s = total_steps - (args.window_days + 1) * (i + 1)
        if s < 0:
            break
        starts.append(s)
    starts.sort()
    print(f"  window starts: {starts}")

    all_obs: list[np.ndarray] = []
    for s in starts:
        window = _slice_window(data, start=s, steps=args.window_days)
        window_obs = build_obs_trace(window, max_steps=args.window_days)
        all_obs.extend(window_obs)
    print(f"  collected {len(all_obs)} obs across {len(starts)} windows")

    ckpts = [DEFAULT_CHECKPOINT] + list(DEFAULT_EXTRA_CHECKPOINTS)
    print(f"  loading {len(ckpts)} policies")
    policies = []
    per_policy = {}
    for ckpt in ckpts:
        name = Path(ckpt).stem
        print(f"  loading {name}...")
        loaded = load_policy(ckpt, num_symbols=num_symbols, device=torch.device("cpu"))
        policies.append(loaded.policy)
        per_policy[name] = measure_policy_on_obs(loaded.policy, all_obs, num_symbols=num_symbols)
        pp = per_policy[name]
        print(f"    {name}: flat_frac={pp['flat_fraction']:.1%} "
              f"mean_flat_p={pp['mean_flat_prob']:.3f} "
              f"top_action={pp['argmax_mode'][:2]}")

    print("  running ensemble...")
    ens = measure_ensemble_on_obs(policies, all_obs, num_symbols=num_symbols)
    print(f"  ENSEMBLE: flat_frac={ens['flat_fraction']:.1%} "
          f"mean_flat_p={ens['mean_flat_prob']:.3f} "
          f"top_action={ens['argmax_mode'][:2]}")

    out = {
        "val_data": str(data_path),
        "num_windows": len(starts),
        "window_days": args.window_days,
        "total_obs": len(all_obs),
        "num_symbols": num_symbols,
        "per_policy": per_policy,
        "ensemble_softmax_avg": ens,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))

    lines = [
        "# Per-policy flat-vote diagnostic",
        "",
        f"val_data={data_path.name} num_windows={len(starts)} window_days={args.window_days} total_obs={len(all_obs)}",
        "",
        "| policy | flat_frac | mean_flat_p | p90_flat_p | top actions |",
        "|---|---:|---:|---:|---|",
    ]
    for name, pp in per_policy.items():
        top = ", ".join(f"a{a}:{c}" for a, c in pp["argmax_mode"])
        lines.append(
            f"| {name} | {pp['flat_fraction']:.1%} | {pp['mean_flat_prob']:.3f} "
            f"| {pp['p90_flat_prob']:.3f} | {top} |"
        )
    top_ens = ", ".join(f"a{a}:{c}" for a, c in ens["argmax_mode"])
    lines.append(
        f"| **ensemble_softmax_avg** | **{ens['flat_fraction']:.1%}** | **{ens['mean_flat_prob']:.3f}** "
        f"| **{ens['p90_flat_prob']:.3f}** | **{top_ens}** |"
    )
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
