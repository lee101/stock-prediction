"""Ensemble confidence-threshold ablation for the v7 screened32 ensemble.

Extends screened32_realism_gate.py with a `--min-top-prob` floor:
when the ensemble's top-action probability is below the threshold,
force action=0 (flat). This is ensemble uncertainty as a gate — only
trade when consensus is strong. Sweeps thresholds over a range and
reports per-threshold median, p10, sortino, max_dd, neg count.

Motivation: softmax_avg already leans flat via member flat-votes
(C_s7 15.6%, D_s28 10% on val), but those are soft votes. A hard
confidence floor is a different knob — some windows where the
argmax BARELY beats flat would shift to flat under a floor.

Usage::

    python scripts/screened32_confidence_gate.py \\
        --val-data pufferlib_market/data/screened32_single_offset_val_full.bin \\
        --window-days 50 --fill-buffer-bps 5 --max-leverage 1.0 \\
        --decision-lag 2 --thresholds 0.0,0.05,0.10,0.15,0.20 \\
        --out docs/confidence_gate/results.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.evaluate_holdout import _mask_all_shorts, _slice_window, load_policy  # noqa: E402
from pufferlib_market.hourly_replay import read_mktd, simulate_daily_policy  # noqa: E402
from src.daily_stock_defaults import DEFAULT_CHECKPOINT, DEFAULT_EXTRA_CHECKPOINTS  # noqa: E402


def _monthly_from_total(total: float, days: int) -> float:
    if days <= 0:
        return 0.0
    try:
        return math.expm1(math.log1p(float(total)) * (21.0 / float(days)))
    except (ValueError, OverflowError):
        return 0.0


def _pct(values, q):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _build_policy_fn(
    *,
    policies,
    num_symbols,
    per_symbol_actions,
    decision_lag,
    disable_shorts,
    device,
    min_top_prob,
):
    pending = collections.deque(maxlen=max(1, decision_lag + 1))

    def reset_buffer():
        pending.clear()

    def policy_fn(obs):
        obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).to(device=device).view(1, -1)
        with torch.no_grad():
            probs_sum = None
            for p in policies:
                lg, _ = p(obs_t)
                pr = torch.softmax(lg, dim=-1)
                probs_sum = pr if probs_sum is None else probs_sum + pr
            avg_probs = probs_sum / float(len(policies))
            # Match prod realism gate: mask shorts AFTER averaging, in log space
            if disable_shorts:
                logits_avg = torch.log(avg_probs + 1e-12)
                logits_masked = _mask_all_shorts(
                    logits_avg,
                    num_symbols=int(num_symbols),
                    per_symbol_actions=int(per_symbol_actions),
                )
                # Renormalize over non-masked actions for honest "top_prob"
                probs_for_top = torch.softmax(logits_masked, dim=-1)
                top_prob, top_action = torch.max(probs_for_top, dim=-1)
            else:
                top_prob, top_action = torch.max(avg_probs, dim=-1)
            top_prob_f = float(top_prob.item())
            action_now = int(top_action.item())
            if min_top_prob > 0.0 and top_prob_f < min_top_prob:
                action_now = 0
        if decision_lag <= 0:
            return action_now
        pending.append(action_now)
        if len(pending) <= decision_lag:
            return 0
        return pending.popleft()

    return policy_fn, reset_buffer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--window-days", type=int, default=50)
    ap.add_argument("--fill-buffer-bps", type=float, default=5.0)
    ap.add_argument("--max-leverage", type=float, default=1.0)
    ap.add_argument("--decision-lag", type=int, default=2)
    ap.add_argument("--fee-rate", type=float, default=0.001)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--disable-shorts", action="store_true", default=True)
    ap.add_argument("--no-disable-shorts", dest="disable_shorts", action="store_false")
    ap.add_argument(
        "--thresholds",
        default="0.0,0.05,0.10,0.15,0.20,0.25,0.30",
        help="Comma-separated min_top_prob thresholds",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    val_path = Path(args.val_data)
    if not val_path.exists():
        print(f"val not found: {val_path}", file=sys.stderr)
        return 2

    thresholds = [float(x) for x in args.thresholds.split(",") if x]

    ckpts = [DEFAULT_CHECKPOINT, *DEFAULT_EXTRA_CHECKPOINTS]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {len(ckpts)} ensemble members on {device}...", flush=True)

    data = read_mktd(val_path)
    total_steps = int(data.num_timesteps)
    num_symbols = int(data.num_symbols)
    start_indices = list(range(0, total_steps - args.window_days))
    print(f"val: {total_steps} steps, {len(start_indices)} windows × {args.window_days}d", flush=True)

    loaded = [load_policy(c, num_symbols, device=device) for c in ckpts]
    head = loaded[0]
    policies = [lp.policy for lp in loaded]
    per_symbol_actions = max(1, int(head.action_allocation_bins)) * max(1, int(head.action_level_bins))

    results = []
    for thr in thresholds:
        policy_fn, reset_buffer = _build_policy_fn(
            policies=policies,
            num_symbols=num_symbols,
            per_symbol_actions=per_symbol_actions,
            decision_lag=args.decision_lag,
            disable_shorts=args.disable_shorts,
            device=device,
            min_top_prob=thr,
        )
        rets, sortinos, maxdds = [], [], []
        flat_forced = 0
        total_decisions = 0
        print(f"[threshold={thr:.2f}] running {len(start_indices)} windows...", flush=True)
        for start in start_indices:
            window = _slice_window(data, start=int(start), steps=int(args.window_days))
            reset_buffer()
            result = simulate_daily_policy(
                window,
                policy_fn,
                max_steps=int(args.window_days),
                fee_rate=float(args.fee_rate),
                slippage_bps=float(args.slippage_bps),
                max_leverage=float(args.max_leverage),
                periods_per_year=365.0,
                fill_buffer_bps=float(args.fill_buffer_bps),
                action_allocation_bins=int(head.action_allocation_bins),
                action_level_bins=int(head.action_level_bins),
                action_max_offset_bps=float(head.action_max_offset_bps),
                enable_drawdown_profit_early_exit=False,
            )
            rets.append(float(result.total_return))
            sortinos.append(float(result.sortino))
            maxdds.append(float(result.max_drawdown))
        med_total = _pct(rets, 50)
        p10_total = _pct(rets, 10)
        med_monthly = _monthly_from_total(med_total, args.window_days)
        p10_monthly = _monthly_from_total(p10_total, args.window_days)
        n_neg = int(sum(1 for r in rets if r < 0.0))
        med_sortino = _pct(sortinos, 50)
        med_maxdd = _pct(maxdds, 50)
        row = {
            "threshold": thr,
            "median_monthly_return": med_monthly,
            "p10_monthly_return": p10_monthly,
            "median_sortino": med_sortino,
            "median_max_dd": med_maxdd,
            "n_neg": n_neg,
            "n_windows": len(rets),
        }
        results.append(row)
        print(
            f"  med={med_monthly * 100:.2f}%/mo  p10={p10_monthly * 100:.2f}%/mo  "
            f"sortino={med_sortino:.2f}  neg={n_neg}/{len(rets)}  "
            f"max_dd={med_maxdd * 100:.2f}%",
            flush=True,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "val_data": str(val_path),
                "window_days": args.window_days,
                "fill_buffer_bps": args.fill_buffer_bps,
                "max_leverage": args.max_leverage,
                "decision_lag": args.decision_lag,
                "fee_rate": args.fee_rate,
                "slippage_bps": args.slippage_bps,
                "disable_shorts": args.disable_shorts,
                "ensemble_size": len(ckpts),
                "checkpoints": [str(c) for c in ckpts],
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out_path}")
    print("\n| threshold | med%/mo | p10%/mo | sortino | neg | max_dd% |")
    print("|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r['threshold']:.2f} | {r['median_monthly_return'] * 100:.2f} | "
            f"{r['p10_monthly_return'] * 100:.2f} | {r['median_sortino']:.2f} | "
            f"{r['n_neg']} | {r['median_max_dd'] * 100:.2f} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
