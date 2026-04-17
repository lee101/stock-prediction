"""Ensemble member-agreement gate for the v7 screened32 ensemble.

Different mechanism from the (rejected) confidence gate: instead of
gating on the post-mask softmax top-probability, gate on how many of
the N ensemble members INDIVIDUALLY pick the same argmax action as
the ensemble's softmax_avg. If fewer than M members agree, force flat.

Motivation: softmax_avg is already post-hoc smoothed. A low-disagreement
day might still have a high top-prob after averaging because softmax
amplifies the biggest chunk. Member-count agreement is a different signal:
"even though the weighted average points here, N members individually
disagree" — could be a regime where experts are split and we should sit out.

Usage::

    python scripts/screened32_agreement_gate.py \\
        --val-data pufferlib_market/data/screened32_single_offset_val_full.bin \\
        --window-days 50 --fill-buffer-bps 5 --max-leverage 1.0 \\
        --decision-lag 2 --min-agree-counts 0,5,6,7,8,9,10 \\
        --out docs/agreement_gate/v7_agreement.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

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
    min_agree_count,
):
    pending = collections.deque(maxlen=max(1, decision_lag + 1))

    def reset_buffer():
        pending.clear()

    def policy_fn(obs):
        obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).to(device=device).view(1, -1)
        with torch.no_grad():
            probs_sum = None
            per_member_argmax = []
            for p in policies:
                lg, _ = p(obs_t)
                pr = torch.softmax(lg, dim=-1)
                probs_sum = pr if probs_sum is None else probs_sum + pr
                # Each member's argmax over its OWN softmax, after the same mask
                if disable_shorts:
                    lg_masked = _mask_all_shorts(
                        torch.log(pr + 1e-12),
                        num_symbols=int(num_symbols),
                        per_symbol_actions=int(per_symbol_actions),
                    )
                    m_top = int(torch.argmax(lg_masked, dim=-1).item())
                else:
                    m_top = int(torch.argmax(pr, dim=-1).item())
                per_member_argmax.append(m_top)
            avg_probs = probs_sum / float(len(policies))
            if disable_shorts:
                logits_avg = torch.log(avg_probs + 1e-12)
                logits_masked = _mask_all_shorts(
                    logits_avg,
                    num_symbols=int(num_symbols),
                    per_symbol_actions=int(per_symbol_actions),
                )
                probs_for_top = torch.softmax(logits_masked, dim=-1)
                _, top_action = torch.max(probs_for_top, dim=-1)
            else:
                _, top_action = torch.max(avg_probs, dim=-1)
            ensemble_action = int(top_action.item())
            # How many members individually pick the ensemble's chosen action?
            n_agree = sum(1 for a in per_member_argmax if a == ensemble_action)
            action_now = ensemble_action
            if min_agree_count > 0 and n_agree < min_agree_count:
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
        "--min-agree-counts",
        default="0,3,4,5,6,7,8,9,10",
        help="Comma-separated minimum number of members that must agree with the ensemble argmax",
    )
    ap.add_argument(
        "--extra-checkpoints",
        nargs="*",
        default=[],
        help="Additional checkpoints appended to DEFAULT (v7) baseline, e.g. to test a 13th-member add-in under the gate.",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    val_path = Path(args.val_data)
    if not val_path.exists():
        print(f"val not found: {val_path}", file=sys.stderr)
        return 2

    counts = [int(x) for x in args.min_agree_counts.split(",") if x]

    ckpts = [DEFAULT_CHECKPOINT, *DEFAULT_EXTRA_CHECKPOINTS, *args.extra_checkpoints]
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
    for mac in counts:
        policy_fn, reset_buffer = _build_policy_fn(
            policies=policies,
            num_symbols=num_symbols,
            per_symbol_actions=per_symbol_actions,
            decision_lag=args.decision_lag,
            disable_shorts=args.disable_shorts,
            device=device,
            min_agree_count=mac,
        )
        rets, sortinos, maxdds = [], [], []
        print(f"[min_agree={mac}] running {len(start_indices)} windows...", flush=True)
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
            "min_agree_count": mac,
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
    print("\n| min_agree | med%/mo | p10%/mo | sortino | neg | max_dd% |")
    print("|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r['min_agree_count']} | {r['median_monthly_return'] * 100:.2f} | "
            f"{r['p10_monthly_return'] * 100:.2f} | {r['median_sortino']:.2f} | "
            f"{r['n_neg']} | {r['median_max_dd'] * 100:.2f} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
