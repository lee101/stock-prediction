"""Swap-in analysis: for each member i of the 13-model ensemble, evaluate
the 13-model ensemble where member i is replaced by a candidate checkpoint.

Output per swap:
  - delta_med, delta_p10, delta_neg vs baseline
  - verdict: win / break-even / worse

Any positive delta_med with delta_neg ≤ 0 is a win (the swap improves the
ensemble on the deploy-gate cell).

Usage::

    python scripts/screened32_swap_in.py \\
        --candidate pufferlib_market/checkpoints/screened32_sweep/AD/s9/update_000350.pt \\
        --val-data pufferlib_market/data/screened32_single_offset_val_full.bin \\
        --window-days 50 --fill-buffer-bps 5 --max-leverage 1.0 \\
        --decision-lag 2 --out-dir docs/swap_in
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

from pufferlib_market.evaluate_holdout import (  # noqa: E402
    _mask_all_shorts,
    _slice_window,
    load_policy,
)
from pufferlib_market.hourly_replay import read_mktd, simulate_daily_policy  # noqa: E402
from src.daily_stock_defaults import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_EXTRA_CHECKPOINTS,
)


def _percentile(values, q):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _monthly_from_total(total, days, trading_days_per_month=21.0):
    if days <= 0:
        return 0.0
    try:
        return math.expm1(math.log1p(float(total)) * (trading_days_per_month / float(days)))
    except (ValueError, OverflowError):
        return 0.0


def _build_subset_policy_fn(
    *, policies, keep_indices, num_symbols, per_symbol_actions,
    decision_lag, disable_shorts, device,
):
    """Ensemble policy that averages softmax probs over keep_indices."""
    pending = collections.deque(maxlen=max(1, decision_lag + 1))
    n_keep = len(keep_indices)

    def reset_buffer():
        pending.clear()

    def policy_fn(obs):
        obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).to(device=device).view(1, -1)
        with torch.no_grad():
            probs_sum = None
            for i in keep_indices:
                lg, _ = policies[i](obs_t)
                pr = torch.softmax(lg, dim=-1)
                probs_sum = pr if probs_sum is None else probs_sum + pr
            logits = torch.log(probs_sum / float(n_keep) + 1e-8)
        if disable_shorts:
            logits = _mask_all_shorts(
                logits,
                num_symbols=int(num_symbols),
                per_symbol_actions=int(per_symbol_actions),
            )
        a = int(torch.argmax(logits, dim=-1).item())
        if decision_lag <= 0:
            return a
        pending.append(a)
        if len(pending) <= decision_lag:
            return 0
        return pending.popleft()

    return policy_fn, reset_buffer


def evaluate_subset(
    *, data, policies, head, keep_indices, num_symbols, per_symbol_actions,
    decision_lag, disable_shorts, device, fill_buffer_bps, max_leverage,
    fee_rate, slippage_bps, window_days, start_indices, label="",
):
    policy_fn, reset_buffer = _build_subset_policy_fn(
        policies=policies, keep_indices=keep_indices,
        num_symbols=num_symbols, per_symbol_actions=per_symbol_actions,
        decision_lag=decision_lag, disable_shorts=disable_shorts, device=device,
    )
    rets, sortinos, maxdds = [], [], []
    n = len(start_indices)
    for i, start in enumerate(start_indices):
        window = _slice_window(data, start=int(start), steps=int(window_days))
        reset_buffer()
        result = simulate_daily_policy(
            window, policy_fn, max_steps=int(window_days),
            fee_rate=float(fee_rate), slippage_bps=float(slippage_bps),
            max_leverage=float(max_leverage), periods_per_year=365.0,
            fill_buffer_bps=float(fill_buffer_bps),
            action_allocation_bins=int(head.action_allocation_bins),
            action_level_bins=int(head.action_level_bins),
            action_max_offset_bps=float(head.action_max_offset_bps),
            enable_drawdown_profit_early_exit=False,
        )
        rets.append(float(result.total_return))
        sortinos.append(float(result.sortino))
        maxdds.append(float(result.max_drawdown))
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  [{label}] {i + 1}/{n} windows", flush=True)
    median_total = _percentile(rets, 50)
    p10_total = _percentile(rets, 10)
    return {
        "median_total": median_total,
        "p10_total": p10_total,
        "median_monthly": _monthly_from_total(median_total, days=window_days),
        "p10_monthly": _monthly_from_total(p10_total, days=window_days),
        "median_sortino": _percentile(sortinos, 50),
        "median_max_dd": _percentile(maxdds, 50),
        "n_neg": sum(1 for v in rets if v < 0),
        "n_windows": n,
        "keep_indices": list(keep_indices),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True, help="candidate checkpoint to swap in")
    ap.add_argument("--val-data", default="pufferlib_market/data/screened32_single_offset_val_full.bin")
    ap.add_argument("--window-days", type=int, default=50)
    ap.add_argument("--fill-buffer-bps", type=float, default=5.0)
    ap.add_argument("--max-leverage", type=float, default=1.0)
    ap.add_argument("--fee-rate", type=float, default=0.001)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--decision-lag", type=int, default=2)
    ap.add_argument("--disable-shorts", action="store_true", default=True)
    ap.add_argument("--no-disable-shorts", dest="disable_shorts", action="store_false")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="docs/swap_in/result.json")
    args = ap.parse_args(argv)

    val_path = Path(args.val_data).resolve()
    if not val_path.exists():
        print(f"swap: val data not found: {val_path}", file=sys.stderr); return 2
    cand_path = Path(args.candidate).resolve()
    if not cand_path.exists():
        print(f"swap: candidate not found: {cand_path}", file=sys.stderr); return 2

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)

    base_ckpts = [Path(DEFAULT_CHECKPOINT), *(Path(p) for p in DEFAULT_EXTRA_CHECKPOINTS)]
    abs_ckpts = [REPO / c for c in base_ckpts]
    member_names = [Path(c).stem for c in base_ckpts]
    N = len(abs_ckpts)

    data = read_mktd(val_path)
    num_symbols = int(data.num_symbols)
    features_per_sym = int(data.features.shape[2])
    window_len = int(args.window_days) + 1
    start_indices = list(range(data.num_timesteps - window_len + 1))
    device = torch.device(args.device)

    print(f"loading {N} base policies + 1 candidate on {device}...")
    loaded = [
        load_policy(str(p), num_symbols, features_per_sym=features_per_sym, device=device)
        for p in abs_ckpts
    ]
    cand_loaded = load_policy(
        str(cand_path), num_symbols, features_per_sym=features_per_sym, device=device,
    )
    head = loaded[0]
    alloc_bins = int(head.action_allocation_bins)
    level_bins = int(head.action_level_bins)
    per_symbol_actions = max(1, alloc_bins) * max(1, level_bins)
    policies = [lp.policy for lp in loaded] + [cand_loaded.policy]
    CAND = N  # index of candidate in the extended list

    common = dict(
        data=data, policies=policies, head=head, num_symbols=num_symbols,
        per_symbol_actions=per_symbol_actions,
        decision_lag=int(args.decision_lag),
        disable_shorts=bool(args.disable_shorts), device=device,
        fill_buffer_bps=float(args.fill_buffer_bps),
        max_leverage=float(args.max_leverage),
        fee_rate=float(args.fee_rate),
        slippage_bps=float(args.slippage_bps),
        window_days=int(args.window_days), start_indices=start_indices,
    )

    print(f"\n=== Baseline ({N}-model) ===")
    baseline = evaluate_subset(keep_indices=list(range(N)), label="baseline", **common)
    print(
        f"  med={baseline['median_monthly']:+.4f}  p10={baseline['p10_monthly']:+.4f}  "
        f"neg={baseline['n_neg']}/{baseline['n_windows']}  sortino={baseline['median_sortino']:.2f}"
    )

    rows = []
    for drop_idx in range(N):
        keep = [i for i in range(N) if i != drop_idx] + [CAND]
        name = member_names[drop_idx]
        print(f"\n=== SWAP: replace[{drop_idx}]={name} with candidate ===")
        r = evaluate_subset(keep_indices=keep, label=f"swap-{drop_idx}", **common)
        dmed = r["median_monthly"] - baseline["median_monthly"]
        dp10 = r["p10_monthly"] - baseline["p10_monthly"]
        dneg = r["n_neg"] - baseline["n_neg"]
        dsort = r["median_sortino"] - baseline["median_sortino"]
        verdict = (
            "win" if (dmed > 0.001 and dneg <= 0 and dp10 >= -0.002)
            else "break-even" if (abs(dmed) <= 0.002 and dneg <= 0)
            else "worse"
        )
        rows.append({
            "drop_idx": drop_idx, "dropped_member": name,
            "median_monthly": r["median_monthly"],
            "p10_monthly": r["p10_monthly"],
            "n_neg": r["n_neg"],
            "median_sortino": r["median_sortino"],
            "delta_median": dmed, "delta_p10": dp10,
            "delta_neg": dneg, "delta_sortino": dsort,
            "verdict": verdict,
        })
        print(
            f"  med={r['median_monthly']:+.4f} ({dmed:+.4f})  "
            f"p10={r['p10_monthly']:+.4f} ({dp10:+.4f})  "
            f"neg={r['n_neg']} ({dneg:+d})  sort={r['median_sortino']:.2f} ({dsort:+.2f})  "
            f"VERDICT={verdict}"
        )

    summary = {
        "candidate": str(cand_path),
        "baseline_members": member_names,
        "cell": {
            "window_days": int(args.window_days),
            "fill_buffer_bps": float(args.fill_buffer_bps),
            "max_leverage": float(args.max_leverage),
            "decision_lag": int(args.decision_lag),
            "slippage_bps": float(args.slippage_bps),
        },
        "baseline": {k: v for k, v in baseline.items() if k != "keep_indices"},
        "swaps": rows,
        "wins": [r for r in rows if r["verdict"] == "win"],
    }
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[saved] {out}")

    n_wins = len(summary["wins"])
    print(f"\n=== SUMMARY: {n_wins} swap wins ===")
    for w in summary["wins"]:
        print(f"  drop {w['dropped_member']} → swap in candidate:  "
              f"dmed={w['delta_median']:+.4f}  dneg={w['delta_neg']:+d}  "
              f"dsort={w['delta_sortino']:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
