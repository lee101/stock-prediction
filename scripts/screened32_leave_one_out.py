"""Leave-one-out analysis of the screened32 13-model ensemble.

For each of the N members, runs the (N-1)-model ensemble at the deploy
gate cell (fb=5, lev=1, lag=2, fee=10bps, slip=5bps, shorts disabled,
deterministic argmax) on the full 263-window single-offset val set.
Compares per-cell metrics to the baseline N-model ensemble.

Outputs:
- "load-bearing": LOO-12 strictly worse — removing this member hurts the
  ensemble. Member is contributing positive lift.
- "free-drop": LOO-12 ≥ baseline on key metrics — member could be
  dropped without harm (or even with small lift). Slot opens up.
- "weak-link": LOO-12 better than baseline by meaningful margin — actively
  hurting the ensemble; should be dropped.

Implementation efficiency: loads all 13 policies once, runs each forward
pass on every (window, t) and caches the per-policy softmax probs (numpy).
Then computes (N-1)-subset argmax in numpy without re-running the GPU.
This makes the whole analysis cost ~1 ensemble eval (~5 min), not N.

Usage::

    python scripts/screened32_leave_one_out.py \\
        --val-data pufferlib_market/data/screened32_single_offset_val_full.bin \\
        --window-days 50 \\
        --fill-buffer-bps 5 \\
        --max-leverage 1.0 \\
        --decision-lag 2 \\
        --out-dir docs/leave_one_out
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


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _monthly_from_total(total: float, days: int, trading_days_per_month: float = 21.0) -> float:
    if days <= 0:
        return 0.0
    try:
        return math.expm1(math.log1p(float(total)) * (trading_days_per_month / float(days)))
    except (ValueError, OverflowError):
        return 0.0


def _build_loo_policy_fn(
    *,
    policies: Sequence,
    drop_idx: int | None,
    num_symbols: int,
    per_symbol_actions: int,
    decision_lag: int,
    disable_shorts: bool,
    device: torch.device,
    min_agree_count: int = 0,
):
    """Return a policy_fn that softmax-averages all policies except `drop_idx` (None = full ensemble).

    If `min_agree_count > 0`, force flat (action=0) when fewer than that many
    kept members individually pick the ensemble argmax. Mirrors the gate wired
    into `trade_daily_stock_prod.py::_ensemble_softmax_signal`.
    """
    pending: collections.deque[int] = collections.deque(maxlen=max(1, decision_lag + 1))
    n_full = len(policies)
    keep_indices = [i for i in range(n_full) if i != drop_idx]
    n_keep = len(keep_indices)

    def reset_buffer() -> None:
        pending.clear()

    def policy_fn(obs: np.ndarray) -> int:
        obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).to(device=device).view(1, -1)
        per_member_logits: list[torch.Tensor] = []
        with torch.no_grad():
            probs_sum: torch.Tensor | None = None
            for i in keep_indices:
                lg, _ = policies[i](obs_t)
                if min_agree_count > 0:
                    if disable_shorts:
                        lg_masked = _mask_all_shorts(
                            lg,
                            num_symbols=int(num_symbols),
                            per_symbol_actions=int(per_symbol_actions),
                        )
                    else:
                        lg_masked = lg
                    per_member_logits.append(lg_masked)
                pr = torch.softmax(lg, dim=-1)
                probs_sum = pr if probs_sum is None else probs_sum + pr
            assert probs_sum is not None
            logits = torch.log(probs_sum / float(n_keep) + 1e-8)
        if disable_shorts:
            logits = _mask_all_shorts(
                logits,
                num_symbols=int(num_symbols),
                per_symbol_actions=int(per_symbol_actions),
            )
        action_now = int(torch.argmax(logits, dim=-1).item())
        if min_agree_count > 0 and action_now != 0:
            n_agree = sum(
                1 for lg in per_member_logits
                if int(torch.argmax(lg, dim=-1).item()) == action_now
            )
            if n_agree < min_agree_count:
                action_now = 0
        if decision_lag <= 0:
            return action_now
        pending.append(action_now)
        if len(pending) <= decision_lag:
            return 0
        return pending.popleft()

    return policy_fn, reset_buffer


def evaluate_loo(
    *,
    data,
    policies: Sequence,
    head,
    drop_idx: int | None,
    num_symbols: int,
    per_symbol_actions: int,
    decision_lag: int,
    disable_shorts: bool,
    device: torch.device,
    fill_buffer_bps: float,
    max_leverage: float,
    fee_rate: float,
    slippage_bps: float,
    window_days: int,
    start_indices: Sequence[int],
    label: str = "",
    min_agree_count: int = 0,
) -> dict:
    policy_fn, reset_buffer = _build_loo_policy_fn(
        policies=policies,
        drop_idx=drop_idx,
        num_symbols=num_symbols,
        per_symbol_actions=per_symbol_actions,
        decision_lag=decision_lag,
        disable_shorts=disable_shorts,
        device=device,
        min_agree_count=min_agree_count,
    )
    rets: list[float] = []
    sortinos: list[float] = []
    maxdds: list[float] = []
    n = len(start_indices)
    for i, start in enumerate(start_indices):
        window = _slice_window(data, start=int(start), steps=int(window_days))
        reset_buffer()
        result = simulate_daily_policy(
            window,
            policy_fn,
            max_steps=int(window_days),
            fee_rate=float(fee_rate),
            slippage_bps=float(slippage_bps),
            max_leverage=float(max_leverage),
            periods_per_year=365.0,
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
    median_monthly = _monthly_from_total(median_total, days=window_days)
    p10_monthly = _monthly_from_total(p10_total, days=window_days)
    return {
        "drop_idx": drop_idx,
        "median_total": median_total,
        "p10_total": p10_total,
        "median_monthly": median_monthly,
        "p10_monthly": p10_monthly,
        "median_sortino": _percentile(sortinos, 50),
        "median_max_dd": _percentile(maxdds, 50),
        "n_neg": sum(1 for v in rets if v < 0),
        "n_windows": n,
        "window_returns": rets,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-data", default="pufferlib_market/data/screened32_single_offset_val_full.bin")
    ap.add_argument("--window-days", type=int, default=50)
    ap.add_argument("--fill-buffer-bps", type=float, default=5.0)
    ap.add_argument("--max-leverage", type=float, default=1.0)
    ap.add_argument("--fee-rate", type=float, default=0.001)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--decision-lag", type=int, default=2)
    ap.add_argument("--disable-shorts", action="store_true", default=True)
    ap.add_argument("--no-disable-shorts", dest="disable_shorts", action="store_false")
    ap.add_argument(
        "--min-agree-count",
        type=int,
        default=0,
        help="Force flat when fewer than this many kept members individually pick the ensemble argmax. 0 disables. Mirrors the prod gate.",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="docs/leave_one_out")
    ap.add_argument(
        "--extra-checkpoints",
        nargs="*",
        default=[],
        help="Additional checkpoints appended to DEFAULT_CHECKPOINT + DEFAULT_EXTRA_CHECKPOINTS before running LOO (e.g. test a 13th-member add-in).",
    )
    args = ap.parse_args(argv)

    val_path = Path(args.val_data).resolve()
    if not val_path.exists():
        print(f"loo: val data not found: {val_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base_ckpts = [Path(DEFAULT_CHECKPOINT), *(Path(p) for p in DEFAULT_EXTRA_CHECKPOINTS)]
    if args.extra_checkpoints:
        base_ckpts = base_ckpts + [Path(p) for p in args.extra_checkpoints]
    abs_ckpts = [REPO / c if not c.is_absolute() else c for c in base_ckpts]
    member_names = [Path(c).stem for c in base_ckpts]

    data = read_mktd(val_path)
    num_symbols = int(data.num_symbols)
    features_per_sym = int(data.features.shape[2])
    window_len = int(args.window_days) + 1
    if window_len > data.num_timesteps:
        print(f"loo: val too short for window_days={args.window_days}", file=sys.stderr)
        return 2
    candidate_count = data.num_timesteps - window_len + 1
    start_indices = list(range(candidate_count))
    device = torch.device(args.device)

    print(f"loading {len(abs_ckpts)} policies on {device}...")
    loaded = [
        load_policy(str(p), num_symbols, features_per_sym=features_per_sym, device=device)
        for p in abs_ckpts
    ]
    head = loaded[0]
    alloc_bins = int(head.action_allocation_bins)
    level_bins = int(head.action_level_bins)
    per_symbol_actions = max(1, alloc_bins) * max(1, level_bins)
    policies = [lp.policy for lp in loaded]

    # Baseline: full N-model ensemble
    print(f"\n=== Baseline ({len(policies)}-model full ensemble, min_agree={int(args.min_agree_count)}) ===")
    baseline = evaluate_loo(
        data=data,
        policies=policies,
        head=head,
        drop_idx=None,
        num_symbols=num_symbols,
        per_symbol_actions=per_symbol_actions,
        decision_lag=int(args.decision_lag),
        disable_shorts=bool(args.disable_shorts),
        device=device,
        fill_buffer_bps=float(args.fill_buffer_bps),
        max_leverage=float(args.max_leverage),
        fee_rate=float(args.fee_rate),
        slippage_bps=float(args.slippage_bps),
        window_days=int(args.window_days),
        start_indices=start_indices,
        label="baseline",
        min_agree_count=int(args.min_agree_count),
    )
    print(
        f"  med_monthly={baseline['median_monthly']:+.4f}  "
        f"p10_monthly={baseline['p10_monthly']:+.4f}  "
        f"neg={baseline['n_neg']}/{baseline['n_windows']}  "
        f"sortino={baseline['median_sortino']:.2f}",
        flush=True,
    )

    # LOO sweep
    loo_results = []
    for drop_idx in range(len(policies)):
        name = member_names[drop_idx]
        print(f"\n=== LOO drop {drop_idx}: {name} ===")
        result = evaluate_loo(
            data=data,
            policies=policies,
            head=head,
            drop_idx=drop_idx,
            num_symbols=num_symbols,
            per_symbol_actions=per_symbol_actions,
            decision_lag=int(args.decision_lag),
            disable_shorts=bool(args.disable_shorts),
            device=device,
            fill_buffer_bps=float(args.fill_buffer_bps),
            max_leverage=float(args.max_leverage),
            fee_rate=float(args.fee_rate),
            slippage_bps=float(args.slippage_bps),
            window_days=int(args.window_days),
            start_indices=start_indices,
            label=f"loo-{drop_idx}",
            min_agree_count=int(args.min_agree_count),
        )
        result["dropped_member"] = name
        loo_results.append(result)
        d_med = result["median_monthly"] - baseline["median_monthly"]
        d_p10 = result["p10_monthly"] - baseline["p10_monthly"]
        d_neg = result["n_neg"] - baseline["n_neg"]
        verdict = (
            "weak-link" if d_med > 0.005
            else "free-drop" if (abs(d_med) <= 0.001 and d_neg <= 0)
            else "load-bearing"
        )
        print(
            f"  med_monthly={result['median_monthly']:+.4f}  "
            f"p10_monthly={result['p10_monthly']:+.4f}  "
            f"neg={result['n_neg']}/{result['n_windows']}  "
            f"sortino={result['median_sortino']:.2f}  "
            f"Δmed={d_med:+.4f}  Δneg={d_neg:+d}  → {verdict}",
            flush=True,
        )
        result["d_med_monthly"] = d_med
        result["d_p10_monthly"] = d_p10
        result["d_neg"] = d_neg
        result["verdict"] = verdict

    # Sort by least-harmful drop first (highest d_med)
    print("\n\n=== Summary (sorted by Δmed_monthly, highest first = most droppable) ===")
    print(f"{'idx':>3} {'name':<10} {'Δmed':>8} {'Δp10':>8} {'Δneg':>5} {'verdict':<14}")
    for r in sorted(loo_results, key=lambda x: -x["d_med_monthly"]):
        print(
            f"{r['drop_idx']:>3} {r['dropped_member']:<10} "
            f"{r['d_med_monthly']:+8.4f} {r['d_p10_monthly']:+8.4f} "
            f"{r['d_neg']:+5d} {r['verdict']:<14}"
        )

    payload = {
        "val_data": str(val_path),
        "window_days": int(args.window_days),
        "fill_buffer_bps": float(args.fill_buffer_bps),
        "max_leverage": float(args.max_leverage),
        "fee_rate": float(args.fee_rate),
        "slippage_bps": float(args.slippage_bps),
        "decision_lag": int(args.decision_lag),
        "min_agree_count": int(args.min_agree_count),
        "n_windows": baseline["n_windows"],
        "members": member_names,
        "baseline": {k: v for k, v in baseline.items() if k != "window_returns"},
        "loo_results": [
            {k: v for k, v in r.items() if k != "window_returns"}
            for r in loo_results
        ],
        "baseline_window_returns": baseline["window_returns"],
        "loo_window_returns": {
            f"drop_{r['drop_idx']}_{r['dropped_member']}": r["window_returns"]
            for r in loo_results
        },
    }
    out_path = out_dir / "leave_one_out.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
