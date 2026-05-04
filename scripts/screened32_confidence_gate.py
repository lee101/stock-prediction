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
from pufferlib_market.realism import (  # noqa: E402
    PRODUCTION_DECISION_LAG,
    PRODUCTION_SHORT_BORROW_APR,
    require_production_decision_lag,
)
from xgbnew.artifacts import write_json_atomic  # noqa: E402

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


def _require_finite_float(value: float, *, name: str, min_value: float | None = None) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if min_value is not None and parsed < float(min_value):
        raise ValueError(f"{name} must be >= {float(min_value):g}, got {parsed!r}")
    return parsed


def _require_int_at_least(value: int, *, name: str, min_value: int) -> int:
    parsed = int(value)
    if parsed < int(min_value):
        raise ValueError(f"{name} must be >= {int(min_value)}, got {parsed}")
    return parsed


def _parse_float_grid(raw: str, *, name: str, min_value: float | None = None) -> list[float]:
    values: list[float] = []
    seen: set[float] = set()
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            continue
        try:
            parsed = _require_finite_float(float(text), name=name, min_value=min_value)
        except ValueError as exc:
            raise ValueError(f"{name} contains invalid value {text!r}: {exc}") from exc
        if parsed in seen:
            raise ValueError(f"{name} contains duplicate value {parsed:g}")
        seen.add(parsed)
        values.append(parsed)
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--window-days", type=int, default=50)
    ap.add_argument("--fill-buffer-bps", type=float, default=5.0)
    ap.add_argument("--max-leverage", type=float, default=1.0)
    ap.add_argument("--decision-lag", type=int, default=PRODUCTION_DECISION_LAG)
    ap.add_argument(
        "--allow-low-lag-diagnostics",
        action="store_true",
        help="Allow decision_lag < 2 for explicit smoke/diagnostic runs only.",
    )
    ap.add_argument("--fee-rate", type=float, default=0.001)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--short-borrow-apr", type=float, default=PRODUCTION_SHORT_BORROW_APR)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--disable-shorts", action="store_true", default=True)
    ap.add_argument("--no-disable-shorts", dest="disable_shorts", action="store_false")
    ap.add_argument(
        "--thresholds",
        default="0.0,0.05,0.10,0.15,0.20,0.25,0.30",
        help="Comma-separated min_top_prob thresholds",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    try:
        window_days = _require_int_at_least(args.window_days, name="window_days", min_value=1)
        fill_buffer_bps = _require_finite_float(args.fill_buffer_bps, name="fill_buffer_bps", min_value=0.0)
        max_leverage = _require_finite_float(args.max_leverage, name="max_leverage", min_value=0.0)
        if max_leverage <= 0.0:
            raise ValueError("max_leverage must be > 0")
        decision_lag = require_production_decision_lag(
            int(args.decision_lag),
            allow_low_lag_diagnostics=bool(args.allow_low_lag_diagnostics),
        )
        fee_rate = _require_finite_float(args.fee_rate, name="fee_rate", min_value=0.0)
        slippage_bps = _require_finite_float(args.slippage_bps, name="slippage_bps", min_value=0.0)
        short_borrow_apr = _require_finite_float(
            args.short_borrow_apr,
            name="short_borrow_apr",
            min_value=0.0,
        )
        thresholds = _parse_float_grid(args.thresholds, name="thresholds", min_value=0.0)
    except ValueError as exc:
        print(f"screened32_confidence_gate: {exc}", file=sys.stderr)
        return 2

    val_path = Path(args.val_data)
    if not val_path.exists():
        print(f"screened32_confidence_gate: val not found: {val_path}", file=sys.stderr)
        return 2

    ckpts = [DEFAULT_CHECKPOINT, *DEFAULT_EXTRA_CHECKPOINTS]
    device = torch.device(args.device)
    print(f"loading {len(ckpts)} ensemble members on {device}...", flush=True)

    data = read_mktd(val_path)
    total_steps = int(data.num_timesteps)
    num_symbols = int(data.num_symbols)
    start_indices = list(range(0, total_steps - window_days))
    if not start_indices:
        print(
            f"screened32_confidence_gate: val too short for window_days={window_days} (T={total_steps})",
            file=sys.stderr,
        )
        return 2
    print(f"val: {total_steps} steps, {len(start_indices)} windows × {window_days}d", flush=True)

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
            decision_lag=decision_lag,
            disable_shorts=args.disable_shorts,
            device=device,
            min_top_prob=thr,
        )
        rets, sortinos, maxdds = [], [], []
        print(f"[threshold={thr:.2f}] running {len(start_indices)} windows...", flush=True)
        for start in start_indices:
            window = _slice_window(data, start=int(start), steps=int(window_days))
            reset_buffer()
            result = simulate_daily_policy(
                window,
                policy_fn,
                max_steps=int(window_days),
                fee_rate=fee_rate,
                slippage_bps=slippage_bps,
                max_leverage=max_leverage,
                periods_per_year=365.0,
                fill_buffer_bps=fill_buffer_bps,
                short_borrow_apr=short_borrow_apr,
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
        med_monthly = _monthly_from_total(med_total, window_days)
        p10_monthly = _monthly_from_total(p10_total, window_days)
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
    write_json_atomic(
        out_path,
        {
            "val_data": str(val_path),
            "window_days": window_days,
            "fill_buffer_bps": fill_buffer_bps,
            "max_leverage": max_leverage,
            "decision_lag": decision_lag,
            "fee_rate": fee_rate,
            "slippage_bps": slippage_bps,
            "short_borrow_apr": short_borrow_apr,
            "disable_shorts": args.disable_shorts,
            "ensemble_size": len(ckpts),
            "checkpoints": [str(c) for c in ckpts],
            "results": results,
        },
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
