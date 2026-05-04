"""Swap-in analysis: for each member i of the 13-model ensemble, evaluate
the 13-model ensemble where member i is replaced by a candidate checkpoint.

Output per swap:
  - delta_med, delta_p10, delta_neg vs baseline
  - verdict: win / break-even / worse

Any positive worst-cell delta_med with delta_neg <= 0 is a win (the swap
improves the ensemble across the production slippage grid).

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
import math
import sys
from collections.abc import Sequence
from pathlib import Path

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
from pufferlib_market.realism import (  # noqa: E402
    PRODUCTION_DECISION_LAG,
    require_production_decision_lag,
)
from xgbnew.artifacts import write_json_atomic  # noqa: E402

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


def _require_finite_float(value: float, *, name: str, min_value: float | None = None) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if min_value is not None and parsed < float(min_value):
        raise ValueError(f"{name} must be >= {float(min_value):g}, got {parsed!r}")
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


def _slippage_values(args: argparse.Namespace) -> list[float]:
    if args.slippage_bps is not None:
        return [
            _require_finite_float(
                float(args.slippage_bps),
                name="slippage_bps",
                min_value=0.0,
            )
        ]
    return _parse_float_grid(str(args.slippage_bps_grid), name="slippage_bps_grid", min_value=0.0)


def _slip_key(slippage_bps: float) -> str:
    return f"{float(slippage_bps):g}"


def _summarize_by_slippage(by_slippage: dict[str, dict]) -> dict:
    if not by_slippage:
        return {"by_slippage": {}, "worst_cell": None}
    worst_key, worst = min(
        by_slippage.items(),
        key=lambda item: float(item[1].get("median_monthly", 0.0)),
    )
    return {
        "by_slippage": by_slippage,
        "worst_cell": {"slippage_bps": float(worst_key), **{k: v for k, v in worst.items() if k != "keep_indices"}},
    }


def _delta_row(*, slippage_bps: float, candidate: dict, baseline: dict) -> dict:
    return {
        "slippage_bps": float(slippage_bps),
        "median_monthly": float(candidate["median_monthly"]),
        "p10_monthly": float(candidate["p10_monthly"]),
        "n_neg": int(candidate["n_neg"]),
        "median_sortino": float(candidate["median_sortino"]),
        "delta_median": float(candidate["median_monthly"]) - float(baseline["median_monthly"]),
        "delta_p10": float(candidate["p10_monthly"]) - float(baseline["p10_monthly"]),
        "delta_neg": int(candidate["n_neg"]) - int(baseline["n_neg"]),
        "delta_sortino": float(candidate["median_sortino"]) - float(baseline["median_sortino"]),
    }


def _classify_swap(deltas: Sequence[dict]) -> tuple[str, dict]:
    if not deltas:
        return "worse", {
            "worst_delta_median": 0.0,
            "worst_delta_p10": 0.0,
            "max_delta_neg": 0,
            "mean_delta_sortino": 0.0,
        }
    worst_delta_median = min(float(row["delta_median"]) for row in deltas)
    worst_delta_p10 = min(float(row["delta_p10"]) for row in deltas)
    max_delta_neg = max(int(row["delta_neg"]) for row in deltas)
    mean_delta_sortino = sum(float(row["delta_sortino"]) for row in deltas) / float(len(deltas))
    verdict = (
        "win" if (worst_delta_median > 0.001 and max_delta_neg <= 0 and worst_delta_p10 >= -0.002)
        else "break-even" if (abs(worst_delta_median) <= 0.002 and max_delta_neg <= 0)
        else "worse"
    )
    return verdict, {
        "worst_delta_median": worst_delta_median,
        "worst_delta_p10": worst_delta_p10,
        "max_delta_neg": int(max_delta_neg),
        "mean_delta_sortino": mean_delta_sortino,
    }


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
    fee_rate, slippage_bps, short_borrow_apr, window_days, start_indices, label="",
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
            short_borrow_apr=float(short_borrow_apr),
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
        "slippage_bps": float(slippage_bps),
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
    ap.add_argument(
        "--slippage-bps-grid",
        default="0,5,10,20",
        help="Production slippage grid. Ignored when --slippage-bps is supplied.",
    )
    ap.add_argument("--slippage-bps", type=float, default=None, help="Single-cell smoke/legacy override")
    ap.add_argument("--short-borrow-apr", type=float, default=0.0625)
    ap.add_argument("--decision-lag", type=int, default=PRODUCTION_DECISION_LAG)
    ap.add_argument(
        "--allow-low-lag-diagnostics",
        action="store_true",
        help="Allow decision_lag < 2 for explicit smoke/diagnostic runs only.",
    )
    ap.add_argument("--disable-shorts", action="store_true", default=True)
    ap.add_argument("--no-disable-shorts", dest="disable_shorts", action="store_false")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="docs/swap_in/result.json")
    args = ap.parse_args(argv)
    try:
        slippages = _slippage_values(args)
        fill_buffer_bps = _require_finite_float(args.fill_buffer_bps, name="fill_buffer_bps", min_value=0.0)
        max_leverage = _require_finite_float(args.max_leverage, name="max_leverage", min_value=0.0)
        if max_leverage <= 0.0:
            raise ValueError("max_leverage must be > 0")
        fee_rate = _require_finite_float(args.fee_rate, name="fee_rate", min_value=0.0)
        short_borrow_apr = _require_finite_float(args.short_borrow_apr, name="short_borrow_apr", min_value=0.0)
        if int(args.window_days) < 1:
            raise ValueError("window_days must be >= 1")
        decision_lag = require_production_decision_lag(
            int(args.decision_lag),
            allow_low_lag_diagnostics=bool(args.allow_low_lag_diagnostics),
        )
    except ValueError as exc:
        print(f"swap: {exc}", file=sys.stderr)
        return 2

    val_path = Path(args.val_data).resolve()
    if not val_path.exists():
        print(f"swap: val data not found: {val_path}", file=sys.stderr)
        return 2
    cand_path = Path(args.candidate).resolve()
    if not cand_path.exists():
        print(f"swap: candidate not found: {cand_path}", file=sys.stderr)
        return 2

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

    common = {
        "data": data,
        "policies": policies,
        "head": head,
        "num_symbols": num_symbols,
        "per_symbol_actions": per_symbol_actions,
        "decision_lag": decision_lag,
        "disable_shorts": bool(args.disable_shorts),
        "device": device,
        "fill_buffer_bps": fill_buffer_bps,
        "max_leverage": max_leverage,
        "fee_rate": fee_rate,
        "short_borrow_apr": short_borrow_apr,
        "window_days": int(args.window_days),
        "start_indices": start_indices,
    }

    print(f"\n=== Baseline ({N}-model) ===")
    baseline_by_slippage = {}
    for slip in slippages:
        baseline_by_slippage[_slip_key(slip)] = evaluate_subset(
            keep_indices=list(range(N)),
            slippage_bps=float(slip),
            label=f"baseline-{_slip_key(slip)}bps",
            **common,
        )
    baseline = _summarize_by_slippage(baseline_by_slippage)
    baseline_worst = baseline["worst_cell"]
    print(
        f"  worst_slip={baseline_worst['slippage_bps']:g}bps  "
        f"med={baseline_worst['median_monthly']:+.4f}  p10={baseline_worst['p10_monthly']:+.4f}  "
        f"neg={baseline_worst['n_neg']}/{baseline_worst['n_windows']}  "
        f"sortino={baseline_worst['median_sortino']:.2f}"
    )

    rows = []
    for drop_idx in range(N):
        keep = [i for i in range(N) if i != drop_idx] + [CAND]
        name = member_names[drop_idx]
        print(f"\n=== SWAP: replace[{drop_idx}]={name} with candidate ===")
        by_slippage = {}
        deltas = []
        for slip in slippages:
            slip_key = _slip_key(slip)
            r = evaluate_subset(
                keep_indices=keep,
                slippage_bps=float(slip),
                label=f"swap-{drop_idx}-{slip_key}bps",
                **common,
            )
            by_slippage[slip_key] = r
            deltas.append(_delta_row(slippage_bps=float(slip), candidate=r, baseline=baseline_by_slippage[slip_key]))
        verdict, delta_summary = _classify_swap(deltas)
        worst_result = _summarize_by_slippage(by_slippage)["worst_cell"]
        rows.append({
            "drop_idx": drop_idx, "dropped_member": name,
            "worst_slippage_bps": worst_result["slippage_bps"],
            "worst_median_monthly": worst_result["median_monthly"],
            "worst_p10_monthly": worst_result["p10_monthly"],
            "max_delta_neg": delta_summary["max_delta_neg"],
            "worst_delta_median": delta_summary["worst_delta_median"],
            "worst_delta_p10": delta_summary["worst_delta_p10"],
            "mean_delta_sortino": delta_summary["mean_delta_sortino"],
            "by_slippage": {
                k: {m: v for m, v in result.items() if m != "keep_indices"}
                for k, result in by_slippage.items()
            },
            "deltas_by_slippage": deltas,
            "verdict": verdict,
        })
        print(
            f"  worst_slip={worst_result['slippage_bps']:g}bps  "
            f"worst_dmed={delta_summary['worst_delta_median']:+.4f}  "
            f"worst_dp10={delta_summary['worst_delta_p10']:+.4f}  "
            f"max_dneg={delta_summary['max_delta_neg']:+d}  "
            f"mean_dsort={delta_summary['mean_delta_sortino']:+.2f}  "
            f"VERDICT={verdict}"
        )

    summary = {
        "candidate": str(cand_path),
        "baseline_members": member_names,
        "cell": {
            "window_days": int(args.window_days),
            "fill_buffer_bps": fill_buffer_bps,
            "max_leverage": max_leverage,
            "decision_lag": decision_lag,
            "slippage_bps_grid": [float(v) for v in slippages],
            "short_borrow_apr": short_borrow_apr,
            "fee_rate": fee_rate,
        },
        "baseline": {
            "by_slippage": {
                k: {m: v for m, v in result.items() if m != "keep_indices"}
                for k, result in baseline_by_slippage.items()
            },
            "worst_cell": baseline["worst_cell"],
        },
        "swaps": rows,
        "wins": [r for r in rows if r["verdict"] == "win"],
    }
    write_json_atomic(out, summary, sort_keys=True)
    print(f"\n[saved] {out}")

    n_wins = len(summary["wins"])
    print(f"\n=== SUMMARY: {n_wins} swap wins ===")
    for w in summary["wins"]:
        print(f"  drop {w['dropped_member']} -> swap in candidate:  "
              f"worst_dmed={w['worst_delta_median']:+.4f}  "
              f"max_dneg={w['max_delta_neg']:+d}  "
              f"mean_dsort={w['mean_delta_sortino']:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
