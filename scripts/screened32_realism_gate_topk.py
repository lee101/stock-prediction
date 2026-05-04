"""Realism gate variant — top-K decision rule instead of argmax.

Same prod ensemble + window logic as `screened32_realism_gate.py`, but the
policy_fn replaces deterministic argmax with the live `_ensemble_top_k_signals`
rule:
    1. Compute softmax-averaged probs across N models.
    2. flat_prob = probs[0]; sym_prob[s] = sum(probs[1+s*per_sym : ...]).
    3. threshold = max(flat_prob, top_sym_prob * min_prob_ratio).
    4. If top_sym_prob >= threshold: emit action for that sym's first bin.
       Else emit flat.

This is what would happen if live were launched with `--multi-position 1`
(or higher) and `_ensemble_top_k_signals` chose a single symbol per step.
Compared against the argmax baseline at fb=5, lev=1.0.
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


def build_topk_policy_fn(*, ckpts, num_symbols, features_per_sym, decision_lag, device, min_prob_ratio):
    loaded = [load_policy(c, num_symbols, features_per_sym=features_per_sym, device=device) for c in ckpts]
    head = loaded[0]
    per_sym = max(1, int(head.action_allocation_bins)) * max(1, int(head.action_level_bins))
    n_models = len(loaded)
    pending: collections.deque[int] = collections.deque(maxlen=max(1, decision_lag + 1))

    def reset_buffer():
        pending.clear()

    def policy_fn(obs: np.ndarray) -> int:
        obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).to(device).view(1, -1)
        with torch.no_grad():
            probs_sum = None
            for lp in loaded:
                lg, _ = lp.policy(obs_t)
                pr = torch.softmax(lg, dim=-1)
                probs_sum = pr if probs_sum is None else probs_sum + pr
            avg = probs_sum / n_models
            logits_masked = _mask_all_shorts(
                torch.log(avg + 1e-8),
                num_symbols=num_symbols,
                per_symbol_actions=per_sym,
            )
            probs_masked = torch.softmax(logits_masked, dim=-1).squeeze(0)
        flat_prob = float(probs_masked[0].item())
        sym_probs = []
        sym_argmax_bin = []
        for s in range(num_symbols):
            start = 1 + s * per_sym
            block = probs_masked[start:start + per_sym]
            sym_probs.append(float(block.sum().item()))
            sym_argmax_bin.append(int(torch.argmax(block).item()) if per_sym > 1 else 0)
        if not sym_probs:
            action_now = 0
        else:
            top_sym = int(np.argmax(sym_probs))
            top_prob = sym_probs[top_sym]
            threshold = max(flat_prob, top_prob * float(min_prob_ratio))
            if top_prob >= threshold and top_prob > 0:
                action_now = 1 + top_sym * per_sym + sym_argmax_bin[top_sym]
            else:
                action_now = 0

        if decision_lag <= 0:
            return action_now
        pending.append(action_now)
        if len(pending) <= decision_lag:
            return 0
        return pending.popleft()

    return policy_fn, reset_buffer, head


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--val-data", default="pufferlib_market/data/screened32_single_offset_val_full.bin")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--decision-lag", type=int, default=PRODUCTION_DECISION_LAG)
    p.add_argument(
        "--allow-low-lag-diagnostics",
        action="store_true",
        help="Allow decision_lag < 2 for explicit smoke/diagnostic runs only.",
    )
    p.add_argument("--fill-buffer-bps", type=float, default=5.0)
    p.add_argument("--fee-rate", type=float, default=0.001)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--short-borrow-apr", type=float, default=0.0625)
    p.add_argument("--max-leverage", type=float, default=1.0)
    p.add_argument("--window-days", type=int, default=50)
    p.add_argument("--min-prob-ratios", default="0.3,0.5,0.7,1.0",
                   help="Comma list of min_prob_ratios to sweep.")
    p.add_argument("--out-json", default="docs/realism_gate_topk/screened32_topk.json")
    args = p.parse_args(argv)

    try:
        decision_lag = require_production_decision_lag(
            int(args.decision_lag),
            allow_low_lag_diagnostics=bool(args.allow_low_lag_diagnostics),
        )
        window_days = _require_int_at_least(args.window_days, name="window_days", min_value=1)
        fill_buffer_bps = _require_finite_float(args.fill_buffer_bps, name="fill_buffer_bps", min_value=0.0)
        fee_rate = _require_finite_float(args.fee_rate, name="fee_rate", min_value=0.0)
        slippage_bps = _require_finite_float(args.slippage_bps, name="slippage_bps", min_value=0.0)
        short_borrow_apr = _require_finite_float(args.short_borrow_apr, name="short_borrow_apr", min_value=0.0)
        max_leverage = _require_finite_float(args.max_leverage, name="max_leverage", min_value=0.0)
        if max_leverage <= 0.0:
            raise ValueError("max_leverage must be > 0")
        ratios = _parse_float_grid(args.min_prob_ratios, name="min_prob_ratios", min_value=0.0)
    except ValueError as exc:
        print(f"screened32_realism_gate_topk: {exc}", file=sys.stderr)
        return 2

    val_path = Path(args.val_data)
    if not val_path.exists():
        print(f"screened32_realism_gate_topk: val data not found: {val_path}", file=sys.stderr)
        return 2

    data = read_mktd(val_path)
    T = int(data.num_timesteps)
    num_symbols = int(data.num_symbols)
    features_per_sym = int(data.features.shape[2])
    print(f"Val: T={T}, S={num_symbols}, F={features_per_sym}")

    ckpts = [DEFAULT_CHECKPOINT, *DEFAULT_EXTRA_CHECKPOINTS]
    print(f"Ensemble: {len(ckpts)} models")

    device = torch.device(args.device)
    starts = list(range(0, T - window_days))
    if not starts:
        print(
            f"screened32_realism_gate_topk: val too short for window_days={window_days} (T={T})",
            file=sys.stderr,
        )
        return 2
    print(f"Windows: {len(starts)} starts × {window_days}d")

    rows = []
    for ratio in ratios:
        policy_fn, reset_buffer, head = build_topk_policy_fn(
            ckpts=ckpts,
            num_symbols=num_symbols,
            features_per_sym=features_per_sym,
            decision_lag=decision_lag,
            device=device,
            min_prob_ratio=ratio,
        )
        rets, sortinos, dds = [], [], []
        for start in starts:
            window = _slice_window(data, start=int(start), steps=int(window_days))
            reset_buffer()
            r = simulate_daily_policy(
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
            rets.append(r.total_return)
            sortinos.append(r.sortino)
            dds.append(r.max_drawdown)
        rets = np.asarray(rets)
        med = float(np.median(rets))
        p10 = float(np.percentile(rets, 10))
        n_neg = int(np.sum(rets < 0))
        n_w = len(rets)
        med_monthly = _monthly_from_total(med, window_days)
        p10_monthly = _monthly_from_total(p10, window_days)
        med_sortino = float(np.median(sortinos))
        med_dd = float(np.median(dds))
        rows.append({
            "min_prob_ratio": ratio,
            "median_total": med,
            "p10_total": p10,
            "median_monthly": med_monthly,
            "p10_monthly": p10_monthly,
            "median_sortino": med_sortino,
            "median_max_dd": med_dd,
            "n_neg": n_neg,
            "n_windows": n_w,
        })
        print(
            f"ratio={ratio:.2f}  med_monthly={med_monthly*100:+.2f}%  p10_monthly={p10_monthly*100:+.2f}%  "
            f"sortino={med_sortino:.2f}  max_dd={med_dd*100:.2f}%  neg={n_neg}/{n_w}"
        )

    write_json_atomic(Path(args.out_json), {
        "ensemble": [Path(c).stem for c in ckpts],
        "decision_lag": decision_lag,
        "fill_buffer_bps": fill_buffer_bps,
        "fee_rate": fee_rate,
        "slippage_bps": slippage_bps,
        "short_borrow_apr": short_borrow_apr,
        "max_leverage": max_leverage,
        "window_days": window_days,
        "rows": rows,
    })
    print(f"\nWrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
