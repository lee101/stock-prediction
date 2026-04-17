"""Test if SPY-regime filter cleanly removes negative windows from realism gate.

Re-runs prod 13-model ensemble on the screened32 val (lag=2, fb=5, lev=1,
fee=10bps, slip=5bps), capturing per-window total returns. Then computes
SPY MA20 at each window's start_idx (using val's own SPY price column,
since SPY is symbol index 28). Splits windows into bull (SPY>MA20 at start)
vs bear, and reports per-bucket stats.

If bear windows account for most/all of the 11 negative windows in the
baseline +6.89%/mo cell, then deploying the SPY regime filter (already
active in live) would dramatically lift the realism gate sortino/median.
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
from pufferlib_market.hourly_replay import P_CLOSE, read_mktd, simulate_daily_policy  # noqa: E402
from src.daily_stock_defaults import DEFAULT_CHECKPOINT, DEFAULT_EXTRA_CHECKPOINTS  # noqa: E402


def _monthly_from_total(total: float, days: int) -> float:
    if days <= 0:
        return 0.0
    try:
        return math.expm1(math.log1p(float(total)) * (21.0 / float(days)))
    except (ValueError, OverflowError):
        return 0.0


def build_argmax_policy_fn(*, ckpts, num_symbols, features_per_sym, decision_lag, device):
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
            action = int(torch.argmax(logits_masked, dim=-1).item())

        if decision_lag <= 0:
            return action
        pending.append(action)
        if len(pending) <= decision_lag:
            return 0
        return pending.popleft()

    return policy_fn, reset_buffer, head


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--val-data", default="pufferlib_market/data/screened32_single_offset_val_full.bin")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--decision-lag", type=int, default=2)
    p.add_argument("--fill-buffer-bps", type=float, default=5.0)
    p.add_argument("--max-leverage", type=float, default=1.0)
    p.add_argument("--window-days", type=int, default=50)
    p.add_argument("--spy-ma", type=int, default=20)
    p.add_argument("--spy-symbol", default="SPY")
    p.add_argument("--out-json", default="docs/regime_filter_gate/screened32_regime_split.json")
    args = p.parse_args()

    data = read_mktd(args.val_data)
    T = int(data.num_timesteps)
    S = int(data.num_symbols)
    F = int(data.features.shape[2])
    print(f"Val: T={T}, S={S}, F={F}")

    # Find SPY index
    syms = list(data.symbols)
    spy_idx = syms.index(args.spy_symbol)
    print(f"SPY index: {spy_idx}")
    spy_close = np.asarray(data.prices[:, spy_idx, P_CLOSE], dtype=np.float64)

    # Compute SPY MA at each timestep (NaN for first ma-1 values)
    ma_window = args.spy_ma
    spy_ma = np.full(T, np.nan)
    for t in range(ma_window - 1, T):
        spy_ma[t] = spy_close[t - ma_window + 1: t + 1].mean()
    bull_mask = (spy_close > spy_ma) & ~np.isnan(spy_ma)

    ckpts = [DEFAULT_CHECKPOINT, *DEFAULT_EXTRA_CHECKPOINTS]
    print(f"Ensemble: {len(ckpts)} models")

    device = torch.device(args.device)
    policy_fn, reset_buffer, head = build_argmax_policy_fn(
        ckpts=ckpts,
        num_symbols=S,
        features_per_sym=F,
        decision_lag=args.decision_lag,
        device=device,
    )

    starts = list(range(0, T - args.window_days))
    print(f"Windows: {len(starts)} starts × {args.window_days}d")

    rows = []
    for start in starts:
        window = _slice_window(data, start=int(start), steps=int(args.window_days))
        reset_buffer()
        r = simulate_daily_policy(
            window,
            policy_fn,
            max_steps=int(args.window_days),
            fee_rate=0.001,
            slippage_bps=5.0,
            max_leverage=float(args.max_leverage),
            periods_per_year=365.0,
            fill_buffer_bps=float(args.fill_buffer_bps),
            action_allocation_bins=int(head.action_allocation_bins),
            action_level_bins=int(head.action_level_bins),
            action_max_offset_bps=float(head.action_max_offset_bps),
            enable_drawdown_profit_early_exit=False,
        )
        regime = "bull" if bull_mask[start] else "bear"
        rows.append({
            "start": start,
            "regime": regime,
            "spy_close": float(spy_close[start]),
            "spy_ma": float(spy_ma[start]) if not np.isnan(spy_ma[start]) else None,
            "total_return": float(r.total_return),
            "sortino": float(r.sortino),
            "max_dd": float(r.max_drawdown),
        })

    # Splits
    rets = np.array([r["total_return"] for r in rows])
    bull_rets = np.array([r["total_return"] for r in rows if r["regime"] == "bull"])
    bear_rets = np.array([r["total_return"] for r in rows if r["regime"] == "bear"])

    def stats(label, arr):
        if len(arr) == 0:
            print(f"  {label}: empty")
            return {}
        med = float(np.median(arr))
        p10 = float(np.percentile(arr, 10))
        n_neg = int(np.sum(arr < 0))
        med_monthly = _monthly_from_total(med, args.window_days)
        p10_monthly = _monthly_from_total(p10, args.window_days)
        print(f"  {label}: n={len(arr)}, med_monthly={med_monthly*100:+.2f}%, "
              f"p10_monthly={p10_monthly*100:+.2f}%, neg={n_neg}/{len(arr)} "
              f"({n_neg/len(arr)*100:.1f}%)")
        return {
            "n": len(arr), "med_total": med, "p10_total": p10,
            "med_monthly": med_monthly, "p10_monthly": p10_monthly,
            "n_neg": n_neg,
        }

    print()
    print(f"=== Regime split (SPY MA{ma_window}, lag=2, fb={args.fill_buffer_bps}, lev={args.max_leverage}) ===")
    all_stats = stats("ALL", rets)
    bull_stats = stats("BULL (SPY>MA)", bull_rets)
    bear_stats = stats("BEAR (SPY<MA)", bear_rets)

    # Where do the negative windows live?
    neg_rows = [r for r in rows if r["total_return"] < 0]
    bull_neg = sum(1 for r in neg_rows if r["regime"] == "bull")
    bear_neg = sum(1 for r in neg_rows if r["regime"] == "bear")
    print()
    print(f"=== Negative windows breakdown (n={len(neg_rows)}) ===")
    print(f"  in BULL regime: {bull_neg}/{len(neg_rows)}")
    print(f"  in BEAR regime: {bear_neg}/{len(neg_rows)}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({
        "ensemble": [Path(c).stem for c in ckpts],
        "decision_lag": args.decision_lag,
        "fill_buffer_bps": args.fill_buffer_bps,
        "max_leverage": args.max_leverage,
        "window_days": args.window_days,
        "spy_ma": ma_window,
        "summary": {"all": all_stats, "bull": bull_stats, "bear": bear_stats},
        "neg_breakdown": {"bull_count": bull_neg, "bear_count": bear_neg, "total": len(neg_rows)},
        "rows": rows,
    }, indent=2))
    print(f"\nWrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
