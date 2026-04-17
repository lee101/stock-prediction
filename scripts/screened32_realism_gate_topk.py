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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--val-data", default="pufferlib_market/data/screened32_single_offset_val_full.bin")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--decision-lag", type=int, default=2)
    p.add_argument("--fill-buffer-bps", type=float, default=5.0)
    p.add_argument("--max-leverage", type=float, default=1.0)
    p.add_argument("--window-days", type=int, default=50)
    p.add_argument("--min-prob-ratios", default="0.3,0.5,0.7,1.0",
                   help="Comma list of min_prob_ratios to sweep.")
    p.add_argument("--out-json", default="docs/realism_gate_topk/screened32_topk.json")
    args = p.parse_args()

    data = read_mktd(args.val_data)
    T = int(data.num_timesteps)
    num_symbols = int(data.num_symbols)
    features_per_sym = int(data.features.shape[2])
    print(f"Val: T={T}, S={num_symbols}, F={features_per_sym}")

    ckpts = [DEFAULT_CHECKPOINT, *DEFAULT_EXTRA_CHECKPOINTS]
    print(f"Ensemble: {len(ckpts)} models")

    device = torch.device(args.device)
    starts = list(range(0, T - args.window_days))
    print(f"Windows: {len(starts)} starts × {args.window_days}d")

    ratios = [float(r) for r in args.min_prob_ratios.split(",")]
    rows = []
    for ratio in ratios:
        policy_fn, reset_buffer, head = build_topk_policy_fn(
            ckpts=ckpts,
            num_symbols=num_symbols,
            features_per_sym=features_per_sym,
            decision_lag=args.decision_lag,
            device=device,
            min_prob_ratio=ratio,
        )
        rets, sortinos, dds = [], [], []
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
            rets.append(r.total_return)
            sortinos.append(r.sortino)
            dds.append(r.max_drawdown)
        rets = np.asarray(rets)
        med = float(np.median(rets))
        p10 = float(np.percentile(rets, 10))
        n_neg = int(np.sum(rets < 0))
        n_w = len(rets)
        med_monthly = _monthly_from_total(med, args.window_days)
        p10_monthly = _monthly_from_total(p10, args.window_days)
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

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({
        "ensemble": [Path(c).stem for c in ckpts],
        "decision_lag": args.decision_lag,
        "fill_buffer_bps": args.fill_buffer_bps,
        "max_leverage": args.max_leverage,
        "window_days": args.window_days,
        "rows": rows,
    }, indent=2))
    print(f"\nWrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
