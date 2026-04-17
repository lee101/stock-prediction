"""Probe the prod ensemble's action distribution across the full val.

Drives the prod ensemble through `simulate_daily_policy` on the full val set
(~313 timesteps) at decision_lag=2 / fb=5 / lev=1 (matches live), and records
the action chosen at every step. Reports:
    - % of timesteps where argmax == 0 (flat)
    - % where ensemble emits a long
    - flat_prob / top_prob distributions
    - last 30 steps in detail

This answers "is the live bot being normally cautious or stuck?" by comparing
the live observation's recent flat run to the model's typical flat rate.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.evaluate_holdout import _mask_all_shorts, _slice_window, load_policy  # noqa: E402
from pufferlib_market.hourly_replay import read_mktd, simulate_daily_policy  # noqa: E402
from src.daily_stock_defaults import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_EXTRA_CHECKPOINTS,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--val-data", default="pufferlib_market/data/screened32_single_offset_val_full.bin")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--decision-lag", type=int, default=2)
    p.add_argument("--fill-buffer-bps", type=float, default=5.0)
    p.add_argument("--max-leverage", type=float, default=1.0)
    p.add_argument("--last-n", type=int, default=30)
    args = p.parse_args()

    data = read_mktd(args.val_data)
    T = int(data.num_timesteps)
    num_symbols = int(data.num_symbols)
    features_per_sym = int(data.features.shape[2])
    print(f"Val: T={T}, S={num_symbols}, F={features_per_sym}")

    ckpts = [DEFAULT_CHECKPOINT, *DEFAULT_EXTRA_CHECKPOINTS]
    n_models = len(ckpts)
    print(f"Loading {n_models}-model prod ensemble (deterministic argmax, lag={args.decision_lag})...")

    device = torch.device(args.device)
    loaded = [load_policy(c, num_symbols, features_per_sym=features_per_sym, device=device) for c in ckpts]
    head = loaded[0]
    per_sym = max(1, int(head.action_allocation_bins)) * max(1, int(head.action_level_bins))
    n_actions = 1 + 2 * num_symbols * per_sym  # flat + per-symbol long+short bins
    print(f"action space: {n_actions} total (per_sym_actions={per_sym}, longs+shorts before mask)")

    flat_probs: list[float] = []
    top_probs: list[float] = []
    actions_chosen: list[int] = []
    pending: collections.deque[int] = collections.deque(maxlen=max(1, args.decision_lag + 1))

    def reset_buffer() -> None:
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
        for s in range(num_symbols):
            start = 1 + s * per_sym
            sym_probs.append(float(probs_masked[start:start + per_sym].sum().item()))
        top_prob = max(sym_probs) if sym_probs else 0.0
        flat_probs.append(flat_prob)
        top_probs.append(top_prob)

        action_now = int(torch.argmax(logits_masked, dim=-1).item())
        actions_chosen.append(action_now)
        if args.decision_lag <= 0:
            return action_now
        pending.append(action_now)
        if len(pending) <= args.decision_lag:
            return 0
        return pending.popleft()

    # Walk the full val by stepping through one giant window
    window = _slice_window(data, start=0, steps=T - 1)
    reset_buffer()
    result = simulate_daily_policy(
        window,
        policy_fn,
        max_steps=int(T - 1),
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

    n_steps = len(actions_chosen)
    n_flat = sum(1 for a in actions_chosen if a == 0)
    print()
    print("=== ACTION DISTRIBUTION (raw policy argmax, before lag) ===")
    print(f"steps polled: {n_steps}")
    print(f"  argmax==flat: {n_flat:4d} ({100*n_flat/n_steps:.1f}%)")
    print(f"  argmax==long: {n_steps-n_flat:4d} ({100*(n_steps-n_flat)/n_steps:.1f}%)")
    print(f"sim total_return={result.total_return*100:+.2f}%  sortino={result.sortino:.2f}  max_dd={result.max_drawdown*100:.2f}%")

    fp = np.asarray(flat_probs); tp = np.asarray(top_probs)
    print()
    print("=== flat_prob (avg_probs[0] across {} models) ===".format(n_models))
    print(f"min={fp.min():.4f}  p10={np.percentile(fp,10):.4f}  p25={np.percentile(fp,25):.4f}  med={np.median(fp):.4f}  p75={np.percentile(fp,75):.4f}  p90={np.percentile(fp,90):.4f}  max={fp.max():.4f}  mean={fp.mean():.4f}")
    print()
    print("=== top_prob (max sym_long_prob_sum) ===")
    print(f"min={tp.min():.4f}  p10={np.percentile(tp,10):.4f}  p25={np.percentile(tp,25):.4f}  med={np.median(tp):.4f}  p75={np.percentile(tp,75):.4f}  p90={np.percentile(tp,90):.4f}  max={tp.max():.4f}  mean={tp.mean():.4f}")

    print()
    print(f"=== LAST {args.last_n} STEPS ===")
    for t in range(max(0, n_steps - args.last_n), n_steps):
        a = "flat" if actions_chosen[t] == 0 else f"a={actions_chosen[t]}"
        print(f"t={t:4d}  flat_prob={flat_probs[t]:.4f}  top_prob={top_probs[t]:.4f}  → {a}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
