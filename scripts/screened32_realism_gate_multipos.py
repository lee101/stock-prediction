"""Multi-position portfolio realism gate.

Simulates the live `--multi-position N` path: pick top-k symbols by ensemble
probability at each step, equal-weight allocate `total_alloc/k` to each,
compute next-day return with `decision_lag=2` timing, apply fees on rebalance
deltas. This is a simplified model — it doesn't simulate intra-bar limit
orders — but it's representative of what `execute_multi_position_signals`
actually does on the trading server (rebalance to target weights at next
day's open/close).

Compare against the single-action argmax baseline to see whether
multi-position mode is worth the complexity.

Usage::

    python scripts/screened32_realism_gate_multipos.py --k 8 --total-alloc 1.0 \
        --window-days 50 --out-json docs/realism_gate_multipos/k8.json
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

from pufferlib_market.evaluate_holdout import _mask_all_shorts, load_policy  # noqa: E402
from pufferlib_market.hourly_replay import P_CLOSE, P_OPEN, read_mktd  # noqa: E402
from src.daily_stock_defaults import DEFAULT_CHECKPOINT, DEFAULT_EXTRA_CHECKPOINTS  # noqa: E402

INITIAL_CASH = 100_000.0


def _monthly_from_total(total: float, days: int) -> float:
    if days <= 0:
        return 0.0
    try:
        return math.expm1(math.log1p(float(total)) * (21.0 / float(days)))
    except (ValueError, OverflowError):
        return 0.0


def load_ensemble(ckpts, num_symbols, features_per_sym, device):
    return [load_policy(c, num_symbols, features_per_sym=features_per_sym, device=device) for c in ckpts]


def compute_top_k_targets(
    *,
    loaded,
    obs: np.ndarray,
    num_symbols: int,
    per_sym: int,
    k: int,
    min_prob_ratio: float,
    total_alloc: float,
    device,
) -> dict[int, float]:
    """Return {sym_idx: weight_fraction} for top-k longs (weights sum to total_alloc).

    Matches `_ensemble_top_k_signals`:
        prob_threshold = max(flat_prob, top_prob * min_prob_ratio)
        symbols passing threshold are weighted by their normalized prob.
    """
    obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).to(device).view(1, -1)
    n_models = len(loaded)
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
        probs_masked = torch.softmax(logits_masked, dim=-1).squeeze(0).cpu().numpy()

    flat_prob = float(probs_masked[0])
    sym_probs = []
    for s in range(num_symbols):
        start = 1 + s * per_sym
        sym_probs.append((s, float(np.sum(probs_masked[start:start + per_sym]))))
    sym_probs.sort(key=lambda x: -x[1])
    if not sym_probs:
        return {}
    top_prob = sym_probs[0][1]
    threshold = max(flat_prob, top_prob * min_prob_ratio)

    selected = []
    for s, p in sym_probs[:k]:
        if p < threshold or p <= 0:
            break
        selected.append((s, p))
    if not selected:
        return {}

    total_p = sum(p for _, p in selected)
    return {s: total_alloc * (p / total_p) for s, p in selected}


def simulate_multipos(
    *,
    data,
    loaded,
    head,
    k: int,
    min_prob_ratio: float,
    total_alloc: float,
    decision_lag: int,
    fee_rate: float,
    slippage_bps: float,
    window_days: int,
    start_idx: int,
    device,
) -> dict:
    """Simulate top-k portfolio over `window_days` starting at `start_idx`."""
    per_sym = max(1, int(head.action_allocation_bins)) * max(1, int(head.action_level_bins))
    S = int(data.num_symbols)
    F = int(data.features.shape[2])
    slip = max(0.0, slippage_bps) / 10_000.0
    effective_fee = float(fee_rate) + slip

    closes = np.asarray(data.prices[:, :, P_CLOSE], dtype=np.float64)
    features = np.asarray(data.features, dtype=np.float32)

    cash = INITIAL_CASH
    positions: dict[int, float] = {}  # sym_idx -> qty
    pending: collections.deque[dict[int, float]] = collections.deque(maxlen=max(1, decision_lag + 1))

    # Policy obs layout matches `_build_obs` in pufferlib_market/hourly_replay.py:
    #   [S*F features at t_obs=t-1] [cash/scale] [pos_val/scale] [unreal/scale]
    #   [hold/max_steps] [step/max_steps] [one-hot pos S]
    # For multi-position mode we present obs as "no current single position":
    #   cash = full equity (so cash/scale = 1.0 at start of each episode)
    #   pos_val = 0, unreal = 0, no one-hot.
    obs_size = S * F + 5 + S
    portfolio_scale = INITIAL_CASH

    equities = [INITIAL_CASH]
    for step_in_window, t in enumerate(range(start_idx, start_idx + window_days)):
        if t + 1 >= len(closes):
            break

        # Mark to market BEFORE rebalance using current closes (this is the equity the
        # policy "sees" for cash normalization purposes).
        prices_t = closes[t]
        equity = float(cash)
        for s, q in positions.items():
            equity += float(q) * float(prices_t[s])

        # Build obs with t_obs = t-1 (1-bar lag, matching C env)
        t_obs = max(0, t - 1)
        obs = np.zeros((obs_size,), dtype=np.float32)
        obs[: S * F] = features[t_obs].reshape(-1)
        base = S * F
        obs[base + 0] = float(equity / max(portfolio_scale, 1e-12))  # cash slot ≈ full equity
        # pos_val, unreal, hold, step left as zero — multi-position has no single-pos analogue
        obs[base + 4] = float(step_in_window / max(window_days, 1))
        # one-hot zeros (no single position)

        targets = compute_top_k_targets(
            loaded=loaded,
            obs=obs,
            num_symbols=S,
            per_sym=per_sym,
            k=k,
            min_prob_ratio=min_prob_ratio,
            total_alloc=total_alloc,
            device=device,
        )

        # Apply lag: targets computed at t are executed after `decision_lag` steps
        pending.append(targets)
        if len(pending) <= decision_lag:
            target_today = {}
        else:
            target_today = pending.popleft()

        # Mark to market with current closes
        prices_t = closes[t]
        equity = float(cash)
        for s, q in positions.items():
            equity += float(q) * float(prices_t[s])

        # Rebalance to target_today using close prices (next-day fill model)
        # Compute desired qty per symbol
        desired_qty: dict[int, float] = {}
        for s, w in target_today.items():
            px = float(prices_t[s])
            if px > 0 and w > 0:
                desired_qty[s] = (equity * w) / px

        all_syms = set(positions) | set(desired_qty)
        for s in all_syms:
            cur = positions.get(s, 0.0)
            tgt = desired_qty.get(s, 0.0)
            delta = tgt - cur
            if abs(delta) < 1e-9:
                continue
            px = float(prices_t[s])
            if px <= 0:
                continue
            notional_delta = delta * px
            fee_cost = abs(notional_delta) * effective_fee
            cash -= notional_delta + fee_cost
            positions[s] = tgt
        positions = {s: q for s, q in positions.items() if abs(q) > 1e-9}

        # Mark to market AFTER rebalance using NEXT day's close (forward return)
        next_prices = closes[t + 1]
        equity_after = float(cash)
        for s, q in positions.items():
            equity_after += float(q) * float(next_prices[s])
        equities.append(equity_after)

    # Compute metrics
    eq = np.asarray(equities)
    if len(eq) < 2:
        return {"start": start_idx, "total_return": 0.0, "sortino": 0.0, "max_dd": 0.0}
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-8)
    total = eq[-1] / eq[0] - 1.0
    running_max = np.maximum.accumulate(eq)
    max_dd = float(np.max(1 - eq / running_max))
    downside = np.minimum(rets, 0.0)
    downside_std = math.sqrt(float((downside ** 2).mean()) + 1e-12)
    sortino = float(rets.mean() / max(downside_std, 1e-9)) * math.sqrt(252)
    return {
        "start": start_idx,
        "total_return": float(total),
        "sortino": sortino,
        "max_dd": max_dd,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--val-data", default="pufferlib_market/data/screened32_single_offset_val_full.bin")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--decision-lag", type=int, default=2)
    p.add_argument("--fill-buffer-bps", type=float, default=5.0)
    p.add_argument("--total-alloc", type=float, default=1.0)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--min-prob-ratio", type=float, default=0.5)
    p.add_argument("--window-days", type=int, default=50)
    p.add_argument("--out-json", default="docs/realism_gate_multipos/screened32_multipos.json")
    args = p.parse_args()

    data = read_mktd(args.val_data)
    T = int(data.num_timesteps)
    num_symbols = int(data.num_symbols)
    features_per_sym = int(data.features.shape[2])
    print(f"Val: T={T}, S={num_symbols}, F={features_per_sym}")

    ckpts = [DEFAULT_CHECKPOINT, *DEFAULT_EXTRA_CHECKPOINTS]
    print(f"Ensemble: {len(ckpts)} models, k={args.k}, ratio={args.min_prob_ratio}, total_alloc={args.total_alloc}")

    device = torch.device(args.device)
    loaded = load_ensemble(ckpts, num_symbols, features_per_sym, device)
    head = loaded[0]

    starts = list(range(0, T - args.window_days - 1))
    print(f"Windows: {len(starts)} starts × {args.window_days}d")

    fee_rate = 0.001
    slippage_bps = 5.0
    results = []
    for i, start in enumerate(starts):
        r = simulate_multipos(
            data=data,
            loaded=loaded,
            head=head,
            k=args.k,
            min_prob_ratio=args.min_prob_ratio,
            total_alloc=args.total_alloc,
            decision_lag=args.decision_lag,
            fee_rate=fee_rate,
            slippage_bps=slippage_bps,
            window_days=args.window_days,
            start_idx=start,
            device=device,
        )
        results.append(r)
        if i % 50 == 0:
            print(f"  win {i+1}/{len(starts)}: tot={r['total_return']*100:+.2f}%  dd={r['max_dd']*100:.2f}%")

    rets = np.array([r["total_return"] for r in results])
    sortinos = np.array([r["sortino"] for r in results])
    dds = np.array([r["max_dd"] for r in results])
    med = float(np.median(rets))
    p10 = float(np.percentile(rets, 10))
    n_neg = int(np.sum(rets < 0))
    med_monthly = _monthly_from_total(med, args.window_days)
    p10_monthly = _monthly_from_total(p10, args.window_days)
    print()
    print(f"=== MULTI-POSITION k={args.k}, total_alloc={args.total_alloc} ===")
    print(f"median_monthly: {med_monthly*100:+.2f}%")
    print(f"p10_monthly:    {p10_monthly*100:+.2f}%")
    print(f"median_sortino: {float(np.median(sortinos)):.2f}")
    print(f"median_max_dd:  {float(np.median(dds))*100:.2f}%")
    print(f"n_neg:          {n_neg}/{len(rets)}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({
        "ensemble": [Path(c).stem for c in ckpts],
        "k": args.k,
        "min_prob_ratio": args.min_prob_ratio,
        "total_alloc": args.total_alloc,
        "decision_lag": args.decision_lag,
        "fill_buffer_bps": args.fill_buffer_bps,
        "fee_rate": fee_rate,
        "slippage_bps": slippage_bps,
        "window_days": args.window_days,
        "summary": {
            "median_monthly": med_monthly,
            "p10_monthly": p10_monthly,
            "median_sortino": float(np.median(sortinos)),
            "median_max_dd": float(np.median(dds)),
            "n_neg": n_neg,
            "n_windows": len(rets),
        },
    }, indent=2))
    print(f"\nWrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
