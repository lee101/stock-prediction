#!/usr/bin/env python3
"""Evaluate all stocks12 daily PPO seeds with the 50-window holdout.

Run after training finishes:
    source .venv313/bin/activate
    python scripts/eval_stocks12_seeds.py --checkpoint-root pufferlib_market/checkpoints/stocks12_new_seeds
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from pufferlib_market.evaluate_holdout import (
    TradingPolicy,
    _infer_arch,
    _infer_hidden_size,
    _infer_num_actions,
    _slice_window,
)
from pufferlib_market.hourly_replay import read_mktd, simulate_daily_policy


def _require_finite_float(value: float, *, name: str, min_value: float = 0.0) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < min_value:
        raise ValueError(f"{name} must be finite and >= {min_value:g}")
    return parsed


def _require_int(value: int, *, name: str, min_value: int = 0) -> int:
    parsed = int(value)
    if parsed < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    return parsed


def eval_checkpoint(
    ckpt_path: Path,
    data,
    n_windows=50,
    eval_steps=90,
    seed=42,
    fee=0.001,
    fill_bps=5.0,
    slippage_bps=20.0,
    short_borrow_apr=0.0625,
    decision_lag=2,
    max_hold_bars=6,
    max_leverage=1.0,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = payload.get("model", payload)
    arch = _infer_arch(state_dict)
    hs = _infer_hidden_size(state_dict, arch=arch)
    obs_size = next(iter(state_dict.values())).shape[1]
    n_actions = _infer_num_actions(state_dict, fallback=25)

    policy = TradingPolicy(
        obs_size=obs_size,
        hidden=hs,
        num_actions=n_actions,
    ).to(device).eval()
    policy.load_state_dict(state_dict, strict=False)

    lag = _require_int(decision_lag, name="decision_lag")
    max_hold = _require_int(max_hold_bars, name="max_hold_bars")
    fee = _require_finite_float(fee, name="fee")
    fill_bps = _require_finite_float(fill_bps, name="fill_bps")
    slippage_bps = _require_finite_float(slippage_bps, name="slippage_bps")
    short_borrow_apr = _require_finite_float(short_borrow_apr, name="short_borrow_apr")
    max_leverage = _require_finite_float(max_leverage, name="max_leverage", min_value=1e-12)

    def infer_action(obs: np.ndarray) -> int:
        obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).to(device).view(1, -1)
        with torch.inference_mode():
            logits, _ = policy(obs_t)
        return int(torch.argmax(logits, dim=-1).item())

    T = data.num_timesteps
    candidate_starts = np.arange(0, T - eval_steps, dtype=np.int64)
    rng = np.random.default_rng(seed)
    starts = rng.choice(candidate_starts, size=n_windows, replace=candidate_starts.size < n_windows)

    returns = []
    sortinos = []
    for start_idx in starts:
        window = _slice_window(data, start=int(start_idx), steps=eval_steps)
        pending_actions: collections.deque[int] = collections.deque(maxlen=max(1, lag + 1))

        def policy_fn(
            obs: np.ndarray,
            pending_actions: collections.deque[int] = pending_actions,
        ) -> int:
            action_now = infer_action(obs)
            if lag <= 0:
                return action_now
            pending_actions.append(action_now)
            if len(pending_actions) <= lag:
                return 0
            return pending_actions.popleft()

        result = simulate_daily_policy(
            window,
            policy_fn,
            max_steps=eval_steps,
            fee_rate=fee,
            fill_buffer_bps=fill_bps,
            slippage_bps=slippage_bps,
            short_borrow_apr=short_borrow_apr,
            max_leverage=max_leverage,
            max_hold_bars=max_hold,
            enable_drawdown_profit_early_exit=False,
        )
        returns.append(result.total_return)
        sortinos.append(result.sortino)

    returns = np.array(returns)
    return {
        "checkpoint": str(ckpt_path),
        "med": float(np.median(returns) * 100),
        "p10": float(np.percentile(returns, 10) * 100),
        "p90": float(np.percentile(returns, 90) * 100),
        "worst": float(returns.min() * 100),
        "best": float(returns.max() * 100),
        "neg": int((returns < 0).sum()),
        "n": len(returns),
        "med_sortino": float(np.median(sortinos)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-root", default="pufferlib_market/checkpoints/stocks12_new_seeds")
    p.add_argument("--data-path", default="pufferlib_market/data/stocks12_daily_val.bin")
    p.add_argument("--n-windows", type=int, default=50)
    p.add_argument("--eval-steps", type=int, default=90)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--fill-bps", type=float, default=5.0)
    p.add_argument("--slippage-bps", type=float, default=20.0)
    p.add_argument("--short-borrow-apr", type=float, default=0.0625)
    p.add_argument("--decision-lag", type=int, default=2)
    p.add_argument("--max-hold-bars", type=int, default=6)
    p.add_argument("--max-leverage", type=float, default=1.0)
    p.add_argument("--out", default="")
    args = p.parse_args()

    try:
        fee = _require_finite_float(args.fee, name="fee")
        fill_bps = _require_finite_float(args.fill_bps, name="fill_bps")
        slippage_bps = _require_finite_float(args.slippage_bps, name="slippage_bps")
        short_borrow_apr = _require_finite_float(args.short_borrow_apr, name="short_borrow_apr")
        decision_lag = _require_int(args.decision_lag, name="decision_lag")
        max_hold_bars = _require_int(args.max_hold_bars, name="max_hold_bars")
        max_leverage = _require_finite_float(args.max_leverage, name="max_leverage", min_value=1e-12)
    except ValueError as exc:
        print(f"eval_stocks12_seeds: {exc}", file=sys.stderr)
        return 2

    root = Path(args.checkpoint_root)
    data = read_mktd(args.data_path)
    print(f"Data: {data.num_symbols} symbols, {data.num_timesteps} timesteps")
    print(f"Evaluating checkpoints in {root}...\n")

    results = []
    ckpt_dirs = sorted(root.glob("*/"))
    for d in ckpt_dirs:
        ckpt = d / "best.pt"
        if not ckpt.exists():
            print(f"  {d.name}: no best.pt, skipping")
            continue
        try:
            res = eval_checkpoint(
                ckpt,
                data,
                n_windows=args.n_windows,
                eval_steps=args.eval_steps,
                fee=fee,
                fill_bps=fill_bps,
                slippage_bps=slippage_bps,
                short_borrow_apr=short_borrow_apr,
                decision_lag=decision_lag,
                max_hold_bars=max_hold_bars,
                max_leverage=max_leverage,
            )
            results.append({**res, "name": d.name})
            neg_str = "✓ 0/50 neg" if res["neg"] == 0 else f"✗ {res['neg']}/50 neg"
            print(
                f"  {d.name:30s} med={res['med']:+.1f}%  p10={res['p10']:+.1f}%  "
                f"worst={res['worst']:+.1f}%  {neg_str}"
            )
        except Exception as e:
            print(f"  {d.name}: ERROR {e}")

    results.sort(key=lambda r: r["p10"], reverse=True)
    print("\n=== RANKED BY P10 ===")
    for r in results:
        neg_str = "✓" if r["neg"] == 0 else f"✗{r['neg']}"
        print(f"  {r['name']:30s} med={r['med']:+.1f}%  p10={r['p10']:+.1f}%  worst={r['worst']:+.1f}%  {neg_str}")

    zero_neg = [r for r in results if r["neg"] == 0]
    print(f"\n{'='*60}")
    print(f"0/50 negative: {len(zero_neg)}/{len(results)} seeds")
    if zero_neg:
        print("DEPLOYABLE CANDIDATES:")
        for r in zero_neg:
            print(f"  {r['name']}: med={r['med']:+.1f}%  p10={r['p10']:+.1f}%")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
