#!/usr/bin/env python3
"""Fast GPU-accelerated exhaustive eval for multiple checkpoints.

Usage:
    source .venv313/bin/activate
    python scripts/fast_batch_eval.py \
        --ckpt-root pufferlib_market/checkpoints/stocks12_v2_sweep \
        --val-data pufferlib_market/data/stocks12_daily_val.bin \
        --device cuda
"""
from __future__ import annotations

import argparse
import collections
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).parent.parent))

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


def eval_checkpoint_fast(
    ckpt_path: str,
    val_data,
    eval_steps: int = 90,
    fee: float = 0.001,
    fill_bps: float = 5.0,
    slippage_bps: float = 20.0,
    short_borrow_apr: float = 0.0625,
    decision_lag: int = 2,
    max_hold_bars: int = 6,
    max_leverage: float = 1.0,
    device: torch.device = torch.device("cuda"),
) -> dict:
    """GPU-accelerated exhaustive eval."""
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = payload.get("model", payload)
    arch = _infer_arch(state_dict)
    hs = _infer_hidden_size(state_dict, arch=arch)
    obs_size = next(iter(state_dict.values())).shape[1]
    n_actions = _infer_num_actions(state_dict, fallback=25)

    policy = TradingPolicy(obs_size=obs_size, hidden=hs, num_actions=n_actions).to(device).eval()
    policy.load_state_dict(state_dict, strict=False)
    if hasattr(policy, "_use_encoder_norm"):
        if isinstance(payload, dict) and "use_encoder_norm" in payload:
            policy._use_encoder_norm = bool(payload["use_encoder_norm"])
        else:
            missing = [k for k in policy.state_dict() if k not in state_dict]
            policy._use_encoder_norm = "encoder_norm.weight" not in missing

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

    T = val_data.num_timesteps
    starts = list(range(T - eval_steps))
    returns = []
    for start in starts:
        w = _slice_window(val_data, start=int(start), steps=eval_steps)
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

        r = simulate_daily_policy(
            w,
            policy_fn,
            max_steps=eval_steps,
            fill_buffer_bps=fill_bps,
            fee_rate=fee,
            slippage_bps=slippage_bps,
            short_borrow_apr=short_borrow_apr,
            max_hold_bars=max_hold,
            max_leverage=max_leverage,
            enable_drawdown_profit_early_exit=False,
        )
        returns.append(r.total_return)
    returns = np.array(returns)
    return {
        "n": len(returns),
        "neg": int((returns < 0).sum()),
        "med": float(np.median(returns) * 100),
        "p10": float(np.percentile(returns, 10) * 100),
        "p90": float(np.percentile(returns, 90) * 100),
        "worst": float(returns.min() * 100),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--fill-bps", type=float, default=5.0)
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=20.0,
        help="Adverse slippage bps. Defaults to the worst production eval cell.",
    )
    parser.add_argument("--short-borrow-apr", type=float, default=0.0625)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--max-hold-bars", type=int, default=6)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--names", type=str, default="",
                        help="Comma-separated subdirectory names to eval (default: all)")
    args = parser.parse_args()

    try:
        fee = _require_finite_float(args.fee, name="fee")
        fill_bps = _require_finite_float(args.fill_bps, name="fill_bps")
        slippage_bps = _require_finite_float(args.slippage_bps, name="slippage_bps")
        short_borrow_apr = _require_finite_float(args.short_borrow_apr, name="short_borrow_apr")
        decision_lag = _require_int(args.decision_lag, name="decision_lag")
        max_hold_bars = _require_int(args.max_hold_bars, name="max_hold_bars")
        max_leverage = _require_finite_float(args.max_leverage, name="max_leverage", min_value=1e-12)
    except ValueError as exc:
        print(f"fast_batch_eval: {exc}", file=sys.stderr)
        return 2

    val_data = read_mktd(args.val_data)
    print(f"Val data: {val_data.num_timesteps} ts, {val_data.num_symbols} sym, {val_data.num_timesteps-90} windows")

    ckpt_root = Path(args.ckpt_root)
    device = torch.device(args.device)

    if args.names:
        names = [n.strip() for n in args.names.split(",") if n.strip()]
    else:
        names = sorted(d.name for d in ckpt_root.iterdir() if d.is_dir())

    print(f"Evaluating {len(names)} checkpoints on {device}...\n")
    print(f"{'Name':<40s} {'Neg/111':>8} {'Med%':>8} {'P10%':>8} {'P90%':>8} {'Worst%':>8}")
    print("-" * 80)

    results = []
    for name in names:
        ckpt = ckpt_root / name / "best.pt"
        if not ckpt.exists():
            print(f"  {name}: MISSING")
            continue
        try:
            t0 = time.time()
            r = eval_checkpoint_fast(str(ckpt), val_data, device=device,
                                     fee=fee, fill_bps=fill_bps,
                                     slippage_bps=slippage_bps,
                                     short_borrow_apr=short_borrow_apr,
                                     decision_lag=decision_lag,
                                     max_hold_bars=max_hold_bars,
                                     max_leverage=max_leverage)
            elapsed = time.time() - t0
            star = " ***" if r["neg"] <= 5 else ""
            print(
                f"{name:<40s} {r['neg']:>5d}/111 {r['med']:>+7.1f}% "
                f"{r['p10']:>+7.1f}% {r['p90']:>+7.1f}% "
                f"{r['worst']:>+7.1f}%  [{elapsed:.0f}s]{star}"
            )
            results.append({"name": name, **r})
        except Exception as e:
            print(f"  {name}: ERROR {e}")

    print("\n=== Summary (sorted by neg, then med) ===")
    results.sort(key=lambda x: (x["neg"], -x["med"]))
    for r in results[:20]:
        star = " ***" if r["neg"] <= 5 else ""
        print(f"{r['name']:<40s} neg={r['neg']:3d}/111 med={r['med']:+7.1f}% p10={r['p10']:+7.1f}%{star}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
