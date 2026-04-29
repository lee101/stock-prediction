#!/usr/bin/env python3
"""Monitor training runs and run exhaustive eval (all 111 windows) when each finishes.

Usage:
    source .venv313/bin/activate
    python scripts/eval_when_done.py \
        --ckpt-root pufferlib_market/checkpoints/stocks12_35m_v2 \
        --val-data pufferlib_market/data/stocks12_daily_val.bin \
        --out pufferlib_market/stocks12_35m_v2_exhaustive.csv \
        --poll-interval 120
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
import sys
import time
from datetime import UTC, datetime
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


def exhaustive_eval(
    ckpt_path: Path,
    val_data,
    eval_steps: int = 90,
    fee: float = 0.001,
    fill_bps: float = 5.0,
    slippage_bps: float = 20.0,
    short_borrow_apr: float = 0.0625,
    decision_lag: int = 2,
    max_hold_bars: int = 6,
    max_leverage: float = 1.0,
) -> dict:
    """Evaluate checkpoint on all possible windows with production-realism defaults."""
    device = torch.device("cpu")
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = payload.get("model", payload)
    arch = _infer_arch(state_dict)
    hs = _infer_hidden_size(state_dict, arch=arch)
    obs_size = next(iter(state_dict.values())).shape[1]
    n_actions = _infer_num_actions(state_dict, fallback=25)

    policy = TradingPolicy(obs_size=obs_size, hidden=hs, num_actions=n_actions).to(device).eval()
    policy.load_state_dict(state_dict, strict=False)
    # Respect stored use_encoder_norm flag if present (for checkpoints from train.py).
    # Old checkpoints don't have this key; infer from missing keys (backward-compat).
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
    all_starts = list(range(T - eval_steps))  # All possible windows
    returns = []
    for start in all_starts:
        w = _slice_window(val_data, start=start, steps=eval_steps)
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
        "med": float(np.median(returns) * 100),
        "p10": float(np.percentile(returns, 10) * 100),
        "p90": float(np.percentile(returns, 90) * 100),
        "worst": float(returns.min() * 100),
        "neg": int((returns < 0).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Monitor + exhaustive-eval training runs as they finish")
    parser.add_argument("--ckpt-root", required=True, help="Root dir with per-seed checkpoint subdirs")
    parser.add_argument("--val-data", required=True, help="stocks12_daily_val.bin path")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--poll-interval", type=int, default=120, help="Polling interval in seconds")
    parser.add_argument("--eval-steps", type=int, default=90)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--fill-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=20.0)
    parser.add_argument("--short-borrow-apr", type=float, default=0.0625)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--max-hold-bars", type=int, default=6)
    parser.add_argument("--max-leverage", type=float, default=1.0)
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
        print(f"eval_when_done: {exc}", file=sys.stderr)
        return 2

    val_data = read_mktd(args.val_data)
    print(f"Val data: {val_data.num_timesteps} timesteps, {val_data.num_symbols} symbols")
    n_windows = val_data.num_timesteps - args.eval_steps
    print(f"All-window exhaustive eval: {n_windows} windows")

    out_path = Path(args.out)
    # Track by (path, mtime) — re-evaluate if file changes (best.pt gets overwritten during training)
    evaluated: dict[str, float] = {}  # path -> mtime_at_eval
    if out_path.exists():
        with open(out_path) as f:
            for _row in csv.DictReader(f):
                # Don't pre-load mtimes — we'll always re-eval if the file changed
                pass
        print(f"Output file exists: {out_path}")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("timestamp,seed_dir,ckpt_type,n_windows,med,p10,p90,worst,neg,checkpoint\n")

    ckpt_root = Path(args.ckpt_root)

    while True:
        seed_dirs = sorted(ckpt_root.iterdir()) if ckpt_root.exists() else []

        for seed_dir in seed_dirs:
            if not seed_dir.is_dir():
                continue
            # Check best.pt, val_best.pt, and best_neg.pt
            for ckpt_name in ["best.pt", "val_best.pt", "best_neg.pt"]:
                ckpt = seed_dir / ckpt_name
                if not ckpt.exists():
                    continue
                key = str(ckpt)
                current_mtime = ckpt.stat().st_mtime
                if evaluated.get(key) == current_mtime:
                    continue  # File hasn't changed since last eval
                # Check if this seed's training might still be running (conservative: skip if log shows recent activity)
                # Just try to eval — worst case we re-eval a partial checkpoint
                try:
                    print(f"  Evaluating {ckpt} ({n_windows} windows)...", flush=True)
                    t0 = time.time()
                    result = exhaustive_eval(ckpt, val_data, eval_steps=args.eval_steps,
                                             fee=fee, fill_bps=fill_bps,
                                             slippage_bps=slippage_bps,
                                             short_borrow_apr=short_borrow_apr,
                                             decision_lag=decision_lag,
                                             max_hold_bars=max_hold_bars,
                                             max_leverage=max_leverage)
                    elapsed = time.time() - t0
                    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
                    row = (
                        f"{ts},{seed_dir.name},{ckpt_name},{result['n']},"
                        f"{result['med']:.2f},{result['p10']:.2f},"
                        f"{result['p90']:.2f},{result['worst']:.2f},"
                        f"{result['neg']},{key}\n"
                    )
                    with open(out_path, "a") as f:
                        f.write(row)
                    evaluated[key] = current_mtime
                    print(
                        f"  {seed_dir.name}/{ckpt_name}: med={result['med']:.1f}% "
                        f"p10={result['p10']:.1f}% neg={result['neg']}/{result['n']} "
                        f"[{elapsed:.0f}s]"
                    )
                except Exception as e:
                    print(f"  ERROR evaluating {ckpt}: {e}")

        print(
            f"[{datetime.now(UTC).strftime('%H:%M:%S')}] "
            f"Waiting {args.poll_interval}s... ({len(evaluated)} done)",
            flush=True,
        )
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())
