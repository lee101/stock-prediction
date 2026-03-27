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
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def exhaustive_eval(ckpt_path: Path, val_data, eval_steps: int = 90, fee: float = 0.001, fill_bps: float = 5.0) -> dict:
    """Evaluate checkpoint on all possible 90-day windows (exhaustive = ground truth)."""
    from pufferlib_market.evaluate_holdout import (
        _infer_arch, _infer_hidden_size, _infer_num_actions, _slice_window, TradingPolicy
    )
    from pufferlib_market.hourly_replay import simulate_daily_policy

    device = torch.device("cpu")
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = payload.get("model", payload)
    arch = _infer_arch(state_dict)
    hs = _infer_hidden_size(state_dict, arch=arch)
    obs_size = list(state_dict.values())[0].shape[1]
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

    def policy_fn(obs: np.ndarray) -> int:
        obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).to(device).view(1, -1)
        with torch.inference_mode():
            logits, _ = policy(obs_t)
        return int(torch.argmax(logits, dim=-1).item())

    T = val_data.num_timesteps
    all_starts = list(range(T - eval_steps))  # All possible windows
    returns = []
    for start in all_starts:
        w = _slice_window(val_data, start=start, steps=eval_steps)
        r = simulate_daily_policy(w, policy_fn, max_steps=eval_steps,
                                  fill_buffer_bps=fill_bps, fee_rate=fee,
                                  enable_drawdown_profit_early_exit=False)
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
    args = parser.parse_args()

    from pufferlib_market.hourly_replay import read_mktd
    val_data = read_mktd(args.val_data)
    print(f"Val data: {val_data.num_timesteps} timesteps, {val_data.num_symbols} symbols")
    n_windows = val_data.num_timesteps - args.eval_steps
    print(f"All-window exhaustive eval: {n_windows} windows")

    out_path = Path(args.out)
    # Track by (path, mtime) — re-evaluate if file changes (best.pt gets overwritten during training)
    evaluated: dict[str, float] = {}  # path -> mtime_at_eval
    if out_path.exists():
        with open(out_path) as f:
            for row in csv.DictReader(f):
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
            # Check both best.pt and val_best.pt
            for ckpt_name in ["best.pt", "val_best.pt"]:
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
                                            fee=args.fee, fill_bps=args.fill_bps)
                    elapsed = time.time() - t0
                    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    row = f"{ts},{seed_dir.name},{ckpt_name},{result['n']},{result['med']:.2f},{result['p10']:.2f},{result['p90']:.2f},{result['worst']:.2f},{result['neg']},{key}\n"
                    with open(out_path, "a") as f:
                        f.write(row)
                    evaluated[key] = current_mtime
                    print(f"  {seed_dir.name}/{ckpt_name}: med={result['med']:.1f}% p10={result['p10']:.1f}% neg={result['neg']}/{result['n']} [{elapsed:.0f}s]")
                except Exception as e:
                    print(f"  ERROR evaluating {ckpt}: {e}")

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting {args.poll_interval}s... ({len(evaluated)} done)", flush=True)
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
