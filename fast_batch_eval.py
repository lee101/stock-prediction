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
import time
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def eval_checkpoint_fast(ckpt_path: str, val_data, eval_steps: int = 90,
                         fee: float = 0.001, fill_bps: float = 5.0,
                         device: torch.device = torch.device("cuda")) -> dict:
    """GPU-accelerated exhaustive eval."""
    from pufferlib_market.evaluate_holdout import (
        _infer_arch, _infer_hidden_size, _infer_num_actions, _slice_window, TradingPolicy
    )
    from pufferlib_market.hourly_replay import simulate_daily_policy

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = payload.get("model", payload)
    arch = _infer_arch(state_dict)
    hs = _infer_hidden_size(state_dict, arch=arch)
    obs_size = list(state_dict.values())[0].shape[1]
    n_actions = _infer_num_actions(state_dict, fallback=25)

    policy = TradingPolicy(obs_size=obs_size, hidden=hs, num_actions=n_actions).to(device).eval()
    policy.load_state_dict(state_dict, strict=False)
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
    starts = list(range(T - eval_steps))
    returns = []
    for start in starts:
        w = _slice_window(val_data, start=int(start), steps=eval_steps)
        r = simulate_daily_policy(w, policy_fn, max_steps=eval_steps,
                                  fill_buffer_bps=fill_bps, fee_rate=fee,
                                  enable_drawdown_profit_early_exit=False)
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
    parser.add_argument("--names", type=str, default="",
                        help="Comma-separated subdirectory names to eval (default: all)")
    args = parser.parse_args()

    from pufferlib_market.hourly_replay import read_mktd
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
                                     fee=args.fee, fill_bps=args.fill_bps)
            elapsed = time.time() - t0
            star = " ***" if r["neg"] <= 5 else ""
            print(f"{name:<40s} {r['neg']:>5d}/111 {r['med']:>+7.1f}% {r['p10']:>+7.1f}% {r['p90']:>+7.1f}% {r['worst']:>+7.1f}%  [{elapsed:.0f}s]{star}")
            results.append({"name": name, **r})
        except Exception as e:
            print(f"  {name}: ERROR {e}")

    print("\n=== Summary (sorted by neg, then med) ===")
    results.sort(key=lambda x: (x["neg"], -x["med"]))
    for r in results[:20]:
        star = " ***" if r["neg"] <= 5 else ""
        print(f"{r['name']:<40s} neg={r['neg']:3d}/111 med={r['med']:+7.1f}% p10={r['p10']:+7.1f}%{star}")


if __name__ == "__main__":
    main()
