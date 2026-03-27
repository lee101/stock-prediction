#!/usr/bin/env python3
"""Evaluate all stocks12 daily PPO seeds with the 50-window holdout.

Run after training finishes:
    source .venv313/bin/activate
    python scripts/eval_stocks12_seeds.py --checkpoint-root pufferlib_market/checkpoints/stocks12_new_seeds
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

import src.market_sim_early_exit as _mse
def _no_early_exit(*a, **k):
    return _mse.EarlyExitDecision(False, 0.0, 0.0, 0.0)
_mse.evaluate_drawdown_vs_profit_early_exit = _no_early_exit

from pufferlib_market.hourly_replay import read_mktd, simulate_daily_policy
from pufferlib_market.evaluate_holdout import _slice_window, TradingPolicy, _infer_arch, _infer_hidden_size, _infer_num_actions

import torch


def eval_checkpoint(ckpt_path: Path, data, n_windows=50, eval_steps=90, seed=42, fee=0.001, fill_bps=5.0) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = payload.get("model", payload)
    arch = _infer_arch(state_dict)
    hs = _infer_hidden_size(state_dict, arch=arch)
    obs_size = list(state_dict.values())[0].shape[1]
    n_actions = _infer_num_actions(state_dict, fallback=25)

    policy = TradingPolicy(
        obs_size=obs_size,
        hidden=hs,
        num_actions=n_actions,
    ).to(device).eval()
    policy.load_state_dict(state_dict, strict=False)

    def policy_fn(obs: np.ndarray) -> int:
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
        result = simulate_daily_policy(
            window, policy_fn,
            max_steps=eval_steps,
            fee_rate=fee, fill_buffer_bps=fill_bps,
            max_leverage=1.0, enable_drawdown_profit_early_exit=False,
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
    p.add_argument("--out", default="")
    args = p.parse_args()

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
            res = eval_checkpoint(ckpt, data, n_windows=args.n_windows, eval_steps=args.eval_steps)
            results.append({**res, "name": d.name})
            neg_str = "✓ 0/50 neg" if res["neg"] == 0 else f"✗ {res['neg']}/50 neg"
            print(f"  {d.name:30s} med={res['med']:+.1f}%  p10={res['p10']:+.1f}%  worst={res['worst']:+.1f}%  {neg_str}")
        except Exception as e:
            print(f"  {d.name}: ERROR {e}")

    results.sort(key=lambda r: r["p10"], reverse=True)
    print(f"\n=== RANKED BY P10 ===")
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


if __name__ == "__main__":
    main()
