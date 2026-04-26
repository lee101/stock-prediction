"""Measure D_s16 softmax entropy across a 120d window.

Question: does D_s16 have dispersable top-K mass (high entropy → portfolio packing
could diversify) or is the softmax peaked on single picks (low entropy → packing
won't help)?

For each decision step we:
- Run the policy forward
- Apply the same long-only mask as evaluate_holdout
- Compute per-symbol mass: sum softmax prob over per_symbol_actions bins for each sym
- Report: top-1 prob, top-2/top-1 ratio, entropy over sym-mass, #syms with >5% mass
"""
from __future__ import annotations

import argparse
import numpy as np
import torch

from pufferlib_market.evaluate_holdout import load_policy, _mask_all_shorts
from pufferlib_market.hourly_replay import read_mktd, simulate_daily_policy


def _apply_long_only_mask(logits: torch.Tensor, num_symbols: int, per_symbol_actions: int) -> torch.Tensor:
    return _mask_all_shorts(logits, num_symbols=num_symbols, per_symbol_actions=per_symbol_actions)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="pufferlib_market/prod_ensemble_screened32/D_s16.pt")
    p.add_argument("--data-path", default="pufferlib_market/data/screened32_full_val.bin")
    p.add_argument("--eval-hours", type=int, default=120)
    p.add_argument("--start", type=int, default=0, help="Start timestep offset")
    args = p.parse_args()

    data = read_mktd(args.data_path)
    S = data.num_symbols
    feats = int(data.features.shape[2])
    loaded = load_policy(args.checkpoint, S, features_per_sym=feats, device=torch.device("cpu"))
    policy = loaded.policy
    per_sym = int(loaded.action_allocation_bins) * int(loaded.action_level_bins)

    records: list[dict] = []

    def policy_fn(obs: np.ndarray) -> int:
        obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).view(1, -1)
        with torch.no_grad():
            logits, _ = policy(obs_t)
            logits = _apply_long_only_mask(logits, num_symbols=S, per_symbol_actions=per_sym)
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        # Action layout: [hold, then S * (2 * per_sym_actions) long/short bins]
        # hold = idx 0, longs = idx 1 .. 1 + S*per_sym, shorts masked to -inf
        hold_prob = float(probs[0])
        long_start = 1
        long_end = 1 + S * per_sym
        long_probs = probs[long_start:long_end].reshape(S, per_sym)
        sym_mass = long_probs.sum(axis=1)  # per-symbol long mass
        action = int(np.argmax(probs))
        total_long = float(sym_mass.sum())
        if total_long > 1e-8:
            p_sym = sym_mass / total_long
            ent = float(-(p_sym * np.log(p_sym + 1e-12)).sum())
        else:
            ent = 0.0
        sorted_mass = np.sort(sym_mass)[::-1]
        top1 = float(sorted_mass[0])
        top2 = float(sorted_mass[1]) if S > 1 else 0.0
        n_above_5pct = int((sym_mass > 0.05).sum())
        records.append({
            "hold_prob": hold_prob,
            "entropy_over_syms": ent,
            "top1_mass": top1,
            "top2_mass": top2,
            "top2_over_top1": top2 / max(top1, 1e-9),
            "n_syms_above_5pct": n_above_5pct,
            "total_long_mass": total_long,
        })
        return action

    _ = simulate_daily_policy(
        data,
        policy_fn,
        max_steps=args.eval_hours,
        fee_rate=0.001,
        slippage_bps=5.0,
        fill_buffer_bps=5.0,
        max_leverage=3.0,
        action_allocation_bins=loaded.action_allocation_bins,
        action_level_bins=loaded.action_level_bins,
        action_max_offset_bps=loaded.action_max_offset_bps,
        enable_drawdown_profit_early_exit=False,
        death_spiral_tolerance_bps=50.0,
        death_spiral_overnight_tolerance_bps=500.0,
        death_spiral_stale_after_bars=8,
    )

    arr = {k: np.array([r[k] for r in records]) for k in records[0].keys()}
    print(f"Steps observed: {len(records)} (over {args.eval_hours} max)")
    print(f"num_symbols={S}, per_sym_actions={per_sym}")
    print()
    print(f"{'metric':30s}  {'mean':>10s}  {'median':>10s}  {'p10':>10s}  {'p90':>10s}  {'max':>10s}")
    for k, v in arr.items():
        print(f"{k:30s}  {v.mean():10.4f}  {np.median(v):10.4f}  {np.percentile(v,10):10.4f}  {np.percentile(v,90):10.4f}  {v.max():10.4f}")
    print()
    # Interpretability: uniform over S syms has entropy log(32) = 3.47
    max_ent = float(np.log(S))
    print(f"Max possible entropy over {S} syms: {max_ent:.3f}")
    print(f"Median entropy fraction: {float(np.median(arr['entropy_over_syms']))/max_ent:.3f}")
    # Diversification potential: syms with >5% mass
    print(f"Steps where >1 sym has >5% mass: {int((arr['n_syms_above_5pct']>1).sum())}/{len(records)}")
    print(f"Steps where top2/top1 > 0.5:     {int((arr['top2_over_top1']>0.5).sum())}/{len(records)}")


if __name__ == "__main__":
    main()
