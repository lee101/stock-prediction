#!/usr/bin/env python3
"""Build crypto30 ensemble from different training stages.

Key insight: The prod 4-model ensemble uses models at updates 150, 400, 750, 1800.
Diversity in convergence stage is what makes the ensemble work.
This script tests all combinations of checkpoints from different stages.
"""
from __future__ import annotations
import collections
import itertools
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pufferlib_market.hourly_replay import read_mktd, INITIAL_CASH, simulate_daily_policy
from pufferlib_market.evaluate_holdout import load_policy, _mask_all_shorts

VAL_BIN = REPO / "pufferlib_market/data/crypto30_daily_val.bin"
SESSAUG_DIR = REPO / "pufferlib_market/checkpoints/crypto30_sessaug"
PROD_DIR = REPO / "pufferlib_market/checkpoints/crypto30_ensemble"


def softmax_ensemble_eval(data, policies, S, decision_lag=2, fee_rate=0.001, slippage_bps=5.0):
    T = data.num_timesteps
    pending = collections.deque(maxlen=max(1, decision_lag + 1))

    def policy_fn(obs):
        obs_t = torch.from_numpy(obs.astype(np.float32)).view(1, -1)
        with torch.no_grad():
            probs = [torch.softmax(p(obs_t)[0], dim=-1) for p in policies]
            avg = torch.stack(probs).mean(dim=0)
            masked = _mask_all_shorts(torch.log(avg + 1e-8), num_symbols=S, per_symbol_actions=1)
            action_now = int(torch.argmax(masked, dim=-1).item())
        if decision_lag <= 0:
            return action_now
        pending.append(action_now)
        if len(pending) <= decision_lag:
            return 0
        return pending.popleft()

    result = simulate_daily_policy(
        data, policy_fn, max_steps=T - 1,
        fee_rate=fee_rate, slippage_bps=slippage_bps,
        fill_buffer_bps=5.0, max_leverage=1.0,
        periods_per_year=365.0, enable_drawdown_profit_early_exit=False,
    )
    eq = np.array(result.equity_curve) if result.equity_curve is not None and len(result.equity_curve) > 0 else np.ones(T) * INITIAL_CASH
    ret = eq[-1] / eq[0] - 1
    dr = np.diff(eq) / np.maximum(eq[:-1], 1e-8)
    neg = dr[dr < 0]
    sort = float(np.mean(dr) / np.sqrt(np.mean(neg**2)) * np.sqrt(365)) if len(neg) > 0 else 0
    peak = np.maximum.accumulate(eq)
    dd = float(np.max((peak - eq) / np.maximum(peak, 1e-8)))
    return ret, sort, dd


def load_all_checkpoints(data_S):
    """Load all available checkpoints (prod + sessaug)."""
    ckpts = {}

    # Prod ensemble
    for pt in sorted(PROD_DIR.glob("*.pt")):
        try:
            loaded = load_policy(pt, data_S, features_per_sym=16, device=torch.device("cpu"))
            ckpt_data = torch.load(pt, map_location="cpu", weights_only=False)
            update = ckpt_data.get("update", 0)
            name = f"prod_{pt.stem}_u{update}"
            ckpts[name] = loaded.policy
        except Exception as e:
            print(f"  SKIP {pt.name}: {e}")

    # Sessaug checkpoints
    for subdir in sorted(SESSAUG_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        for pt in sorted(subdir.glob("*.pt")):
            if pt.suffix != ".pt":
                continue
            try:
                loaded = load_policy(pt, data_S, features_per_sym=16, device=torch.device("cpu"))
                ckpt_data = torch.load(pt, map_location="cpu", weights_only=False)
                update = ckpt_data.get("update", 0)
                name = f"{subdir.name}_{pt.stem}_u{update}"
                ckpts[name] = loaded.policy
            except Exception:
                continue

    return ckpts


def main():
    data = read_mktd(str(VAL_BIN))
    S = data.num_symbols
    print(f"data: {data.num_timesteps}d x {S}sym\n")

    ckpts = load_all_checkpoints(S)
    print(f"Loaded {len(ckpts)} checkpoints:")
    for name in sorted(ckpts.keys()):
        print(f"  {name}")
    print()

    if len(ckpts) < 2:
        print("Need at least 2 checkpoints for ensemble testing.")
        return

    # Evaluate all individual models
    print("=== Individual models ===")
    print(f"{'Name':<35} {'Ret%':>10} {'Sort':>8} {'MaxDD%':>8}")
    print("-" * 65)
    individual = {}
    for name, policy in sorted(ckpts.items()):
        ret, sort, dd = softmax_ensemble_eval(data, [policy], S)
        print(f"{name:<35} {ret*100:>+10.2f}% {sort:>8.2f} {dd*100:>8.2f}%")
        individual[name] = (ret, sort, dd)

    # Prod baseline
    prod_names = [n for n in ckpts if n.startswith("prod_")]
    if len(prod_names) >= 2:
        prod_policies = [ckpts[n] for n in sorted(prod_names)]
        ret, sort, dd = softmax_ensemble_eval(data, prod_policies, S)
        print(f"\nProd ensemble ({len(prod_names)} models): ret={ret*100:+.2f}% sort={sort:.2f} dd={dd*100:.2f}%")
        prod_ret = ret
    else:
        prod_ret = -999

    # Test all 2, 3, 4-model ensembles (if not too many)
    names = sorted(ckpts.keys())
    if len(names) <= 12:
        for k in [2, 3, 4]:
            print(f"\n=== Top {k}-model ensembles ===")
            combos = list(itertools.combinations(names, k))
            results = []
            for combo in combos:
                policies = [ckpts[n] for n in combo]
                ret, sort, dd = softmax_ensemble_eval(data, policies, S)
                results.append((ret, sort, dd, combo))
            results.sort(key=lambda x: x[0], reverse=True)
            for ret, sort, dd, combo in results[:5]:
                label = "+".join(c.split("_u")[0] for c in combo)
                delta = f"(delta={100*(ret-prod_ret):+.2f}%)" if prod_ret > -998 else ""
                print(f"  {label}: ret={ret*100:+.2f}% sort={sort:.2f} dd={dd*100:.2f}% {delta}")
    else:
        # Too many for exhaustive -- use greedy
        print(f"\n=== Greedy ensemble building (too many for exhaustive) ===")
        # Start with best individual
        best_name = max(individual, key=lambda n: individual[n][0])
        ensemble = [best_name]
        print(f"  Start: {best_name} ret={individual[best_name][0]*100:+.2f}%")

        for step in range(min(7, len(names) - 1)):
            best_add = None
            best_ret = -999
            for name in names:
                if name in ensemble:
                    continue
                test = ensemble + [name]
                policies = [ckpts[n] for n in test]
                ret, sort, dd = softmax_ensemble_eval(data, policies, S)
                if ret > best_ret:
                    best_ret = ret
                    best_add = name
                    best_sort = sort
                    best_dd = dd

            if best_add:
                ensemble.append(best_add)
                delta = f"(delta={100*(best_ret-prod_ret):+.2f}%)" if prod_ret > -998 else ""
                print(f"  +{best_add}: ret={best_ret*100:+.2f}% sort={best_sort:.2f} dd={best_dd*100:.2f}% {delta}")
            else:
                break


if __name__ == "__main__":
    main()
