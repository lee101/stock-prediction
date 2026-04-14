#!/usr/bin/env python3
"""Evaluate all crypto30 sessaug checkpoints and test ensemble combinations.

Runs after training to find if any new model improves the prod ensemble.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pufferlib_market.hourly_replay import read_mktd, INITIAL_CASH
from pufferlib_market.evaluate_holdout import load_policy, _mask_all_shorts


VAL_BIN = REPO / "pufferlib_market/data/crypto30_daily_val.bin"
SESSAUG_DIR = REPO / "pufferlib_market/checkpoints/crypto30_sessaug"
PROD_ENSEMBLE = [
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s2.pt",
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s19.pt",
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s21.pt",
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s23.pt",
]


def softmax_ensemble_eval(data, policies, S, decision_lag=2, fee_rate=0.001, slippage_bps=5.0):
    """Run softmax-average ensemble on val data, return equity curve."""
    import collections
    from pufferlib_market.hourly_replay import simulate_daily_policy

    T = data.num_timesteps
    pending = collections.deque(maxlen=max(1, decision_lag + 1))

    def policy_fn(obs):
        obs_t = torch.from_numpy(obs.astype(np.float32)).view(1, -1)
        with torch.no_grad():
            probs_raw = []
            for p in policies:
                lg, _ = p(obs_t)
                probs_raw.append(torch.softmax(lg, dim=-1))
            avg = torch.stack(probs_raw).mean(dim=0)
            log_avg = torch.log(avg + 1e-8)
            masked = _mask_all_shorts(log_avg, num_symbols=S, per_symbol_actions=1)
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
    return eq


def metrics(eq):
    ret = eq[-1] / eq[0] - 1
    dr = np.diff(eq) / np.maximum(eq[:-1], 1e-8)
    neg = dr[dr < 0]
    sort = float(np.mean(dr) / np.sqrt(np.mean(neg**2)) * np.sqrt(365)) if len(neg) > 0 else 0
    peak = np.maximum.accumulate(eq)
    dd = float(np.max((peak - eq) / np.maximum(peak, 1e-8)))
    return ret, sort, dd


def main():
    data = read_mktd(str(VAL_BIN))
    S = data.num_symbols
    print(f"data: {data.num_timesteps}d x {S}sym\n")

    # 1. Evaluate prod ensemble baseline
    print("=== Prod ensemble baseline ===")
    prod_policies = []
    for cp in PROD_ENSEMBLE:
        if not cp.exists():
            print(f"  SKIP {cp.name} (missing)")
            continue
        loaded = load_policy(cp, S, features_per_sym=16, device=torch.device("cpu"))
        prod_policies.append(loaded.policy)

    if prod_policies:
        eq = softmax_ensemble_eval(data, prod_policies, S)
        ret, sort, dd = metrics(eq)
        print(f"  Prod 4-model: ret={ret*100:+.2f}% sort={sort:.2f} dd={dd*100:.2f}%\n")
        prod_ret = ret
    else:
        print("  No prod models found!\n")
        prod_ret = -999

    # 2. Evaluate all sessaug checkpoints individually
    print("=== Sessaug individual models ===")
    print(f"{'Tag':<25} {'Ckpt':<10} {'Ret%':>10} {'Sort':>8} {'MaxDD%':>8}")
    print("-" * 65)

    sessaug_results = {}
    for subdir in sorted(SESSAUG_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        for pt_name in ["val_best.pt", "best.pt", "final.pt"]:
            ckpt = subdir / pt_name
            if not ckpt.exists():
                continue
            try:
                loaded = load_policy(ckpt, S, features_per_sym=16, device=torch.device("cpu"))
                policy = loaded.policy
                # Single model eval (just softmax ensemble of 1)
                eq = softmax_ensemble_eval(data, [policy], S)
                ret, sort, dd = metrics(eq)
                tag = f"{subdir.name}/{pt_name.replace('.pt','')}"
                print(f"{tag:<25} {'':>10} {ret*100:>+10.2f}% {sort:>8.2f} {dd*100:>8.2f}%")
                sessaug_results[ckpt] = (policy, ret, sort, dd)
            except Exception as e:
                print(f"  {subdir.name}/{pt_name}: FAIL ({e})")

    if not sessaug_results:
        print("  No sessaug models found yet. Training may still be running.\n")
        return

    # 3. Test greedy addition to prod ensemble
    print(f"\n=== Greedy ensemble addition (starting from prod {len(prod_policies)}-model) ===")
    best_improvement = 0
    best_candidate = None

    candidates = sorted(sessaug_results.items(), key=lambda x: x[1][1], reverse=True)
    for ckpt, (policy, ind_ret, _, _) in candidates[:10]:  # top 10 by individual return
        test_policies = prod_policies + [policy]
        eq = softmax_ensemble_eval(data, test_policies, S)
        ret, sort, dd = metrics(eq)
        improvement = ret - prod_ret
        tag = f"{ckpt.parent.name}/{ckpt.stem}"
        print(f"  +{tag}: ret={ret*100:+.2f}% (delta={improvement*100:+.2f}%) sort={sort:.2f}")
        if improvement > best_improvement:
            best_improvement = improvement
            best_candidate = tag

    if best_improvement > 0:
        print(f"\n  BEST: +{best_candidate} improves ensemble by {best_improvement*100:+.2f}%")
    else:
        print(f"\n  No model improves prod ensemble.")

    # 4. Test all-sessaug ensemble (no prod models)
    print(f"\n=== Sessaug-only ensembles ===")
    all_sessaug_policies = [(ckpt, info[0], info[1]) for ckpt, info in candidates]

    if len(all_sessaug_policies) >= 2:
        # Top-2
        top2 = [p for _, p, _ in all_sessaug_policies[:2]]
        eq = softmax_ensemble_eval(data, top2, S)
        ret, sort, dd = metrics(eq)
        names = [str(c.parent.name) for c, _, _ in all_sessaug_policies[:2]]
        print(f"  Top-2 ({'+'.join(names)}): ret={ret*100:+.2f}% sort={sort:.2f}")

    if len(all_sessaug_policies) >= 4:
        top4 = [p for _, p, _ in all_sessaug_policies[:4]]
        eq = softmax_ensemble_eval(data, top4, S)
        ret, sort, dd = metrics(eq)
        names = [str(c.parent.name) for c, _, _ in all_sessaug_policies[:4]]
        print(f"  Top-4 ({'+'.join(names)}): ret={ret*100:+.2f}% sort={sort:.2f}")

    # 5. Slippage stress test on best
    if best_improvement > 0 and best_candidate:
        print(f"\n=== Slippage stress test: prod + {best_candidate} ===")
        best_ckpt = [c for c, _ in candidates if f"{c.parent.name}/{c.stem}" == best_candidate][0]
        best_policy = sessaug_results[best_ckpt][0]
        test_policies = prod_policies + [best_policy]
        for slip in [0, 5, 10, 20]:
            eq = softmax_ensemble_eval(data, test_policies, S, slippage_bps=slip)
            ret, sort, dd = metrics(eq)
            print(f"  {slip:>4}bp: ret={ret*100:+.2f}% sort={sort:.2f} dd={dd*100:.2f}%")


if __name__ == "__main__":
    main()
