#!/usr/bin/env python3
"""Evaluate crypto30_cpu checkpoints for ensemble improvement over prod."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pufferlib_market.hourly_replay import read_mktd, INITIAL_CASH
from pufferlib_market.evaluate_holdout import load_policy, _mask_all_shorts
from scripts.eval_crypto30_sessaug import softmax_ensemble_eval, metrics

VAL_BIN = REPO / "pufferlib_market/data/crypto30_daily_val.bin"
CPU_DIR = REPO / "pufferlib_market/checkpoints/crypto30_cpu"
PROD_ENSEMBLE = [
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s2.pt",
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s19.pt",
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s21.pt",
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s23.pt",
]


def main():
    data = read_mktd(str(VAL_BIN))
    S = data.num_symbols
    print(f"data: {data.num_timesteps}d x {S}sym\n")

    # Load prod
    prod_policies = []
    for cp in PROD_ENSEMBLE:
        if cp.exists():
            loaded = load_policy(cp, S, features_per_sym=16, device=torch.device("cpu"))
            prod_policies.append(loaded.policy)
    eq = softmax_ensemble_eval(data, prod_policies, S)
    prod_ret, prod_sort, prod_dd = metrics(eq)
    print(f"Prod 4-model: ret={prod_ret*100:+.2f}% sort={prod_sort:.2f} dd={prod_dd*100:.2f}%\n")

    # Eval all cpu checkpoints
    print(f"{'Tag':<30} {'Ret%':>10} {'Sort':>8} {'MaxDD%':>8}")
    print("-" * 60)

    results = {}
    for subdir in sorted(CPU_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        for pt in sorted(subdir.glob("*.pt")):
            try:
                loaded = load_policy(pt, S, features_per_sym=16, device=torch.device("cpu"))
                policy = loaded.policy
                eq = softmax_ensemble_eval(data, [policy], S)
                ret, sort, dd = metrics(eq)
                tag = f"{subdir.name}/{pt.stem}"
                print(f"{tag:<30} {ret*100:>+10.2f}% {sort:>8.2f} {dd*100:>8.2f}%")
                results[tag] = (policy, ret, sort, dd)
            except Exception as e:
                print(f"  {subdir.name}/{pt.stem}: FAIL ({e})")

    if not results:
        print("No models found")
        return

    # Greedy ensemble addition to prod
    print(f"\n=== Greedy addition to prod ensemble ===")
    candidates = sorted(results.items(), key=lambda x: x[1][1], reverse=True)
    best_improvement = 0
    best_candidate = None
    for tag, (policy, ind_ret, _, _) in candidates[:15]:
        test_policies = prod_policies + [policy]
        eq = softmax_ensemble_eval(data, test_policies, S)
        ret, sort, dd = metrics(eq)
        improvement = ret - prod_ret
        marker = " ***" if improvement > 0 else ""
        print(f"  +{tag}: ret={ret*100:+.2f}% (delta={improvement*100:+.2f}%) sort={sort:.2f}{marker}")
        if improvement > best_improvement:
            best_improvement = improvement
            best_candidate = tag

    if best_improvement > 0:
        print(f"\nBEST: +{best_candidate} improves by {best_improvement*100:+.2f}%")
    else:
        print(f"\nNo model improves prod ensemble.")

    # Try replacing each prod model with best cpu candidates
    print(f"\n=== Replacement test (swap each prod model) ===")
    prod_names = ["s2", "s19", "s21", "s23"]
    for i, pname in enumerate(prod_names):
        best_swap_ret = prod_ret
        best_swap_tag = None
        for tag, (policy, _, _, _) in candidates[:5]:
            test_policies = prod_policies[:i] + [policy] + prod_policies[i+1:]
            eq = softmax_ensemble_eval(data, test_policies, S)
            ret, _, _ = metrics(eq)
            if ret > best_swap_ret:
                best_swap_ret = ret
                best_swap_tag = tag
        if best_swap_tag:
            print(f"  Replace {pname}: +{best_swap_tag} -> ret={best_swap_ret*100:+.2f}% (delta={100*(best_swap_ret-prod_ret):+.2f}%)")
        else:
            print(f"  Replace {pname}: no improvement")


if __name__ == "__main__":
    main()
