#!/usr/bin/env python3
"""Deeper analysis of crypto30 ensemble: subset testing + weighted softmax."""
from __future__ import annotations
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pufferlib_market.hourly_replay import read_mktd, MktdData, INITIAL_CASH
from scripts.meta_strategy_backtest import simulate_ensemble, ModelTrace

VAL_BIN = REPO / "pufferlib_market/data/crypto30_daily_val.bin"

PROD = [
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s2.pt",
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s19.pt",
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s21.pt",
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s23.pt",
]

LABELS = ["s2", "s19", "s21", "s23"]


def metrics(eq):
    ret = eq[-1] / eq[0] - 1
    dr = np.diff(eq) / np.maximum(eq[:-1], 1e-8)
    neg = dr[dr < 0]
    sort = float(np.mean(dr) / np.sqrt(np.mean(neg**2)) * np.sqrt(365)) if len(neg) > 0 else 0
    peak = np.maximum.accumulate(eq)
    dd = float(np.max((peak - eq) / np.maximum(peak, 1e-8)))
    return ret, sort, dd


def eval_ens(data, paths, **kw):
    tr = simulate_ensemble(data, paths, **kw)
    return metrics(tr.equity_curve)


def main():
    data = read_mktd(str(VAL_BIN))
    print(f"data: {data.num_timesteps}d x {data.num_symbols}sym\n")

    kw = dict(fee_rate=0.001, slippage_bps=5.0, fill_buffer_bps=5.0, decision_lag=2, device="cpu")

    # Full ensemble
    ret4, sort4, dd4 = eval_ens(data, PROD, **kw)
    print(f"4-model: ret={ret4*100:+.2f}% sort={sort4:.2f} dd={dd4*100:.2f}%\n")

    # All 3-model subsets (leave-one-out)
    print("Leave-one-out (3-model subsets):")
    print(f"{'Removed':<10} {'Ret%':>10} {'Sort':>10} {'MaxDD%':>10} {'Delta':>10}")
    print("-" * 55)
    for i in range(4):
        subset = [p for j, p in enumerate(PROD) if j != i]
        ret, sort, dd = eval_ens(data, subset, **kw)
        delta = ret - ret4
        print(f"{LABELS[i]:<10} {ret*100:>+10.2f}% {sort:>10.2f} {dd*100:>10.2f}% {delta*100:>+10.2f}%")

    # All 2-model subsets
    print(f"\n2-model subsets:")
    print(f"{'Pair':<15} {'Ret%':>10} {'Sort':>10} {'MaxDD%':>10}")
    print("-" * 50)
    for combo in itertools.combinations(range(4), 2):
        subset = [PROD[i] for i in combo]
        lbl = "+".join(LABELS[i] for i in combo)
        ret, sort, dd = eval_ens(data, subset, **kw)
        print(f"{lbl:<15} {ret*100:>+10.2f}% {sort:>10.2f} {dd*100:>10.2f}%")

    # Slippage stress test on prod ensemble
    print(f"\nSlippage stress (4-model):")
    for slip in [0, 5, 10, 15, 20, 30, 50]:
        kw2 = dict(kw, slippage_bps=slip)
        ret, sort, dd = eval_ens(data, PROD, **kw2)
        print(f"  {slip:>3}bp: ret={ret*100:>+8.2f}% sort={sort:>7.2f} dd={dd*100:>7.2f}%")

    # Decision lag sensitivity
    print(f"\nDecision lag sensitivity (4-model, 5bps):")
    for lag in [0, 1, 2, 3]:
        kw2 = dict(kw, decision_lag=lag)
        ret, sort, dd = eval_ens(data, PROD, **kw2)
        print(f"  lag={lag}: ret={ret*100:>+8.2f}% sort={sort:>7.2f} dd={dd*100:>7.2f}%")


if __name__ == "__main__":
    main()
