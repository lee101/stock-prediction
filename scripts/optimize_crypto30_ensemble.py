#!/usr/bin/env python3
"""Greedy forward selection to find optimal crypto30 softmax ensemble.

Starts with the best single model (or current prod 4-model), greedily adds
models that improve the ensemble, stopping when no addition helps.

Usage:
    python scripts/optimize_crypto30_ensemble.py
    python scripts/optimize_crypto30_ensemble.py --start-from-prod
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pufferlib_market.hourly_replay import read_mktd, MktdData, INITIAL_CASH
from scripts.meta_strategy_backtest import simulate_ensemble, ModelTrace

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("opt_ens")

VAL_BIN = "pufferlib_market/data/crypto30_daily_val.bin"

PROD_ENSEMBLE = [
    "pufferlib_market/checkpoints/crypto30_ensemble/s2.pt",
    "pufferlib_market/checkpoints/crypto30_ensemble/s19.pt",
    "pufferlib_market/checkpoints/crypto30_ensemble/s21.pt",
    "pufferlib_market/checkpoints/crypto30_ensemble/s23.pt",
]

CHECKPOINT_DIRS = [
    "pufferlib_market/checkpoints/crypto30_ensemble",
    "pufferlib_market/checkpoints/crypto30_daily",
    "pufferlib_market/checkpoints/crypto30_cpu",
]


def discover_all(dirs: list[str]) -> list[Path]:
    found = []
    for d in dirs:
        dp = REPO / d
        if not dp.exists():
            continue
        for pt in sorted(dp.glob("*.pt")):
            found.append(pt)
        for sub in sorted(dp.iterdir()):
            if sub.is_dir():
                for name in ["best.pt", "val_best.pt"]:
                    p = sub / name
                    if p.exists():
                        found.append(p)
                        break
    seen = set()
    unique = []
    for p in found:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def label(p: Path) -> str:
    parts = p.parts
    for i, part in enumerate(parts):
        if "crypto30_ensemble" in part:
            return "ens_" + p.stem
        if "crypto30_daily" in part and i + 1 < len(parts):
            return "daily_" + parts[i + 1]
        if "crypto30_cpu" in part and i + 1 < len(parts):
            return "cpu_" + parts[i + 1]
    return p.stem


def eval_ensemble(data, checkpoint_paths, fee_rate=0.001, slippage_bps=5.0,
                  fill_buffer_bps=5.0, decision_lag=2, device="cpu"):
    """Evaluate a softmax ensemble and return (return%, sortino, maxdd)."""
    trace = simulate_ensemble(
        data, checkpoint_paths,
        fee_rate=fee_rate, slippage_bps=slippage_bps,
        fill_buffer_bps=fill_buffer_bps, decision_lag=decision_lag,
        device=device,
    )
    eq = trace.equity_curve
    ret = eq[-1] / eq[0] - 1
    dr = np.diff(eq) / np.maximum(eq[:-1], 1e-8)
    neg = dr[dr < 0]
    sort = float(np.mean(dr) / np.sqrt(np.mean(neg**2)) * np.sqrt(365)) if len(neg) > 0 else 0
    peak = np.maximum.accumulate(eq)
    dd = float(np.max((peak - eq) / np.maximum(peak, 1e-8)))
    return ret, sort, dd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-bin", default=VAL_BIN)
    parser.add_argument("--start-from-prod", action="store_true")
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-size", type=int, default=12)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    data = read_mktd(str(REPO / args.data_bin))
    log.info("data: %d days x %d symbols", data.num_timesteps, data.num_symbols)

    all_cps = discover_all(CHECKPOINT_DIRS)
    log.info("discovered %d checkpoints", len(all_cps))

    kw = dict(fee_rate=args.fee_rate, slippage_bps=args.slippage_bps,
              fill_buffer_bps=5.0, decision_lag=args.decision_lag, device=args.device)

    # Phase 1: Evaluate current prod ensemble
    prod_paths = [REPO / p for p in PROD_ENSEMBLE]
    prod_ret, prod_sort, prod_dd = eval_ensemble(data, prod_paths, **kw)
    print(f"\nProd 4-model ensemble: ret={prod_ret*100:+.2f}% sort={prod_sort:.2f} dd={prod_dd*100:.2f}%")

    # Phase 2: Evaluate each individual model as singleton ensemble
    print(f"\nIndividual model evaluation:")
    print(f"{'Model':<25} {'Ret%':>10} {'Sort':>10} {'MaxDD%':>10}")
    print("-" * 57)
    individual_results = []
    for cp in all_cps:
        try:
            ret, sort, dd = eval_ensemble(data, [cp], **kw)
            lbl = label(cp)
            print(f"{lbl:<25} {ret*100:>+10.2f}% {sort:>10.2f} {dd*100:>10.2f}%")
            individual_results.append((cp, lbl, ret, sort, dd))
        except Exception as e:
            log.warning("  %s failed: %s", label(cp), e)

    # Phase 3: Greedy forward selection
    if args.start_from_prod:
        current = list(prod_paths)
        current_ret = prod_ret
        current_sort = prod_sort
        log.info("\nStarting from prod ensemble (%d models, ret=%+.2f%%)", len(current), current_ret * 100)
    else:
        best_single = max(individual_results, key=lambda x: x[2])
        current = [best_single[0]]
        current_ret = best_single[2]
        current_sort = best_single[3]
        log.info("\nStarting from best single: %s (ret=%+.2f%%)", best_single[1], current_ret * 100)

    remaining = [cp for cp in all_cps if cp.resolve() not in {c.resolve() for c in current}]

    print(f"\n{'Step':<6} {'Added':<25} {'Size':>5} {'Ret%':>10} {'Sort':>10} {'MaxDD%':>10} {'Delta':>10}")
    print("-" * 80)
    print(f"{'init':<6} {'':>25} {len(current):>5} {current_ret*100:>+10.2f}% {current_sort:>10.2f} {'':>10} {'':>10}")

    history = [{"size": len(current), "models": [label(c) for c in current],
                "return": current_ret, "sortino": current_sort}]

    step = 0
    while remaining and len(current) < args.max_size:
        step += 1
        best_add = None
        best_add_ret = current_ret
        best_add_sort = 0
        best_add_dd = 0

        for cp in remaining:
            try:
                candidate = current + [cp]
                ret, sort, dd = eval_ensemble(data, candidate, **kw)
                if ret > best_add_ret:
                    best_add = cp
                    best_add_ret = ret
                    best_add_sort = sort
                    best_add_dd = dd
            except Exception:
                continue

        if best_add is None:
            print(f"{'stop':<6} no improvement found")
            break

        delta = best_add_ret - current_ret
        current.append(best_add)
        remaining = [cp for cp in remaining if cp.resolve() != best_add.resolve()]
        current_ret = best_add_ret
        current_sort = best_add_sort

        lbl = label(best_add)
        print(f"{step:<6} {lbl:<25} {len(current):>5} {current_ret*100:>+10.2f}% {current_sort:>10.2f} {best_add_dd*100:>10.2f}% {delta*100:>+10.2f}%")

        history.append({"size": len(current), "added": lbl,
                        "models": [label(c) for c in current],
                        "return": current_ret, "sortino": current_sort,
                        "delta": delta})

    # Phase 4: Slippage sweep on best ensemble
    print(f"\nBest ensemble ({len(current)} models):")
    for c in current:
        print(f"  {label(c)}: {c}")

    print(f"\nSlippage sweep:")
    print(f"{'Slip':>6} {'Ret%':>10} {'Sort':>10} {'MaxDD%':>10}")
    print("-" * 40)
    for slip in [0, 5, 10, 20]:
        ret, sort, dd = eval_ensemble(data, current,
            fee_rate=args.fee_rate, slippage_bps=slip,
            fill_buffer_bps=5.0, decision_lag=args.decision_lag, device=args.device)
        print(f"{slip:>4}bp {ret*100:>+10.2f}% {sort:>10.2f} {dd*100:>10.2f}%")

    # Compare vs prod
    print(f"\nProd ensemble:     ret={prod_ret*100:+.2f}%")
    print(f"Optimized ensemble: ret={current_ret*100:+.2f}% ({len(current)} models)")
    if current_ret > prod_ret:
        print(f"IMPROVEMENT: +{(current_ret-prod_ret)*100:.2f}%")
    else:
        print(f"No improvement over prod")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "prod": {"return": prod_ret, "sortino": prod_sort, "maxdd": prod_dd},
            "best": {"models": [str(c) for c in current],
                     "labels": [label(c) for c in current],
                     "return": current_ret, "sortino": current_sort},
            "history": history,
            "individual": {r[1]: {"return": r[2], "sortino": r[3], "maxdd": r[4]}
                           for r in individual_results},
        }, indent=2))
        log.info("saved %s", args.out)


if __name__ == "__main__":
    main()
