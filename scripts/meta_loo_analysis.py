#!/usr/bin/env python3
"""Leave-one-out analysis for meta-selector model pool."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--ensemble-dir", required=True)
    parser.add_argument("--eval-days", type=int, default=95)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--lookback", type=int, default=3)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    ens_dir = Path(args.ensemble_dir)
    all_models = sorted(ens_dir.glob("*.pt"))
    print(f"Pool: {len(all_models)} models")

    from scripts.meta_strategy_backtest import (
        build_mktd_from_csvs,
        run_meta_portfolio,
        simulate_single_model,
        DEFAULT_SYMBOLS,
    )
    from pufferlib_market.hourly_replay import MktdData

    data_dir = Path(args.data_dir)
    data, dates = build_mktd_from_csvs(data_dir, DEFAULT_SYMBOLS)

    if args.eval_days and args.eval_days < data.num_timesteps:
        start = data.num_timesteps - args.eval_days
        data = MktdData(
            version=data.version,
            symbols=list(data.symbols),
            features=data.features[start:].copy(),
            prices=data.prices[start:].copy(),
            tradable=data.tradable[start:].copy() if data.tradable is not None else None,
        )
        dates = dates[start:]
    print(f"Data: {data.num_symbols} symbols, {data.num_timesteps} days")

    def run_pool(model_paths):
        traces = []
        for mp in model_paths:
            tr = simulate_single_model(data, mp, device=args.device,
                                       slippage_bps=args.slippage_bps)
            traces.append(tr)
        result = run_meta_portfolio(
            data, traces,
            top_k=args.top_k,
            lookback=args.lookback,
            warmup=args.lookback,
            fee_rate=0.001,
            slippage_bps=args.slippage_bps,
            selector="momentum",
        )
        return result.total_return, result.sortino, result.max_drawdown

    full_ret, full_sort, full_dd = run_pool(all_models)
    print(f"\nFull pool ({len(all_models)}m): ret={full_ret:+.2%} sort={full_sort:.2f} dd={full_dd:.2%}")
    print(f"\n{'Model':<30s} {'Drop Ret':>10s} {'Drop Sort':>10s} {'Delta Ret':>10s} {'Delta Sort':>10s} {'Impact':>8s}")
    print("-" * 80)

    results = []
    for i, model in enumerate(all_models):
        pool = [m for j, m in enumerate(all_models) if j != i]
        ret, sort, dd = run_pool(pool)
        delta_ret = ret - full_ret
        delta_sort = sort - full_sort
        impact = "HELPS" if delta_ret < 0 else "HURTS"
        print(f"{model.stem:<30s} {ret:>+10.2%} {sort:>10.2f} {delta_ret:>+10.2%} {delta_sort:>+10.2f} {impact:>8s}")
        results.append({
            "model": model.stem,
            "drop_return": ret,
            "drop_sortino": sort,
            "drop_maxdd": dd,
            "delta_return": delta_ret,
            "delta_sortino": delta_sort,
        })

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "full_pool": {"return": full_ret, "sortino": full_sort, "maxdd": full_dd},
            "loo_results": results,
        }, indent=2))
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
