#!/usr/bin/env python3
"""Meta-strategy backtest for Crypto30 Daily PPO models.

Reuses the stock meta-selector infrastructure (simulate_single_model,
run_meta_portfolio) but loads from MKTD binary files instead of CSV.

Usage:
    python scripts/meta_strategy_crypto30_backtest.py
    python scripts/meta_strategy_crypto30_backtest.py --top-k 1 --lookback 3
    python scripts/meta_strategy_crypto30_backtest.py --sweep
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
from scripts.meta_strategy_backtest import (
    simulate_single_model,
    simulate_ensemble,
    run_meta_portfolio,
    ModelTrace,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("meta_crypto30")

DEFAULT_VAL_BIN = "pufferlib_market/data/crypto30_daily_val.bin"

CRYPTO30_CHECKPOINT_DIRS = [
    "pufferlib_market/checkpoints/crypto30_ensemble",
    "pufferlib_market/checkpoints/crypto30_daily",
    "pufferlib_market/checkpoints/crypto30_cpu",
]


def discover_checkpoints(dirs: list[str], prefer: str = "best.pt") -> list[Path]:
    """Find all unique checkpoints across directories."""
    found = []
    for d in dirs:
        dp = REPO / d
        if not dp.exists():
            continue
        # Direct .pt files in the directory
        for pt in sorted(dp.glob("*.pt")):
            found.append(pt)
        # Subdirectories with best.pt
        for sub in sorted(dp.iterdir()):
            if sub.is_dir():
                pref = sub / prefer
                if pref.exists():
                    found.append(pref)
    # Dedupe by resolved path
    seen = set()
    unique = []
    for p in found:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def label(p: Path) -> str:
    """Short label for a checkpoint path."""
    parts = p.parts
    # crypto30_ensemble/s2.pt -> ens_s2
    # crypto30_daily/s3/best.pt -> daily_s3
    # crypto30_cpu/wd005_s1/best.pt -> cpu_wd005_s1
    for i, part in enumerate(parts):
        if "crypto30_ensemble" in part:
            return "ens_" + p.stem
        if "crypto30_daily" in part and i + 1 < len(parts):
            return "daily_" + parts[i + 1]
        if "crypto30_cpu" in part and i + 1 < len(parts):
            return "cpu_" + parts[i + 1]
    return p.stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-bin", default=DEFAULT_VAL_BIN)
    parser.add_argument("--checkpoint-dirs", nargs="+", default=CRYPTO30_CHECKPOINT_DIRS)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--lookback", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sweep", action="store_true", help="Sweep lookback and top-k")
    parser.add_argument("--slippage-sweep", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--ensemble-only", nargs="+", default=None,
                        help="Only use these checkpoint paths (for prod ensemble comparison)")
    args = parser.parse_args()

    data = read_mktd(str(REPO / args.data_bin))
    log.info("data: %d days x %d symbols", data.num_timesteps, data.num_symbols)

    if args.ensemble_only:
        checkpoints = [Path(p) for p in args.ensemble_only]
    else:
        checkpoints = discover_checkpoints(args.checkpoint_dirs)
    log.info("discovered %d checkpoints", len(checkpoints))
    for cp in checkpoints:
        log.info("  %s -> %s", label(cp), cp)

    # Simulate each model individually
    traces = []
    for i, cp in enumerate(checkpoints):
        lbl = label(cp)
        log.info("[%d/%d] simulating %s...", i + 1, len(checkpoints), lbl)
        try:
            tr = simulate_single_model(
                data, cp,
                fee_rate=args.fee_rate,
                slippage_bps=args.slippage_bps,
                fill_buffer_bps=args.fill_buffer_bps,
                decision_lag=args.decision_lag,
                device=args.device,
            )
            tr.name = lbl
            ret = tr.equity_curve[-1] / tr.equity_curve[0] * 100 - 100
            traces.append(tr)
            log.info("  %s: ret=%+.2f%%", lbl, ret)
        except Exception as e:
            log.warning("  %s failed: %s", lbl, e)

    if len(traces) < 2:
        log.error("need >= 2 models, got %d", len(traces))
        return

    # Run softmax ensemble baseline
    log.info("simulating %d-model softmax ensemble...", len(checkpoints))
    try:
        ens_trace = simulate_ensemble(
            data, checkpoints,
            fee_rate=args.fee_rate,
            slippage_bps=args.slippage_bps,
            fill_buffer_bps=args.fill_buffer_bps,
            decision_lag=args.decision_lag,
            device=args.device,
        )
        ens_ret = ens_trace.equity_curve[-1] / ens_trace.equity_curve[0] * 100 - 100
        log.info("ensemble: ret=%+.2f%%", ens_ret)
    except Exception as e:
        log.warning("ensemble failed: %s", e)
        ens_trace = None

    if args.sweep:
        _run_sweep(data, traces, ens_trace, args)
        return

    if args.slippage_sweep:
        _run_slippage_sweep(data, traces, checkpoints, args)
        return

    # Single meta-portfolio run
    meta = run_meta_portfolio(
        data, traces,
        top_k=args.top_k,
        lookback=args.lookback,
        warmup=args.warmup,
        fee_rate=args.fee_rate,
        slippage_bps=args.slippage_bps,
        selector="momentum",
    )

    _print_results(data, traces, ens_trace, meta, args)

    if args.out:
        _save_results(args.out, data, traces, ens_trace, meta, args)


def _run_sweep(data, traces, ens_trace, args):
    print(f"\n{'lb':>4} {'k':>3} {'Return%':>10} {'Sortino':>10} {'MaxDD%':>10} {'Trades':>8}")
    print("-" * 50)
    best = None
    for lb in [2, 3, 5, 7, 10, 15, 20]:
        for k in [1, 2, 3]:
            if k > len(traces):
                continue
            meta = run_meta_portfolio(
                data, traces,
                top_k=k, lookback=lb, warmup=max(lb, 3),
                fee_rate=args.fee_rate, slippage_bps=args.slippage_bps,
                selector="momentum",
            )
            print(f"{lb:>4} {k:>3} {meta.total_return*100:>+10.2f}% {meta.sortino:>10.2f} {meta.max_drawdown*100:>10.2f}% {meta.num_trades:>8}")
            if best is None or meta.total_return > best[2].total_return:
                best = (lb, k, meta)

    if ens_trace is not None:
        ens_ret = ens_trace.equity_curve[-1] / ens_trace.equity_curve[0] - 1
        ens_dr = np.diff(ens_trace.equity_curve) / np.maximum(ens_trace.equity_curve[:-1], 1e-8)
        neg = ens_dr[ens_dr < 0]
        ens_sort = float(np.mean(ens_dr) / np.sqrt(np.mean(neg**2)) * np.sqrt(252)) if len(neg) > 0 else 0
        print(f"\nEnsemble baseline: ret={ens_ret*100:+.2f}% sortino={ens_sort:.2f}")

    if best:
        lb, k, m = best
        print(f"\nBest: lb={lb} k={k} ret={m.total_return*100:+.2f}% sortino={m.sortino:.2f} maxdd={m.max_drawdown*100:.2f}%")


def _run_slippage_sweep(data, traces, checkpoints, args):
    print(f"\n{'Slip':>6} {'Meta Ret%':>10} {'Meta Sort':>10} {'Meta DD%':>10} {'Ens Ret%':>10} {'Ens Sort':>10}")
    print("-" * 62)
    for slip in [0, 5, 10, 20]:
        meta = run_meta_portfolio(
            data, traces,
            top_k=args.top_k, lookback=args.lookback, warmup=max(args.lookback, 3),
            fee_rate=args.fee_rate, slippage_bps=slip,
            selector="momentum",
        )
        try:
            ens_trace = simulate_ensemble(
                data, checkpoints,
                fee_rate=args.fee_rate, slippage_bps=slip,
                fill_buffer_bps=args.fill_buffer_bps,
                decision_lag=args.decision_lag, device=args.device,
            )
            ens_ret = ens_trace.equity_curve[-1] / ens_trace.equity_curve[0] - 1
            ens_dr = np.diff(ens_trace.equity_curve) / np.maximum(ens_trace.equity_curve[:-1], 1e-8)
            neg = ens_dr[ens_dr < 0]
            ens_sort = float(np.mean(ens_dr) / np.sqrt(np.mean(neg**2)) * np.sqrt(252)) if len(neg) > 0 else 0
        except Exception:
            ens_ret = 0.0
            ens_sort = 0.0
        print(f"{slip:>4}bp {meta.total_return*100:>+10.2f}% {meta.sortino:>10.2f} {meta.max_drawdown*100:>10.2f}% {ens_ret*100:>+10.2f}% {ens_sort:>10.2f}")


def _print_results(data, traces, ens_trace, meta, args):
    print("\n" + "=" * 70)
    print("CRYPTO30 META-STRATEGY BACKTEST")
    print("=" * 70)
    print(f"Data: {data.num_timesteps} days, {data.num_symbols} symbols")
    print(f"Models: {len(traces)}, Fee: {args.fee_rate*100:.0f}bps, Slip: {args.slippage_bps:.0f}bps, Lag: {args.decision_lag}")

    print(f"\nPER-MODEL RESULTS:")
    print(f"{'Model':<25} {'Return%':>10}")
    print("-" * 37)
    for tr in sorted(traces, key=lambda t: t.equity_curve[-1], reverse=True):
        ret = tr.equity_curve[-1] / tr.equity_curve[0] * 100 - 100
        print(f"{tr.name:<25} {ret:>+10.2f}%")

    monthly = 21.0
    n = data.num_timesteps
    def _mo(r):
        m = n / monthly
        return ((1 + r) ** (1.0 / max(m, 0.1)) - 1) * 100 if m > 0 else 0

    print(f"\nCOMPARISON:")
    print(f"{'Strategy':<30} {'Total%':>10} {'Monthly%':>10} {'Sortino':>10} {'MaxDD%':>10} {'Trades':>8}")
    print("-" * 80)

    best_tr = max(traces, key=lambda t: t.equity_curve[-1])
    best_ret = best_tr.equity_curve[-1] / best_tr.equity_curve[0] - 1
    print(f"{'Best model ('+best_tr.name+')':<30} {best_ret*100:>+10.2f}% {_mo(best_ret):>+10.2f}% {'':>10} {'':>10} {'':>8}")

    if ens_trace is not None:
        er = ens_trace.equity_curve[-1] / ens_trace.equity_curve[0] - 1
        ed = np.diff(ens_trace.equity_curve) / np.maximum(ens_trace.equity_curve[:-1], 1e-8)
        neg = ed[ed < 0]
        es = float(np.mean(ed) / np.sqrt(np.mean(neg**2)) * np.sqrt(252)) if len(neg) > 0 else 0
        ep = np.maximum.accumulate(ens_trace.equity_curve)
        edd = float(np.max((ep - ens_trace.equity_curve) / np.maximum(ep, 1e-8)))
        print(f"{'Ensemble ('+str(len(traces))+'m softmax)':<30} {er*100:>+10.2f}% {_mo(er):>+10.2f}% {es:>10.2f} {edd*100:>10.2f}% {'':>8}")

    print(f"{'Meta lb='+str(args.lookback)+' k='+str(args.top_k):<30} {meta.total_return*100:>+10.2f}% {_mo(meta.total_return):>+10.2f}% {meta.sortino:>10.2f} {meta.max_drawdown*100:>10.2f}% {meta.num_trades:>8}")


def _save_results(out_path, data, traces, ens_trace, meta, args):
    results = {
        "config": {
            "n_models": len(traces),
            "n_days": data.num_timesteps,
            "n_symbols": data.num_symbols,
            "top_k": args.top_k,
            "lookback": args.lookback,
            "fee_rate": args.fee_rate,
            "slippage_bps": args.slippage_bps,
            "decision_lag": args.decision_lag,
        },
        "per_model": {
            tr.name: float(tr.equity_curve[-1] / tr.equity_curve[0] - 1)
            for tr in traces
        },
        "meta": {
            "total_return": float(meta.total_return),
            "sortino": float(meta.sortino),
            "max_drawdown": float(meta.max_drawdown),
            "num_trades": int(meta.num_trades),
        },
    }
    if ens_trace is not None:
        er = ens_trace.equity_curve[-1] / ens_trace.equity_curve[0] - 1
        results["ensemble"] = {"total_return": float(er)}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2))
    log.info("saved %s", out_path)


if __name__ == "__main__":
    main()
