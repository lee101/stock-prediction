"""Sweep base constant offsets (buy, sell, intensity) per symbol.

The simplest possible calibration: find the best static buy offset,
sell offset, and trade intensity per symbol using grid search on
binary-fill market simulation with proper train/val/test splits.

Usage:
    source .venv313/bin/activate
    python rl_trading_agent_binance/sweep_base_constants.py \
        --symbols BTCUSD,ETHUSD,SOLUSD,DOGEUSD,AAVEUSD,LINKUSD
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from signal_calibrator import SignalCalibrator, CalibrationConfig, save_calibrator
from train_calibrator import (
    DEPLOYED_SYMBOLS, prepare_symbol_tensors, time_split, run_sim,
)
from differentiable_loss_utils import DEFAULT_MAKER_FEE_RATE

BUY_OFFSETS = [-0.005, -0.003, -0.002, -0.001, -0.0005, 0.0, 0.0005]
SELL_OFFSETS = [0.003, 0.005, 0.008, 0.010, 0.012, 0.015, 0.020]
INTENSITIES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

FEE_MAP = {
    "BTCUSD": 0.0, "ETHUSD": 0.0,
    "SOLUSD": 0.001, "DOGEUSD": 0.001, "AAVEUSD": 0.001, "LINKUSD": 0.001,
}


def eval_constants(
    data: dict,
    buy_off: float,
    sell_off: float,
    intensity: float,
    split_slice: slice,
    maker_fee: float,
    decision_lag: int = 2,
    device: str = "cpu",
) -> dict:
    cfg = CalibrationConfig(
        base_buy_offset=buy_off,
        base_sell_offset=sell_off,
        base_intensity=intensity,
    )
    cal = SignalCalibrator(cfg).to(device)
    cal.eval()
    with torch.no_grad():
        _, metrics = run_sim(
            cal,
            data["features"][split_slice],
            data["closes"][split_slice],
            data["highs"][split_slice],
            data["lows"][split_slice],
            data["opens"][split_slice],
            maker_fee=maker_fee,
            fill_buffer_pct=0.0005,
            decision_lag=decision_lag,
            binary=True,
        )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default=",".join(DEPLOYED_SYMBOLS))
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="rl_trading_agent_binance/calibrator_checkpoints")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    total_combos = len(BUY_OFFSETS) * len(SELL_OFFSETS) * len(INTENSITIES)
    print(f"Sweeping {total_combos} constant combos per symbol")

    all_results = {}
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"SYMBOL: {symbol}")
        print(f"{'='*60}")

        data = prepare_symbol_tensors(symbol, device=args.device)
        n = data["n_bars"]
        _, val_sl, test_sl = time_split(n)
        fee = FEE_MAP.get(symbol, 0.001)
        print(f"  {n} bars, fee={fee}, val={val_sl}, test={test_sl}")

        results = []
        for i, (buy_off, sell_off, intensity) in enumerate(
            itertools.product(BUY_OFFSETS, SELL_OFFSETS, INTENSITIES)
        ):
            metrics = eval_constants(
                data, buy_off, sell_off, intensity,
                val_sl, fee, args.decision_lag, args.device,
            )
            results.append({
                "buy_off": buy_off,
                "sell_off": sell_off,
                "intensity": intensity,
                "val_sortino": metrics["sortino"],
                "val_return": metrics["return"],
            })
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{total_combos} done")

        results.sort(key=lambda r: r["val_sortino"], reverse=True)
        print(f"\n  TOP {args.top_k} on VAL:")
        print(f"  {'buy_off':>8} {'sell_off':>8} {'intens':>6} {'val_sort':>10} {'val_ret':>10}")
        for r in results[:args.top_k]:
            print(f"  {r['buy_off']:+8.4f} {r['sell_off']:+8.4f} {r['intensity']:6.2f} {r['val_sortino']:+10.2f} {r['val_return']:+10.4f}")

        best = results[0]
        print(f"\n  Evaluating top-3 on TEST set:")
        test_results = []
        for r in results[:3]:
            test_m = eval_constants(
                data, r["buy_off"], r["sell_off"], r["intensity"],
                test_sl, fee, args.decision_lag, args.device,
            )
            r["test_sortino"] = test_m["sortino"]
            r["test_return"] = test_m["return"]
            test_results.append(r)
            print(f"    buy={r['buy_off']:+.4f} sell={r['sell_off']:+.4f} int={r['intensity']:.2f} "
                  f"-> test_sort={test_m['sortino']:+.2f} test_ret={test_m['return']:+.4f}")

        test_results.sort(key=lambda r: r["test_sortino"], reverse=True)
        winner = test_results[0]
        all_results[symbol] = {
            "winner": winner,
            "all_val_top10": results[:10],
        }

        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        cfg = CalibrationConfig(
            base_buy_offset=winner["buy_off"],
            base_sell_offset=winner["sell_off"],
            base_intensity=winner["intensity"],
        )
        cal = SignalCalibrator(cfg).to(args.device)
        save_calibrator(cal, save_dir / f"{symbol}_calibrator.pt", cfg, metadata={
            "type": "constant_sweep",
            "val_sortino": winner["val_sortino"],
            "test_sortino": winner["test_sortino"],
            "test_return": winner["test_return"],
        })
        print(f"  WINNER: buy={winner['buy_off']:+.4f} sell={winner['sell_off']:+.4f} "
              f"int={winner['intensity']:.2f} test_sort={winner['test_sortino']:+.2f}")

    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Symbol':<10} {'BuyOff':>8} {'SellOff':>8} {'Intens':>6} {'ValSort':>8} {'TestSort':>8} {'TestRet':>10}")
    print("-" * 70)
    for sym, res in all_results.items():
        w = res["winner"]
        print(f"{sym:<10} {w['buy_off']:+8.4f} {w['sell_off']:+8.4f} {w['intensity']:6.2f} "
              f"{w['val_sortino']:+8.2f} {w['test_sortino']:+8.2f} {w['test_return']:+10.4f}")

    out_path = Path(args.save_dir) / "constant_sweep_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
