"""Sweep calibrator hyperparameters to find optimal settings.

Sweeps: hidden_size, max_price_bps, max_amount_adj, lr, base offsets.
Evaluates on holdout with binary fills.

Usage:
    source .venv313/bin/activate
    python rl_trading_agent_binance/sweep_calibrator.py --symbols BTCUSD
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from signal_calibrator import CalibrationConfig
from train_calibrator import (
    compute_baseline,
    prepare_symbol_tensors,
    train_one_symbol,
)


SWEEP_GRID = {
    "hidden": [16, 32, 64],
    "max_price_bps": [10, 25, 50, 100],
    "max_amount_adj": [0.15, 0.30, 0.50],
    "lr": [5e-4, 1e-3, 3e-3],
    "base_buy_offset": [-0.001, -0.002, -0.003],
    "base_sell_offset": [0.005, 0.008, 0.012],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTCUSD")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--sweep-keys",
        type=str,
        default="max_price_bps,base_sell_offset,lr",
        help="Comma-separated keys to sweep (others use default)",
    )
    parser.add_argument("--save-dir", type=str, default="rl_trading_agent_binance/calibrator_sweep")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    sweep_keys = [k.strip() for k in args.sweep_keys.split(",")]

    defaults = {
        "hidden": 32,
        "max_price_bps": 25,
        "max_amount_adj": 0.30,
        "lr": 1e-3,
        "base_buy_offset": -0.001,
        "base_sell_offset": 0.008,
    }

    sweep_values = {}
    for k in sweep_keys:
        if k in SWEEP_GRID:
            sweep_values[k] = SWEEP_GRID[k]
    fixed_keys = [k for k in defaults if k not in sweep_keys]

    combos = list(itertools.product(*[sweep_values[k] for k in sweep_keys]))
    print(f"Sweeping {len(combos)} combinations over keys: {sweep_keys}")
    print(f"Fixed: {', '.join(f'{k}={defaults[k]}' for k in fixed_keys)}")

    all_results = []
    for symbol in symbols:
        print(f"\n{'=' * 60}")
        print(f"SYMBOL: {symbol}")
        print(f"{'=' * 60}")

        data = prepare_symbol_tensors(symbol, device=args.device)
        print(f"  {data['n_bars']} bars loaded")

        for combo in combos:
            params = dict(defaults)
            for k, v in zip(sweep_keys, combo):
                params[k] = v

            config = CalibrationConfig(
                hidden=params["hidden"],
                max_price_adj_bps=params["max_price_bps"],
                max_amount_adj=params["max_amount_adj"],
                base_buy_offset=params["base_buy_offset"],
                base_sell_offset=params["base_sell_offset"],
            )
            label = "_".join(f"{k}{v}" for k, v in zip(sweep_keys, combo))
            print(f"\n  --- {label} ---")

            try:
                result = train_one_symbol(
                    symbol,
                    data,
                    config,
                    epochs=args.epochs,
                    lr=params["lr"],
                    device=args.device,
                    save_dir=Path(args.save_dir) / label,
                )
                baseline = compute_baseline(data, config)
                result["config_label"] = label
                result["params"] = params
                result["baseline"] = baseline
                all_results.append(result)
            except Exception as e:
                print(f"  FAILED: {e}")

    all_results.sort(key=lambda r: r.get("test_metrics", {}).get("sortino", -999), reverse=True)
    print(f"\n{'=' * 70}")
    print("SWEEP RESULTS (sorted by test Sortino)")
    print(f"{'=' * 70}")
    print(f"{'Label':<35} {'Sym':<8} {'Base':>6} {'Test Sort':>10} {'Test Ret':>10} {'Ep':>4}")
    print("-" * 70)
    for r in all_results:
        ts = r.get("test_metrics", {}).get("sortino", 0)
        tr = r.get("test_metrics", {}).get("return", 0)
        bs = r.get("baseline", {}).get("sortino", 0)
        print(
            f"{r.get('config_label', '?'):<35} {r['symbol']:<8} {bs:+6.2f} {ts:+10.2f} {tr:+10.4f} {r['best_epoch']:4d}"
        )

    out_path = Path(args.save_dir) / "sweep_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            [{k: v for k, v in r.items() if k != "history"} for r in all_results],
            indent=2,
            default=str,
        )
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
