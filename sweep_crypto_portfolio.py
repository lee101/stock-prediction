#!/usr/bin/env python3
"""Sweep crypto portfolio checkpoints across symbols, lags, and time windows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch

from binanceneural.config import DatasetConfig, TrainingConfig
from binanceneural.data import BinanceHourlyDataModule, FeatureNormalizer
from binanceneural.inference import generate_actions_from_frame
from binanceneural.marketsimulator import (
    BinanceMarketSimulator,
    SimulationConfig,
)
from binanceneural.model import align_state_dict_input_dim, build_policy, policy_config_from_payload
from src.torch_load_utils import torch_load_compat

SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "AAVEUSD", "LINKUSD"]
LAGS = [0, 1, 2]
WINDOWS_DAYS = [14, 30, 60]


def load_model(ckpt_path, input_dim, device="cpu"):
    payload = torch_load_compat(ckpt_path, map_location=device, weights_only=False)
    state_dict = payload.get("state_dict", payload)
    state_dict = align_state_dict_input_dim(state_dict, input_dim=input_dim)
    cfg = payload.get("config", TrainingConfig())
    if hasattr(cfg, "__dict__"):
        cfg = cfg.__dict__
    policy_cfg = policy_config_from_payload(cfg, input_dim=input_dim, state_dict=state_dict)
    model = build_policy(policy_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, payload


def eval_symbol_window(model, data_module, symbol, seq_len, lag, window_hours, fee_rate, max_hold, fill_buffer_bps, device):
    val_frame = data_module.val_dataset.frame.copy()
    if "symbol" not in val_frame.columns:
        val_frame["symbol"] = symbol
    if window_hours and len(val_frame) > window_hours:
        val_frame = val_frame.iloc[-window_hours:].reset_index(drop=True)
    if len(val_frame) < seq_len + 2:
        return None

    actions = generate_actions_from_frame(
        model=model,
        frame=val_frame,
        feature_columns=data_module.feature_columns,
        normalizer=data_module.normalizer,
        sequence_length=seq_len,
        horizon=1,
        device=torch.device(device),
    )
    if "symbol" not in actions.columns:
        actions["symbol"] = symbol

    sim = BinanceMarketSimulator(SimulationConfig(
        maker_fee=fee_rate,
        initial_cash=10_000.0,
        max_hold_hours=max_hold,
        fill_buffer_bps=fill_buffer_bps,
        decision_lag_bars=lag,
        one_side_per_bar=True,
    ))
    result = sim.run(val_frame, actions)
    m = result.metrics
    trades = []
    for sr in result.per_symbol.values():
        trades.extend(sr.trades)

    eq = result.combined_equity
    if len(eq) > 1:
        peak = np.maximum.accumulate(eq.values)
        dd = ((eq.values - peak) / (peak + 1e-10)).min()
    else:
        dd = 0.0

    return {
        "total_return": m["total_return"],
        "sortino": m["sortino"],
        "max_drawdown": float(dd),
        "num_trades": len(trades),
        "hours": len(val_frame),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("binanceneural/checkpoints/crypto_portfolio_6sym"))
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--lags", nargs="+", type=int, default=LAGS)
    parser.add_argument("--windows", nargs="+", type=int, default=WINDOWS_DAYS)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--max-hold", type=int, default=24)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-only", action="store_true", default=True)
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir
    ckpt_files = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not ckpt_files:
        print(f"No checkpoints in {ckpt_dir}")
        return

    data_modules = {}
    for sym in args.symbols:
        try:
            cfg = DatasetConfig(
                symbol=sym, sequence_length=args.seq_len, cache_only=args.cache_only,
                forecast_horizons=(1,),
            )
            dm = BinanceHourlyDataModule(cfg)
            data_modules[sym] = dm
            print(f"  {sym}: {len(dm.val_dataset.frame)} val bars")
        except Exception as e:
            print(f"  {sym}: SKIP ({e})")

    if not data_modules:
        print("No symbols loaded")
        return

    all_results = []

    for ckpt_path in ckpt_files:
        ep_name = ckpt_path.stem
        print(f"\n=== {ep_name} ===")

        first_dm = next(iter(data_modules.values()))
        model, payload = load_model(str(ckpt_path), len(first_dm.feature_columns), device=args.device)
        model = model.to(args.device)

        for lag in args.lags:
            for window_d in args.windows:
                window_h = window_d * 24
                per_sym = {}
                for sym, dm in data_modules.items():
                    r = eval_symbol_window(
                        model, dm, sym, args.seq_len, lag, window_h,
                        args.fee_rate, args.max_hold, args.fill_buffer_bps, args.device,
                    )
                    per_sym[sym] = r

                valid = {s: r for s, r in per_sym.items() if r is not None}
                if not valid:
                    continue

                rets = [r["total_return"] for r in valid.values()]
                sorts = [r["sortino"] for r in valid.values()]
                total_trades = sum(r["num_trades"] for r in valid.values())
                pos_syms = sum(1 for r in rets if r > 0)
                mean_ret = np.mean(rets)
                mean_sort = np.mean(sorts)
                worst_dd = min(r["max_drawdown"] for r in valid.values())

                row = {
                    "epoch": ep_name,
                    "lag": lag,
                    "window_days": window_d,
                    "mean_return_pct": round(mean_ret * 100, 3),
                    "mean_sortino": round(mean_sort, 2),
                    "positive_symbols": pos_syms,
                    "total_trades": total_trades,
                    "worst_dd_pct": round(worst_dd * 100, 2),
                    "per_symbol": {s: {
                        "ret_pct": round(r["total_return"] * 100, 3),
                        "sortino": round(r["sortino"], 2),
                        "trades": r["num_trades"],
                        "dd_pct": round(r["max_drawdown"] * 100, 2),
                    } for s, r in valid.items()},
                }
                all_results.append(row)

                sym_str = " | ".join(
                    f"{s}:{r['total_return']*100:+.2f}%"
                    for s, r in sorted(valid.items())
                )
                print(
                    f"  lag={lag} {window_d}d: ret={mean_ret*100:+.3f}% sort={mean_sort:.2f} "
                    f"pos={pos_syms}/{len(valid)} trades={total_trades} dd={worst_dd*100:.1f}% [{sym_str}]"
                )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = ckpt_dir / "portfolio_sweep_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 80)
    print("SUMMARY: BEST EPOCH PER LAG @ 30d")
    print("=" * 80)
    for lag in args.lags:
        lag_results = [r for r in all_results if r["lag"] == lag and r["window_days"] == 30]
        if lag_results:
            best = max(lag_results, key=lambda r: r["mean_sortino"])
            print(
                f"  lag={lag}: {best['epoch']} ret={best['mean_return_pct']:+.3f}% "
                f"sort={best['mean_sortino']:.2f} pos={best['positive_symbols']}/{len(args.symbols)} "
                f"dd={best['worst_dd_pct']:.1f}%"
            )

    print("\n" + "=" * 80)
    print("SUMMARY: LAG=1 ALL WINDOWS PER EPOCH")
    print("=" * 80)
    lag1 = [r for r in all_results if r["lag"] == 1]
    if lag1:
        by_epoch = {}
        for r in lag1:
            by_epoch.setdefault(r["epoch"], []).append(r)
        for ep, rows in sorted(by_epoch.items()):
            avg_sort = np.mean([r["mean_sortino"] for r in rows])
            avg_ret = np.mean([r["mean_return_pct"] for r in rows])
            pos_all = all(r["mean_return_pct"] > 0 for r in rows)
            print(f"  {ep}: avg_sort={avg_sort:.2f} avg_ret={avg_ret:+.3f}% all_pos={'Y' if pos_all else 'N'}")

    print("\n" + "=" * 80)
    print("SUMMARY: LAG=2 ALL WINDOWS PER EPOCH")
    print("=" * 80)
    lag2 = [r for r in all_results if r["lag"] == 2]
    if lag2:
        by_epoch = {}
        for r in lag2:
            by_epoch.setdefault(r["epoch"], []).append(r)
        for ep, rows in sorted(by_epoch.items()):
            avg_sort = np.mean([r["mean_sortino"] for r in rows])
            avg_ret = np.mean([r["mean_return_pct"] for r in rows])
            pos_all = all(r["mean_return_pct"] > 0 for r in rows)
            print(f"  {ep}: avg_sort={avg_sort:.2f} avg_ret={avg_ret:+.3f}% all_pos={'Y' if pos_all else 'N'}")


if __name__ == "__main__":
    main()
