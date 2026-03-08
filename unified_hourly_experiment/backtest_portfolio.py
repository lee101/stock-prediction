#!/usr/bin/env python3
"""Backtest portfolio policy with per-symbol metrics and directional constraints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch

from binanceneural.config import DatasetConfig
from binanceneural.data import BinanceHourlyDataModule, FeatureNormalizer
from binanceneural.inference import generate_actions_from_frame
from binanceneural.model import build_policy, PolicyConfig, policy_config_from_payload
from differentiable_loss_utils import simulate_hourly_trades_binary
from src.trade_directions import (
    DEFAULT_ALPACA_LIVE8_STOCKS,
    resolve_trade_directions,
    trade_direction_name,
)
from src.torch_load_utils import torch_load_compat

def load_model(ckpt_path, config, device):
    ckpt = torch_load_compat(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    feature_columns = config["feature_columns"]
    policy_cfg = policy_config_from_payload(config, input_dim=len(feature_columns), state_dict=state_dict)
    model = build_policy(policy_cfg)
    model.load_state_dict(state_dict, strict=False)
    return model.eval().to(device)


def backtest_symbol(model, dm, symbol, config, device, maker_fee):
    feature_columns = config["feature_columns"]
    seq_len = config.get("sequence_length", 512)

    frame = dm.frame.copy()
    val_hours = 30 * 24
    if len(frame) <= val_hours + seq_len:
        val_frame = frame
    else:
        val_frame = frame.iloc[-(val_hours + seq_len):].reset_index(drop=True)

    actions = generate_actions_from_frame(
        model=model, frame=val_frame, feature_columns=feature_columns,
        normalizer=dm.normalizer, sequence_length=seq_len, horizon=1, device=device,
    )

    n = len(actions)
    highs = torch.tensor(val_frame["high"].iloc[-n:].values, dtype=torch.float32).unsqueeze(0)
    lows = torch.tensor(val_frame["low"].iloc[-n:].values, dtype=torch.float32).unsqueeze(0)
    closes = torch.tensor(val_frame["close"].iloc[-n:].values, dtype=torch.float32).unsqueeze(0)
    buy_prices = torch.tensor(actions["buy_price"].values, dtype=torch.float32).unsqueeze(0)
    sell_prices = torch.tensor(actions["sell_price"].values, dtype=torch.float32).unsqueeze(0)
    buy_amt = torch.tensor(actions["buy_amount"].values, dtype=torch.float32).unsqueeze(0) / 100.0
    sell_amt = torch.tensor(actions["sell_amount"].values, dtype=torch.float32).unsqueeze(0) / 100.0
    trade_amt = torch.maximum(buy_amt, sell_amt)

    directions = resolve_trade_directions(symbol, allow_short=True)
    can_long = 1.0 if directions.can_long else 0.0
    can_short = 1.0 if directions.can_short else 0.0

    sim = simulate_hourly_trades_binary(
        highs=highs, lows=lows, closes=closes,
        buy_prices=buy_prices, sell_prices=sell_prices,
        trade_intensity=trade_amt,
        buy_trade_intensity=buy_amt,
        sell_trade_intensity=sell_amt,
        maker_fee=maker_fee,
        initial_cash=1.0,
        can_long=can_long, can_short=can_short,
    )

    returns = sim.returns.squeeze(0).numpy()
    pv = sim.portfolio_values.squeeze(0).numpy()
    final_val = pv[-1] if len(pv) > 0 else 1.0
    total_return = (final_val - 1.0) * 100

    downside = np.minimum(returns, 0.0)
    downside_std = np.sqrt(np.mean(downside**2))
    sortino = (np.mean(returns) / downside_std * np.sqrt(252 * 6.5)) if downside_std > 1e-8 else 0.0

    peak = np.maximum.accumulate(pv)
    drawdown = (pv - peak) / np.where(peak > 0, peak, 1.0)
    max_dd = float(drawdown.min()) * 100

    hold_hours = float(actions["hold_hours"].mean()) if "hold_hours" in actions.columns else 0.0
    alloc_frac = float(actions["allocation_fraction"].mean()) if "allocation_fraction" in actions.columns else 0.0

    return {
        "symbol": symbol,
        "return_pct": float(round(total_return, 2)),
        "sortino": float(round(sortino, 2)),
        "max_drawdown_pct": float(round(max_dd, 2)),
        "direction": trade_direction_name(symbol, allow_short=True),
        "avg_hold_hours": float(round(hold_hours, 1)),
        "avg_alloc_frac": float(round(alloc_frac, 3)),
        "n_bars": int(n),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--symbols", default=None)
    p.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    p.add_argument("--cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    p.add_argument("--maker-fee", type=float, default=0.001)
    p.add_argument("--validation-days", type=int, default=30)
    p.add_argument("--sweep-epochs", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.sweep_epochs:
        ckpt_dir = args.checkpoint
        checkpoints = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    else:
        ckpt_dir = args.checkpoint.parent
        checkpoints = [args.checkpoint]

    meta_path = ckpt_dir / "training_meta.json"
    config_path = ckpt_dir / "config.json"
    if meta_path.exists():
        with open(meta_path) as f:
            config = json.load(f)
    elif config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        ckpt = torch_load_compat(checkpoints[0], map_location="cpu", weights_only=False)
        config = ckpt.get("config", {})

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = config.get("symbols", list(DEFAULT_ALPACA_LIVE8_STOCKS))

    seq_len = config.get("sequence_length", 512)

    from binanceneural.data import build_default_feature_columns
    ckpt0 = torch_load_compat(checkpoints[0], map_location="cpu", weights_only=False)
    sd0 = ckpt0.get("state_dict", ckpt0)
    if any(k.startswith("_orig_mod.") for k in sd0):
        sd0 = {k.replace("_orig_mod.", ""): v for k, v in sd0.items()}
    embed_w = sd0.get("embed.weight")
    model_input_dim = int(embed_w.shape[1]) if embed_w is not None and embed_w.ndim == 2 else None

    feature_columns = config.get("feature_columns")
    if feature_columns is not None and model_input_dim and len(feature_columns) != model_input_dim:
        feature_columns = None
    if feature_columns is None:
        for h_try in [[1], [1, 24]]:
            fc = build_default_feature_columns(h_try)
            if model_input_dim is None or len(fc) == model_input_dim:
                feature_columns = fc
                break
        if feature_columns is None:
            feature_columns = build_default_feature_columns([1])

    horizons = [1]
    if any("h24" in c for c in feature_columns):
        horizons = [1, 24]
    config["feature_columns"] = feature_columns

    print(f"Loading data for {len(symbols)} symbols...")
    data_modules = {}
    for symbol in symbols:
        try:
            ds_cfg = DatasetConfig(
                symbol=symbol, data_root=args.data_root,
                forecast_cache_root=args.cache_root,
                forecast_horizons=tuple(horizons),
                sequence_length=seq_len,
                validation_days=args.validation_days,
                cache_only=True, min_history_hours=seq_len + 48,
            )
            data_modules[symbol] = BinanceHourlyDataModule(ds_cfg)
        except Exception as e:
            print(f"  Skip {symbol}: {e}")

    print(f"Loaded {len(data_modules)}/{len(symbols)} symbols")

    all_results = []
    for ckpt_path in checkpoints:
        epoch = int(ckpt_path.stem.split("_")[1])
        model = load_model(ckpt_path, config, device)

        results = []
        for symbol, dm in data_modules.items():
            r = backtest_symbol(model, dm, symbol, config, device, args.maker_fee)
            results.append(r)

        avg_return = np.mean([r["return_pct"] for r in results])
        total_sortino = np.mean([r["sortino"] for r in results])
        avg_dd = np.mean([r["max_drawdown_pct"] for r in results])

        long_results = [r for r in results if r["direction"] == "long"]
        short_results = [r for r in results if r["direction"] == "short"]

        print(f"\n{'='*70}")
        print(f"Epoch {epoch}: avg_return={avg_return:+.2f}%, avg_sortino={total_sortino:.2f}, avg_dd={avg_dd:.2f}%")
        if long_results:
            lr = np.mean([r["return_pct"] for r in long_results])
            print(f"  Long  ({len(long_results)} syms): avg_return={lr:+.2f}%")
        if short_results:
            sr = np.mean([r["return_pct"] for r in short_results])
            print(f"  Short ({len(short_results)} syms): avg_return={sr:+.2f}%")

        print(f"\n{'Symbol':<8} {'Dir':<6} {'Return%':>8} {'Sortino':>8} {'MaxDD%':>8} {'Hold_h':>7} {'Alloc':>6}")
        print("-" * 60)
        for r in sorted(results, key=lambda x: x["return_pct"], reverse=True):
            print(f"{r['symbol']:<8} {r['direction']:<6} {r['return_pct']:>+8.2f} {r['sortino']:>8.2f} "
                  f"{r['max_drawdown_pct']:>8.2f} {r['avg_hold_hours']:>7.1f} {r['avg_alloc_frac']:>6.3f}")

        all_results.append({
            "epoch": epoch,
            "avg_return": round(avg_return, 2),
            "avg_sortino": round(total_sortino, 2),
            "avg_max_dd": round(avg_dd, 2),
            "per_symbol": results,
        })

    if args.sweep_epochs:
        best = max(all_results, key=lambda x: x["avg_sortino"])
        print(f"\n{'='*70}")
        print(f"BEST: Epoch {best['epoch']}, return={best['avg_return']:+.2f}%, sortino={best['avg_sortino']:.2f}")

    out_path = ckpt_dir / "portfolio_backtest.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
