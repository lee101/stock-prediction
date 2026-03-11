#!/usr/bin/env python3
"""Sweep multi-asset crypto checkpoints across multiple time windows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch

from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binanceneural.data import FeatureNormalizer
from src.forecast_horizon_utils import resolve_required_forecast_horizons
from unified_hourly_experiment.multiasset_policy import (
    MultiAssetConfig,
    DifferentiablePortfolioSim,
    build_multiasset_policy,
)

REPO = Path(__file__).resolve().parents[1]

WINDOWS = [
    (30, 14, "14d"),
    (30, 30, "30d"),
    (30, 60, "60d"),
    (30, 90, "90d"),
    (30, 120, "120d"),
]


def load_checkpoint(ckpt_path: str, device: str = "cuda"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    config = MultiAssetConfig(
        num_assets=cfg["num_assets"],
        feature_dim=cfg["feature_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        max_len=cfg["max_len"],
        dropout=cfg.get("dropout", 0.3),
    )
    include_cash = cfg.get("include_cash", False)
    from unified_hourly_experiment.multiasset_policy import MultiAssetPolicy
    model = MultiAssetPolicy(config, include_cash=include_cash)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    normalizer = FeatureNormalizer.from_dict(ckpt["normalizer"])
    return model, normalizer, ckpt


def simulate_multiasset_window(
    model,
    normalizer,
    symbols,
    feat_cols,
    seq_len,
    horizon,
    maker_fee,
    margin_hourly_rate,
    data_root,
    forecast_cache,
    val_days,
    test_days,
    device="cuda",
):
    """Run multi-asset allocation sim on a test window, return metrics."""
    forecast_horizons = resolve_required_forecast_horizons((1,), fallback_horizons=(1,))
    frames = {}
    for sym in symbols:
        dm = ChronosSolDataModule(
            symbol=sym,
            data_root=data_root,
            forecast_cache_root=forecast_cache,
            forecast_horizons=forecast_horizons,
            context_hours=512,
            quantile_levels=(0.1, 0.5, 0.9),
            batch_size=32,
            model_id="amazon/chronos-t5-small",
            sequence_length=seq_len,
            split_config=SplitConfig(val_days=val_days, test_days=test_days),
            cache_only=True,
            max_history_days=365,
        )
        frames[sym] = dm.test_frame

    # align timestamps
    ts_sets = [set(f["timestamp"].tolist()) for f in frames.values()]
    common_ts = sorted(set.intersection(*ts_sets))
    if len(common_ts) < seq_len + horizon:
        return None

    # extract features and returns
    features_by_sym = {}
    returns_by_sym = {}
    for sym in symbols:
        f = frames[sym].set_index("timestamp").loc[common_ts]
        available = [c for c in feat_cols if c in f.columns]
        feats = normalizer.transform(f[available].to_numpy(dtype=np.float32))
        close = f["close"].to_numpy(dtype=np.float32)
        ret = np.zeros_like(close)
        ret[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-10)
        features_by_sym[sym] = feats
        returns_by_sym[sym] = ret

    # run allocation sim
    n = len(common_ts)
    cash = 10000.0
    equity_curve = [cash]
    num_rebalances = 0
    prev_alloc = np.zeros(len(symbols))

    model.eval()
    with torch.inference_mode():
        for t in range(seq_len, n - 1):
            # build input
            feat_list = []
            for sym in symbols:
                feat_list.append(torch.from_numpy(features_by_sym[sym][t - seq_len + 1 : t + 1]))
            feat_tensor = torch.stack(feat_list).unsqueeze(0).to(device)  # (1, num_assets, seq, feat)
            alloc, _ = model(feat_tensor)
            alloc_np = alloc[0].cpu().numpy()
            # strip cash slot if present
            asset_alloc = alloc_np[:len(symbols)]

            asset_returns = np.array([returns_by_sym[sym][t + 1] for sym in symbols])
            port_return = (asset_alloc * asset_returns).sum()
            turnover = np.abs(asset_alloc - prev_alloc).sum()
            tx_cost = turnover * maker_fee
            margin_cost = margin_hourly_rate * max(0, asset_alloc.sum() - 1.0)
            net_return = port_return - tx_cost - margin_cost
            cash *= (1 + net_return)
            equity_curve.append(cash)
            prev_alloc = asset_alloc
            if turnover > 0.01:
                num_rebalances += 1

    eq = np.array(equity_curve)
    total_return = eq[-1] / eq[0] - 1
    peak = np.maximum.accumulate(eq)
    dd = ((eq - peak) / (peak + 1e-10)).min()
    rets = np.diff(eq) / (np.abs(eq[:-1]) + 1e-10)
    neg = rets[rets < 0]
    dd_std = np.std(neg) if len(neg) > 1 else 1e-10
    sortino = float(np.mean(rets)) / (dd_std + 1e-10) if dd_std > 1e-10 else 0.0

    return {
        "total_return": float(total_return),
        "sortino": float(sortino),
        "max_drawdown": float(dd),
        "num_rebalances": int(num_rebalances),
        "hours": len(equity_curve) - 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--data-root", type=Path, default=REPO / "trainingdatahourlybinance")
    parser.add_argument("--forecast-cache", type=Path, default=REPO / "binanceneural/forecast_cache")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir
    config_path = ckpt_dir / "config.json"
    if not config_path.exists():
        print(f"no config.json in {ckpt_dir}")
        sys.exit(1)

    with open(config_path) as f:
        run_config = json.load(f)

    symbols = run_config["symbols"]
    feat_cols = run_config["feature_columns"]
    seq_len = run_config["sequence_length"]
    horizon = run_config["horizon"]
    maker_fee = run_config.get("maker_fee", 0.001)
    margin_rate = run_config.get("margin_hourly_rate", 0.0000025457)

    ckpt_files = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not ckpt_files:
        print("no checkpoints found")
        sys.exit(1)

    print(f"sweeping {len(ckpt_files)} checkpoints x {len(WINDOWS)} windows for {symbols}")
    results = []

    for ckpt_path in ckpt_files:
        model, normalizer, ckpt = load_checkpoint(str(ckpt_path), device=args.device)
        per_window = {}

        for val_days, test_days, label in WINDOWS:
            m = simulate_multiasset_window(
                model, normalizer, symbols, feat_cols, seq_len, horizon,
                maker_fee, margin_rate,
                args.data_root, args.forecast_cache,
                val_days, test_days, device=args.device,
            )
            per_window[label] = m or {"total_return": 0, "sortino": 0, "max_drawdown": 0, "num_rebalances": 0}

        rets = [v["total_return"] for v in per_window.values()]
        sorts = [v["sortino"] for v in per_window.values()]
        dds = [v["max_drawdown"] for v in per_window.values()]
        pos_windows = sum(r > 0 for r in rets)

        row = {
            "checkpoint": ckpt_path.name,
            "positive_windows": pos_windows,
            "mean_return_pct": round(np.mean(rets) * 100, 2),
            "min_return_pct": round(np.min(rets) * 100, 2),
            "mean_sortino": round(np.mean(sorts), 2),
            "min_sortino": round(np.min(sorts), 2),
            "worst_dd_pct": round(np.min(dds) * 100, 2),
            "per_window": per_window,
        }
        results.append(row)
        print(
            f"  {ckpt_path.name}: pos={pos_windows}/{len(WINDOWS)} "
            f"ret={row['mean_return_pct']:.1f}% sort={row['mean_sortino']:.2f} dd={row['worst_dd_pct']:.1f}%"
        )

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    results_sorted = sorted(results, key=lambda r: (r["positive_windows"], r["min_return_pct"]), reverse=True)
    out_path = ckpt_dir / "sweep_results.json"
    out_path.write_text(json.dumps(results_sorted, indent=2))
    print(f"\nresults saved to {out_path}")
    if results_sorted:
        b = results_sorted[0]
        print(f"best: {b['checkpoint']} pos={b['positive_windows']} ret={b['mean_return_pct']}% sort={b['mean_sortino']}")


if __name__ == "__main__":
    main()
