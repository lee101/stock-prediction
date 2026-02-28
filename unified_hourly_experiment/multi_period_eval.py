#!/usr/bin/env python3
"""Multi-period evaluation: evaluate epochs 5-10 across 7d/30d/60d/90d holdout."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
from loguru import logger
from binanceneural.data import BinanceHourlyDataModule, FeatureNormalizer
from binanceneural.config import DatasetConfig, PolicyConfig
from binanceneural.model import build_policy
from binanceneural.inference import generate_actions_from_frame
from unified_hourly_experiment.marketsimulator import PortfolioConfig, run_portfolio_simulation
from src.torch_load_utils import torch_load_compat

CONFIGS = [
    ("realistic_rw015", "NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT"),
    ("sweep_rw035_wd04", "NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT"),
    ("sweep_rw025_wd04", "NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT"),
    ("sweep_rw035_wd05", "NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT"),
    ("sweep_rw020_wd03_fb", "NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT"),
    ("sweep_9sym_rw035_wd04", "NVDA,PLTR,GOOG,NET,DBX,TRIP,EBAY,MTCH,NYT"),
    ("sweep_rw015_wd04_seq32", "NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT"),
]
EPOCHS = list(range(5, 13))
HOLDOUT_DAYS = [7, 30, 60, 90]
CKPT_ROOT = Path("unified_hourly_experiment/checkpoints")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for config_name, symbols_str in CONFIGS:
        symbols = [s.strip().upper() for s in symbols_str.split(",")]
        ckpt_dir = CKPT_ROOT / config_name

        if not ckpt_dir.exists():
            logger.warning("Skip missing {}", config_name)
            continue

        with open(ckpt_dir / "config.json") as f:
            config = json.load(f)

        feature_columns = config.get("feature_columns", [])
        horizons = sorted({int(c.split("_h")[1]) for c in feature_columns
                           if "_h" in c and c.split("_h")[1].isdigit()}) or [1, 24]
        seq_len = config.get("sequence_length", 32)

        data_modules = {}
        for symbol in symbols:
            data_config = DatasetConfig(
                symbol=symbol, data_root="trainingdatahourly/stocks",
                forecast_cache_root="unified_hourly_experiment/forecast_cache",
                forecast_horizons=horizons, sequence_length=seq_len,
                min_history_hours=100, validation_days=30, cache_only=True,
            )
            try:
                data_modules[symbol] = BinanceHourlyDataModule(data_config)
            except Exception as e:
                logger.warning("Skip {}: {}", symbol, e)

        if "normalizer" in config:
            normalizer = FeatureNormalizer.from_dict(config["normalizer"])
        else:
            normalizer = list(data_modules.values())[0].normalizer

        for epoch in EPOCHS:
            ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
            if not ckpt_path.exists():
                continue

            ckpt = torch_load_compat(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)

            pe_key = "pos_encoding.pe"
            max_len = seq_len
            if pe_key in state_dict:
                max_len = max(max_len, state_dict[pe_key].shape[0])
            policy_cfg = PolicyConfig(
                input_dim=len(feature_columns),
                hidden_dim=config.get("transformer_dim", 128),
                num_heads=config.get("transformer_heads", 4),
                num_layers=config.get("transformer_layers", 3),
                num_outputs=config.get("num_outputs", 4),
                model_arch=config.get("model_arch", "gemma"),
                max_len=max_len,
            )
            model = build_policy(policy_cfg)
            if any(k.startswith("_orig_mod.") for k in state_dict):
                state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            model.eval().to(device)

            # Generate actions ONCE (full data)
            all_bars, all_actions = [], []
            for symbol, dm in data_modules.items():
                frame = dm.frame.copy()
                frame["symbol"] = symbol
                all_bars.append(frame)
                actions_df = generate_actions_from_frame(
                    model=model, frame=frame, feature_columns=feature_columns,
                    normalizer=normalizer, sequence_length=seq_len,
                    horizon=1, device=device,
                )
                all_actions.append(actions_df)

            full_bars = pd.concat(all_bars, ignore_index=True)
            full_actions = pd.concat(all_actions, ignore_index=True)

            # Simulate across multiple holdout periods
            for holdout in HOLDOUT_DAYS:
                cutoff = full_bars["timestamp"].max() - pd.Timedelta(days=holdout)
                bars = full_bars[full_bars["timestamp"] >= cutoff].reset_index(drop=True)
                actions = full_actions[full_actions["timestamp"] >= cutoff].reset_index(drop=True)

                cfg = PortfolioConfig(
                    initial_cash=10000, max_positions=5, min_edge=0.012,
                    max_hold_hours=6, enforce_market_hours=True, close_at_eod=True,
                    symbols=symbols, decision_lag_bars=1, market_order_entry=False,
                    bar_margin=0.0005, max_leverage=2.0, force_close_slippage=0.003,
                    int_qty=True,
                    fee_by_symbol={s: 0.001 for s in symbols},
                    margin_annual_rate=0.0625,
                )
                r = run_portfolio_simulation(bars, actions, cfg, horizon=1)
                ret = r.metrics["total_return"] * 100
                sort = r.metrics["sortino"]
                buys = r.metrics["num_buys"]
                dd = r.metrics.get("max_drawdown", 0) * 100

                logger.info("{:25s} ep{:2d} {:3d}d: ret={:+7.2f}% sort={:6.2f} dd={:+5.1f}% buys={}",
                             config_name, epoch, holdout, ret, sort, dd, buys)
                results.append({
                    "config": config_name, "epoch": epoch, "holdout_days": holdout,
                    "return": round(ret, 4), "sortino": round(sort, 4),
                    "max_drawdown": round(dd, 4), "buys": buys,
                })

            del model
            torch.cuda.empty_cache()

    out_path = Path("unified_hourly_experiment/multi_period_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved {} results to {}", len(results), out_path)

    # Print summary table
    logger.info("\n" + "=" * 90)
    logger.info("{:25s} {:>4s} {:>6s} {:>8s} {:>8s} {:>8s} {:>8s}",
                "Config", "Ep", "Period", "Return%", "Sortino", "MaxDD%", "Buys")
    logger.info("-" * 90)
    for r in sorted(results, key=lambda x: (-x["sortino"])):
        logger.info("{:25s} {:4d} {:5d}d {:+8.2f} {:8.2f} {:+8.1f} {:>6d}",
                     r["config"], r["epoch"], r["holdout_days"],
                     r["return"], r["sortino"], r["max_drawdown"], r["buys"])

if __name__ == "__main__":
    main()
