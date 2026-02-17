#!/usr/bin/env python3
"""Sweep min_edge across top model checkpoints to find best deployment config."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import torch
from loguru import logger

from binanceneural.data import BinanceHourlyDataModule
from binanceneural.config import DatasetConfig
from binanceneural.model import build_policy, PolicyConfig
from binanceneural.inference import generate_actions_from_frame
from unified_hourly_experiment.marketsimulator import UnifiedSelectionConfig, run_unified_simulation
from src.torch_load_utils import torch_load_compat

SYMBOLS = ["NVDA", "MSFT", "META", "GOOG", "NET", "PLTR", "NYT", "YELP", "DBX", "TRIP", "KIND", "EBAY", "MTCH", "ANGI", "Z", "EXPE", "BKNG", "NWSA"]
DATA_ROOT = Path("trainingdatahourly/stocks")
CACHE_ROOT = Path("unified_hourly_experiment/forecast_cache")

MODELS = [
    ("exp_512h_4L_lr5e5", "epoch_033.pt"),
    ("exp_512h_4L_lr5e5", "epoch_031.pt"),
    ("exp_512h_6L", "epoch_006.pt"),
    ("exp_512h_6L", "epoch_007.pt"),
    ("exp_512h_4L_seq64", "epoch_010.pt"),
    ("nas_512h_4L", "epoch_028.pt"),
    ("nas_512h_4L", "epoch_003.pt"),
]

MIN_EDGES = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015]

def load_model(checkpoint_dir: Path, checkpoint_name: str):
    ckpt_path = checkpoint_dir / checkpoint_name
    ckpt = torch_load_compat(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    with open(checkpoint_dir / "config.json") as f:
        config = json.load(f)
    feature_columns = config.get("feature_columns", [])
    policy_cfg = PolicyConfig(
        input_dim=len(feature_columns),
        hidden_dim=config.get("transformer_dim", 128),
        num_heads=config.get("transformer_heads", 4),
        num_layers=config.get("transformer_layers", 3),
        model_arch="gemma",
        max_len=config.get("sequence_length", 32),
    )
    model = build_policy(policy_cfg)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, feature_columns, config.get("sequence_length", 32)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_modules = {}
    for symbol in SYMBOLS:
        data_config = DatasetConfig(
            symbol=symbol, data_root=str(DATA_ROOT), forecast_cache_root=str(CACHE_ROOT),
            forecast_horizons=[1, 24], sequence_length=64, min_history_hours=100,
            validation_days=30, cache_only=True,
        )
        try:
            data_modules[symbol] = BinanceHourlyDataModule(data_config)
        except Exception as e:
            logger.warning("Skip {}: {}", symbol, e)

    normalizer = list(data_modules.values())[0].normalizer

    all_results = []
    ckpt_root = Path("unified_hourly_experiment/checkpoints")

    for model_name, ckpt_name in MODELS:
        ckpt_dir = ckpt_root / model_name
        if not (ckpt_dir / ckpt_name).exists():
            logger.warning("Missing {}/{}", model_name, ckpt_name)
            continue

        logger.info("=== {} / {} ===", model_name, ckpt_name)
        model, feature_columns, seq_len = load_model(ckpt_dir, ckpt_name)
        model = model.to(device)

        all_bars = []
        all_actions = []
        for symbol in data_modules:
            frame = data_modules[symbol].frame.copy()
            frame["symbol"] = symbol
            all_bars.append(frame)
            actions_df = generate_actions_from_frame(
                model=model, frame=frame, feature_columns=feature_columns,
                normalizer=normalizer, sequence_length=seq_len, horizon=1, device=device,
            )
            all_actions.append(actions_df)

        bars = pd.concat(all_bars, ignore_index=True)
        actions = pd.concat(all_actions, ignore_index=True)

        for min_edge in MIN_EDGES:
            sim_config = UnifiedSelectionConfig(
                initial_cash=10000, min_edge=min_edge, enforce_market_hours=True,
                close_at_eod=True, symbols=SYMBOLS, max_leverage_stock=1.0, max_leverage_crypto=1.0,
            )
            result = run_unified_simulation(bars, actions, sim_config, horizon=1)
            ret = (result.equity_curve.iloc[-1] / 10000 - 1) * 100
            sortino = result.metrics.get("sortino", 0)
            dd = result.metrics.get("max_drawdown", 0)
            trades = len(result.trades)
            logger.info("  me={:.4f}: ret={:+.1f}% sort={:.2f} dd={:.1f}% trades={}",
                       min_edge, ret, sortino, dd * 100, trades)
            all_results.append({
                "model": model_name, "checkpoint": ckpt_name, "min_edge": min_edge,
                "return": ret, "sortino": sortino, "max_drawdown": dd, "trades": trades,
            })

    out_path = Path("unified_hourly_experiment/best_models_sweep.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("\n" + "=" * 80)
    best_sortino = max(all_results, key=lambda x: x["sortino"])
    best_return = max(all_results, key=lambda x: x["return"])
    logger.info("BEST SORTINO: {}/{} me={} -> ret={:.1f}% sort={:.2f}",
                best_sortino["model"], best_sortino["checkpoint"], best_sortino["min_edge"],
                best_sortino["return"], best_sortino["sortino"])
    logger.info("BEST RETURN:  {}/{} me={} -> ret={:.1f}% sort={:.2f}",
                best_return["model"], best_return["checkpoint"], best_return["min_edge"],
                best_return["return"], best_return["sortino"])

if __name__ == "__main__":
    main()
