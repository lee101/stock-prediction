#!/usr/bin/env python3
"""Sweep min_edge threshold to find optimal value."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import torch
from loguru import logger

from binanceneural.data import BinanceHourlyDataModule, FeatureNormalizer
from binanceneural.config import DatasetConfig
from binanceneural.model import build_policy, PolicyConfig
from binanceneural.inference import generate_actions_from_frame
from unified_hourly_experiment.marketsimulator import UnifiedSelectionConfig, run_unified_simulation
from src.torch_load_utils import torch_load_compat

SYMBOLS = ["NVDA", "MSFT", "META", "GOOG", "NET", "PLTR", "NYT", "YELP", "DBX", "TRIP", "KIND", "EBAY", "MTCH", "ANGI", "Z", "EXPE", "BKNG", "NWSA"]
CHECKPOINT_DIR = Path("unified_hourly_experiment/checkpoints/nas_512h_4L")
DATA_ROOT = Path("trainingdatahourly/stocks")
CACHE_ROOT = Path("unified_hourly_experiment/forecast_cache")

def load_model(checkpoint_dir: Path):
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    ckpt = torch_load_compat(checkpoints[-1], map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    with open(checkpoint_dir / "config.json") as f:
        config = json.load(f)
    feature_columns = config.get("feature_columns", [])
    input_dim = len(feature_columns)
    sequence_length = config.get("sequence_length", 32)
    hidden_dim = config.get("transformer_dim", 128)
    num_heads = config.get("transformer_heads", 4)
    num_layers = config.get("transformer_layers", 3)
    policy_cfg = PolicyConfig(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers, model_arch="gemma", max_len=sequence_length)
    model = build_policy(policy_cfg)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, feature_columns, sequence_length

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, feature_columns, seq_len = load_model(CHECKPOINT_DIR)
    model = model.to(device)

    all_bars = []
    all_actions = []
    data_modules = {}

    for symbol in SYMBOLS:
        data_config = DatasetConfig(
            symbol=symbol, data_root=str(DATA_ROOT), forecast_cache_root=str(CACHE_ROOT),
            forecast_horizons=[1, 24], sequence_length=seq_len, min_history_hours=100,
            validation_days=30, cache_only=True,
        )
        try:
            dm = BinanceHourlyDataModule(data_config)
            data_modules[symbol] = dm
            frame = dm.frame.copy()
            frame["symbol"] = symbol
            all_bars.append(frame)
        except Exception as e:
            logger.warning("Failed {}: {}", symbol, e)

    first_symbol = list(data_modules.keys())[0]
    normalizer = data_modules[first_symbol].normalizer

    for symbol in data_modules:
        frame = data_modules[symbol].frame.copy()
        frame["symbol"] = symbol
        actions_df = generate_actions_from_frame(
            model=model, frame=frame, feature_columns=feature_columns,
            normalizer=normalizer, sequence_length=seq_len, horizon=1, device=device,
        )
        all_actions.append(actions_df)

    bars = pd.concat(all_bars, ignore_index=True)
    actions = pd.concat(all_actions, ignore_index=True)

    results = []
    for min_edge in [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01]:
        sim_config = UnifiedSelectionConfig(
            initial_cash=10000, min_edge=min_edge, enforce_market_hours=True,
            close_at_eod=True, symbols=SYMBOLS, max_leverage_stock=1.0, max_leverage_crypto=1.0,
        )
        result = run_unified_simulation(bars, actions, sim_config, horizon=1)
        ret = (result.equity_curve.iloc[-1] / 10000 - 1) * 100
        logger.info("min_edge={:.4f}: Return={:.2f}%, Sortino={:.2f}, Trades={}",
                   min_edge, ret, result.metrics["sortino"], len(result.trades))
        results.append({"min_edge": min_edge, "return": ret, "sortino": result.metrics["sortino"], "trades": len(result.trades)})

    logger.info("=" * 60)
    best = max(results, key=lambda x: x["return"])
    logger.info("Best: min_edge={}, Return={:.2f}%, Sortino={:.2f}", best["min_edge"], best["return"], best["sortino"])

if __name__ == "__main__":
    main()
