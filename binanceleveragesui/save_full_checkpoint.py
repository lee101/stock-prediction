#!/usr/bin/env python3
"""Save full checkpoint with normalizer+feature_columns from epoch checkpoint."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig

REPO = Path(__file__).resolve().parents[1]

symbol = sys.argv[1] if len(sys.argv) > 1 else "DOGEUSD"
epoch_ckpt = sys.argv[2] if len(sys.argv) > 2 else str(REPO / "binanceleveragesui/checkpoints/DOGEUSD_rw30_wd03/binanceneural_20260222_063258/epoch_004.pt")
out_path = sys.argv[3] if len(sys.argv) > 3 else str(REPO / "binanceleveragesui/checkpoints/DOGEUSD_rw30_ep4_full.pt")

forecast_cache = REPO / "binanceneural" / "forecast_cache"

dm = ChronosSolDataModule(
    symbol=symbol,
    data_root=REPO / "trainingdatahourlybinance",
    forecast_cache_root=forecast_cache,
    forecast_horizons=(1,),
    context_hours=512,
    quantile_levels=(0.1, 0.5, 0.9),
    batch_size=32,
    model_id="amazon/chronos-t5-small",
    sequence_length=72,
    split_config=SplitConfig(val_days=30, test_days=30),
    cache_only=True,
    max_history_days=365,
)

payload = torch.load(epoch_ckpt, map_location="cpu", weights_only=False)
payload["normalizer"] = dm.normalizer.to_dict()
payload["feature_columns"] = list(dm.feature_columns)
torch.save(payload, out_path)
print(f"Saved full checkpoint to {out_path}")
print(f"  Features: {len(payload['feature_columns'])}")
print(f"  Keys: {list(payload.keys())}")
