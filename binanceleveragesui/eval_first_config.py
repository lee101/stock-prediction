#!/usr/bin/env python3
"""Evaluate the first completed config's epochs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binanceleveragesui.train_high_rw_sweep import evaluate_epoch
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig

REPO = Path(__file__).resolve().parents[1]

dm = ChronosSolDataModule(
    symbol="SUIUSDT",
    data_root=REPO / "trainingdatahourlybinance",
    forecast_cache_root=REPO / "binancechronossolexperiment" / "forecast_cache_sui_10bp",
    forecast_horizons=(1, 4, 24),
    context_hours=512,
    quantile_levels=(0.1, 0.5, 0.9),
    batch_size=32,
    model_id="amazon/chronos-t5-small",
    sequence_length=72,
    split_config=SplitConfig(val_days=30, test_days=30),
    cache_only=True,
    max_history_days=365,
)

ckpt_dir = REPO / "binanceleveragesui" / "checkpoints" / "SUIUSDT_rw10_wd03" / "binanceneural_20260221_230946"
for ckpt_path in sorted(ckpt_dir.glob("epoch_*.pt")):
    ep = ckpt_path.stem
    results = evaluate_epoch(ckpt_path, dm, (1, 4, 24), "rw10_wd03")
    e0 = results.get("edge0", {})
    best_k = max(results.keys(), key=lambda k: results[k].get("sortino", -999))
    best_s = results[best_k]["sortino"]
    best_r = results[best_k]["return"]
    best_t = results[best_k]["trades"]
    print(f"{ep}: edge0 sort={e0.get('sortino',0):.2f} ret={e0.get('return',0):+.4f} t={e0.get('trades',0)} | best={best_k} sort={best_s:.2f} ret={best_r:+.4f} t={best_t}")
