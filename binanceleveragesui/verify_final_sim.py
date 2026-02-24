#!/usr/bin/env python3
"""Final sim verification with all prod-matching params."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceneural.inference import generate_actions_from_frame
from binanceleveragesui.run_leverage_sweep import simulate_with_margin_cost, LeverageConfig

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "binanceleveragesui/checkpoints/DOGEUSD_rw30_ep4_full.pt"

dm = ChronosSolDataModule(
    symbol="DOGEUSD",
    data_root=REPO / "trainingdatahourlybinance",
    forecast_cache_root=REPO / "binanceneural/forecast_cache",
    forecast_horizons=(1,), context_hours=512, quantile_levels=(0.1, 0.5, 0.9),
    batch_size=32, model_id="amazon/chronos-t5-small", sequence_length=72,
    split_config=SplitConfig(val_days=30, test_days=30),
    cache_only=True, max_history_days=365,
)

model, normalizer, feature_columns, meta = load_policy_checkpoint(CKPT, device="cuda")
test_frame = dm.test_frame.copy()
actions = generate_actions_from_frame(
    model=model, frame=test_frame, feature_columns=feature_columns,
    normalizer=normalizer, sequence_length=meta.get("sequence_length", 72), horizon=1,
)
bars = test_frame[["timestamp", "symbol", "open", "high", "low", "close"]].copy()

print(f"{'Config':>40} {'Trades':>7} {'Return':>8} {'Sortino':>8} {'MaxDD':>8}")
print("-" * 80)

configs = [
    ("old: buy-first, no hold limit, is=1", dict(max_hold_bars=0, intensity_scale=1.0)),
    ("sell-first, no hold limit, is=1", dict(max_hold_bars=0, intensity_scale=1.0)),
    ("sell-first, hold=6h, is=1", dict(max_hold_bars=6, intensity_scale=1.0)),
    ("sell-first, hold=6h, is=5 (PROD)", dict(max_hold_bars=6, intensity_scale=5.0)),
    ("sell-first, no hold, is=5", dict(max_hold_bars=0, intensity_scale=5.0)),
]

for name, overrides in configs:
    cfg = LeverageConfig(
        symbol="DOGEUSD", max_leverage=1.0, can_short=False,
        maker_fee=0.001, margin_hourly_rate=0.0,
        initial_cash=10000.0, fill_buffer_pct=0.0013,
        decision_lag_bars=1, min_edge=0.0,
        **overrides,
    )
    r = simulate_with_margin_cost(bars, actions, cfg)
    print(f"{name:>40} {r['num_trades']:>7} {r['total_return']*100:>7.1f}% {r['sortino']:>8.2f} {r['max_drawdown']*100:>7.1f}%")
