#!/usr/bin/env python3
"""Independent verification of DOGE results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import pandas as pd
from binanceneural.inference import generate_actions_from_frame
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import _build_policy
from binanceleveragesui.run_leverage_sweep import (
    LeverageConfig, SUI_HOURLY_MARGIN_RATE, MAKER_FEE_10BP,
    simulate_with_margin_cost,
)

REPO = Path(__file__).resolve().parents[1]
FILL_BUFFER = 0.0013

dm = ChronosSolDataModule(
    symbol="DOGEUSD",
    data_root=REPO / "trainingdatahourlybinance",
    forecast_cache_root=REPO / "binanceneural" / "forecast_cache",
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

ckpt = REPO / "binanceleveragesui/checkpoints/DOGEUSD_rw30_wd03/binanceneural_20260222_063258/epoch_004.pt"
payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
sd = payload.get("state_dict", payload)
cfg = payload.get("config", {})
feature_columns = list(dm.feature_columns)
model = _build_policy(sd, cfg, len(feature_columns))

test_frame = dm.test_frame.copy()
actions = generate_actions_from_frame(
    model=model, frame=test_frame, feature_columns=feature_columns,
    normalizer=dm.normalizer, sequence_length=72, horizon=1,
)
test_start = dm.test_window_start
bars = test_frame[test_frame["timestamp"] >= test_start].copy()
actions_test = actions[actions["timestamp"] >= test_start].copy()

print(f"Test period: {bars['timestamp'].min()} to {bars['timestamp'].max()}")
print(f"Test bars: {len(bars)}, Actions: {len(actions_test)}")
print(f"Price range: {bars['close'].min():.4f} - {bars['close'].max():.4f}")
print(f"Buy/hold return: {(bars['close'].iloc[-1] / bars['close'].iloc[0] - 1)*100:.1f}%")
print()

# Action distribution
print(f"Action columns: {list(actions_test.columns)}")
act_col = [c for c in actions_test.columns if 'act' in c.lower() or 'pos' in c.lower() or 'signal' in c.lower()]
print(f"Likely action cols: {act_col}")
if act_col:
    col = act_col[0]
    print(f"{col} distribution: {dict(actions_test[col].value_counts().head(10))}")
    print(f"Mean {col}: {actions_test[col].mean():.3f}")
print()

for min_edge in [0.0, 0.006, 0.010]:
    lcfg = LeverageConfig(
        max_leverage=1.0, initial_cash=5000.0,
        decision_lag_bars=1, fill_buffer_pct=FILL_BUFFER,
        margin_hourly_rate=SUI_HOURLY_MARGIN_RATE,
        maker_fee=MAKER_FEE_10BP,
        min_edge=min_edge,
    )
    r = simulate_with_margin_cost(bars, actions_test, lcfg)
    print(f"edge={min_edge:.3f}: sort={r['sortino']:.2f} ret={r['total_return']*100:+.1f}% trades={r['num_trades']} final=${r['final_equity']:.0f}")
