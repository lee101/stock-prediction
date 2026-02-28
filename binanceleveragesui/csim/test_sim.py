#!/usr/bin/env python3
"""Test C sim matches Python sim, benchmark speed."""
import sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from binanceneural.inference import generate_actions_from_frame
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceleveragesui.run_leverage_sweep import LeverageConfig, simulate_with_margin_cost
from binanceleveragesui.csim.fast_sim import simulate_fast

DATA_ROOT = REPO / "trainingdatahourlybinance"
FORECAST_CACHE = REPO / "binanceneural/forecast_cache"
CKPT = REPO / "binanceleveragesui/checkpoints/DOGEUSD_r4_R4_h384_cosine/binanceneural_20260301_065549/epoch_001.pt"

model, normalizer, feature_columns, meta = load_policy_checkpoint(CKPT, device="cuda")
seq_len = meta.get("sequence_length", 72)

dm = ChronosSolDataModule(
    symbol="DOGEUSD", data_root=DATA_ROOT,
    forecast_cache_root=FORECAST_CACHE, forecast_horizons=(1,),
    context_hours=512, quantile_levels=(0.1, 0.5, 0.9),
    batch_size=32, model_id="amazon/chronos-t5-small",
    sequence_length=seq_len,
    split_config=SplitConfig(val_days=30, test_days=30),
    cache_only=True, max_history_days=365,
)

actions = generate_actions_from_frame(
    model=model, frame=dm.test_frame, feature_columns=feature_columns,
    normalizer=normalizer, sequence_length=seq_len, horizon=1,
)
bars = dm.test_frame[["timestamp", "symbol", "open", "high", "low", "close"]].copy()

cfg = LeverageConfig(
    symbol="DOGEUSD", max_leverage=1.0, can_short=False,
    maker_fee=0.001, margin_hourly_rate=0.0, initial_cash=10000.0,
    fill_buffer_pct=0.0013, decision_lag_bars=1, min_edge=0.0,
    max_hold_bars=6, intensity_scale=5.0,
)

# Python sim
t0 = time.perf_counter()
for _ in range(10):
    py_result = simulate_with_margin_cost(bars, actions, cfg)
py_time = (time.perf_counter() - t0) / 10

# C sim
t0 = time.perf_counter()
for _ in range(10):
    c_result = simulate_fast(bars, actions, cfg, decision_lag_bars=cfg.decision_lag_bars)
c_time = (time.perf_counter() - t0) / 10

print(f"Bars: {len(bars)}")
print(f"\nPython: {py_time*1000:.1f}ms")
print(f"C:      {c_time*1000:.1f}ms")
print(f"Speedup: {py_time/c_time:.1f}x")

print(f"\nPython: sort={py_result['sortino']:.4f} ret={py_result['total_return']:.6f} dd={py_result['max_drawdown']:.6f} trades={py_result['num_trades']}")
print(f"C:      sort={c_result['sortino']:.4f} ret={c_result['total_return']:.6f} dd={c_result['max_drawdown']:.6f} trades={c_result['num_trades']}")

match = (
    abs(py_result['sortino'] - c_result['sortino']) < 0.01 and
    abs(py_result['total_return'] - c_result['total_return']) < 0.001 and
    py_result['num_trades'] == c_result['num_trades']
)
print(f"\nMATCH: {'YES' if match else 'NO'}")
