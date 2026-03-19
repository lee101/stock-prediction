#!/usr/bin/env python3
"""Multi-window evaluation of deployed DOGE checkpoint."""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
from loguru import logger
from binanceneural.inference import generate_actions_from_frame
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceleveragesui.run_leverage_sweep import LeverageConfig, simulate_with_margin_cost

CHECKPOINT = str(REPO / "binanceleveragesui/checkpoints/DOGEUSD_r4_R4_h384_cosine/binanceneural_20260301_065549/epoch_001.pt")
EVAL_SYMBOLS = ["DOGEUSD", "BTCUSD", "ETHUSD", "SOLUSD", "LINKUSD", "AAVEUSD"]
WINDOWS = [
    (30, 3, "3d"),
    (30, 7, "7d"),
    (30, 14, "14d"),
    (30, 30, "30d"),
    (30, 60, "60d"),
    (30, 90, "90d"),
    (30, 120, "120d"),
]
LEVERAGE_LEVELS = [1.0, 1.5, 2.0, 3.0]

model, normalizer, feature_columns, meta = load_policy_checkpoint(CHECKPOINT, device="cuda")
seq_len = meta.get("sequence_length", 72)

all_results = {}

for val_days, test_days, label in WINDOWS:
    logger.info(f"\n=== Window: {label} (val={val_days}d, test={test_days}d) ===")
    window_results = {}

    for symbol in EVAL_SYMBOLS:
        try:
            dm = ChronosSolDataModule(
                symbol=symbol, data_root=REPO / "trainingdatahourlybinance",
                forecast_cache_root=REPO / "binanceneural/forecast_cache",
                forecast_horizons=(1,), context_hours=512,
                quantile_levels=(0.1, 0.5, 0.9), batch_size=32,
                model_id="amazon/chronos-t5-small", sequence_length=seq_len,
                split_config=SplitConfig(val_days=val_days, test_days=test_days),
                cache_only=True, max_history_days=365,
            )
            actions = generate_actions_from_frame(
                model=model, frame=dm.test_frame, feature_columns=feature_columns,
                normalizer=normalizer, sequence_length=seq_len, horizon=1,
            )
            bars = dm.test_frame[["timestamp", "symbol", "open", "high", "low", "close"]].copy()
            test_start = dm.test_window_start
            bars_test = bars[bars["timestamp"] >= test_start]
            actions_test = actions[actions["timestamp"] >= test_start]

            sym_results = {}
            for lev in LEVERAGE_LEVELS:
                cfg = LeverageConfig(
                    symbol=symbol, max_leverage=lev, can_short=False,
                    maker_fee=0.001, margin_hourly_rate=0.0000025457 if lev > 1 else 0.0,
                    initial_cash=3300, fill_buffer_pct=0.0005,
                    decision_lag_bars=1, min_edge=0.0, max_hold_bars=6,
                    intensity_scale=5.0,
                )
                r = simulate_with_margin_cost(bars_test, actions_test, cfg)
                sym_results[f"{lev}x"] = {
                    "return": round(r["total_return"] * 100, 2),
                    "sortino": round(r["sortino"], 2),
                    "max_dd": round(r["max_drawdown"] * 100, 2),
                    "trades": r["num_trades"],
                    "margin_cost": round(r["margin_cost_pct"], 3),
                }

            window_results[symbol] = sym_results
            s1 = sym_results["1.0x"]
            logger.info(f"  {symbol}: Sort={s1['sortino']:.1f} Ret={s1['return']:+.1f}% DD={s1['max_dd']:.1f}% Trades={s1['trades']}")

        except Exception as e:
            logger.warning(f"  {symbol}: SKIP ({e})")
            window_results[symbol] = {"error": str(e)}

    # Summary for window
    doge = window_results.get("DOGEUSD", {}).get("1.0x", {})
    sortinos_1x = [v.get("1.0x", {}).get("sortino", -999) for v in window_results.values() if "error" not in v]
    positive = sum(1 for s in sortinos_1x if s > 0)
    mean_sort = np.mean(sortinos_1x) if sortinos_1x else 0

    logger.info(f"  SUMMARY {label}: DOGE={doge.get('sortino',0):.1f} Mean={mean_sort:.1f} Pos={positive}/{len(sortinos_1x)}")

    # Leverage comparison for DOGE
    doge_all = window_results.get("DOGEUSD", {})
    if "error" not in doge_all:
        for lk in ["1.0x", "1.5x", "2.0x", "3.0x"]:
            d = doge_all.get(lk, {})
            logger.info(f"    DOGE {lk}: Sort={d.get('sortino',0):.1f} Ret={d.get('return',0):+.1f}% DD={d.get('max_dd',0):.1f}%")

    all_results[label] = window_results

# Final summary table
logger.info(f"\n{'='*80}\nFINAL MULTI-WINDOW SUMMARY (DOGE deployed checkpoint ep001)\n{'='*80}")
logger.info(f"{'Window':<8} {'Sort@1x':>8} {'Ret@1x':>8} {'DD@1x':>8} {'Sort@2x':>8} {'Ret@2x':>8} {'DD@2x':>8}")
logger.info("-" * 60)
for label in [w[2] for w in WINDOWS]:
    d = all_results.get(label, {}).get("DOGEUSD", {})
    if "error" in d:
        logger.info(f"{label:<8} {'SKIP':>8}")
        continue
    d1 = d.get("1.0x", {})
    d2 = d.get("2.0x", {})
    logger.info(f"{label:<8} {d1.get('sortino',0):>8.1f} {d1.get('return',0):>+7.1f}% {d1.get('max_dd',0):>7.1f}% "
                f"{d2.get('sortino',0):>8.1f} {d2.get('return',0):>+7.1f}% {d2.get('max_dd',0):>7.1f}%")

out_path = REPO / "binanceleveragesui" / "doge_multiwindow_eval.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
logger.info(f"\nSaved: {out_path}")
