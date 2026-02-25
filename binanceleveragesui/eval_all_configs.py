#!/usr/bin/env python3
"""Evaluate all SUI sweep checkpoints with per-epoch + min_edge sweep."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from loguru import logger
from binanceneural.inference import generate_actions_from_frame
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import _build_policy
from binanceleveragesui.run_leverage_sweep import (
    LeverageConfig, SUI_HOURLY_MARGIN_RATE, MAKER_FEE_10BP,
    simulate_with_margin_cost,
)

REPO = Path(__file__).resolve().parents[1]
CKPT_ROOT = REPO / "binanceleveragesui" / "checkpoints"
FILL_BUFFER = 0.0005
MIN_EDGES = [0.0, 0.002, 0.004, 0.006, 0.008, 0.010]
HORIZONS = (1, 4, 24)

dm = ChronosSolDataModule(
    symbol="SUIUSDT",
    data_root=REPO / "trainingdatahourlybinance",
    forecast_cache_root=REPO / "binancechronossolexperiment" / "forecast_cache_sui_10bp",
    forecast_horizons=HORIZONS,
    context_hours=512,
    quantile_levels=(0.1, 0.5, 0.9),
    batch_size=32,
    model_id="amazon/chronos-t5-small",
    sequence_length=72,
    split_config=SplitConfig(val_days=30, test_days=30),
    cache_only=True,
    max_history_days=365,
)

feature_columns = list(dm.feature_columns)
normalizer = dm.normalizer
test_frame = dm.test_frame.copy()
test_start = dm.test_window_start

configs = sorted(CKPT_ROOT.glob("SUIUSDT_*"))
all_results = []
global_best_sort = -999
global_best_info = ""

for cfg_dir in configs:
    tag = cfg_dir.name
    ckpt_files = sorted(cfg_dir.rglob("epoch_*.pt"))
    if not ckpt_files:
        logger.warning(f"{tag}: no checkpoints")
        continue

    config_best_sort = -999
    config_best_ep = ""
    config_best_edge = ""

    for ckpt_path in ckpt_files:
        ep = ckpt_path.stem
        try:
            payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            sd = payload.get("state_dict", payload)
            cfg = payload.get("config", {})
            model = _build_policy(sd, cfg, len(feature_columns))

            actions = generate_actions_from_frame(
                model=model, frame=test_frame, feature_columns=feature_columns,
                normalizer=normalizer, sequence_length=72, horizon=HORIZONS[0],
            )
            bars = test_frame[test_frame["timestamp"] >= test_start].copy()
            actions_test = actions[actions["timestamp"] >= test_start].copy()

            edge_results = {}
            for min_edge in MIN_EDGES:
                lcfg = LeverageConfig(
                    max_leverage=1.0, initial_cash=5000.0,
                    decision_lag_bars=1, fill_buffer_pct=FILL_BUFFER,
                    margin_hourly_rate=SUI_HOURLY_MARGIN_RATE,
                    maker_fee=MAKER_FEE_10BP,
                    min_edge=min_edge,
                )
                r = simulate_with_margin_cost(bars, actions_test, lcfg)
                ek = f"edge{int(min_edge*1000)}"
                edge_results[ek] = {
                    "return": r["total_return"],
                    "sortino": r["sortino"],
                    "trades": r["num_trades"],
                }

            best_ek = max(edge_results.keys(), key=lambda k: edge_results[k]["sortino"])
            best_s = edge_results[best_ek]["sortino"]
            best_r = edge_results[best_ek]["return"]
            best_t = edge_results[best_ek]["trades"]
            e0 = edge_results["edge0"]

            logger.info(f"{tag} {ep}: e0 s={e0['sortino']:.2f} r={e0['return']:+.4f} t={e0['trades']} | best={best_ek} s={best_s:.2f} r={best_r:+.4f} t={best_t}")

            if best_s > config_best_sort:
                config_best_sort = best_s
                config_best_ep = ep
                config_best_edge = best_ek

        except Exception as e:
            logger.warning(f"{tag} {ep}: FAILED {e}")

    logger.info(f"  >> {tag} BEST: {config_best_ep} {config_best_edge} sort={config_best_sort:.2f}")
    all_results.append({
        "tag": tag, "best_epoch": config_best_ep,
        "best_edge": config_best_edge, "best_sortino": config_best_sort,
    })

    if config_best_sort > global_best_sort:
        global_best_sort = config_best_sort
        global_best_info = f"{tag} {config_best_ep} {config_best_edge}"

logger.info(f"\n=== GLOBAL BEST: {global_best_info} sort={global_best_sort:.2f} ===")

out_path = REPO / "binanceleveragesui" / "all_configs_eval.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
logger.info(f"Saved to {out_path}")
