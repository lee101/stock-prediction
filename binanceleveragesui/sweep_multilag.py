#!/usr/bin/env python3
"""Multi-lag robust training: average loss across lag=0,1,2 so model
learns strategies that degrade gracefully with execution timing uncertainty.

Queued to run after sweep_lag1.py finishes.
"""
from __future__ import annotations
import json, sys, time
from dataclasses import asdict
from pathlib import Path
import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from binanceneural.config import TrainingConfig
from binanceneural.trainer import BinanceHourlyTrainer
from binanceneural.inference import generate_actions_from_frame
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceleveragesui.run_leverage_sweep import LeverageConfig, simulate_with_margin_cost

FORECAST_CACHE = Path("binancechronossolexperiment/forecast_cache_sui_10bp")
DATA_ROOT = Path("trainingdatahourlybinance")
HORIZONS = (1, 4, 24)

EXPERIMENTS = [
    # lag=0 is the lookahead bias -- never include it in training.
    # Multi-lag "1,2" -- robust across realistic execution delays
    {"name": "multilag12_rw016", "return_weight": 0.016, "lag_range": "1,2"},
    {"name": "multilag12_rw020", "return_weight": 0.020, "lag_range": "1,2"},
    {"name": "multilag12_rw025", "return_weight": 0.025, "lag_range": "1,2"},
    {"name": "multilag12_rw012", "return_weight": 0.012, "lag_range": "1,2"},
    # Multi-lag "1,2,3" -- even more pessimistic timing
    {"name": "multilag123_rw016", "return_weight": 0.016, "lag_range": "1,2,3"},
    {"name": "multilag123_rw020", "return_weight": 0.020, "lag_range": "1,2,3"},
    # Higher rw to compensate for multi-lag drag
    {"name": "multilag12_rw030", "return_weight": 0.030, "lag_range": "1,2"},
    {"name": "multilag12_rw040", "return_weight": 0.040, "lag_range": "1,2"},
    # More epochs since multi-lag is harder to optimize
    {"name": "multilag12_rw016_ep35", "return_weight": 0.016, "lag_range": "1,2", "epochs": 35},
    {"name": "multilag12_rw020_ep35", "return_weight": 0.020, "lag_range": "1,2", "epochs": 35},
    # Cosine LR + multi-lag
    {"name": "multilag12_rw016_cosine", "return_weight": 0.016, "lag_range": "1,2", "lr_schedule": "cosine", "lr_min_ratio": 0.1},
    {"name": "multilag12_rw020_cosine", "return_weight": 0.020, "lag_range": "1,2", "lr_schedule": "cosine", "lr_min_ratio": 0.1},
]


def run_experiment(exp: dict, seed: int = 1337, val_days: int = 20, test_days: int = 10) -> dict:
    name = exp["name"]
    lag_range = exp.get("lag_range", "1")
    logger.info("=== {} (lag_range={}) ===", name, lag_range)

    dm = ChronosSolDataModule(
        symbol="SUIUSDT", data_root=DATA_ROOT,
        forecast_cache_root=FORECAST_CACHE, forecast_horizons=HORIZONS,
        context_hours=512, quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32, model_id="amazon/chronos-t5-small",
        sequence_length=72, split_config=SplitConfig(val_days=val_days, test_days=test_days),
        cache_only=True,
    )

    checkpoint_root = Path(f"binanceleveragesui/checkpoints/sweep_{name}")
    tc = TrainingConfig(
        epochs=exp.get("epochs", 25),
        batch_size=16, sequence_length=72,
        learning_rate=exp.get("learning_rate", 1e-4),
        weight_decay=1e-4,
        return_weight=exp.get("return_weight", 0.016),
        smoothness_penalty=exp.get("smoothness_penalty", 0.0),
        decision_lag_bars=1,
        decision_lag_range=lag_range,
        seed=seed,
        transformer_dim=256, transformer_layers=4, transformer_heads=8,
        maker_fee=0.001,
        checkpoint_root=checkpoint_root,
        log_dir=Path("tensorboard_logs/binanceleveragesui_multilag"),
        use_compile=False,
        lr_schedule=exp.get("lr_schedule", "none"),
        lr_min_ratio=exp.get("lr_min_ratio", 0.0),
    )

    trainer = BinanceHourlyTrainer(tc, dm)
    artifacts = trainer.train()

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    sd = artifacts.state_dict
    if artifacts.best_checkpoint and artifacts.best_checkpoint.exists():
        ckpt = torch.load(artifacts.best_checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
    ckpt_path = checkpoint_root / "policy_checkpoint.pt"
    torch.save({
        "state_dict": sd,
        "config": asdict(tc),
        "feature_columns": list(artifacts.feature_columns),
        "normalizer": artifacts.normalizer.to_dict(),
    }, ckpt_path)

    model, normalizer, feature_columns, _ = load_policy_checkpoint(str(ckpt_path))
    test_frame = dm.test_frame.copy()
    actions = generate_actions_from_frame(
        model=model, frame=test_frame, feature_columns=feature_columns,
        normalizer=normalizer, sequence_length=72, horizon=1,
    )
    test_start = dm.test_window_start
    bars = test_frame[test_frame["timestamp"] >= test_start].copy()
    actions = actions[actions["timestamp"] >= test_start].copy()

    result = {"name": name, "config": exp, "seed": seed, "lag_range": lag_range}
    for lev in [1.0, 3.0, 5.0]:
        for lag in [0, 1, 2]:
            cfg = LeverageConfig(max_leverage=lev, initial_cash=5000.0, decision_lag_bars=lag)
            m = simulate_with_margin_cost(bars, actions, cfg)
            key = f"lev_{lev:.0f}x_lag{lag}"
            result[key] = m
            mult = m["final_equity"] / 5000
            if lag == 1:
                logger.info("  {}x lag={}: {:.3f}x sort={:.1f} dd={:.3f}", lev, lag, mult, m["sortino"], m["max_drawdown"])

    result["best_epoch"] = artifacts.best_epoch if hasattr(artifacts, "best_epoch") else -1
    result["val_sortino"] = artifacts.best_val_sortino if hasattr(artifacts, "best_val_sortino") else 0.0
    return result


def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"binanceleveragesui/sweep_multilag_results_{timestamp}.json")
    all_results = []

    for exp in EXPERIMENTS:
        try:
            result = run_experiment(exp)
            all_results.append(result)
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2)
        except Exception as e:
            logger.error("{} failed: {}", exp["name"], e)
            import traceback; traceback.print_exc()
            all_results.append({"name": exp["name"], "error": str(e)})

    logger.info("\n=== MULTI-LAG SWEEP SUMMARY (lag=1 eval) ===")
    logger.info(f"{'Name':<30} {'1x':>8} {'3x':>8} {'5x':>8} {'5x sort':>8} {'5x dd':>8}")
    for r in all_results:
        if "error" in r:
            logger.info("{}: FAILED", r["name"])
            continue
        def _m(key):
            return r.get(key, {}).get("final_equity", 0) / 5000
        m5 = r.get("lev_5x_lag1", {})
        logger.info("{:<30} {:>8.3f}x {:>8.3f}x {:>8.3f}x {:>8.1f} {:>8.3f}",
                     r["name"], _m("lev_1x_lag1"), _m("lev_3x_lag1"), _m("lev_5x_lag1"),
                     m5.get("sortino", 0), m5.get("max_drawdown", 0))

    # Robustness: show lag0 vs lag1 vs lag2 degradation
    logger.info("\n=== ROBUSTNESS: 5x across lags ===")
    logger.info(f"{'Name':<30} {'lag0':>10} {'lag1':>10} {'lag2':>10} {'degradation':>12}")
    for r in all_results:
        if "error" in r:
            continue
        s0 = r.get("lev_5x_lag0", {}).get("sortino", 0)
        s1 = r.get("lev_5x_lag1", {}).get("sortino", 0)
        s2 = r.get("lev_5x_lag2", {}).get("sortino", 0)
        deg = (s0 - s2) / max(abs(s0), 1e-6) * 100
        logger.info("{:<30} {:>10.1f} {:>10.1f} {:>10.1f} {:>11.1f}%", r["name"], s0, s1, s2, deg)

    logger.info("Saved: {}", output_path)


if __name__ == "__main__":
    main()
