#!/usr/bin/env python3
"""Full pipeline: retrain Chronos-2 LoRA -> regenerate forecast cache -> retrain policy.

Run on remote 5090:
  .venv313/bin/python -u binanceleveragesui/retrain_full_pipeline.py
"""
from __future__ import annotations

import json, subprocess, sys, time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SYMBOL = "SUIUSDT"
DATA_ROOT = REPO / "trainingdatahourlybinance"
FORECAST_CACHE = REPO / "binancechronossolexperiment" / "forecast_cache_sui_10bp"
LORA_OUTPUT_ROOT = REPO / "chronos2_finetuned"
CHECKPOINT_ROOT = REPO / "binanceleveragesui" / "checkpoints"

LORA_STEPS = 2000
LORA_LR = 1e-4
LORA_CTX = 1024
LORA_BATCH = 64
HORIZONS = [1, 4, 24]
CONTEXT_HOURS = 512
QUANTILES = [0.1, 0.5, 0.9]
FORECAST_BATCH = 32

POLICY_EPOCHS = 25
POLICY_LR = 1e-4
POLICY_RW_VALUES = [0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.025, 0.030]


def step1_retrain_lora():
    from loguru import logger
    run_id = time.strftime("%Y%m%d_%H%M%S")
    save_name = f"binance_lora_{run_id}_{SYMBOL}"
    logger.info(f"Step 1: Retrain LoRA -> {save_name}")

    cmd = [
        sys.executable, str(REPO / "chronos2_trainer.py"),
        "--symbol", SYMBOL,
        "--data-root", str(DATA_ROOT),
        "--output-root", str(LORA_OUTPUT_ROOT),
        "--finetune-mode", "lora",
        "--learning-rate", str(LORA_LR),
        "--num-steps", str(LORA_STEPS),
        "--context-length", str(LORA_CTX),
        "--batch-size", str(LORA_BATCH),
        "--val-hours", "168",
        "--test-hours", "168",
        "--torch-dtype", "bfloat16",
        "--save-name", save_name,
    ]
    logger.info(f"  cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO))
    if result.returncode != 0:
        raise RuntimeError(f"LoRA training failed (exit {result.returncode})")

    ckpt_dir = LORA_OUTPUT_ROOT / save_name / "finetuned-ckpt"
    if not ckpt_dir.exists():
        raise RuntimeError(f"LoRA checkpoint not found: {ckpt_dir}")

    report_dir = REPO / "hyperparams" / "chronos2" / "hourly_lora"
    reports = sorted(report_dir.glob(f"{SYMBOL}_lora_{save_name}*.json"))
    if reports:
        with open(reports[-1]) as f:
            rpt = json.load(f)
        val_mae = rpt.get("val_metrics", {}).get("mae_percent", "?")
        test_mae = rpt.get("test_metrics", {}).get("mae_percent", "?")
        logger.info(f"  LoRA done. val_mae={val_mae}% test_mae={test_mae}%")
    else:
        logger.info(f"  LoRA done. checkpoint: {ckpt_dir}")
    return str(ckpt_dir)


def step2_regenerate_forecasts(lora_model_id: str):
    from loguru import logger
    logger.info(f"Step 2: Regenerate forecast cache with {lora_model_id}")

    from binancechronossolexperiment.forecasts import build_forecast_bundle
    result = build_forecast_bundle(
        symbol=SYMBOL,
        data_root=DATA_ROOT,
        cache_root=FORECAST_CACHE,
        horizons=HORIZONS,
        context_hours=CONTEXT_HOURS,
        quantile_levels=QUANTILES,
        batch_size=FORECAST_BATCH,
        model_id=lora_model_id,
        cache_only=False,
    )
    logger.info(f"  Forecast cache: {len(result)} rows, last={result['timestamp'].max()}")
    return result


def step3_retrain_policies():
    from loguru import logger
    logger.info(f"Step 3: Retrain policies (rw sweep: {POLICY_RW_VALUES})")

    from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
    from binanceleveragesui.run_leverage_sweep import (
        LeverageConfig, SUI_HOURLY_MARGIN_RATE, MAKER_FEE_10BP,
        simulate_with_margin_cost,
    )
    from binanceneural.trainer import BinanceHourlyTrainer
    from binanceneural.config import TrainingConfig
    from binanceneural.inference import generate_actions_from_frame
    from binancechronossolexperiment.inference import load_policy_checkpoint

    horizons = tuple(HORIZONS)
    results = {}

    for rw in POLICY_RW_VALUES:
        tag = f"retrain_rw{str(rw).replace('.','')}"
        logger.info(f"\n=== Training {tag} (rw={rw}) ===")

        dm = ChronosSolDataModule(
            symbol=SYMBOL,
            data_root=DATA_ROOT,
            forecast_cache_root=FORECAST_CACHE,
            forecast_horizons=horizons,
            context_hours=CONTEXT_HOURS,
            quantile_levels=tuple(QUANTILES),
            batch_size=32,
            model_id="amazon/chronos-t5-small",
            sequence_length=72,
            split_config=SplitConfig(val_days=20, test_days=10),
            cache_only=True,
        )

        ckpt_root = CHECKPOINT_ROOT / f"sweep_{tag}"
        tc = TrainingConfig(
            epochs=POLICY_EPOCHS,
            batch_size=16,
            sequence_length=72,
            learning_rate=POLICY_LR,
            weight_decay=1e-4,
            return_weight=rw,
            seed=1337,
            transformer_dim=256,
            transformer_layers=4,
            transformer_heads=8,
            maker_fee=MAKER_FEE_10BP,
            checkpoint_root=ckpt_root,
            log_dir=Path("tensorboard_logs/binanceleveragesui"),
            use_compile=False,
        )

        trainer = BinanceHourlyTrainer(tc, dm)
        artifacts = trainer.train()

        ckpt_root.mkdir(parents=True, exist_ok=True)
        sd = artifacts.state_dict
        if artifacts.best_checkpoint and artifacts.best_checkpoint.exists():
            ckpt = torch.load(artifacts.best_checkpoint, map_location="cpu", weights_only=False)
            sd = ckpt.get("state_dict", ckpt)
        ckpt_path = ckpt_root / "policy_checkpoint.pt"
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
            normalizer=normalizer, sequence_length=72, horizon=horizons[0],
        )
        test_start = dm.test_window_start
        bars = test_frame[test_frame["timestamp"] >= test_start].copy()
        actions_test = actions[actions["timestamp"] >= test_start].copy()

        row_results = {}
        for lev in [1.0, 3.0, 5.0]:
            for lag in [0, 1, 2]:
                lcfg = LeverageConfig(
                    max_leverage=lev, initial_cash=5000.0,
                    decision_lag_bars=lag,
                    margin_hourly_rate=SUI_HOURLY_MARGIN_RATE,
                    maker_fee=MAKER_FEE_10BP,
                )
                r = simulate_with_margin_cost(bars, actions_test, lcfg)
                key = f"{lev:.0f}x_lag{lag}"
                row_results[key] = {
                    "return": r["total_return"] + 1,
                    "sortino": r["sortino"],
                    "max_dd": r["max_drawdown"],
                }
                logger.info(f"  {key}: {r['total_return']+1:.3f}x sort={r['sortino']:.1f} dd={r['max_drawdown']:.4f}")

        results[tag] = {
            "rw": rw,
            "checkpoint": str(ckpt_path),
            "evals": row_results,
        }

        out_path = REPO / "binanceleveragesui" / f"result_{tag}.json"
        with open(out_path, "w") as f:
            json.dump(results[tag], f, indent=2)

    summary_path = REPO / "binanceleveragesui" / "retrain_sweep_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nAll results saved to {summary_path}")

    best_tag = max(results, key=lambda t: results[t]["evals"].get("1.0x_lag1", {}).get("sortino", -999))
    best = results[best_tag]
    logger.info(f"\nBest at 1x lag=1: {best_tag} -> sort={best['evals']['1.0x_lag1']['sortino']:.1f}")
    return results


def main():
    from loguru import logger
    logger.info("=== FULL RETRAIN PIPELINE ===")

    lora_path = step1_retrain_lora()
    step2_regenerate_forecasts(lora_path)
    step3_retrain_policies()

    logger.info("=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
