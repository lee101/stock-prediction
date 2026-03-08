#!/usr/bin/env python3
"""Train and evaluate DOGE/AAVE meta candidates with alternate Chronos feature horizons."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.forecasts import build_forecast_bundle
from binanceneural.config import DatasetConfig, TrainingConfig
from binanceneural.trainer import BinanceHourlyTrainer

DATA_ROOT = REPO / "trainingdatahourlybinance"
LIVE_DOGE_CKPT = REPO / "deployments/binance-meta-margin/20260307_gate24_calmar_short016_full/checkpoints/doge_epoch_010.pt"
LIVE_AAVE_CKPT = REPO / "deployments/binance-meta-margin/20260307_gate24_calmar_short016_full/checkpoints/aave_epoch_002.pt"
DEFAULT_EXPERIMENT_ROOT = REPO / "experiments" / "meta_feature_horizon_sweep_20260307"

SYMBOL_CONFIGS = {
    "doge": {
        "data_symbol": "DOGEUSD",
        "live_checkpoint": LIVE_DOGE_CKPT,
        "training": {
            "epochs": 6,
            "batch_size": 16,
            "sequence_length": 72,
            "learning_rate": 1e-4,
            "weight_decay": 0.04,
            "maker_fee": 0.001,
            "return_weight": 0.10,
            "fill_temperature": 0.1,
            "fill_buffer_pct": 0.0005,
            "loss_type": "sortino",
            "decision_lag_bars": 1,
            "lr_schedule": "cosine",
            "lr_min_ratio": 0.01,
            "transformer_dim": 384,
            "transformer_layers": 6,
            "transformer_heads": 8,
            "transformer_dropout": 0.1,
            "model_arch": "nano",
            "num_memory_tokens": 8,
            "dilated_strides": "1,2,6,24",
            "num_outputs": 4,
            "max_hold_hours": 24.0,
            "use_compile": False,
            "seed": 1337,
        },
    },
    "aave": {
        "data_symbol": "AAVEUSD",
        "live_checkpoint": LIVE_AAVE_CKPT,
        "training": {
            "epochs": 6,
            "batch_size": 16,
            "sequence_length": 72,
            "learning_rate": 1e-4,
            "weight_decay": 0.03,
            "maker_fee": 0.001,
            "return_weight": 0.03,
            "fill_temperature": 0.1,
            "fill_buffer_pct": 0.0005,
            "loss_type": "sortino",
            "decision_lag_bars": 1,
            "lr_schedule": "cosine",
            "lr_min_ratio": 0.01,
            "transformer_dim": 384,
            "transformer_layers": 6,
            "transformer_heads": 8,
            "transformer_dropout": 0.1,
            "model_arch": "nano",
            "num_memory_tokens": 8,
            "dilated_strides": "1,2,6,24",
            "num_outputs": 4,
            "max_hold_hours": 24.0,
            "use_compile": False,
            "seed": 1337,
        },
    },
}


def _parse_horizon_sets(raw: str) -> list[tuple[int, ...]]:
    candidates: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for part in str(raw).split(";"):
        horizons = tuple(sorted({int(token.strip()) for token in part.split(",") if token.strip()}))
        if not horizons or horizons in seen:
            continue
        seen.add(horizons)
        candidates.append(horizons)
    return candidates


def _parse_windows(raw: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        day = int(token)
        if day <= 0 or day in seen:
            continue
        seen.add(day)
        values.append(day)
    return values


def _horizon_label(horizons: tuple[int, ...]) -> str:
    return "_".join(f"h{int(h)}" for h in horizons)


def _latest_price_timestamp(data_symbol: str) -> pd.Timestamp:
    path = DATA_ROOT / f"{data_symbol}.csv"
    frame = pd.read_csv(path, usecols=["timestamp"])
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if ts.isna().all():
        raise ValueError(f"No valid timestamps found in {path}")
    return ts.max()


def ensure_forecast_cache(
    *,
    data_symbol: str,
    horizons: tuple[int, ...],
    forecast_cache_root: Path,
    forecast_model_id: str,
    max_history_days: int,
) -> dict:
    end = _latest_price_timestamp(data_symbol)
    lookback_hours = int(max_history_days) * 24 + 24 * 7
    start = end - pd.Timedelta(hours=float(lookback_hours))
    started = time.time()
    frame = build_forecast_bundle(
        symbol=data_symbol,
        data_root=DATA_ROOT,
        cache_root=forecast_cache_root,
        horizons=horizons,
        context_hours=512,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32,
        model_id=forecast_model_id,
        cache_only=False,
        start=start,
        end=end,
    )
    return {
        "data_symbol": data_symbol,
        "horizons": [int(h) for h in horizons],
        "rows": int(len(frame)),
        "start": pd.Timestamp(frame["timestamp"].min()).isoformat(),
        "end": pd.Timestamp(frame["timestamp"].max()).isoformat(),
        "seconds": round(time.time() - started, 2),
    }


def _training_config_for(
    *,
    symbol_key: str,
    horizons: tuple[int, ...],
    forecast_cache_root: Path,
    checkpoint_root: Path,
) -> TrainingConfig:
    symbol_cfg = SYMBOL_CONFIGS[symbol_key]
    training_kwargs = dict(symbol_cfg["training"])
    epochs = int(training_kwargs.pop("epochs"))
    batch_size = int(training_kwargs["batch_size"])
    sequence_length = int(training_kwargs["sequence_length"])
    dataset_cfg = DatasetConfig(
        symbol=str(symbol_cfg["data_symbol"]),
        data_root=DATA_ROOT,
        forecast_cache_root=forecast_cache_root,
        forecast_horizons=tuple(int(h) for h in horizons),
        sequence_length=sequence_length,
        validation_days=30,
        cache_only=False,
    )
    return TrainingConfig(
        epochs=epochs,
        checkpoint_root=checkpoint_root,
        log_dir=checkpoint_root / "tensorboard",
        preload_checkpoint_path=Path(symbol_cfg["live_checkpoint"]),
        dataset=dataset_cfg,
        **training_kwargs,
    )


def train_symbol_candidate(
    *,
    symbol_key: str,
    horizons: tuple[int, ...],
    forecast_cache_root: Path,
    experiment_root: Path,
    max_history_days: int,
) -> dict:
    label = _horizon_label(horizons)
    checkpoint_root = experiment_root / "checkpoints" / f"{symbol_key}_{label}"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    cfg = _training_config_for(
        symbol_key=symbol_key,
        horizons=horizons,
        forecast_cache_root=forecast_cache_root,
        checkpoint_root=checkpoint_root,
    )
    dm = ChronosSolDataModule(
        symbol=str(SYMBOL_CONFIGS[symbol_key]["data_symbol"]),
        data_root=DATA_ROOT,
        forecast_cache_root=forecast_cache_root,
        forecast_horizons=tuple(int(h) for h in horizons),
        context_hours=512,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32,
        model_id="amazon/chronos-2",
        sequence_length=int(cfg.sequence_length),
        split_config=SplitConfig(val_days=30, test_days=30),
        cache_only=False,
        max_history_days=max_history_days,
    )
    trainer = BinanceHourlyTrainer(cfg, dm)
    started = time.time()
    artifacts = trainer.train()
    rows = []
    for path in sorted(trainer.checkpoint_dir.glob("epoch_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metrics = payload.get("metrics") or {}
        rows.append(
            {
                "path": str(path),
                "epoch": int(payload.get("epoch", 0) or 0),
                "val_score": float(metrics.get("score", 0.0) or 0.0),
                "val_return": float(metrics.get("return", 0.0) or 0.0),
                "val_sortino": float(metrics.get("sortino", 0.0) or 0.0),
            }
        )
    rows.sort(key=lambda row: row["val_score"], reverse=True)
    best_checkpoint = Path(artifacts.best_checkpoint) if artifacts.best_checkpoint else Path(rows[0]["path"])
    return {
        "symbol": symbol_key,
        "data_symbol": str(SYMBOL_CONFIGS[symbol_key]["data_symbol"]),
        "horizons": [int(h) for h in horizons],
        "checkpoint_dir": str(trainer.checkpoint_dir),
        "best_checkpoint": str(best_checkpoint),
        "epochs": rows,
        "seconds": round(time.time() - started, 2),
        "training_config": asdict(cfg),
    }


def run_backtest(
    *,
    name: str,
    doge_checkpoint: Path,
    aave_checkpoint: Path,
    forecast_cache_root: Path,
    experiment_root: Path,
    days: int,
    metric: str,
    lookback: int,
    short_max_leverage: float,
) -> dict:
    output_path = experiment_root / "backtests" / name / f"{days}d_{metric}_lb{lookback}_s{short_max_leverage:.2f}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "binanceleveragesui.backtest_trade_margin_meta",
        "--days",
        str(days),
        "--doge-checkpoint",
        str(doge_checkpoint),
        "--aave-checkpoint",
        str(aave_checkpoint),
        "--selection-mode",
        "winner_cash",
        "--selection-metric",
        metric,
        "--lookback",
        str(lookback),
        "--allow-short",
        "--long-max-leverage",
        "2.3",
        "--short-max-leverage",
        f"{short_max_leverage:.2f}",
        "--cash-threshold",
        "0.0",
        "--switch-margin",
        "0.0",
        "--min-score-gap",
        "0.0",
        "--profit-gate-lookback-hours",
        "24",
        "--profit-gate-min-return",
        "0.0",
        "--data-root",
        str(DATA_ROOT),
        "--forecast-cache",
        str(forecast_cache_root),
        "--output-json",
        str(output_path),
    ]
    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    subprocess.run(cmd, check=True, cwd=REPO, env=env)
    return json.loads(output_path.read_text())


def summarize_candidate(name: str, reports: dict[int, dict]) -> dict:
    windows = {}
    for day, report in reports.items():
        meta = report["meta"]
        windows[f"{int(day)}d"] = {
            "return_pct": float(meta["return_pct"]),
            "max_drawdown_pct": float(meta["max_drawdown_pct"]),
            "trade_count": int(meta["trade_count"]),
            "switch_count": int(meta["switch_count"]),
        }
    return {"name": name, "windows": windows}


def candidate_score(summary: dict) -> tuple:
    windows = summary["windows"]
    ret_1d = float(windows.get("1d", {}).get("return_pct", -999.0))
    ret_7d = float(windows.get("7d", {}).get("return_pct", -999.0))
    ret_30d = float(windows.get("30d", {}).get("return_pct", -999.0))
    dd_30d = float(windows.get("30d", {}).get("max_drawdown_pct", -999.0))
    profitable_gate = 1 if ret_1d > 0.0 and dd_30d > -20.0 else 0
    return (
        profitable_gate,
        ret_7d,
        ret_1d,
        ret_30d,
        dd_30d,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Chronos feature horizons for DOGE/AAVE meta deployment")
    parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--forecast-cache-root", default=str(DEFAULT_EXPERIMENT_ROOT / "forecast_cache_chronos2"))
    parser.add_argument("--forecast-model-id", default="amazon/chronos-2")
    parser.add_argument("--candidate-horizon-sets", default="1,6;1,12;1,6,12")
    parser.add_argument("--windows", default="1,7,30")
    parser.add_argument("--max-history-days", type=int, default=365)
    parser.add_argument("--refine-short-caps", default="0.12,0.16,0.20")
    parser.add_argument("--refine-metrics", default="calmar,sortino")
    args = parser.parse_args()

    experiment_root = Path(args.experiment_root)
    forecast_cache_root = Path(args.forecast_cache_root)
    experiment_root.mkdir(parents=True, exist_ok=True)
    forecast_cache_root.mkdir(parents=True, exist_ok=True)

    candidate_horizon_sets = _parse_horizon_sets(args.candidate_horizon_sets)
    windows = _parse_windows(args.windows)
    refine_short_caps = [float(value) for value in args.refine_short_caps.split(",") if value.strip()]
    refine_metrics = [value.strip() for value in args.refine_metrics.split(",") if value.strip()]

    all_required_horizons = tuple(sorted({1, *[h for horizons in candidate_horizon_sets for h in horizons]}))
    forecast_builds = []
    for symbol_key, symbol_cfg in SYMBOL_CONFIGS.items():
        forecast_builds.append(
            ensure_forecast_cache(
                data_symbol=str(symbol_cfg["data_symbol"]),
                horizons=all_required_horizons,
                forecast_cache_root=forecast_cache_root,
                forecast_model_id=str(args.forecast_model_id),
                max_history_days=int(args.max_history_days),
            )
        )

    torch.use_deterministic_algorithms(True)
    training_runs: dict[str, dict] = {}
    for horizons in candidate_horizon_sets:
        label = _horizon_label(horizons)
        for symbol_key in SYMBOL_CONFIGS:
            key = f"{symbol_key}_{label}"
            training_runs[key] = train_symbol_candidate(
                symbol_key=symbol_key,
                horizons=horizons,
                forecast_cache_root=forecast_cache_root,
                experiment_root=experiment_root,
                max_history_days=int(args.max_history_days),
            )

    candidate_pairs: dict[str, tuple[Path, Path]] = {
        "live_baseline_existing_cache": (LIVE_DOGE_CKPT, LIVE_AAVE_CKPT),
        "live_baseline_chronos2_h1": (LIVE_DOGE_CKPT, LIVE_AAVE_CKPT),
    }
    for horizons in candidate_horizon_sets:
        label = _horizon_label(horizons)
        candidate_pairs[f"ft_{label}"] = (
            Path(training_runs[f"doge_{label}"]["best_checkpoint"]),
            Path(training_runs[f"aave_{label}"]["best_checkpoint"]),
        )

    candidate_summaries = []
    candidate_reports: dict[str, dict] = {}
    for name, (doge_ckpt, aave_ckpt) in candidate_pairs.items():
        reports = {}
        cache_root = REPO / "binanceneural/forecast_cache" if name == "live_baseline_existing_cache" else forecast_cache_root
        for day in windows:
            reports[day] = run_backtest(
                name=name,
                doge_checkpoint=doge_ckpt,
                aave_checkpoint=aave_ckpt,
                forecast_cache_root=cache_root,
                experiment_root=experiment_root,
                days=day,
                metric="calmar",
                lookback=1,
                short_max_leverage=0.16,
            )
        candidate_reports[name] = reports
        candidate_summaries.append(summarize_candidate(name, reports))

    ranked_candidates = sorted(candidate_summaries, key=candidate_score, reverse=True)
    best_name = ranked_candidates[0]["name"]
    best_pair = candidate_pairs[best_name]

    refinement = []
    for metric in refine_metrics:
        for short_cap in refine_short_caps:
            name = f"refine_{best_name}_{metric}_s{short_cap:.2f}"
            reports = {}
            for day in windows:
                reports[day] = run_backtest(
                    name=name,
                    doge_checkpoint=best_pair[0],
                    aave_checkpoint=best_pair[1],
                    forecast_cache_root=(REPO / "binanceneural/forecast_cache") if best_name == "live_baseline_existing_cache" else forecast_cache_root,
                    experiment_root=experiment_root,
                    days=day,
                    metric=metric,
                    lookback=1,
                    short_max_leverage=short_cap,
                )
            summary = summarize_candidate(name, reports)
            summary["base_pair"] = best_name
            summary["metric"] = metric
            summary["short_max_leverage"] = short_cap
            refinement.append(summary)

    ranked_refinement = sorted(refinement, key=candidate_score, reverse=True)
    selected_deploy = ranked_refinement[0] if ranked_refinement else None

    final_summary = {
        "forecast_cache_root": str(forecast_cache_root),
        "forecast_model_id": str(args.forecast_model_id),
        "forecast_builds": forecast_builds,
        "training_runs": training_runs,
        "candidate_summaries": candidate_summaries,
        "ranked_candidates": ranked_candidates,
        "ranked_refinement": ranked_refinement,
        "selected_pair": best_name,
        "selected_pair_checkpoints": {
            "doge": str(best_pair[0]),
            "aave": str(best_pair[1]),
        },
        "selected_deploy": selected_deploy,
    }
    (experiment_root / "summary.json").write_text(json.dumps(final_summary, indent=2))
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
