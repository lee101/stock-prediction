from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from RLgpt.config import DailyPlanDataConfig, PlannerConfig, SimulatorConfig, TrainingConfig
from RLgpt.config import (
    DEFAULT_RLGPT_BATCH_SIZE,
    DEFAULT_RLGPT_DATA_ROOT,
    DEFAULT_RLGPT_DEPTH,
    DEFAULT_RLGPT_DROPOUT,
    DEFAULT_RLGPT_EPOCHS,
    DEFAULT_RLGPT_FORECAST_CACHE_ROOT,
    DEFAULT_RLGPT_HEADS,
    DEFAULT_RLGPT_HIDDEN_DIM,
    DEFAULT_RLGPT_SEQUENCE_LENGTH,
    DEFAULT_RLGPT_VALIDATION_DAYS,
    default_forecast_horizons_csv,
)
from RLgpt.data import DailyPlanTensors
from RLgpt.model import CrossAssetDailyPlanner
import RLgpt.train as train_module
from RLgpt.train import build_training_config, parse_args, run_training


def test_run_training_writes_checkpoint_and_metrics(tmp_path, monkeypatch):
    bundle = DailyPlanTensors(
        symbols=("AAA", "BBB"),
        feature_names=("f0", "f1", "f2", "f3"),
        days=tuple(pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")),
        features=torch.tensor(
            [
                [[0.1, 0.0, 0.2, -0.1], [0.0, 0.1, -0.2, 0.0]],
                [[0.2, -0.1, 0.1, -0.2], [-0.1, 0.2, -0.1, 0.1]],
                [[0.3, -0.2, 0.0, -0.1], [-0.2, 0.3, 0.0, 0.2]],
                [[0.2, 0.0, 0.1, 0.0], [0.0, 0.1, -0.1, 0.1]],
                [[0.1, 0.1, 0.2, 0.1], [0.1, 0.0, -0.2, 0.0]],
            ],
            dtype=torch.float32,
        ),
        daily_anchor=torch.full((5, 2), 100.0),
        prev_close=torch.full((5, 2), 99.0),
        hourly_open=torch.full((5, 2, 2), 100.0),
        hourly_high=torch.tensor(
            [
                [[100.5, 100.2], [101.0, 100.4]],
                [[100.3, 100.1], [101.2, 100.5]],
                [[100.4, 100.3], [101.1, 100.6]],
                [[100.2, 100.3], [101.0, 100.5]],
                [[100.1, 100.2], [101.0, 100.4]],
            ],
            dtype=torch.float32,
        ),
        hourly_low=torch.tensor(
            [
                [[99.2, 99.5], [99.8, 99.7]],
                [[99.0, 99.4], [99.7, 99.6]],
                [[99.1, 99.3], [99.8, 99.5]],
                [[99.3, 99.4], [99.9, 99.7]],
                [[99.4, 99.5], [99.8, 99.6]],
            ],
            dtype=torch.float32,
        ),
        hourly_close=torch.tensor(
            [
                [[100.0, 100.0], [100.8, 100.2]],
                [[100.1, 100.0], [100.9, 100.3]],
                [[100.2, 100.1], [101.0, 100.4]],
                [[100.1, 100.2], [100.8, 100.3]],
                [[100.0, 100.1], [100.7, 100.2]],
            ],
            dtype=torch.float32,
        ),
        hourly_mask=torch.ones(5, 2, 2),
    )
    monkeypatch.setattr("RLgpt.train.prepare_daily_plan_tensors", lambda _config: bundle)

    config = TrainingConfig(
        data=DailyPlanDataConfig(symbols=("AAA", "BBB"), validation_days=1),
        planner=PlannerConfig(hidden_dim=32, depth=1, heads=4, dropout=0.0),
        simulator=SimulatorConfig(
            initial_cash=1_000.0,
            maker_fee_bps=0.0,
            slippage_bps=0.0,
            fill_buffer_bps=0.0,
            fill_temperature_bps=0.1,
        ),
        epochs=2,
        batch_size=2,
        output_root=tmp_path,
        run_name="synthetic_rlgpt",
        device="cpu",
    )

    result = run_training(config)

    out_dir = Path(result["output_dir"])
    assert (out_dir / "best.pt").exists()
    assert (out_dir / "metrics.json").exists()
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["run_name"] == "synthetic_rlgpt"
    assert len(metrics["history"]) == 2


def test_run_training_falls_back_to_cpu_on_auto_cuda_oom(tmp_path, monkeypatch):
    bundle = DailyPlanTensors(
        symbols=("AAA", "BBB"),
        feature_names=("f0", "f1", "f2", "f3"),
        days=tuple(pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")),
        features=torch.zeros(4, 2, 4, dtype=torch.float32),
        daily_anchor=torch.full((4, 2), 100.0),
        prev_close=torch.full((4, 2), 99.0),
        hourly_open=torch.full((4, 2, 2), 100.0),
        hourly_high=torch.full((4, 2, 2), 101.0),
        hourly_low=torch.full((4, 2, 2), 99.0),
        hourly_close=torch.full((4, 2, 2), 100.0),
        hourly_mask=torch.ones(4, 2, 2),
    )
    monkeypatch.setattr("RLgpt.train.prepare_daily_plan_tensors", lambda _config: bundle)

    original_to = CrossAssetDailyPlanner.to
    calls: list[str] = []

    def flaky_to(self, device, *args, **kwargs):
        resolved = torch.device(device)
        calls.append(resolved.type)
        if resolved.type == "cuda":
            raise torch.OutOfMemoryError("CUDA out of memory")
        return original_to(self, device, *args, **kwargs)

    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(CrossAssetDailyPlanner, "to", flaky_to)

    config = TrainingConfig(
        data=DailyPlanDataConfig(symbols=("AAA", "BBB"), validation_days=1),
        planner=PlannerConfig(hidden_dim=32, depth=1, heads=4, dropout=0.0),
        simulator=SimulatorConfig(
            initial_cash=1_000.0,
            maker_fee_bps=0.0,
            slippage_bps=0.0,
            fill_buffer_bps=0.0,
            fill_temperature_bps=0.1,
        ),
        epochs=1,
        batch_size=2,
        output_root=tmp_path,
        run_name="synthetic_rlgpt_auto_fallback",
    )

    result = run_training(config)

    assert calls[:2] == ["cuda", "cpu"]
    assert Path(result["best_checkpoint"]).exists()


def test_build_training_config_normalizes_symbols_and_horizons() -> None:
    args = parse_args(
        [
            "--symbols",
            "btcusd,BTCUSD,ethusd,,solusd",
            "--forecast-horizons",
            "24,1,24,6",
        ]
    )

    config = build_training_config(args)

    assert config.data.symbols == ("BTCUSD", "ETHUSD", "SOLUSD")
    assert config.data.forecast_horizons == (24, 1, 6)


def test_parse_args_defaults_match_rlgpt_config_defaults() -> None:
    args = parse_args(["--symbols", "BTCUSD"])

    assert args.data_root == str(DEFAULT_RLGPT_DATA_ROOT)
    assert args.forecast_cache_root == str(DEFAULT_RLGPT_FORECAST_CACHE_ROOT)
    assert args.forecast_horizons == default_forecast_horizons_csv()
    assert args.validation_days == DEFAULT_RLGPT_VALIDATION_DAYS
    assert args.epochs == DEFAULT_RLGPT_EPOCHS
    assert args.batch_size == DEFAULT_RLGPT_BATCH_SIZE
    assert args.hidden_dim == DEFAULT_RLGPT_HIDDEN_DIM
    assert args.depth == DEFAULT_RLGPT_DEPTH
    assert args.heads == DEFAULT_RLGPT_HEADS
    assert args.dropout == DEFAULT_RLGPT_DROPOUT
    assert args.sequence_length == DEFAULT_RLGPT_SEQUENCE_LENGTH


def test_main_print_config_reports_normalized_setup(tmp_path, capsys) -> None:
    train_module.main(
        [
            "--symbols",
            "btcusd,BTCUSD,ethusd",
            "--forecast-horizons",
            "24,1,24",
            "--data-root",
            str(tmp_path / "prices"),
            "--forecast-cache-root",
            str(tmp_path / "forecasts"),
            "--print-config",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["symbols"] == ["BTCUSD", "ETHUSD"]
    assert payload["data"]["forecast_horizons"] == [24, 1]
    assert payload["symbol_count"] == 2
    assert payload["forecast_horizon_count"] == 2
    assert payload["resolved_device"] == "auto"


def test_main_check_config_reports_missing_files(tmp_path, capsys) -> None:
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    (prices_dir / "BTCUSD.csv").write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="1"):
        train_module.main(
            [
                "--symbols",
                "BTCUSD,ETHUSD",
                "--data-root",
                str(prices_dir),
                "--forecast-cache-root",
                str(tmp_path / "missing_forecasts"),
                "--cache-only",
                "--check-config",
            ]
        )

    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    assert payload["data_root_exists"] is True
    assert payload["forecast_cache_root_exists"] is False
    assert any(path.endswith("ETHUSD.csv") for path in payload["missing_price_files"])
    assert any("Forecast cache root does not exist in cache-only mode" in error for error in payload["errors"])
