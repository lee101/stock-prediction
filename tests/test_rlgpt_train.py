from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from RLgpt.config import DailyPlanDataConfig, PlannerConfig, SimulatorConfig, TrainingConfig
from RLgpt.data import DailyPlanTensors
from RLgpt.train import run_training


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
    )

    result = run_training(config)

    out_dir = Path(result["output_dir"])
    assert (out_dir / "best.pt").exists()
    assert (out_dir / "metrics.json").exists()
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["run_name"] == "synthetic_rlgpt"
    assert len(metrics["history"]) == 2
