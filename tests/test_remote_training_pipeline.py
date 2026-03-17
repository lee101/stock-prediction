from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.remote_training_pipeline import (
    build_remote_hourly_chronos_rl_plan,
    compute_hourly_train_val_window,
    render_remote_pipeline_script,
)


def _write_hourly_csv(path: Path, start: str, periods: int) -> None:
    index = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": index,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_compute_hourly_train_val_window_uses_latest_common_overlap(tmp_path: Path) -> None:
    root = tmp_path / "data" / "crypto"
    _write_hourly_csv(root / "AAA.csv", "2026-01-01 00:00:00+00:00", 240)
    _write_hourly_csv(root / "BBB.csv", "2026-01-02 00:00:00+00:00", 220)

    window = compute_hourly_train_val_window(
        symbols=["AAA", "BBB"],
        data_root=root,
        train_hours=96,
        val_hours=48,
        gap_hours=4,
    )

    assert window.latest_common == "2026-01-10T23:00:00+00:00"
    assert window.val_end == "2026-01-10T23:00:00+00:00"
    assert window.val_start == "2026-01-09T00:00:00+00:00"
    assert window.train_end == "2026-01-08T19:00:00+00:00"
    assert window.train_start == "2026-01-04T20:00:00+00:00"


def test_compute_hourly_train_val_window_rejects_insufficient_shared_history(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_hourly_csv(root / "AAA.csv", "2026-01-01 00:00:00+00:00", 40)
    _write_hourly_csv(root / "BBB.csv", "2026-01-01 00:00:00+00:00", 40)

    with pytest.raises(ValueError, match="Not enough shared history"):
        compute_hourly_train_val_window(
            symbols=["AAA", "BBB"],
            data_root=root,
            train_hours=30,
            val_hours=20,
            gap_hours=0,
        )


def test_build_remote_hourly_chronos_rl_plan_generates_expected_paths(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_hourly_csv(root / "AAA.csv", "2026-01-01 00:00:00+00:00", 600)
    _write_hourly_csv(root / "BBB.csv", "2026-01-01 00:00:00+00:00", 600)

    plan = build_remote_hourly_chronos_rl_plan(
        run_id="probe123",
        symbols=["AAA", "BBB"],
        local_data_root=root,
        remote_data_root="trainingdatahourly/crypto",
        train_hours=240,
        val_hours=72,
        gap_hours=0,
        preaugs=["baseline", "percent_change"],
        context_lengths=[128],
        learning_rates=[5e-5],
        num_steps=100,
        prediction_length=24,
        lora_r=16,
        feature_lag=1,
        min_coverage=0.95,
        time_budget=300,
        max_trials=2,
        descriptions=["baseline_anneal_lr", "ent_anneal"],
    )

    assert plan.remote_run_dir == "analysis/remote_runs/probe123"
    assert plan.train_data_path == "pufferlib_market/data/probe123_train.bin"
    assert plan.val_data_path == "pufferlib_market/data/probe123_val.bin"
    assert plan.leaderboard_path == "pufferlib_market/probe123_leaderboard.csv"
    assert len(plan.commands) == 6
    assert list(plan.commands[0][:4]) == ["python", "-u", "scripts/run_crypto_lora_batch.py", "--run-id"]
    assert "--descriptions" in plan.commands[-1]


def test_build_remote_hourly_chronos_rl_plan_honors_overlap_override(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_hourly_csv(root / "AAA.csv", "2026-01-01 00:00:00+00:00", 600)

    plan = build_remote_hourly_chronos_rl_plan(
        run_id="probe125",
        symbols=["AAA"],
        local_data_root=root,
        remote_data_root="trainingdatahourly/crypto",
        train_hours=48,
        val_hours=24,
        gap_hours=0,
        preaugs=["baseline"],
        context_lengths=[128],
        learning_rates=[5e-5],
        num_steps=100,
        prediction_length=24,
        lora_r=16,
        feature_lag=1,
        min_coverage=0.95,
        time_budget=60,
        max_trials=1,
        descriptions=["baseline_anneal_lr"],
        earliest_common_override="2026-01-05T00:00:00+00:00",
        latest_common_override="2026-01-08T23:00:00+00:00",
    )

    assert plan.window.earliest_common == "2026-01-05T00:00:00+00:00"
    assert plan.window.latest_common == "2026-01-08T23:00:00+00:00"
    assert plan.window.val_end == "2026-01-08T23:00:00+00:00"


def test_render_remote_pipeline_script_activates_env_and_runs_commands(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_hourly_csv(root / "AAA.csv", "2026-01-01 00:00:00+00:00", 400)

    plan = build_remote_hourly_chronos_rl_plan(
        run_id="probe124",
        symbols=["AAA"],
        local_data_root=root,
        remote_data_root="trainingdatahourly/crypto",
        train_hours=200,
        val_hours=48,
        gap_hours=0,
        preaugs=["baseline"],
        context_lengths=[128],
        learning_rates=[5e-5],
        num_steps=50,
        prediction_length=24,
        lora_r=8,
        feature_lag=1,
        min_coverage=0.95,
        time_budget=120,
        max_trials=1,
        descriptions=["baseline_anneal_lr"],
    )

    script = render_remote_pipeline_script(
        remote_dir="/nvme0n1-disk/code/stock-prediction",
        remote_env=".venv313",
        plan=plan,
    )

    assert "source .venv313/bin/activate" in script
    assert 'export PYTHONPATH="$PWD:$PWD/PufferLib:${PYTHONPATH:-}"' in script
    assert "scripts/run_crypto_lora_batch.py" in script
    assert "pufferlib_market.autoresearch_rl" in script
