from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pufferlib_market.export_data_daily import export_binary as export_daily_binary
from pufferlib_market.hourly_replay import read_mktd
from scripts.run_augmented_daily_stock_pipeline import (
    build_daily_train_val_window,
    build_eval_command,
    build_train_command,
    concat_mktd_files,
    _resolve_eval_checkpoint,
    main as augmented_pipeline_main,
)
from scripts.wandboard_c_dashboard import load_pipeline_runs, render_runs_table


def _write_daily_csv(root: Path, symbol: str, start: str, periods: int) -> None:
    index = pd.date_range(start, periods=periods, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": index,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000.0,
            "symbol": symbol,
        }
    )
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / f"{symbol}.csv", index=False)


def _write_hourly_csv(root: Path, symbol: str, start: str, periods: int) -> None:
    index = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": index,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000.0,
            "symbol": symbol,
        }
    )
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / f"{symbol}.csv", index=False)


def test_build_daily_train_val_window_uses_overlap_and_holdout(tmp_path: Path) -> None:
    daily_root = tmp_path / "daily"
    _write_daily_csv(daily_root, "AAPL", "2026-01-01", 40)
    _write_daily_csv(daily_root, "MSFT", "2026-01-05", 32)

    window = build_daily_train_val_window(
        symbols=["AAPL", "MSFT"],
        daily_root=daily_root,
        hourly_root=None,
        train_start=None,
        train_end=None,
        val_start=None,
        val_end=None,
        val_days=10,
        gap_days=3,
    )

    assert window.earliest_common == "2026-01-05T00:00:00+00:00"
    assert window.latest_common == "2026-02-05T00:00:00+00:00"
    assert window.val_start == "2026-01-27T00:00:00+00:00"
    assert window.val_end == "2026-02-05T00:00:00+00:00"
    assert window.train_end == "2026-01-23T00:00:00+00:00"


def test_build_daily_train_val_window_clips_to_hourly_overlap(tmp_path: Path) -> None:
    daily_root = tmp_path / "daily"
    hourly_root = tmp_path / "hourly"
    _write_daily_csv(daily_root, "AAPL", "2026-01-01", 60)
    _write_daily_csv(daily_root, "MSFT", "2026-01-01", 60)
    _write_hourly_csv(hourly_root, "AAPL", "2026-01-20", 24 * 25)
    _write_hourly_csv(hourly_root, "MSFT", "2026-01-22", 24 * 23)

    window = build_daily_train_val_window(
        symbols=["AAPL", "MSFT"],
        daily_root=daily_root,
        hourly_root=hourly_root,
        train_start=None,
        train_end=None,
        val_start=None,
        val_end=None,
        val_days=10,
        gap_days=2,
    )

    assert window.earliest_common == "2026-01-22T00:00:00+00:00"
    assert window.latest_common == "2026-02-13T00:00:00+00:00"


def test_concat_mktd_files_appends_time_dimension(tmp_path: Path) -> None:
    data_a = tmp_path / "a"
    data_b = tmp_path / "b"
    out_a = tmp_path / "a.bin"
    out_b = tmp_path / "b.bin"
    merged_path = tmp_path / "merged.bin"
    _write_daily_csv(data_a, "AAPL", "2026-01-01", 6)
    _write_daily_csv(data_b, "AAPL", "2026-02-01", 4)

    export_daily_binary(symbols=["AAPL"], data_root=data_a, output_path=out_a, min_days=3)
    export_daily_binary(symbols=["AAPL"], data_root=data_b, output_path=out_b, min_days=3)

    merged = concat_mktd_files(input_paths=[out_a, out_b], output_path=merged_path)
    reloaded = read_mktd(merged_path)

    assert merged.num_timesteps == read_mktd(out_a).num_timesteps + read_mktd(out_b).num_timesteps
    assert reloaded.num_timesteps == merged.num_timesteps
    assert reloaded.symbols == ["AAPL"]


def test_build_train_command_passes_wandb_resume_args() -> None:
    cmd = build_train_command(
        train_data=Path("train.bin"),
        val_data=Path("val.bin"),
        checkpoint_dir=Path("ckpts"),
        wandb_project="stock",
        wandb_entity="team",
        wandb_group="group1",
        wandb_run_name="aug-run",
        wandb_run_id="abc123",
        wandb_mode="offline",
        extra_args=["--total-timesteps", "1000"],
    )

    assert "--wandb-project" in cmd
    assert "--wandb-run-id" in cmd
    assert cmd[cmd.index("--wandb-run-id") + 1] == "abc123"
    assert cmd[cmd.index("--wandb-resume") + 1] == "allow"
    assert "--lr-schedule" in cmd
    assert cmd[cmd.index("--lr-schedule") + 1] == "cosine"
    assert "--obs-norm" in cmd
    assert "--cuda-graph-ppo" in cmd
    assert cmd[-2:] == ["--total-timesteps", "1000"]


def test_build_eval_command_uses_hourly_intrabar_defaults() -> None:
    cmd = build_eval_command(
        checkpoint=Path("ckpts/best.pt"),
        val_data=Path("val.bin"),
        out_dir=Path("eval"),
        hourly_data_root=Path("trainingdatahourly/stocks"),
        daily_start_date="2025-07-29T00:00:00+00:00",
        extra_args=[],
    )

    assert "--execution-granularity" in cmd
    assert cmd[cmd.index("--execution-granularity") + 1] == "hourly_intrabar"
    assert cmd[cmd.index("--hourly-data-root") + 1] == "trainingdatahourly/stocks"
    assert cmd[cmd.index("--daily-start-date") + 1] == "2025-07-29T00:00:00+00:00"


def test_wandboard_c_dashboard_renders_local_manifests(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    manifest_dir = run_root / "run_one"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_name": "run_one",
                "status": "completed",
                "symbols": ["AAPL", "MSFT"],
                "offsets": [0, 1, 2],
                "window": {
                    "train_start": "2026-01-01T00:00:00+00:00",
                    "train_end": "2026-02-01T00:00:00+00:00",
                    "val_days": 30,
                },
                "wandb": {"run_id": "wandb123"},
                "eval_100d": {"aggregate": {"worst_slip_monthly": 0.31}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    runs = load_pipeline_runs(run_root)
    table = render_runs_table(runs)

    assert len(runs) == 1
    assert "run_one" in table
    assert "+31.00%" in table
    assert "wandb123" in table


def test_resolve_eval_checkpoint_falls_back_to_final(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    final_path = ckpt_dir / "final.pt"
    final_path.write_bytes(b"dummy")

    assert _resolve_eval_checkpoint(ckpt_dir) == final_path


def test_prepare_only_builds_bins_without_training(tmp_path: Path, monkeypatch) -> None:
    daily_root = tmp_path / "daily"
    hourly_root = tmp_path / "hourly"
    for symbol in ["AAPL", "MSFT"]:
        _write_daily_csv(daily_root, symbol, "2026-01-01", 60)
        _write_hourly_csv(hourly_root, symbol, "2026-01-01", 24 * 60)

    monkeypatch.setattr(
        "scripts.run_augmented_daily_stock_pipeline._log_stage",
        lambda **kwargs: None,
    )

    rc = augmented_pipeline_main(
        [
            "--run-name",
            "prep_only",
            "--symbols",
            "AAPL,MSFT",
            "--daily-root",
            str(daily_root),
            "--hourly-root",
            str(hourly_root),
            "--work-root",
            str(tmp_path / "runs"),
            "--data-output-root",
            str(tmp_path / "data"),
            "--checkpoint-root",
            str(tmp_path / "ckpts"),
            "--val-days",
            "10",
            "--offsets",
            "0,1",
            "--prepare-only",
        ]
    )

    assert rc == 0
    assert (tmp_path / "data" / "prep_only_train.bin").exists()
    assert (tmp_path / "data" / "prep_only_val.bin").exists()
