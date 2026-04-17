from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.remote_training_pipeline import (
    LARGE_UNIVERSE_STOCK_A40_DESCRIPTIONS,
    build_export_daily_fused_cmd,
    build_remote_chronos_compare_plan,
    build_remote_autoresearch_plan,
    build_remote_hourly_chronos_rl_plan,
    build_remote_large_universe_stock_plan,
    compute_daily_overlap_bounds,
    compute_hourly_train_val_window,
    normalize_symbols,
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


def test_normalize_symbols_accepts_single_string() -> None:
    assert normalize_symbols("aapl") == ["AAPL"]


def test_normalize_symbols_splits_comma_delimited_entries() -> None:
    assert normalize_symbols(["aapl,msft", " nvda ", "AAPL"]) == ["AAPL", "MSFT", "NVDA"]


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


def test_compute_daily_overlap_bounds_uses_shared_days(tmp_path: Path) -> None:
    root = tmp_path / "daily"
    root.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000.0,
        }
    ).to_csv(root / "AAA.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-03", periods=8, freq="D", tz="UTC"),
            "open": 200.0,
            "high": 201.0,
            "low": 199.0,
            "close": 200.5,
            "volume": 2_000.0,
        }
    ).to_csv(root / "BBB.csv", index=False)

    earliest, latest = compute_daily_overlap_bounds(symbols=["AAA", "BBB"], data_root=root)

    assert earliest == "2026-01-03T00:00:00+00:00"
    assert latest == "2026-01-10T00:00:00+00:00"


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
        lora_seeds=[1337, 2027],
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
    assert "--seeds" in plan.commands[0]
    seeds_idx = list(plan.commands[0]).index("--seeds")
    assert plan.commands[0][seeds_idx + 1] == "1337,2027"
    promote_cmd = list(plan.commands[1])
    assert "--selection-strategy" in promote_cmd
    strategy_idx = promote_cmd.index("--selection-strategy")
    assert promote_cmd[strategy_idx + 1] == "stable_family"
    assert "--stability-penalty" in promote_cmd
    assert "--min-family-size" in promote_cmd
    assert "--descriptions" in plan.commands[-1]


def test_build_export_daily_fused_cmd_uses_single_output_and_roots() -> None:
    cmd = build_export_daily_fused_cmd(
        symbols=["AAA", "BBB"],
        data_root="trainingdatadaily",
        hourly_root="trainingdatahourly",
        daily_forecast_root="strategytraining/forecast_cache",
        hourly_forecast_root="analysis/remote_runs/probe/forecast_cache",
        output_path="pufferlib_market/data/probe_daily_train.bin",
        start_date="2026-01-01",
        end_date="2026-03-01",
        min_days=30,
        zscore_window=45,
    )

    assert cmd[:4] == ["python", "-u", "-m", "pufferlib_market.export_data_daily_v4"]
    assert "--single-output" in cmd
    assert "--hourly-forecast-root" in cmd
    assert "--daily-forecast-root" in cmd
    assert "pufferlib_market/data/probe_daily_train.bin" in cmd


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
        lora_seeds=[1337],
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


def test_build_remote_chronos_compare_plan_generates_dual_branch_paths(tmp_path: Path) -> None:
    hourly_root = tmp_path / "hourly"
    daily_root = tmp_path / "daily"
    _write_hourly_csv(hourly_root / "AAA.csv", "2026-01-01 00:00:00+00:00", 600)
    _write_hourly_csv(hourly_root / "BBB.csv", "2026-01-01 00:00:00+00:00", 600)
    daily_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=40, freq="D", tz="UTC"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000.0,
        }
    ).to_csv(daily_root / "AAA.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=40, freq="D", tz="UTC"),
            "open": 200.0,
            "high": 201.0,
            "low": 199.0,
            "close": 200.5,
            "volume": 2_000.0,
        }
    ).to_csv(daily_root / "BBB.csv", index=False)

    plan = build_remote_chronos_compare_plan(
        run_id="compare123",
        symbols=["AAA", "BBB"],
        local_hourly_data_root=hourly_root,
        remote_hourly_data_root="trainingdatahourly",
        local_daily_data_root=daily_root,
        remote_daily_data_root="trainingdatadaily",
        train_hours=24 * 20,
        val_hours=24 * 10,
        gap_hours=0,
        preaugs=["baseline"],
        context_lengths=[128],
        learning_rates=[5e-5],
        num_steps=100,
        prediction_length=24,
        lora_r=16,
        feature_lag=1,
        min_coverage=0.95,
        time_budget=300,
        max_trials=2,
        descriptions=["sortino_rc3_tp08", "robust_reg_tp01"],
        earliest_common_override="2026-01-01T00:00:00+00:00",
        latest_common_override="2026-02-09T23:00:00+00:00",
    )

    assert plan.remote_run_dir == "analysis/remote_runs/compare123"
    assert plan.hourly_train_data_path == "pufferlib_market/data/compare123_hourly_train.bin"
    assert plan.daily_train_data_path == "pufferlib_market/data/compare123_daily_train.bin"
    assert plan.hourly_leaderboard_path == "analysis/remote_runs/compare123/hourly_leaderboard.csv"
    assert plan.daily_leaderboard_path == "analysis/remote_runs/compare123/daily_leaderboard.csv"
    assert len(plan.commands) == 9
    assert list(plan.commands[0][:4]) == ["python", "-u", "scripts/run_crypto_lora_batch.py", "--run-id"]
    assert list(plan.commands[5][:4]) == ["python", "-u", "-m", "pufferlib_market.export_data_daily_v4"]
    assert "compare123_hourly" in " ".join(plan.commands[7])
    assert "compare123_daily" in " ".join(plan.commands[8])


def test_build_remote_large_universe_stock_plan_uses_ctx256_and_a40_descriptions(tmp_path: Path) -> None:
    hourly_root = tmp_path / "hourly"
    daily_root = tmp_path / "daily"
    _write_hourly_csv(hourly_root / "AAA.csv", "2026-01-01 00:00:00+00:00", 900)
    _write_hourly_csv(hourly_root / "BBB.csv", "2026-01-01 00:00:00+00:00", 900)
    daily_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=60, freq="D", tz="UTC"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000.0,
        }
    ).to_csv(daily_root / "AAA.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=60, freq="D", tz="UTC"),
            "open": 200.0,
            "high": 201.0,
            "low": 199.0,
            "close": 200.5,
            "volume": 2_000.0,
        }
    ).to_csv(daily_root / "BBB.csv", index=False)

    plan = build_remote_large_universe_stock_plan(
        run_id="stocks_a40_probe",
        symbols=["AAA", "BBB"],
        local_hourly_data_root=hourly_root,
        remote_hourly_data_root="trainingdatahourly/stocks",
        local_daily_data_root=daily_root,
        remote_daily_data_root="trainingdatadaily/stocks",
        train_hours=24 * 30,
        val_hours=24 * 15,
        gap_hours=24,
        earliest_common_override="2026-01-01T00:00:00+00:00",
        latest_common_override="2026-03-01T23:00:00+00:00",
    )

    assert plan.run_id == "stocks_a40_probe"
    assert plan.symbols == ("AAA", "BBB")
    lora_cmd = list(plan.commands[0])
    assert lora_cmd[lora_cmd.index("--context-lengths") + 1] == "256"
    assert lora_cmd[lora_cmd.index("--learning-rates") + 1] == "5e-05,0.0001"
    assert lora_cmd[lora_cmd.index("--seeds") + 1] == "1337,2027,31415"
    hourly_cmd = list(plan.commands[7])
    daily_cmd = list(plan.commands[8])
    assert hourly_cmd[hourly_cmd.index("--descriptions") + 1] == ",".join(LARGE_UNIVERSE_STOCK_A40_DESCRIPTIONS)
    assert daily_cmd[daily_cmd.index("--descriptions") + 1] == ",".join(LARGE_UNIVERSE_STOCK_A40_DESCRIPTIONS)
    assert daily_cmd[daily_cmd.index("--rank-metric") + 1] == "holdout_robust_score"
    assert daily_cmd[daily_cmd.index("--max-steps-override") + 1] == "252"
    assert daily_cmd[daily_cmd.index("--holdout-n-windows") + 1] == "24"


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
        lora_seeds=[1337],
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
    assert "python pufferlib_market/setup.py build_ext --inplace" in script
    assert "scripts/run_crypto_lora_batch.py" in script
    assert "pufferlib_market.autoresearch_rl" in script


def test_build_remote_autoresearch_plan_includes_replay_and_post_eval() -> None:
    plan = build_remote_autoresearch_plan(
        run_id="mixed23_probe",
        train_data_path="pufferlib_market/data/mixed23_fresh_train.bin",
        val_data_path="pufferlib_market/data/mixed23_fresh_val.bin",
        time_budget=1800,
        max_trials=4,
        descriptions=["reg_combo_2", "robust_reg_tp01"],
        rank_metric="replay_hourly_return_pct",
        periods_per_year=365.0,
        max_steps_override=90,
        holdout_data="pufferlib_market/data/mixed23_fresh_val.bin",
        holdout_eval_steps=90,
        holdout_n_windows=20,
        holdout_fee_rate=0.001,
        holdout_max_leverage=0.5,
        eval_tradable_symbols=["BTCUSD", "ETHUSD"],
        eval_disable_shorts=True,
        replay_eval_hourly_root="trainingdatahourly",
        replay_eval_start_date="2025-06-01",
        replay_eval_end_date="2026-02-05",
        replay_eval_run_hourly_policy=True,
        replay_eval_robust_start_states="flat,long:BTCUSD:0.25",
        post_eval_periods=[30, 60, 90, 120],
        post_eval_sort_period=120,
    )

    assert plan.remote_run_dir == "analysis/remote_runs/mixed23_probe"
    assert plan.post_eval_output_path == "analysis/remote_runs/mixed23_probe/marketsim_30_60_90_120.csv"
    assert len(plan.commands) == 2
    autoresearch_cmd = list(plan.commands[0])
    assert "--holdout-data" in autoresearch_cmd
    assert "--replay-eval-hourly-root" in autoresearch_cmd
    assert "--replay-eval-run-hourly-policy" in autoresearch_cmd
    assert "--replay-eval-robust-start-states" in autoresearch_cmd
    assert "--eval-tradable-symbols" in autoresearch_cmd
    assert autoresearch_cmd[autoresearch_cmd.index("--eval-tradable-symbols") + 1] == "BTCUSD,ETHUSD"
    assert "--eval-disable-shorts" in autoresearch_cmd
    assert "--rank-metric" in autoresearch_cmd
    assert autoresearch_cmd[0:4] == ["python", "-u", "-m", "pufferlib_market.autoresearch_rl"]
    post_eval_cmd = list(plan.commands[1])
    assert post_eval_cmd[0:3] == ["python", "-u", "pufferlib_market/fast_marketsim_eval.py"]
    assert "--checkpoint-dirs" in post_eval_cmd
    assert "--max-leverage" in post_eval_cmd
    assert post_eval_cmd[post_eval_cmd.index("--max-leverage") + 1] == "0.5"
    assert "--tradable-symbols" in post_eval_cmd
    assert post_eval_cmd[post_eval_cmd.index("--tradable-symbols") + 1] == "BTCUSD,ETHUSD"
    assert "--disable-shorts" in post_eval_cmd
    assert "--sequential" in post_eval_cmd


def test_render_remote_pipeline_script_supports_autoresearch_plan() -> None:
    plan = build_remote_autoresearch_plan(
        run_id="mixed23_probe2",
        train_data_path="train.bin",
        val_data_path="val.bin",
        time_budget=300,
        max_trials=1,
        descriptions=["reg_combo_2"],
        holdout_data="val.bin",
        holdout_eval_steps=30,
        holdout_n_windows=4,
        replay_eval_hourly_root="trainingdatahourly",
        replay_eval_start_date="2025-06-01",
        replay_eval_end_date="2026-02-05",
        post_eval_periods=[30, 120],
    )

    script = render_remote_pipeline_script(
        remote_dir="/nvme0n1-disk/code/stock-prediction",
        remote_env=".venv313",
        plan=plan,
    )

    assert "source .venv313/bin/activate" in script
    assert "python pufferlib_market/setup.py build_ext --inplace" in script
    assert "pufferlib_market.autoresearch_rl" in script
    assert "pufferlib_market/fast_marketsim_eval.py" in script
