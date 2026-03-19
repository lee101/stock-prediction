from __future__ import annotations

from pufferlib_market.sweep_stock_focus import STOCK_BASELINES, STOCK_EXPERIMENTS, build_default_argv


def test_stock_focus_experiments_target_replay_winners() -> None:
    assert STOCK_EXPERIMENTS
    assert all(exp.get("arch") == "resmlp" for exp in STOCK_EXPERIMENTS)
    assert all(exp.get("hidden_size") == 256 for exp in STOCK_EXPERIMENTS)
    assert all(exp.get("disable_shorts") is True for exp in STOCK_EXPERIMENTS)
    assert all(exp.get("resume_from") for exp in STOCK_EXPERIMENTS)
    assert any(exp.get("trade_penalty") for exp in STOCK_EXPERIMENTS)
    assert any(exp.get("lr") == 1e-5 for exp in STOCK_EXPERIMENTS)
    assert any(exp.get("time_budget") == 240 for exp in STOCK_EXPERIMENTS)
    assert any(exp.get("time_budget") == 120 for exp in STOCK_EXPERIMENTS)
    assert {exp.get("train_data") for exp in STOCK_EXPERIMENTS} == {
        baseline["train_data"] for baseline in STOCK_BASELINES.values()
    }


def test_build_default_argv_enables_stock_validation_settings() -> None:
    argv = build_default_argv("prog")

    assert argv[0] == "prog"
    assert "--market-validation-asset-class" in argv
    assert argv[argv.index("--market-validation-asset-class") + 1] == "stock"
    assert "--disable-shorts-override" in argv
    assert "--fee-rate-override" in argv
    assert argv[argv.index("--fee-rate-override") + 1] == "0.0005"
    assert "--max-leverage-override" in argv
    assert argv[argv.index("--max-leverage-override") + 1] == "2.0"
    assert "--resume-from-override" not in argv
    assert "--train-data" in argv
    assert "stocks13_hourly_forecast_mktd_v2_start20250915_featlag1.bin" in argv[argv.index("--train-data") + 1]


def test_build_default_argv_can_be_extended_by_callers() -> None:
    argv = build_default_argv("prog") + ["--max-trials", "3", "--time-budget", "120"]

    assert argv[-4:] == ["--max-trials", "3", "--time-budget", "120"]
