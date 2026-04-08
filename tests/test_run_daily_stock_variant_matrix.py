from __future__ import annotations

import json
from pathlib import Path

import scripts.run_daily_stock_variant_matrix as sweep_mod
from src.daily_stock_variant_presets import preset_choices, resolve_variant_preset


def test_parse_args_defaults_to_current_vs_candidates() -> None:
    args = sweep_mod.parse_args([])

    assert args.list_presets is False
    assert args.preset == "current_vs_candidates"
    assert args.days == 120
    assert args.window is None
    assert args.output_json is None
    assert args.run_log_json is None
    assert args.symbols_file is None
    assert args.checkpoint == sweep_mod.daily_stock.DEFAULT_CHECKPOINT


def test_resolve_preset_returns_named_preset() -> None:
    preset = resolve_variant_preset("promising_only")

    assert preset.name == "promising_only"
    assert "beat the current live-equivalent baseline" in preset.description
    assert [variant.name for variant in preset.variants] == [
        "single_static_25",
        "portfolio2_static_50",
        "portfolio3_static_50",
    ]


def test_preset_choices_are_stable_and_sorted() -> None:
    assert preset_choices() == ["current_only", "current_vs_candidates", "promising_only"]


def test_normalize_symbols_uses_defaults_when_omitted() -> None:
    assert sweep_mod._normalize_symbols(None) == list(sweep_mod.daily_stock.DEFAULT_SYMBOLS)


def test_normalize_symbols_dedupes_and_uppercases() -> None:
    assert sweep_mod._normalize_symbols(["aapl,msft", "AAPL", " nvda "]) == ["AAPL", "MSFT", "NVDA"]


def test_normalize_symbols_rejects_unsafe_values() -> None:
    try:
        sweep_mod._normalize_symbols(["../../etc/passwd"])
    except ValueError as exc:
        assert "Unsupported symbol" in str(exc)
    else:
        raise AssertionError("expected invalid symbol input to be rejected")


def test_load_symbols_supports_symbols_file(tmp_path) -> None:
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("nvda, msft\nAAPL\n", encoding="utf-8")

    assert sweep_mod._load_symbols(None, symbols_file=str(symbols_file)) == ["NVDA", "MSFT", "AAPL"]


def test_main_dry_run_prints_resolved_config(capsys) -> None:
    exit_code = sweep_mod.main(["--dry-run", "--preset", "promising_only", "--symbols", "nvda,msft"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["preset"] == "promising_only"
    assert "beat the current live-equivalent baseline" in payload["preset_description"]
    assert payload["days"] == 120
    assert payload["days_list"] == [120]
    assert payload["symbols"] == ["NVDA", "MSFT"]
    assert [item["name"] for item in payload["variants"]] == [
        "single_static_25",
        "portfolio2_static_50",
        "portfolio3_static_50",
    ]


def test_main_dry_run_supports_symbols_file(tmp_path, capsys) -> None:
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("nvda, msft\nAAPL\n", encoding="utf-8")

    exit_code = sweep_mod.main(["--dry-run", "--preset", "promising_only", "--symbols-file", str(symbols_file)])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["symbols"] == ["NVDA", "MSFT", "AAPL"]
    assert payload["symbols_file"] == str(symbols_file)


def test_main_dry_run_writes_json_report(tmp_path) -> None:
    output_path = tmp_path / "reports" / "sweep.json"

    exit_code = sweep_mod.main(
        [
            "--dry-run",
            "--preset",
            "promising_only",
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["preset"] == "promising_only"
    assert payload["days_list"] == [120]


def test_main_list_presets_prints_catalog(capsys) -> None:
    exit_code = sweep_mod.main(["--list-presets"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "current_vs_candidates:" in output
    assert "promising_only:" in output
    assert "current_live_12p5" in output


def test_main_list_presets_prints_json_catalog(capsys) -> None:
    exit_code = sweep_mod.main(["--list-presets", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["name"] == "current_only"
    assert any(item["name"] == "current_vs_candidates" for item in payload)


def test_resolve_run_log_json_path_defaults_next_to_output_json() -> None:
    assert sweep_mod._resolve_run_log_json_path(
        output_json="reports/sweep.json",
        run_log_json=None,
    ) == "reports/sweep.run.json"


def test_main_delegates_to_variant_matrix_runner(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    def _fake_runner(**kwargs):
        captured.update(kwargs)
        return [
            {
                "name": "current_live_12p5",
                "allocation_pct": 12.5,
                "allocation_sizing_mode": "static",
                "multi_position": 0,
                "multi_position_min_prob_ratio": 0.3,
                "buying_power_multiplier": 1.0,
                "total_return": -0.01,
                "annualized_return": -0.02,
                "sortino": -0.5,
                "max_drawdown": -0.03,
                "trades": 8.0,
            },
            {
                "name": "single_static_25",
                "allocation_pct": 25.0,
                "allocation_sizing_mode": "static",
                "multi_position": 0,
                "multi_position_min_prob_ratio": 0.3,
                "buying_power_multiplier": 1.0,
                "total_return": 0.02,
                "annualized_return": 0.04,
                "sortino": 0.8,
                "max_drawdown": -0.02,
                "trades": 7.0,
            },
        ]

    monkeypatch.setattr(sweep_mod.daily_stock, "run_backtest_variant_matrix_via_trading_server", _fake_runner)

    exit_code = sweep_mod.main(["--preset", "current_vs_candidates", "--json", "--days", "60"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert captured["days"] == 60
    assert captured["symbols"] == list(sweep_mod.daily_stock.DEFAULT_SYMBOLS)
    assert len(captured["variants"]) == 4
    assert payload["results"][0]["name"] == "single_static_25"
    assert payload["results"][1]["name"] == "current_live_12p5"
    assert payload["baseline_comparison"]["baseline_name"] == "current_live_12p5"
    assert payload["baseline_comparison"]["beaters"][0]["name"] == "single_static_25"


def test_resolve_days_prefers_explicit_windows() -> None:
    assert sweep_mod._resolve_days(120, [60, 120, 60, 252]) == [60, 120, 252]
    assert sweep_mod._resolve_days(120, None) == [120]


def test_main_multi_window_json_reports_summary(monkeypatch, capsys) -> None:
    captured_days_list: list[list[int]] = []

    def _fake_multi_runner(**kwargs):
        days_list = [int(day) for day in kwargs["days_list"]]
        captured_days_list.append(days_list)
        windows = []
        for days in days_list:
            windows.append(
                {
                    "days": days,
                    "results": [
                        {
                            "name": "current_live_12p5",
                            "allocation_pct": 12.5,
                            "allocation_sizing_mode": "static",
                            "multi_position": 0,
                            "multi_position_min_prob_ratio": 0.3,
                            "buying_power_multiplier": 1.0,
                            "total_return": 0.01 if days == 60 else -0.01,
                            "annualized_return": 0.02,
                            "sortino": 0.4 if days == 60 else -0.2,
                            "max_drawdown": -0.03 if days == 60 else -0.05,
                            "trades": 8.0,
                        },
                        {
                            "name": "single_static_25",
                            "allocation_pct": 25.0,
                            "allocation_sizing_mode": "static",
                            "multi_position": 0,
                            "multi_position_min_prob_ratio": 0.3,
                            "buying_power_multiplier": 1.0,
                            "total_return": 0.02 if days == 60 else 0.03,
                            "annualized_return": 0.04,
                            "sortino": 0.8 if days == 60 else 1.0,
                            "max_drawdown": -0.02 if days == 60 else -0.04,
                            "trades": 7.0,
                        },
                    ],
                }
            )
        return windows

    monkeypatch.setattr(sweep_mod.daily_stock, "run_backtest_multi_window_variant_matrix_via_trading_server", _fake_multi_runner)

    exit_code = sweep_mod.main(["--preset", "current_vs_candidates", "--json", "--window", "60", "--window", "120"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert captured_days_list == [[60, 120]]
    assert payload["config"]["days_list"] == [60, 120]
    assert payload["summary"][0]["name"] == "single_static_25"
    assert payload["summary"][0]["window_count"] == 2
    assert payload["baseline_comparison"]["baseline_name"] == "current_live_12p5"
    assert payload["baseline_comparison"]["beaters"][0]["name"] == "single_static_25"
    assert len(payload["windows"]) == 2


def test_main_multi_window_json_writes_report_file(monkeypatch, tmp_path) -> None:
    def _fake_multi_runner(**kwargs):
        return [
            {
                "days": 60,
                "results": [
                    {
                        "name": "current_live_12p5",
                        "allocation_pct": 12.5,
                        "allocation_sizing_mode": "static",
                        "multi_position": 0,
                        "multi_position_min_prob_ratio": 0.3,
                        "buying_power_multiplier": 1.0,
                        "total_return": 0.0,
                        "annualized_return": -0.02,
                        "sortino": -0.5,
                        "max_drawdown": -0.03,
                        "trades": 8.0,
                    },
                    {
                        "name": "single_static_25",
                        "allocation_pct": 25.0,
                        "allocation_sizing_mode": "static",
                        "multi_position": 0,
                        "multi_position_min_prob_ratio": 0.3,
                        "buying_power_multiplier": 1.0,
                        "total_return": 0.03,
                        "annualized_return": 0.04,
                        "sortino": 0.8,
                        "max_drawdown": -0.02,
                        "trades": 7.0,
                    },
                ],
            },
            {
                "days": 120,
                "results": [
                    {
                        "name": "current_live_12p5",
                        "allocation_pct": 12.5,
                        "allocation_sizing_mode": "static",
                        "multi_position": 0,
                        "multi_position_min_prob_ratio": 0.3,
                        "buying_power_multiplier": 1.0,
                        "total_return": -0.01,
                        "annualized_return": -0.02,
                        "sortino": -0.5,
                        "max_drawdown": -0.03,
                        "trades": 8.0,
                    },
                    {
                        "name": "single_static_25",
                        "allocation_pct": 25.0,
                        "allocation_sizing_mode": "static",
                        "multi_position": 0,
                        "multi_position_min_prob_ratio": 0.3,
                        "buying_power_multiplier": 1.0,
                        "total_return": 0.02,
                        "annualized_return": 0.04,
                        "sortino": 0.8,
                        "max_drawdown": -0.02,
                        "trades": 7.0,
                    },
                ],
            },
        ]

    monkeypatch.setattr(sweep_mod.daily_stock, "run_backtest_multi_window_variant_matrix_via_trading_server", _fake_multi_runner)
    output_path = tmp_path / "reports" / "multi_window.json"

    exit_code = sweep_mod.main(
        [
            "--preset",
            "current_vs_candidates",
            "--json",
            "--window",
            "60",
            "--window",
            "120",
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["config"]["days_list"] == [60, 120]
    assert payload["summary"][0]["name"] == "single_static_25"
    assert payload["baseline_comparison"]["beaters"][0]["name"] == "single_static_25"
    run_log = json.loads(output_path.with_suffix(".run.json").read_text(encoding="utf-8"))
    assert run_log["status"] == "success"
    assert run_log["mode"] == "multi_window"
    assert run_log["top_result_name"] == "single_static_25"


def test_main_text_mode_writes_json_report(monkeypatch, tmp_path, capsys) -> None:
    def _fake_runner(**kwargs):
        return [
            {
                "name": "current_live_12p5",
                "allocation_pct": 12.5,
                "allocation_sizing_mode": "static",
                "multi_position": 0,
                "multi_position_min_prob_ratio": 0.3,
                "buying_power_multiplier": 1.0,
                "total_return": -0.01,
                "annualized_return": -0.02,
                "sortino": -0.5,
                "max_drawdown": -0.03,
                "trades": 8.0,
            },
            {
                "name": "single_static_25",
                "allocation_pct": 25.0,
                "allocation_sizing_mode": "static",
                "multi_position": 0,
                "multi_position_min_prob_ratio": 0.3,
                "buying_power_multiplier": 1.0,
                "total_return": 0.02,
                "annualized_return": 0.04,
                "sortino": 0.8,
                "max_drawdown": -0.02,
                "trades": 7.0,
            },
        ]

    monkeypatch.setattr(sweep_mod.daily_stock, "run_backtest_variant_matrix_via_trading_server", _fake_runner)
    output_path = tmp_path / "reports" / "single_window.json"

    exit_code = sweep_mod.main(
        [
            "--preset",
            "current_vs_candidates",
            "--days",
            "60",
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert "Beat baseline" in capsys.readouterr().out
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["results"][0]["name"] == "single_static_25"
    assert payload["baseline_comparison"]["beaters"][0]["name"] == "single_static_25"
    run_log = json.loads(output_path.with_suffix(".run.json").read_text(encoding="utf-8"))
    assert run_log["status"] == "success"
    assert run_log["mode"] == "single_window"


def test_parse_args_rejects_nonpositive_windows() -> None:
    try:
        sweep_mod.parse_args(["--window", "0", "--window", "-5"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected parse_args to reject empty resolved window set")


def test_main_returns_error_when_output_json_write_fails(monkeypatch, capsys) -> None:
    def _boom(self, rendered: str, *, encoding: str) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", _boom)

    exit_code = sweep_mod.main(
        [
            "--dry-run",
            "--preset",
            "promising_only",
            "--output-json",
            "reports/sweep.json",
        ]
    )

    assert exit_code == 1
    assert "Failed to write JSON report" in capsys.readouterr().err


def test_main_returns_error_when_window_backtest_fails(monkeypatch, capsys) -> None:
    def _boom(**kwargs):
        raise RuntimeError("transient broker simulation failure")

    monkeypatch.setattr(sweep_mod.daily_stock, "run_backtest_variant_matrix_via_trading_server", _boom)

    exit_code = sweep_mod.main(["--preset", "current_vs_candidates", "--days", "60"])

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "Backtest sweep failed" in stderr
    assert "60 trading days" in stderr


def test_main_failure_writes_run_log_when_configured(monkeypatch, tmp_path, capsys) -> None:
    def _boom(**kwargs):
        raise RuntimeError("transient broker simulation failure")

    monkeypatch.setattr(sweep_mod.daily_stock, "run_backtest_variant_matrix_via_trading_server", _boom)
    run_log_path = tmp_path / "logs" / "failure.run.json"

    exit_code = sweep_mod.main(
        [
            "--preset",
            "current_vs_candidates",
            "--days",
            "60",
            "--run-log-json",
            str(run_log_path),
        ]
    )

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert f"run log: {run_log_path}" in stderr
    payload = json.loads(run_log_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failure"
    assert payload["error_type"] == "RuntimeError"
    assert "transient broker simulation failure" in payload["error"]


def test_main_returns_error_for_invalid_symbol_input(capsys) -> None:
    exit_code = sweep_mod.main(["--dry-run", "--symbols", "../../etc/passwd"])

    assert exit_code == 1
    assert "Unsupported symbol" in capsys.readouterr().err


def test_table_for_results_contains_headers() -> None:
    table = sweep_mod._table_for_results(
        [
            {
                "name": "demo",
                "allocation_pct": 25.0,
                "multi_position": 2,
                "total_return": 0.02,
                "monthly_return": 0.003,
                "sortino": 0.8,
                "max_drawdown": -0.02,
                "trades": 10.0,
            }
        ]
    )

    assert "name" in table
    assert "monthly_return" in table
    assert "demo" in table


def test_table_for_multi_window_summary_contains_headers() -> None:
    table = sweep_mod._table_for_multi_window_summary(
        [
            {
                "name": "demo",
                "allocation_pct": 25.0,
                "multi_position": 2,
                "avg_monthly_return": 0.003,
                "min_monthly_return": -0.001,
                "avg_sortino": 0.8,
                "worst_max_drawdown": -0.02,
                "window_count": 3,
            }
        ]
    )

    assert "avg_monthly" in table
    assert "min_monthly" in table
    assert "demo" in table


def test_format_single_window_baseline_comparison_lists_beaters() -> None:
    message = sweep_mod._format_single_window_baseline_comparison(
        {
            "baseline_name": "current_live_12p5",
            "baseline_monthly_return": -0.001,
            "beaters": [
                {
                    "name": "single_static_25",
                    "candidate_monthly_return": 0.003,
                    "delta_monthly_return": 0.004,
                }
            ],
        }
    )

    assert message is not None
    assert "current_live_12p5" in message
    assert "single_static_25" in message


def test_format_multi_window_baseline_comparison_lists_beaters() -> None:
    message = sweep_mod._format_multi_window_baseline_comparison(
        {
            "baseline_name": "current_live_12p5",
            "baseline_avg_monthly_return": -0.001,
            "baseline_min_monthly_return": -0.003,
            "beaters": [
                {
                    "name": "portfolio2_static_50",
                    "candidate_avg_monthly_return": 0.004,
                    "candidate_min_monthly_return": 0.001,
                    "delta_avg_monthly_return": 0.005,
                    "delta_min_monthly_return": 0.004,
                }
            ],
        }
    )

    assert message is not None
    assert "current_live_12p5" in message
    assert "portfolio2_static_50" in message
