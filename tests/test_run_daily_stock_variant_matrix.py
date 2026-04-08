from __future__ import annotations

import json

import scripts.run_daily_stock_variant_matrix as sweep_mod
from src.daily_stock_variant_presets import preset_choices, resolve_variant_preset


def test_parse_args_defaults_to_current_vs_candidates() -> None:
    args = sweep_mod.parse_args([])

    assert args.preset == "current_vs_candidates"
    assert args.days == 120
    assert args.window is None
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
    captured_days: list[int] = []

    def _fake_runner(**kwargs):
        days = int(kwargs["days"])
        captured_days.append(days)
        return [
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
        ]

    monkeypatch.setattr(sweep_mod.daily_stock, "run_backtest_variant_matrix_via_trading_server", _fake_runner)

    exit_code = sweep_mod.main(["--preset", "current_vs_candidates", "--json", "--window", "60", "--window", "120"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert captured_days == [60, 120]
    assert payload["config"]["days_list"] == [60, 120]
    assert payload["summary"][0]["name"] == "single_static_25"
    assert payload["summary"][0]["window_count"] == 2
    assert payload["baseline_comparison"]["baseline_name"] == "current_live_12p5"
    assert payload["baseline_comparison"]["beaters"][0]["name"] == "single_static_25"
    assert len(payload["windows"]) == 2


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
