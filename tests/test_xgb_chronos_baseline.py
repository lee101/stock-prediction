from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import unified_hourly_experiment.xgb_chronos_baseline as baseline

from unified_hourly_experiment.xgb_chronos_baseline import (
    ConstantProbabilityModel,
    DEFAULT_MAX_SEARCH_CONFIGS,
    EFFECTIVE_ARGS_FILENAME,
    RUN_EVENTS_FILENAME,
    SearchConfig,
    WindowMetrics,
    build_effective_args_manifest,
    blend_forecast_price,
    build_parser,
    build_markdown_report,
    build_results_leaderboard,
    build_search_space_report,
    build_actions_and_bars,
    build_action_row,
    build_labeled_rows,
    iter_search_configs,
    parse_entry_allocator_mode_list,
    parse_entry_selection_mode_list,
    parse_label_basis_list,
    save_models,
    train_classifier,
    validate_search_space_budget,
)


def _cfg(**overrides) -> SearchConfig:
    payload = {
        "label_horizon_hours": 4,
        "label_basis": "reference_close",
        "residual_scale": 1.0,
        "risk_penalty": 1.0,
        "min_trade_probability": 0.45,
        "probability_power": 1.0,
        "entry_alpha": 0.5,
        "exit_alpha": 0.75,
        "edge_threshold": 0.002,
        "edge_to_full_size": 0.02,
        "max_positions": 5,
        "max_hold_hours": 4,
        "close_at_eod": False,
        "market_order_entry": False,
        "entry_selection_mode": "edge_rank",
        "entry_allocator_mode": "legacy",
        "entry_allocator_edge_power": 2.0,
    }
    payload.update(overrides)
    return SearchConfig(**payload)


def _read_jsonl(path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _main_cli_args(output_dir) -> list[str]:
    return [
        "xgb_chronos_baseline.py",
        "--symbols",
        "NET",
        "--forecast-horizons",
        "1,24",
        "--eval-windows",
        "7",
        "--label-horizon-grid",
        "2",
        "--label-basis-grid",
        "reference_close",
        "--residual-scale-grid",
        "0.5",
        "--risk-penalty-grid",
        "1.0",
        "--min-trade-probability-grid",
        "0.45",
        "--probability-power-grid",
        "1.0",
        "--entry-alpha-grid",
        "0.5",
        "--exit-alpha-grid",
        "0.75",
        "--edge-threshold-grid",
        "0.002",
        "--edge-to-full-size-grid",
        "0.02",
        "--max-positions-grid",
        "5",
        "--max-hold-hours-grid",
        "2",
        "--close-at-eod-grid",
        "0",
        "--market-order-entry-grid",
        "0",
        "--entry-selection-mode-grid",
        "edge_rank",
        "--entry-allocator-mode-grid",
        "legacy",
        "--entry-allocator-edge-power-grid",
        "2.0",
        "--output-dir",
        str(output_dir),
    ]


def test_build_action_row_prefers_long_when_long_edge_wins() -> None:
    row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-03-28T14:00:00Z"),
            "symbol": "NET",
            "reference_close": 100.0,
            "pred_high_ret_xgb": 0.03,
            "pred_low_ret_xgb": -0.005,
            "pred_close_ret_xgb": 0.015,
            "pred_long_prob_xgb": 0.85,
            "pred_short_prob_xgb": 0.20,
        }
    )

    action = build_action_row(row, _cfg())

    assert action["buy_amount"] > 0.0
    assert action["sell_amount"] == 0.0
    assert action["sell_price"] > action["buy_price"]
    assert action["xgb_long_edge"] > action["xgb_short_edge"]


def test_build_action_row_prefers_short_for_short_only_symbol() -> None:
    row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-03-28T14:00:00Z"),
            "symbol": "DBX",
            "reference_close": 50.0,
            "pred_high_ret_xgb": 0.01,
            "pred_low_ret_xgb": -0.04,
            "pred_close_ret_xgb": -0.02,
            "pred_long_prob_xgb": 0.15,
            "pred_short_prob_xgb": 0.90,
        }
    )

    action = build_action_row(row, _cfg())

    assert action["sell_amount"] > 0.0
    assert action["buy_amount"] == 0.0
    assert action["sell_price"] > action["buy_price"]
    assert action["xgb_short_edge"] > action["xgb_long_edge"]


def test_build_action_row_skips_when_edge_below_threshold() -> None:
    row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-03-28T14:00:00Z"),
            "symbol": "NET",
            "reference_close": 100.0,
            "pred_high_ret_xgb": 0.004,
            "pred_low_ret_xgb": -0.003,
            "pred_close_ret_xgb": 0.001,
            "pred_long_prob_xgb": 0.80,
            "pred_short_prob_xgb": 0.20,
        }
    )

    action = build_action_row(row, _cfg(edge_threshold=0.01))

    assert action["buy_amount"] == 0.0
    assert action["sell_amount"] == 0.0
    assert action["trade_amount"] == 0.0


def test_build_action_row_skips_when_probability_below_threshold() -> None:
    row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-03-28T14:00:00Z"),
            "symbol": "NET",
            "reference_close": 100.0,
            "pred_high_ret_xgb": 0.03,
            "pred_low_ret_xgb": -0.005,
            "pred_close_ret_xgb": 0.015,
            "pred_long_prob_xgb": 0.40,
            "pred_short_prob_xgb": 0.10,
        }
    )

    action = build_action_row(row, _cfg(min_trade_probability=0.60))

    assert action["buy_amount"] == 0.0
    assert action["sell_amount"] == 0.0
    assert action["trade_amount"] == 0.0


def test_build_action_row_accepts_itertuples_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-03-28T14:00:00Z"),
                "symbol": "net",
                "reference_close": 100.0,
                "pred_high_ret_xgb": 0.03,
                "pred_low_ret_xgb": -0.005,
                "pred_close_ret_xgb": 0.015,
                "pred_long_prob_xgb": 0.85,
                "pred_short_prob_xgb": 0.20,
            }
        ]
    )

    action = build_action_row(next(frame.itertuples(index=False)), _cfg())

    assert action["symbol"] == "NET"
    assert action["buy_amount"] > 0.0
    assert action["sell_amount"] == 0.0


def test_save_models_falls_back_to_booster_on_typeerror(tmp_path: Path) -> None:
    class _Booster:
        def save_model(self, path: str) -> None:
            Path(path).write_text("booster")

    class _Wrapper:
        def save_model(self, path: str) -> None:
            raise TypeError("_estimator_type undefined")

        def get_booster(self) -> _Booster:
            return _Booster()

    wrapper = _Wrapper()
    const = ConstantProbabilityModel(probability=0.5)

    save_models(
        {
            "high": wrapper,
            "low": wrapper,
            "close": wrapper,
            "long_cls": const,
            "short_cls": const,
        },
        tmp_path,
    )

    assert (tmp_path / "models" / "high.json").read_text() == "booster"
    assert (tmp_path / "models" / "long_cls.json").exists()


def test_build_actions_and_bars_assigns_predictions_without_merge() -> None:
    scored = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-03-28T15:00:00Z"),
                "symbol": "net",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000.0,
                "reference_close": 100.0,
                "pred_high_ret_xgb": 0.03,
                "pred_low_ret_xgb": -0.005,
                "pred_close_ret_xgb": 0.015,
                "pred_long_prob_xgb": 0.85,
                "pred_short_prob_xgb": 0.20,
            },
            {
                "timestamp": pd.Timestamp("2026-03-28T14:00:00Z"),
                "symbol": "dbx",
                "open": 50.0,
                "high": 51.0,
                "low": 49.0,
                "close": 50.5,
                "volume": 1200.0,
                "reference_close": 50.0,
                "pred_high_ret_xgb": 0.01,
                "pred_low_ret_xgb": -0.04,
                "pred_close_ret_xgb": -0.02,
                "pred_long_prob_xgb": 0.15,
                "pred_short_prob_xgb": 0.90,
            },
        ]
    )

    bars, actions = build_actions_and_bars(scored, _cfg())

    assert list(bars["symbol"]) == ["DBX", "NET"]
    assert list(actions["symbol"]) == ["DBX", "NET"]
    assert list(bars["predicted_high_p50_h1"]) == pytest.approx([50.5, 103.0])
    assert list(bars["predicted_low_p50_h1"]) == pytest.approx([48.0, 99.5])
    assert list(bars["predicted_close_p50_h1"]) == pytest.approx([49.0, 101.5])
    assert list(bars["predicted_high_p50_h24"]) == list(bars["predicted_high_p50_h1"])
    assert list(bars["predicted_low_p50_h24"]) == list(bars["predicted_low_p50_h1"])
    assert list(bars["predicted_close_p50_h24"]) == list(bars["predicted_close_p50_h1"])


def test_build_actions_and_bars_handles_empty_frames() -> None:
    scored = pd.DataFrame(
        columns=[
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "reference_close",
            "pred_high_ret_xgb",
            "pred_low_ret_xgb",
            "pred_close_ret_xgb",
            "pred_long_prob_xgb",
            "pred_short_prob_xgb",
        ]
    )

    bars, actions = build_actions_and_bars(scored, _cfg())

    assert bars.empty
    assert actions.empty
    assert "predicted_high_p50_h1" in bars.columns
    assert "predicted_high_p50_h1" in actions.columns


def test_build_actions_and_bars_preserves_rows_with_duplicate_timestamp_symbol_keys() -> None:
    duplicate_ts = pd.Timestamp("2026-03-28T14:00:00Z")
    scored = pd.DataFrame(
        [
            {
                "timestamp": duplicate_ts,
                "symbol": "net",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000.0,
                "reference_close": 100.0,
                "pred_high_ret_xgb": 0.01,
                "pred_low_ret_xgb": -0.01,
                "pred_close_ret_xgb": 0.005,
                "pred_long_prob_xgb": 0.80,
                "pred_short_prob_xgb": 0.10,
            },
            {
                "timestamp": duplicate_ts,
                "symbol": "net",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000.0,
                "reference_close": 100.0,
                "pred_high_ret_xgb": 0.04,
                "pred_low_ret_xgb": -0.02,
                "pred_close_ret_xgb": 0.015,
                "pred_long_prob_xgb": 0.90,
                "pred_short_prob_xgb": 0.10,
            },
        ]
    )

    bars, actions = build_actions_and_bars(scored, _cfg())

    assert len(bars) == 2
    assert len(actions) == 2
    assert list(bars["predicted_high_p50_h1"]) == pytest.approx([101.0, 104.0])
    assert list(bars["predicted_low_p50_h1"]) == pytest.approx([99.0, 98.0])
    assert list(bars["predicted_close_p50_h1"]) == pytest.approx([100.5, 101.5])
    assert list(actions["predicted_high_p50_h1"]) == pytest.approx([101.0, 104.0])


def test_search_config_rejects_invalid_probability_threshold() -> None:
    with pytest.raises(ValueError, match="min_trade_probability"):
        _cfg(min_trade_probability=1.5)


def test_search_config_rejects_invalid_entry_allocator_mode() -> None:
    with pytest.raises(ValueError, match="entry_allocator_mode"):
        _cfg(entry_allocator_mode="unknown")


@pytest.mark.parametrize(("labels", "expected_prob"), [([1, 1, 1], 1.0), ([0, 0, 0], 0.0)])
def test_train_classifier_falls_back_to_constant_probability_model(tmp_path, labels, expected_prob) -> None:
    X = pd.DataFrame({"feature": [0.1, 0.2, 0.3]})
    y = pd.Series(labels)

    model = train_classifier(
        X,
        y,
        sample_weight=np.ones(len(y), dtype=np.float64),
        seed=7,
    )

    assert isinstance(model, ConstantProbabilityModel)
    probs = model.predict_proba(X)
    assert probs.shape == (3, 2)
    assert np.allclose(probs[:, 1], expected_prob)
    assert np.allclose(probs[:, 0], 1.0 - expected_prob)

    output_path = tmp_path / "constant_model.json"
    model.save_model(str(output_path))

    assert json.loads(output_path.read_text()) == {"constant_probability": expected_prob}


def test_iter_search_configs_rejects_invalid_label_basis_token() -> None:
    args = SimpleNamespace(
        label_horizon_grid="2",
        label_basis_grid="bad_basis",
        residual_scale_grid="1.0",
        risk_penalty_grid="1.0",
        min_trade_probability_grid="0.45",
        probability_power_grid="1.0",
        entry_alpha_grid="0.5",
        exit_alpha_grid="0.75",
        edge_threshold_grid="0.002",
        edge_to_full_size_grid="0.02",
        max_positions_grid="5",
        max_hold_hours_grid="4",
        close_at_eod_grid="0",
        market_order_entry_grid="0",
        entry_selection_mode_grid="edge_rank",
        entry_allocator_mode_grid="legacy",
        entry_allocator_edge_power_grid="2.0",
    )

    with pytest.raises(ValueError, match="label_basis"):
        iter_search_configs(args)


@pytest.mark.parametrize(
    ("parser_fn", "value", "expected"),
    [
        (parse_label_basis_list, "reference_close,next_open", ["reference_close", "next_open"]),
        (parse_entry_selection_mode_list, "edge_rank,first_trigger", ["edge_rank", "first_trigger"]),
        (parse_entry_allocator_mode_list, "legacy,concentrated", ["legacy", "concentrated"]),
    ],
)
def test_choice_parsers_accept_supported_tokens(parser_fn, value, expected) -> None:
    assert parser_fn(value) == expected


def test_build_parser_exposes_describe_search_space_flags() -> None:
    parser = build_parser()

    args = parser.parse_args(
        ["--describe-search-space", "--describe-limit", "3", "--max-configs", "99", "--allow-large-search"]
    )

    assert args.describe_search_space is True
    assert args.describe_limit == 3
    assert args.max_configs == 99
    assert args.allow_large_search is True


def test_build_parser_supports_args_files_with_comments(tmp_path) -> None:
    args_file = tmp_path / "baseline.args"
    args_file.write_text(
        "\n".join(
            [
                "# baseline search file",
                "--symbols NET,DBX",
                "--forecast-horizons 1,24",
                "--describe-search-space",
                "--describe-limit 2",
                "",
                "--output-dir run_dir  # inline comment",
            ]
        )
    )
    parser = build_parser()

    args = parser.parse_args([f"@{args_file}"])

    assert args.symbols == "NET,DBX"
    assert args.forecast_horizons == "1,24"
    assert args.describe_search_space is True
    assert args.describe_limit == 2
    assert args.output_dir == Path("run_dir")


def test_build_effective_args_manifest_serializes_paths(monkeypatch, tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(["--symbols", "NET", "--output-dir", str(tmp_path)])
    monkeypatch.setattr("sys.argv", ["xgb_chronos_baseline.py", "--symbols", "NET", "--output-dir", str(tmp_path)])

    manifest = build_effective_args_manifest(args)

    assert manifest["argv"] == ["--symbols", "NET", "--output-dir", str(tmp_path)]
    assert manifest["resolved_args"]["symbols"] == "NET"
    assert manifest["resolved_args"]["output_dir"] == str(tmp_path)


def test_build_search_space_report_summarizes_grid_and_preview() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--symbols",
            "NET,DBX",
            "--forecast-horizons",
            "1,24",
            "--eval-windows",
            "7,14",
            "--sequence-length",
            "48",
            "--validation-days",
            "30",
            "--label-horizon-grid",
            "2,4",
            "--label-basis-grid",
            "reference_close,next_open",
            "--residual-scale-grid",
            "0.5",
            "--risk-penalty-grid",
            "1.0",
            "--min-trade-probability-grid",
            "0.45",
            "--probability-power-grid",
            "1.0",
            "--entry-alpha-grid",
            "0.5",
            "--exit-alpha-grid",
            "0.75",
            "--edge-threshold-grid",
            "0.002",
            "--edge-to-full-size-grid",
            "0.02",
            "--max-positions-grid",
            "5",
            "--max-hold-hours-grid",
            "2",
            "--close-at-eod-grid",
            "0",
            "--market-order-entry-grid",
            "0",
            "--entry-selection-mode-grid",
            "edge_rank",
            "--entry-allocator-mode-grid",
            "legacy",
            "--entry-allocator-edge-power-grid",
            "2.0",
        ]
    )
    search_configs = iter_search_configs(args)

    report = build_search_space_report(
        args,
        symbols=["NET", "DBX"],
        forecast_horizons=(1, 24),
        eval_windows=[7, 14],
        search_configs=search_configs,
        describe_limit=2,
    )

    assert report["symbol_count"] == 2
    assert report["config_count"] == 4
    assert report["dataset_count"] == 4
    assert report["forecast_horizons"] == [1, 24]
    assert report["eval_windows"] == [7, 14]
    assert report["max_configs"] == DEFAULT_MAX_SEARCH_CONFIGS
    assert report["allow_large_search"] is False
    assert report["over_budget"] is False
    assert report["describe_limit"] == 2
    assert len(report["config_preview"]) == 2
    assert report["datasets"] == [
        {"label_horizon_hours": 2, "label_basis": "next_open"},
        {"label_horizon_hours": 2, "label_basis": "reference_close"},
        {"label_horizon_hours": 4, "label_basis": "next_open"},
        {"label_horizon_hours": 4, "label_basis": "reference_close"},
    ]


def test_validate_search_space_budget_rejects_oversized_run() -> None:
    report = {
        "config_count": 1024,
        "max_configs": 128,
        "allow_large_search": False,
    }

    with pytest.raises(ValueError, match="Resolved 1024 search configs"):
        validate_search_space_budget(report)


def test_validate_search_space_budget_allows_explicit_override() -> None:
    report = {
        "config_count": 1024,
        "max_configs": 128,
        "allow_large_search": True,
    }

    validate_search_space_budget(report)


def test_build_results_leaderboard_flattens_and_sorts_results() -> None:
    results = [
        {
            "config": asdict(_cfg(label_horizon_hours=2, residual_scale=0.25)),
            "selection_score": 1.25,
            "window_metrics": [
                {
                    "window_days": 7,
                    "total_return_pct": 2.0,
                    "sortino": 3.5,
                    "max_drawdown_pct": 1.2,
                    "pnl_smoothness": 0.1,
                    "pnl_smoothness_score": 0.9,
                    "goodness_score": 0.8,
                    "num_buys": 4,
                    "num_sells": 4,
                }
            ],
            "encoded_feature_count": 42,
            "train_rows": 100,
            "holdout_rows": 20,
        },
        {
            "config": asdict(_cfg(label_horizon_hours=4, residual_scale=0.5)),
            "selection_score": 2.5,
            "window_metrics": [
                {
                    "window_days": 7,
                    "total_return_pct": 4.0,
                    "sortino": 5.5,
                    "max_drawdown_pct": 0.8,
                    "pnl_smoothness": 0.2,
                    "pnl_smoothness_score": 1.1,
                    "goodness_score": 1.0,
                    "num_buys": 6,
                    "num_sells": 6,
                }
            ],
            "encoded_feature_count": 45,
            "train_rows": 120,
            "holdout_rows": 25,
        },
    ]

    leaderboard = build_results_leaderboard(results)

    assert list(leaderboard["selection_score"]) == [2.5, 1.25]
    assert list(leaderboard["label_horizon_hours"]) == [4, 2]
    assert list(leaderboard["w7d_return_pct"]) == [4.0, 2.0]
    assert list(leaderboard["w7d_sortino"]) == [5.5, 3.5]
    assert list(leaderboard["w7d_num_buys"]) == [6, 4]


def test_build_markdown_report_includes_best_and_top_configs() -> None:
    leaderboard = pd.DataFrame(
        [
            {
                "selection_score": 2.5,
                "label_horizon_hours": 4,
                "label_basis": "reference_close",
                "residual_scale": 0.5,
                "risk_penalty": 1.0,
                "min_trade_probability": 0.45,
                "edge_threshold": 0.002,
                "max_hold_hours": 2,
            },
            {
                "selection_score": 1.25,
                "label_horizon_hours": 2,
                "label_basis": "next_open",
                "residual_scale": 0.25,
                "risk_penalty": 1.5,
                "min_trade_probability": 0.55,
                "edge_threshold": 0.004,
                "max_hold_hours": 4,
            },
        ]
    )
    report = {
        "created_at_utc": "2026-03-29T00:00:00+00:00",
        "output_dir": "experiments/demo",
        "symbols": ["NET", "DBX"],
        "forecast_horizons": [1, 24],
        "validation_days": 30,
        "eval_windows": [7, 14],
        "search_space": {"config_count": 8, "dataset_count": 4},
        "best": {
            "config": asdict(_cfg(label_horizon_hours=4, residual_scale=0.5)),
            "selection_score": 2.5,
            "window_metrics": [
                {
                    "window_days": 7,
                    "total_return_pct": 4.0,
                    "sortino": 5.5,
                    "max_drawdown_pct": 0.8,
                    "pnl_smoothness": 0.2,
                    "pnl_smoothness_score": 1.1,
                    "goodness_score": 1.0,
                    "num_buys": 6,
                    "num_sells": 6,
                }
            ],
        },
    }

    markdown = build_markdown_report(report, leaderboard, top_n=2)

    assert "# XGB Chronos Baseline Report" in markdown
    assert "## Best Config" in markdown
    assert "## Best Window Metrics" in markdown
    assert "## Top 2 Configs" in markdown
    assert "| Window | Return % | Sortino | Max DD % | Smoothness | Goodness | Buys | Sells |" in markdown
    assert "| Rank | Score | Horizon | Basis | Residual | Risk | Min Prob | Edge | Max Hold |" in markdown
    assert "`experiments/demo`" in markdown


def test_main_writes_leaderboard_and_markdown_report(monkeypatch, tmp_path) -> None:
    train_df = pd.DataFrame({"feature": [1.0], "target_long_quality": [0.1], "target_short_quality": [0.0]})
    holdout_df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-03-28T14:00:00Z"),
                "symbol": "NET",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000.0,
                "reference_close": 100.0,
                "pred_high_ret_xgb": 0.03,
                "pred_low_ret_xgb": -0.005,
                "pred_close_ret_xgb": 0.015,
                "pred_long_prob_xgb": 0.85,
                "pred_short_prob_xgb": 0.20,
            }
        ]
    )

    monkeypatch.setattr(
        baseline,
        "prepare_dataset",
        lambda **kwargs: (train_df.copy(), holdout_df.copy()),
    )
    monkeypatch.setattr(baseline, "resolve_feature_columns", lambda forecast_horizons, frame: ["feature"])
    monkeypatch.setattr(
        baseline,
        "train_models",
        lambda train_df, **kwargs: ({}, pd.DataFrame({"feature": [1.0]}), ["feature"]),
    )
    monkeypatch.setattr(
        baseline,
        "score_rows",
        lambda holdout_df, train_df, **kwargs: holdout_df.copy(),
    )
    monkeypatch.setattr(
        baseline,
        "evaluate_windows",
        lambda *args, **kwargs: [
            WindowMetrics(
                window_days=7,
                total_return_pct=1.5,
                sortino=2.0,
                max_drawdown_pct=0.5,
                pnl_smoothness=0.1,
                pnl_smoothness_score=0.8,
                goodness_score=0.9,
                num_buys=1,
                num_sells=1,
            )
        ],
    )
    monkeypatch.setattr(baseline, "save_models", lambda models, output_dir: None)
    monkeypatch.setattr("sys.argv", _main_cli_args(tmp_path))

    baseline.main()

    leaderboard = pd.read_csv(tmp_path / "leaderboard.csv")
    report_json = json.loads((tmp_path / "report.json").read_text())
    markdown = (tmp_path / "report.md").read_text()
    effective_args = json.loads((tmp_path / EFFECTIVE_ARGS_FILENAME).read_text())
    events = _read_jsonl(tmp_path / RUN_EVENTS_FILENAME)
    event_types = [str(event["event_type"]) for event in events]

    assert leaderboard.loc[0, "selection_score"] == pytest.approx(3.915)
    assert leaderboard.loc[0, "w7d_return_pct"] == pytest.approx(1.5)
    assert report_json["leaderboard_path"] == str(tmp_path / "leaderboard.csv")
    assert report_json["markdown_report_path"] == str(tmp_path / "report.md")
    assert report_json["run_events_path"] == str(tmp_path / RUN_EVENTS_FILENAME)
    assert report_json["effective_args_path"] == str(tmp_path / EFFECTIVE_ARGS_FILENAME)
    assert report_json["score_cache_entries"] == 1
    assert report_json["score_cache_hits"] == 1
    assert report_json["score_cache_misses"] == 1
    assert effective_args["resolved_args"]["symbols"] == "NET"
    assert effective_args["resolved_args"]["output_dir"] == str(tmp_path)
    assert "# XGB Chronos Baseline Report" in markdown
    assert "## Top 1 Configs" in markdown
    assert event_types == [
        "run_start",
        "dataset_prepare_start",
        "dataset_prepared",
        "config_start",
        "config_complete",
        "run_complete",
    ]
    assert events[4]["selection_score"] == pytest.approx(3.915)
    assert events[4]["best_selection_score_so_far"] == pytest.approx(3.915)
    assert events[4]["score_cache_hit"] is False
    assert events[0]["effective_args_path"] == str(tmp_path / EFFECTIVE_ARGS_FILENAME)
    assert events[0]["argv"] == _main_cli_args(tmp_path)[1:]


def test_main_reuses_scored_holdout_for_duplicate_residual_scale_configs(monkeypatch, tmp_path) -> None:
    train_df = pd.DataFrame({"feature": [1.0], "target_long_quality": [0.1], "target_short_quality": [0.0]})
    holdout_df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-03-28T14:00:00Z"),
                "symbol": "NET",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000.0,
                "reference_close": 100.0,
                "pred_high_ret_xgb": 0.03,
                "pred_low_ret_xgb": -0.005,
                "pred_close_ret_xgb": 0.015,
                "pred_long_prob_xgb": 0.85,
                "pred_short_prob_xgb": 0.20,
            }
        ]
    )
    score_call_count = 0

    def fake_score_rows(holdout_df, train_df, **kwargs):
        nonlocal score_call_count
        score_call_count += 1
        return holdout_df.copy()

    monkeypatch.setattr(
        baseline,
        "prepare_dataset",
        lambda **kwargs: (train_df.copy(), holdout_df.copy()),
    )
    monkeypatch.setattr(baseline, "resolve_feature_columns", lambda forecast_horizons, frame: ["feature"])
    monkeypatch.setattr(
        baseline,
        "train_models",
        lambda train_df, **kwargs: ({}, pd.DataFrame({"feature": [1.0]}), ["feature"]),
    )
    monkeypatch.setattr(baseline, "score_rows", fake_score_rows)
    monkeypatch.setattr(
        baseline,
        "evaluate_windows",
        lambda *args, **kwargs: [
            WindowMetrics(
                window_days=7,
                total_return_pct=1.5,
                sortino=2.0,
                max_drawdown_pct=0.5,
                pnl_smoothness=0.1,
                pnl_smoothness_score=0.8,
                goodness_score=0.9,
                num_buys=1,
                num_sells=1,
            )
        ],
    )
    monkeypatch.setattr(baseline, "save_models", lambda models, output_dir: None)
    monkeypatch.setattr(
        "sys.argv",
        _main_cli_args(tmp_path)[:-2]
        + [
            "--risk-penalty-grid",
            "1.0,1.5",
            "--output-dir",
            str(tmp_path),
        ],
    )

    baseline.main()

    report_json = json.loads((tmp_path / "report.json").read_text())
    events = _read_jsonl(tmp_path / RUN_EVENTS_FILENAME)
    config_complete_events = [event for event in events if event["event_type"] == "config_complete"]

    assert score_call_count == 1
    assert report_json["score_cache_entries"] == 1
    assert report_json["score_cache_hits"] == 2
    assert report_json["score_cache_misses"] == 1
    assert len(config_complete_events) == 2
    assert config_complete_events[0]["score_cache_hit"] is False
    assert config_complete_events[1]["score_cache_hit"] is True


def test_main_logs_run_error_event_before_reraising(monkeypatch, tmp_path) -> None:
    train_df = pd.DataFrame({"feature": [1.0], "target_long_quality": [0.1], "target_short_quality": [0.0]})
    holdout_df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-03-28T14:00:00Z"),
                "symbol": "NET",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000.0,
                "reference_close": 100.0,
                "pred_high_ret_xgb": 0.03,
                "pred_low_ret_xgb": -0.005,
                "pred_close_ret_xgb": 0.015,
                "pred_long_prob_xgb": 0.85,
                "pred_short_prob_xgb": 0.20,
            }
        ]
    )

    monkeypatch.setattr(
        baseline,
        "prepare_dataset",
        lambda **kwargs: (train_df.copy(), holdout_df.copy()),
    )
    monkeypatch.setattr(baseline, "resolve_feature_columns", lambda forecast_horizons, frame: ["feature"])
    monkeypatch.setattr(
        baseline,
        "train_models",
        lambda train_df, **kwargs: ({}, pd.DataFrame({"feature": [1.0]}), ["feature"]),
    )
    monkeypatch.setattr(
        baseline,
        "score_rows",
        lambda holdout_df, train_df, **kwargs: holdout_df.copy(),
    )
    monkeypatch.setattr(
        baseline,
        "evaluate_windows",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("sim failed")),
    )
    monkeypatch.setattr("sys.argv", _main_cli_args(tmp_path))

    with pytest.raises(RuntimeError, match="sim failed"):
        baseline.main()

    events = _read_jsonl(tmp_path / RUN_EVENTS_FILENAME)
    event_types = [str(event["event_type"]) for event in events]

    assert event_types == [
        "run_start",
        "dataset_prepare_start",
        "dataset_prepared",
        "config_start",
        "config_error",
        "run_error",
    ]
    assert events[4]["error_type"] == "RuntimeError"
    assert "sim failed" in str(events[4]["error_message"])
    assert "evaluate_windows" in str(events[4]["traceback"])
    assert events[5]["error_type"] == "RuntimeError"
    assert float(events[5]["duration_seconds"]) >= 0.0


def test_main_logs_dataset_error_event_before_reraising(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        baseline,
        "prepare_dataset",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("dataset load failed")),
    )
    monkeypatch.setattr("sys.argv", _main_cli_args(tmp_path))

    with pytest.raises(RuntimeError, match="dataset load failed"):
        baseline.main()

    events = _read_jsonl(tmp_path / RUN_EVENTS_FILENAME)
    event_types = [str(event["event_type"]) for event in events]

    assert event_types == [
        "run_start",
        "dataset_prepare_start",
        "dataset_error",
        "run_error",
    ]
    assert events[2]["error_type"] == "RuntimeError"
    assert "dataset load failed" in str(events[2]["error_message"])
    assert events[2]["label_horizon_hours"] == 2
    assert events[2]["label_basis"] == "reference_close"


def test_main_rejects_oversized_search_before_dataset_prep(monkeypatch) -> None:
    monkeypatch.setattr(
        baseline,
        "prepare_dataset",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("prepare_dataset should not run")),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "xgb_chronos_baseline.py",
            "--label-horizon-grid",
            "2,4",
            "--label-basis-grid",
            "reference_close",
            "--residual-scale-grid",
            "0.0,0.5,1.0",
            "--risk-penalty-grid",
            "0.75,1.0",
            "--min-trade-probability-grid",
            "0.45,0.55",
            "--probability-power-grid",
            "1.0,1.5",
            "--entry-alpha-grid",
            "0.25,0.5",
            "--exit-alpha-grid",
            "0.75",
            "--edge-threshold-grid",
            "0.002,0.004",
            "--edge-to-full-size-grid",
            "0.02",
            "--max-positions-grid",
            "5",
            "--max-hold-hours-grid",
            "2,4",
            "--close-at-eod-grid",
            "0,1",
            "--market-order-entry-grid",
            "0,1",
            "--entry-selection-mode-grid",
            "edge_rank,first_trigger",
            "--entry-allocator-mode-grid",
            "legacy,concentrated",
            "--entry-allocator-edge-power-grid",
            "2.0",
            "--max-configs",
            "128",
        ],
    )

    with pytest.raises(ValueError, match="exceeds --max-configs=128"):
        baseline.main()


def test_blend_forecast_price_interpolates_between_horizons() -> None:
    row = pd.Series(
        {
            "reference_close": 100.0,
            "predicted_close_p50_h1": 101.0,
            "predicted_close_p50_h24": 124.0,
        }
    )

    blended = blend_forecast_price(
        row,
        kind="close",
        target_horizon_hours=4,
        forecast_horizons=(1, 24),
    )

    assert 101.0 < blended < 124.0


def test_build_labeled_rows_uses_reference_close_basis() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-03-28T10:00:00Z",
                    "2026-03-28T11:00:00Z",
                    "2026-03-28T12:00:00Z",
                ],
                utc=True,
            ),
            "symbol": ["NET", "NET", "NET"],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.0, 102.0, 103.0],
            "reference_close": [100.0, 102.0, 103.0],
            "predicted_high_p50_h1": [102.0, 104.0, 105.0],
            "predicted_low_p50_h1": [99.0, 101.0, 102.0],
            "predicted_close_p50_h1": [101.0, 103.0, 104.0],
            "predicted_high_p50_h24": [110.0, 111.0, 112.0],
            "predicted_low_p50_h24": [95.0, 96.0, 97.0],
            "predicted_close_p50_h24": [108.0, 109.0, 110.0],
        }
    )

    labeled = build_labeled_rows(
        frame,
        label_horizon_hours=1,
        forecast_horizons=(1, 24),
        label_basis="reference_close",
    )

    assert labeled.loc[0, "target_close_ret"] == pytest.approx(0.02)
    assert labeled.loc[0, "prior_close_ret"] == pytest.approx(0.01)


def test_build_labeled_rows_uses_next_open_basis() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-03-28T10:00:00Z",
                    "2026-03-28T11:00:00Z",
                    "2026-03-28T12:00:00Z",
                ],
                utc=True,
            ),
            "symbol": ["NET", "NET", "NET"],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.0, 102.0, 103.0],
            "reference_close": [100.0, 102.0, 103.0],
            "predicted_high_p50_h1": [102.0, 104.0, 105.0],
            "predicted_low_p50_h1": [99.0, 101.0, 102.0],
            "predicted_close_p50_h1": [101.0, 103.0, 104.0],
            "predicted_high_p50_h24": [110.0, 111.0, 112.0],
            "predicted_low_p50_h24": [95.0, 96.0, 97.0],
            "predicted_close_p50_h24": [108.0, 109.0, 110.0],
        }
    )

    labeled = build_labeled_rows(
        frame,
        label_horizon_hours=1,
        forecast_horizons=(1, 24),
        label_basis="next_open",
    )

    assert labeled.loc[0, "target_close_ret"] == pytest.approx((102.0 / 101.0) - 1.0)
    assert labeled.loc[0, "prior_close_ret"] == pytest.approx(0.0)
