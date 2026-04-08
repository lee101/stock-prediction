from __future__ import annotations

import json
from pathlib import Path

import pytest
import scripts.run_replay_stock_audit as audit_mod


def test_resolve_artifact_paths_defaults_to_timestamped_name() -> None:
    report_path, html_path = audit_mod._resolve_artifact_paths(
        output_dir=Path("artifacts/replay_audit"),
        name="",
        now=audit_mod.datetime(2026, 4, 8, 12, 34, 56, tzinfo=audit_mod.UTC),
    )

    assert report_path == Path("artifacts/replay_audit/stock_replay_audit_20260408_123456.json")
    assert html_path == Path("artifacts/replay_audit/stock_replay_audit_20260408_123456.html")


def test_resolve_artifact_paths_normalizes_name_suffix() -> None:
    report_path, html_path = audit_mod._resolve_artifact_paths(
        output_dir=Path("artifacts/replay_audit"),
        name="custom_run.json",
    )

    assert report_path == Path("artifacts/replay_audit/custom_run.json")
    assert html_path == Path("artifacts/replay_audit/custom_run.html")


def test_resolve_artifact_paths_rejects_nested_name_paths() -> None:
    with pytest.raises(ValueError, match="simple artifact stem"):
        audit_mod._resolve_artifact_paths(
            output_dir=Path("artifacts/replay_audit"),
            name="nested/custom_run",
        )


def test_build_replay_args_includes_artifact_paths_and_optional_values() -> None:
    args = audit_mod.parse_args(
        [
            "--symbols",
            "NVDA,MSFT",
            "--start",
            "2026-04-01T00:00:00Z",
            "--end",
            "2026-04-02T00:00:00Z",
            "--initial-cash",
            "25000",
            "--max-positions",
            "5",
            "--sim-backend",
            "native",
        ]
    )

    replay_args = audit_mod.build_replay_args(
        args,
        report_path=Path("out/report.json"),
        html_path=Path("out/report.html"),
    )

    assert replay_args[:8] == [
        "--trade-log",
        "strategy_state/stock_trade_log.jsonl",
        "--event-log",
        "strategy_state/stock_event_log.jsonl",
        "--data-root",
        "trainingdatahourly/stocks",
        "--output",
        "out/report.json",
    ]
    assert "--visualize-html" in replay_args
    assert "out/report.html" in replay_args
    assert "--symbols" in replay_args
    assert "NVDA,MSFT" in replay_args
    assert "--initial-cash" in replay_args
    assert "25000" in replay_args
    assert "--max-positions" in replay_args
    assert "5" in replay_args
    assert "--sim-backend" in replay_args
    assert "native" in replay_args


def test_main_dry_run_prints_resolved_payload(capsys) -> None:
    audit_mod.main(
        [
            "--dry-run",
            "--output-dir",
            "artifacts/custom",
            "--name",
            "nvda_window",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["report_path"] == "artifacts/custom/nvda_window.json"
    assert payload["visualization_path"] == "artifacts/custom/nvda_window.html"
    assert "--output" in payload["replay_args"]
    assert "artifacts/custom/nvda_window.json" in payload["replay_args"]
    assert "--visualize-html" in payload["replay_args"]
    assert "artifacts/custom/nvda_window.html" in payload["replay_args"]


def test_main_delegates_to_replay_module(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(audit_mod.replay_mod, "main", lambda argv: captured.setdefault("argv", list(argv)))

    audit_mod.main(
        [
            "--output-dir",
            "artifacts/custom",
            "--name",
            "nvda_window",
            "--symbols",
            "NVDA",
        ]
    )

    assert captured["argv"] == [
        "--trade-log",
        "strategy_state/stock_trade_log.jsonl",
        "--event-log",
        "strategy_state/stock_event_log.jsonl",
        "--data-root",
        "trainingdatahourly/stocks",
        "--output",
        "artifacts/custom/nvda_window.json",
        "--visualize-html",
        "artifacts/custom/nvda_window.html",
        "--visualize-num-pairs",
        "6",
        "--max-hold-hours",
        "6",
        "--min-edge",
        "-1",
        "--fee-rate",
        "0.001",
        "--leverage",
        "2",
        "--decision-lag-bars",
        "0",
        "--bar-margins",
        "0.0005,0.001,0.002",
        "--entry-order-ttls",
        "0,1,2",
        "--market-order-entries",
        "0,1",
        "--sim-backend",
        "python",
        "--cancel-ack-delays",
        "1",
        "--partial-fill-on-touch",
        "0,1",
        "--symbols",
        "NVDA",
    ]
