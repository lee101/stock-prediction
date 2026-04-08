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


def test_latest_alias_paths_use_stable_names() -> None:
    paths = audit_mod._latest_alias_paths(Path("artifacts/replay_audit"))

    assert paths == {
        "report": Path("artifacts/replay_audit/latest.json"),
        "visualization": Path("artifacts/replay_audit/latest.html"),
        "trace": Path("artifacts/replay_audit/latest.trace.json"),
        "manifest": Path("artifacts/replay_audit/latest_manifest.json"),
    }


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
    assert payload["latest_alias_paths"] == {
        "manifest": "artifacts/custom/latest_manifest.json",
        "report": "artifacts/custom/latest.json",
        "trace": "artifacts/custom/latest.trace.json",
        "visualization": "artifacts/custom/latest.html",
    }
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


def test_main_updates_latest_aliases_after_success(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "replay_audit" / "nvda_window.json"
    html_path = tmp_path / "replay_audit" / "nvda_window.html"
    trace_path = tmp_path / "replay_audit" / "nvda_window.trace.json"

    def _fake_main(argv: list[str]) -> None:
        del argv
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text("<html>ok</html>", encoding="utf-8")
        trace_path.write_text('{"trace":true}', encoding="utf-8")
        report_path.write_text(
            json.dumps(
                {
                    "visualization": {
                        "generated_html_path": str(html_path),
                        "trace_json_path": str(trace_path),
                    }
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(audit_mod.replay_mod, "main", _fake_main)

    audit_mod.main(
        [
            "--output-dir",
            str(report_path.parent),
            "--name",
            "nvda_window",
        ]
    )

    assert (report_path.parent / "latest.json").read_text(encoding="utf-8") == report_path.read_text(encoding="utf-8")
    assert (report_path.parent / "latest.html").read_text(encoding="utf-8") == "<html>ok</html>"
    assert (report_path.parent / "latest.trace.json").read_text(encoding="utf-8") == '{"trace":true}'

    manifest = json.loads((report_path.parent / "latest_manifest.json").read_text(encoding="utf-8"))
    assert manifest["report_path"] == str(report_path)
    assert manifest["latest_report_path"] == str(report_path.parent / "latest.json")
    assert manifest["latest_visualization_path"] == str(report_path.parent / "latest.html")
    assert manifest["latest_trace_json_path"] == str(report_path.parent / "latest.trace.json")


def test_main_clears_stale_latest_visualization_aliases_when_not_generated(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "replay_audit"
    report_path = output_dir / "nvda_window.json"
    stale_html = output_dir / "latest.html"
    stale_trace = output_dir / "latest.trace.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_html.write_text("stale html", encoding="utf-8")
    stale_trace.write_text("stale trace", encoding="utf-8")

    def _fake_main(argv: list[str]) -> None:
        del argv
        report_path.write_text(
            json.dumps(
                {
                    "visualization": {
                        "generated_html_path": None,
                        "trace_json_path": None,
                    }
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(audit_mod.replay_mod, "main", _fake_main)

    audit_mod.main(
        [
            "--output-dir",
            str(output_dir),
            "--name",
            "nvda_window",
        ]
    )

    assert not stale_html.exists()
    assert not stale_trace.exists()
    manifest = json.loads((output_dir / "latest_manifest.json").read_text(encoding="utf-8"))
    assert manifest["latest_visualization_path"] is None
    assert manifest["latest_trace_json_path"] is None


def test_main_does_not_fail_when_latest_alias_update_breaks(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    report_path = tmp_path / "replay_audit" / "nvda_window.json"

    def _fake_main(argv: list[str]) -> None:
        del argv
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"visualization": {}}), encoding="utf-8")

    monkeypatch.setattr(audit_mod.replay_mod, "main", _fake_main)
    monkeypatch.setattr(
        audit_mod,
        "_write_latest_aliases",
        lambda **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    audit_mod.main(
        [
            "--output-dir",
            str(report_path.parent),
            "--name",
            "nvda_window",
        ]
    )

    stderr = capsys.readouterr().err
    assert "failed to update latest replay aliases" in stderr
    assert "disk full" in stderr
    assert report_path.exists()


def test_main_rejects_unexpected_visualization_source_path(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    output_dir = tmp_path / "replay_audit"
    report_path = output_dir / "nvda_window.json"
    html_path = output_dir / "nvda_window.html"
    external_html = tmp_path / "outside.html"
    external_html.write_text("sensitive", encoding="utf-8")

    def _fake_main(argv: list[str]) -> None:
        del argv
        output_dir.mkdir(parents=True, exist_ok=True)
        html_path.write_text("<html>expected</html>", encoding="utf-8")
        report_path.write_text(
            json.dumps(
                {
                    "visualization": {
                        "generated_html_path": str(external_html),
                        "trace_json_path": None,
                    }
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(audit_mod.replay_mod, "main", _fake_main)

    audit_mod.main(
        [
            "--output-dir",
            str(output_dir),
            "--name",
            "nvda_window",
        ]
    )

    stderr = capsys.readouterr().err
    assert "failed to update latest replay aliases" in stderr
    assert "unexpected visualization path" in stderr
    assert not (output_dir / "latest.html").exists()
