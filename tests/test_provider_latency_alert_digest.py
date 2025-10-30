from __future__ import annotations

from pathlib import Path

from scripts.provider_latency_alert_digest import load_alerts, summarise, summarise_details


def test_load_alerts_parses_lines(tmp_path):
    log = tmp_path / "alerts.log"
    log.write_text(
        "2025-10-24T20:00:00+00:00 Rolling latency for YAHOO shifted +45.0 ms\n",
        encoding="utf-8",
    )
    alerts = load_alerts(log)
    assert alerts[0][1].startswith("Rolling latency")


def test_summarise_outputs_markdown():
    alerts = [
        ("2025-10-24T20:00:00+00:00", "Rolling latency for YAHOO shifted +45.0 ms"),
        ("2025-10-24T21:00:00+00:00", "Rolling latency for YAHOO shifted +50.0 ms"),
    ]
    digest = summarise(alerts)
    assert "Latency Alert Digest" in digest
    assert "Total alerts" in digest
    assert "Severity Counts" in digest


def test_summarise_details_tracks_provider_severity():
    alerts = [
        ("2025-10-24T20:00:00+00:00", "Rolling latency for YAHOO exceeded threshold +45.0 ms"),
        ("2025-10-24T21:00:00+00:00", "Rolling latency for YAHOO warn limit"),
    ]
    _, provider_severity, severity_counter = summarise_details(alerts)
    assert provider_severity["YAHOO"]["CRIT"] >= 1
    assert severity_counter["WARN"] >= 1
