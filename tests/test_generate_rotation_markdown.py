from __future__ import annotations

from pathlib import Path

from scripts.generate_rotation_markdown import render_markdown


def test_render_markdown_with_latency_section(tmp_path):
    rows = [
        {
            "type": "removal",
            "symbol": "XYZ",
            "detail": "streak=10;trend_pnl=-200;last_escalation=2025-10-24",
            "timestamp": "2025-10-24T20:00:00+00:00",
        }
    ]
    latency = {"yahoo": {"avg_ms": 320.0, "delta_avg_ms": 5.0, "p95_ms": 340.0, "delta_p95_ms": 3.0}}
    digest_path = tmp_path / "digest.md"
    digest_path.write_text("# Latency Alert Digest\n- alert", encoding="utf-8")
    leaderboard = tmp_path / "leaderboard.md"
    leaderboard.write_text(
        "| Provider | INFO | WARN | CRIT | Total |\n|----------|------|------|------|-------|\n| YAHOO | 0 | 1 | 2 | 3 |\n",
        encoding="utf-8",
    )

    markdown = render_markdown(
        rows,
        streak_threshold=8,
        latency_snapshot=latency,
        latency_png=Path("thumb.png"),
        latency_digest=digest_path,
        latency_leaderboard=leaderboard,
    )
    assert "Data Feed Health" in markdown
    assert "yahoo" in markdown
    assert "320.00" in markdown
    assert "thumb.png" in markdown
    assert "Recent Latency Alerts" in markdown
    assert "Latency Status" in markdown
    assert "Latency Offenders Leaderboard" in markdown
