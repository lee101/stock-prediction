from __future__ import annotations

from pathlib import Path

from scripts.provider_usage_sparkline import default_token_map, render_markdown


def write_log(tmp_path: Path, entries: list[tuple[str, str, int]]) -> Path:
    log = tmp_path / "provider_usage.csv"
    with log.open("w", encoding="utf-8") as handle:
        handle.write("timestamp,provider,count\n")
        for timestamp, provider, count in entries:
            handle.write(f"{timestamp},{provider},{count}\n")
    return log


def test_render_markdown_outputs_table(tmp_path):
    log = write_log(
        tmp_path,
        [
            ("2025-10-23T00:00:00+00:00", "stooq", 16),
            ("2025-10-24T00:00:00+00:00", "yahoo", 16),
        ],
    )
    markdown = render_markdown(log, window=2, token_map=default_token_map())
    assert "Sparkline" in markdown
    assert "🟥🟦" in markdown
    assert "Legend:" in markdown


def test_render_markdown_handles_empty(tmp_path):
    log = write_log(tmp_path, [])
    markdown = render_markdown(log, window=5, token_map=default_token_map())
    assert "No provider usage data" in markdown
