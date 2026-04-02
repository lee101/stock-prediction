from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import build_stock_forecast_cache_batches as batch_mod


def test_load_symbols_ignores_comments_and_deduplicates(tmp_path: Path) -> None:
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("aapl, msft\n# skip\nAAPL\n nvda \n", encoding="utf-8")

    assert batch_mod._load_symbols(symbols_file) == ["AAPL", "MSFT", "NVDA"]


def test_main_writes_requested_batch_range_and_prints_commands(tmp_path: Path, capsys, monkeypatch) -> None:
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("AAPL\nMSFT\nNVDA\nAMZN\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    def _unexpected_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called without --execute")

    monkeypatch.setattr(batch_mod.subprocess, "run", _unexpected_run)

    result = batch_mod.main(
        [
            "--symbols-file",
            str(symbols_file),
            "--batch-size",
            "2",
            "--start-batch",
            "2",
            "--end-batch",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    batch_file = output_dir / "batch_002_symbols.txt"
    summary_file = output_dir / "batch_002_mae.json"
    assert batch_file.read_text(encoding="utf-8") == "NVDA\nAMZN\n"

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines[0] == "[batch 2/2] 2 symbols"
    assert str(batch_file) in lines[1]
    assert str(summary_file) in lines[1]


def test_main_execute_runs_subprocess_for_each_selected_batch(tmp_path: Path, monkeypatch) -> None:
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("AAPL\nMSFT\nNVDA\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    calls: list[list[str]] = []

    def _fake_run(cmd, check=False):
        assert check is True
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(batch_mod.subprocess, "run", _fake_run)

    result = batch_mod.main(
        [
            "--symbols-file",
            str(symbols_file),
            "--batch-size",
            "2",
            "--output-dir",
            str(output_dir),
            "--execute",
            "--force-rebuild",
        ]
    )

    assert result == 0
    assert len(calls) == 2
    assert "--force-rebuild" in calls[0]
    assert calls[0][0] == batch_mod.sys.executable
    assert calls[0][1].endswith("scripts/build_hourly_forecast_caches.py")
    assert calls[1][calls[1].index("--symbols-file") + 1].endswith("batch_002_symbols.txt")


def test_main_rejects_invalid_batch_range(tmp_path: Path) -> None:
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("AAPL\nMSFT\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="Invalid batch range"):
        batch_mod.main(
            [
                "--symbols-file",
                str(symbols_file),
                "--batch-size",
                "2",
                "--start-batch",
                "3",
            ]
        )
