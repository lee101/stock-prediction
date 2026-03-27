from __future__ import annotations

from pathlib import Path

from scripts.train_crypto_lora_sweep import resolve_data_path


def test_resolve_data_path_supports_mixed_hourly_root(tmp_path: Path) -> None:
    stocks_dir = tmp_path / "stocks"
    stocks_dir.mkdir(parents=True)
    target = stocks_dir / "GOOG.csv"
    target.write_text("timestamp,open,high,low,close,volume\n")

    resolved = resolve_data_path("GOOG", tmp_path)

    assert resolved == target
