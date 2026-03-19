from __future__ import annotations

import struct
from pathlib import Path

from unified_orchestrator import orchestrator


def _write_mktd_header(path: Path, symbols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack(
        "<4sIIIII40s",
        b"MKTD",
        2,
        len(symbols),
        1,
        16,
        5,
        b"\x00" * 40,
    )
    with path.open("wb") as handle:
        handle.write(header)
        for sym in symbols:
            handle.write(sym.encode("ascii").ljust(16, b"\x00"))


def test_read_trained_symbols_uses_hint_for_stocks13(monkeypatch, tmp_path: Path) -> None:
    data_path = tmp_path / "stocks13.bin"
    _write_mktd_header(data_path, ["AAPL", "MSFT", "NVDA"])

    checkpoint = tmp_path / "checkpoints" / "stocks13_custom_run7" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"")

    monkeypatch.setitem(orchestrator._CHECKPOINT_DATA_HINTS, checkpoint.parent.name, data_path)

    symbols = orchestrator._read_trained_symbols_for_checkpoint(checkpoint, expected_num_symbols=3)

    assert symbols == ["AAPL", "MSFT", "NVDA"]


def test_checkpoint_data_path_uses_explicit_hint(monkeypatch, tmp_path: Path) -> None:
    explicit_path = tmp_path / "explicit.bin"
    _write_mktd_header(explicit_path, ["AAPL"])

    checkpoint = tmp_path / "checkpoints" / "stocks13_featlag1_fee5bps_longonly_run4" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"")

    monkeypatch.setitem(orchestrator._CHECKPOINT_DATA_HINTS, checkpoint.parent.name, explicit_path)

    assert orchestrator._checkpoint_data_path(checkpoint) == explicit_path
