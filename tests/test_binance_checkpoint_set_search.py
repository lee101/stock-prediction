from __future__ import annotations

from pathlib import Path

import pytest

from binanceexp1.search_checkpoint_sets_robust import (
    _load_candidates,
    _parse_candidate_spec,
    _resolve_checkpoint_path,
)


def test_parse_candidate_spec_requires_all_symbols() -> None:
    with pytest.raises(ValueError, match="Missing checkpoint candidates"):
        _parse_candidate_spec("BTCUSD=a.pt|b.pt;ETHUSD=c.pt", ["BTCUSD", "ETHUSD", "SOLUSD"])


def test_resolve_checkpoint_path_accepts_latest_epoch_directory(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "btc"
    ckpt_dir.mkdir()
    (ckpt_dir / "epoch_001.pt").write_text("x")
    (ckpt_dir / "epoch_003.pt").write_text("x")
    (ckpt_dir / "epoch_002.pt").write_text("x")

    resolved = _resolve_checkpoint_path(str(ckpt_dir))

    assert resolved.name == "epoch_003.pt"


def test_load_candidates_deduplicates_resolved_paths(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "btc"
    ckpt_dir.mkdir()
    ckpt = ckpt_dir / "epoch_001.pt"
    ckpt.write_text("x")

    raw = f"BTCUSD={ckpt}|{ckpt_dir};ETHUSD={ckpt};SOLUSD={ckpt}"
    loaded = _load_candidates(raw, ["BTCUSD", "ETHUSD", "SOLUSD"])

    assert len(loaded["BTCUSD"]) == 1
    assert loaded["BTCUSD"][0].checkpoint.endswith("epoch_001.pt")
