from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_binanceneural_robustness_sweep import (
    build_common_window,
    build_start_state_maps,
    resolve_checkpoint_candidates,
)


def test_resolve_checkpoint_candidates_uses_latest_epoch_by_default(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "candidate_a"
    ckpt_dir.mkdir()
    for epoch in (1, 3, 5):
        (ckpt_dir / f"epoch_{epoch:03d}.pt").write_bytes(b"ckpt")

    candidates = resolve_checkpoint_candidates([str(ckpt_dir)])

    assert len(candidates) == 1
    assert candidates[0].checkpoint_path.name == "epoch_005.pt"


def test_resolve_checkpoint_candidates_supports_sample_epochs(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "candidate_b"
    ckpt_dir.mkdir()
    for epoch in (1, 2, 4):
        (ckpt_dir / f"epoch_{epoch:03d}.pt").write_bytes(b"ckpt")

    candidates = resolve_checkpoint_candidates([str(ckpt_dir)], sample_epochs=[1, 4])

    assert [candidate.checkpoint_path.name for candidate in candidates] == [
        "epoch_001.pt",
        "epoch_004.pt",
    ]


def test_build_start_state_maps_allocates_seeded_symbol_position() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
            "close": [200.0, 201.0, 202.0],
        }
    )

    start = build_start_state_maps(
        {"BTCUSD": frame},
        initial_cash=10_000.0,
        start_symbol="BTCUSD",
        position_fraction=0.25,
    )

    assert start["initial_cash"] == 7_500.0
    assert start["initial_cost_basis_by_symbol"] == {"BTCUSD": 200.0}
    assert start["initial_inventory_by_symbol"]["BTCUSD"] == 12.5


def test_build_common_window_restricts_to_shared_timestamps() -> None:
    ts_a = pd.date_range("2026-01-01T00:00:00Z", periods=5, freq="h", tz="UTC")
    ts_b = pd.date_range("2026-01-01T01:00:00Z", periods=4, freq="h", tz="UTC")
    merged_by_symbol = {
        "BTCUSD": pd.DataFrame({"timestamp": ts_a, "symbol": "BTCUSD", "close": range(5)}),
        "ETHUSD": pd.DataFrame({"timestamp": ts_b, "symbol": "ETHUSD", "close": range(4)}),
    }

    restricted = build_common_window(merged_by_symbol, window_hours=0)

    assert list(restricted["BTCUSD"]["timestamp"]) == list(ts_b)
    assert list(restricted["ETHUSD"]["timestamp"]) == list(ts_b)
