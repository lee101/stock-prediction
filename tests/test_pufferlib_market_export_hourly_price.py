from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pufferlib_market.export_data_hourly_price import compute_hourly_features, export_binary
from pufferlib_market.hourly_replay import read_mktd


def _write_symbol_csv(path: Path, index: pd.DatetimeIndex) -> None:
    base = np.linspace(100.0, 120.0, num=len(index), dtype=np.float64)
    frame = pd.DataFrame(
        {
            "timestamp": index,
            "open": base,
            "high": base * 1.01,
            "low": base * 0.99,
            "close": base * 1.002,
            "volume": np.linspace(10_000, 20_000, num=len(index), dtype=np.float64),
        }
    )
    frame.to_csv(path, index=False)


def test_compute_hourly_features_returns_16_finite_columns() -> None:
    index = pd.date_range("2026-01-01 00:00:00+00:00", periods=120, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": np.linspace(100.0, 150.0, num=len(index)),
            "high": np.linspace(101.0, 151.0, num=len(index)),
            "low": np.linspace(99.0, 149.0, num=len(index)),
            "close": np.linspace(100.5, 150.5, num=len(index)),
            "volume": np.linspace(50_000, 100_000, num=len(index)),
        },
        index=index,
    )
    feat = compute_hourly_features(frame)
    assert feat.shape == (len(index), 16)
    assert np.isfinite(feat.to_numpy()).all()


def test_export_binary_writes_v2_and_tradable_mask(tmp_path: Path) -> None:
    root = tmp_path / "data" / "crypto"
    root.mkdir(parents=True)
    idx = pd.date_range("2026-01-01 00:00:00+00:00", periods=72, freq="h", tz="UTC")

    _write_symbol_csv(root / "BTCUSD.csv", idx)

    # ETHUSD intentionally misses one bar so tradable mask should contain a 0.
    eth_idx = idx.delete(10)
    _write_symbol_csv(root / "ETHUSD.csv", eth_idx)

    out = tmp_path / "export.bin"
    report = export_binary(
        symbols=["BTCUSD", "ETHUSD"],
        data_root=tmp_path / "data",
        output_path=out,
        min_hours=24,
        min_coverage=0.95,
    )
    assert report["num_symbols"] == 2
    assert report["num_timesteps"] == 72

    parsed = read_mktd(out)
    assert parsed.version == 2
    assert parsed.symbols == ["BTCUSD", "ETHUSD"]
    assert parsed.features.shape == (72, 2, 16)
    assert parsed.prices.shape == (72, 2, 5)
    assert parsed.tradable is not None
    assert parsed.tradable.shape == (72, 2)
    assert parsed.tradable[10, 1] == 0
    assert parsed.tradable[:, 0].min() == 1


def test_export_binary_applies_min_coverage_filter(tmp_path: Path) -> None:
    root = tmp_path / "data" / "crypto"
    root.mkdir(parents=True)
    idx = pd.date_range("2026-01-01 00:00:00+00:00", periods=72, freq="h", tz="UTC")

    _write_symbol_csv(root / "BTCUSD.csv", idx)

    sparse_idx = idx[::8]
    _write_symbol_csv(root / "ETHUSD.csv", sparse_idx)

    out = tmp_path / "coverage.bin"
    report = export_binary(
        symbols=["BTCUSD", "ETHUSD"],
        data_root=tmp_path / "data",
        output_path=out,
        min_hours=24,
        min_coverage=0.9,
    )
    assert report["num_symbols"] == 1
    assert report["symbols"] == ["BTCUSD"]

    parsed = read_mktd(out)
    assert parsed.symbols == ["BTCUSD"]
    assert parsed.features.shape[1] == 1
