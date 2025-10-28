from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from pufferlibtraining2.config import DataConfig
from pufferlibtraining2.data.loader import load_asset_frames


def _make_frame(days: int = 64) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    base = np.linspace(100, 120, days, dtype=np.float32)
    return pd.DataFrame(
        {
            "date": dates,
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.5,
            "volume": np.full(days, 1_000_000, dtype=np.float32),
        }
    )


def test_load_asset_frames(tmp_path: Path) -> None:
    for symbol in ("AAPL", "MSFT"):
        frame = _make_frame()
        frame.to_csv(tmp_path / f"{symbol}.csv", index=False)

    cfg = DataConfig(data_dir=tmp_path, symbols=("AAPL", "MSFT"), window_size=8, min_history=32)
    frames = load_asset_frames(cfg)
    assert set(frames.keys()) == {"AAPL", "MSFT"}
    for df in frames.values():
        assert len(df) >= 32
        assert df["date"].is_monotonic_increasing
