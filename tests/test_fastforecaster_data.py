from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from FastForecaster.config import FastForecasterConfig
from FastForecaster.data import build_data_bundle


def _write_symbol_csv(path: Path, symbol: str, periods: int = 360, *, include_vwap: bool = True) -> None:
    idx = pd.date_range("2024-01-01", periods=periods, freq="h", tz="UTC")
    base = np.linspace(100.0, 130.0, periods) + np.sin(np.arange(periods) / 5.0)
    close = base + 0.3 * np.sin(np.arange(periods) / 3.0)
    open_ = close - 0.2
    high = close + 0.5
    low = close - 0.6
    volume = 10_000 + 500 * np.sin(np.arange(periods) / 11.0)
    vwap = close + 0.1

    payload = {
        "timestamp": idx,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "symbol": symbol,
    }
    if include_vwap:
        payload["vwap"] = vwap
    df = pd.DataFrame(payload)
    df.to_csv(path, index=False)


def test_build_data_bundle_produces_train_val_test(tmp_path: Path):
    _write_symbol_csv(tmp_path / "NVDA.csv", "NVDA")
    _write_symbol_csv(tmp_path / "GOOG.csv", "GOOG")

    cfg = FastForecasterConfig(
        data_dir=tmp_path,
        symbols=("NVDA", "GOOG"),
        max_symbols=0,
        lookback=32,
        horizon=4,
        min_rows_per_symbol=120,
        batch_size=8,
        epochs=1,
        num_workers=0,
        torch_compile=False,
        precision="fp32",
    )

    bundle = build_data_bundle(cfg)
    assert bundle.feature_dim > 4
    assert len(bundle.symbols) == 2
    assert len(bundle.train_dataset) > 0
    assert len(bundle.val_dataset) > 0
    assert len(bundle.test_dataset) > 0

    x, target_ret, target_close, base_close, symbol_idx = bundle.train_dataset[0]
    assert x.shape == (cfg.lookback, bundle.feature_dim)
    assert target_ret.shape == (cfg.horizon,)
    assert target_close.shape == (cfg.horizon,)
    assert torch.isfinite(x).all()
    assert torch.isfinite(target_ret).all()
    assert torch.isfinite(target_close).all()
    assert isinstance(float(base_close), float)
    assert int(symbol_idx) in {0, 1}


def test_build_data_bundle_skips_invalid_symbol_filenames(tmp_path: Path):
    _write_symbol_csv(tmp_path / "NVDA.csv", "NVDA")
    _write_symbol_csv(tmp_path / "AAPL,MSFT.csv", "AAPL")

    cfg = FastForecasterConfig(
        data_dir=tmp_path,
        max_symbols=0,
        lookback=32,
        horizon=4,
        min_rows_per_symbol=120,
        batch_size=8,
        epochs=1,
        num_workers=0,
        torch_compile=False,
        precision="fp32",
    )

    bundle = build_data_bundle(cfg)
    assert bundle.symbols == ("NVDA",)


def test_build_data_bundle_without_vwap_column(tmp_path: Path):
    _write_symbol_csv(tmp_path / "NVDA.csv", "NVDA", include_vwap=False)

    cfg = FastForecasterConfig(
        data_dir=tmp_path,
        max_symbols=0,
        lookback=32,
        horizon=4,
        min_rows_per_symbol=120,
        batch_size=8,
        epochs=1,
        num_workers=0,
        torch_compile=False,
        precision="fp32",
    )

    bundle = build_data_bundle(cfg)
    assert bundle.symbols == ("NVDA",)
    x, *_ = bundle.train_dataset[0]
    assert torch.isfinite(x).all()
