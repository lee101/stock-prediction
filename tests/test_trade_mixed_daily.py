from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

from trade_mixed_daily import _resolve_symbols


def _write_mktd(path: Path, symbols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_symbols = len(symbols)
    num_timesteps = 2
    features_per_sym = 16
    price_features = 5

    header = struct.pack(
        "<4sIIIII40s",
        b"MKTD",
        2,
        num_symbols,
        num_timesteps,
        features_per_sym,
        price_features,
        b"\x00" * 40,
    )
    symbol_table = b"".join(sym.encode("ascii").ljust(16, b"\x00") for sym in symbols)
    features = np.zeros((num_timesteps, num_symbols, features_per_sym), dtype=np.float32)
    prices = np.zeros((num_timesteps, num_symbols, price_features), dtype=np.float32)
    mask = np.ones((num_timesteps, num_symbols), dtype=np.uint8)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(symbol_table)
        handle.write(features.tobytes())
        handle.write(prices.tobytes())
        handle.write(mask.tobytes())


def test_resolve_symbols_prefers_matching_local_mktd(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "pufferlib_market" / "checkpoints" / "mixed23_fresh_replay" / "ent_anneal" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {"encoder.0.weight": torch.zeros((8, 39))}}, checkpoint)

    data_dir = tmp_path / "pufferlib_market" / "data"
    _write_mktd(data_dir / "legacy23_val.bin", ["OLDA", "OLDB"])
    _write_mktd(data_dir / "mixed23_fresh_val.bin", ["AAPL", "BTCUSD"])

    monkeypatch.setattr("trade_mixed_daily.REPO", tmp_path)

    args = argparse.Namespace(checkpoint=str(checkpoint), symbols=None)

    assert _resolve_symbols(args) == ["AAPL", "BTCUSD"]
