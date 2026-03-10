from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.frontiermarketsim import (
    FrontierMarketSimulator,
    FrontierSimConfig,
    SymbolDataset,
    align_symbol_lengths,
    load_symbol_datasets,
)


def _synthetic_ohlcv(length: int, *, base: float) -> torch.Tensor:
    idx = torch.arange(length, dtype=torch.float32)
    opens = base + idx * 0.1
    highs = opens + 0.3
    lows = opens - 0.3
    closes = opens + 0.05 * torch.sin(idx / 4.0)
    volume = torch.full_like(opens, 1_000_000.0)
    return torch.stack([opens, highs, lows, closes, volume], dim=-1).contiguous()


def test_align_symbol_lengths_uses_minimum_tail():
    data_a = SymbolDataset("AAPL", _synthetic_ohlcv(16, base=100.0), False)
    data_b = SymbolDataset("BTCUSD", _synthetic_ohlcv(10, base=30000.0), True)
    aligned = align_symbol_lengths([data_a, data_b])

    assert len(aligned) == 2
    assert aligned[0].ohlcv.shape == (10, 5)
    assert aligned[1].ohlcv.shape == (10, 5)
    torch.testing.assert_close(aligned[0].ohlcv[-1], data_a.ohlcv[-1])
    torch.testing.assert_close(aligned[1].ohlcv[-1], data_b.ohlcv[-1])


def test_torch_backend_step_clamps_crypto_shorting():
    datasets = [
        SymbolDataset("AAPL", _synthetic_ohlcv(64, base=100.0), False),
        SymbolDataset("BTCUSD", _synthetic_ohlcv(64, base=25000.0), True),
    ]
    sim = FrontierMarketSimulator(
        datasets,
        num_envs=6,
        cfg=FrontierSimConfig(context_len=8, horizon=1, device="cpu"),
        use_fast_backend=False,
    )
    obs = sim.reset()
    assert obs.shape == (6, 8, 5)

    actions = torch.full((6,), -10.0, dtype=torch.float32)
    result = sim.step(actions)
    assert result["reward"].shape == (6,)
    assert result["done"].shape == (6,)
    assert result["obs"].shape == (6, 8, 5)

    crypto_mask = sim.is_crypto.detach().cpu().numpy().astype(bool)
    positions = result["position"].detach().cpu().numpy()
    assert np.all(positions[crypto_mask] >= -1e-6)


def test_run_benchmark_reports_metrics():
    datasets = [
        SymbolDataset("AAPL", _synthetic_ohlcv(80, base=100.0), False),
        SymbolDataset("MSFT", _synthetic_ohlcv(80, base=200.0), False),
        SymbolDataset("ETHUSD", _synthetic_ohlcv(80, base=2500.0), True),
    ]
    sim = FrontierMarketSimulator(
        datasets,
        num_envs=12,
        cfg=FrontierSimConfig(context_len=12, horizon=1, device="cpu"),
        use_fast_backend=False,
    )
    metrics = sim.run_benchmark(num_steps=20)

    assert metrics["backend"] == "torch"
    assert metrics["num_envs"] == 12
    assert metrics["steps_executed"] > 0
    assert metrics["env_steps_per_sec"] > 0.0
    assert metrics["final_equity_p95"] >= metrics["final_equity_p05"]


def test_load_symbol_datasets_reads_valid_csvs(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir(parents=True)

    dates = pd.date_range("2024-01-01", periods=24, freq="D")
    for symbol, base in (("AAPL", 100.0), ("BTCUSD", 30000.0)):
        frame = pd.DataFrame(
            {
                "date": dates,
                "open": base + np.arange(len(dates), dtype=np.float32),
                "high": base + np.arange(len(dates), dtype=np.float32) + 1.0,
                "low": base + np.arange(len(dates), dtype=np.float32) - 1.0,
                "close": base + np.arange(len(dates), dtype=np.float32) + 0.25,
                "volume": np.full(len(dates), 1000.0, dtype=np.float32),
            }
        )
        frame.to_csv(root / f"{symbol}.csv", index=False)

    loaded = load_symbol_datasets(root, max_symbols=2, min_rows=12)
    assert len(loaded) == 2
    loaded_symbols = {item.symbol for item in loaded}
    assert {"AAPL", "BTCUSD"} <= loaded_symbols
    for item in loaded:
        assert item.ohlcv.shape[1] == 5
        assert item.ohlcv.shape[0] >= 12
