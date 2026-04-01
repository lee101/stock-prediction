from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from FastForecaster.config import FastForecasterConfig
from FastForecaster.trainer import FastForecasterTrainer


def _write_symbol_csv(path: Path, symbol: str, periods: int = 420) -> None:
    idx = pd.date_range("2024-01-01", periods=periods, freq="h", tz="UTC")
    trend = np.linspace(50.0, 80.0, periods)
    seasonal = 0.8 * np.sin(np.arange(periods) / 12.0)
    noise = 0.1 * np.cos(np.arange(periods) / 7.0)
    close = trend + seasonal + noise
    open_ = close - 0.2
    high = close + 0.4
    low = close - 0.5
    volume = 20000 + 1000 * np.sin(np.arange(periods) / 17.0)
    vwap = close + 0.05

    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "vwap": vwap,
            "symbol": symbol,
        }
    )
    df.to_csv(path, index=False)


def test_trainer_smoke_run(tmp_path: Path):
    _write_symbol_csv(tmp_path / "NVDA.csv", "NVDA")
    _write_symbol_csv(tmp_path / "GOOG.csv", "GOOG")

    cfg = FastForecasterConfig(
        data_dir=tmp_path,
        output_dir=tmp_path / "out",
        symbols=("NVDA", "GOOG"),
        max_symbols=0,
        lookback=48,
        horizon=6,
        min_rows_per_symbol=128,
        max_train_windows_per_symbol=256,
        max_eval_windows_per_symbol=64,
        batch_size=16,
        epochs=1,
        num_workers=0,
        torch_compile=False,
        precision="fp32",
        use_cpp_kernels=False,
        log_interval=1000,
        pin_memory=False,
    )

    trainer = FastForecasterTrainer(cfg)
    summary = trainer.train()

    assert summary["best_val_mae"] >= 0.0
    assert summary["test_mae"] >= 0.0
    assert (cfg.output_dir / "checkpoints" / "best.pt").exists()
    assert (cfg.output_dir / "metrics" / "summary.json").exists()
    assert (cfg.output_dir / "metrics" / "epoch_metrics.json").exists()
    assert (cfg.output_dir / "metrics" / "test_per_symbol.json").exists()
    assert isinstance(summary["epoch_history"], list)
    assert len(summary["epoch_history"]) >= 1


def test_trainer_falls_back_to_cpu_when_auto_cuda_is_unavailable(tmp_path: Path, monkeypatch):
    calls: list[str] = []

    class DummyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def to(self, device, *args, **kwargs):
            target = torch.device(device)
            calls.append(target.type)
            if target.type == "cuda":
                accelerator_error = getattr(torch, "AcceleratorError", RuntimeError)
                raise accelerator_error("CUDA error: out of memory")
            return super().to(target, *args, **kwargs)

    sample = {
        "x": torch.zeros(2, 4),
        "target_ret": torch.zeros(2, 2),
        "target_close": torch.zeros(2, 2),
        "base_close": torch.zeros(2),
        "symbol_idx": torch.zeros(2, dtype=torch.long),
    }
    bundle = SimpleNamespace(
        feature_dim=4,
        symbols=("NVDA",),
        train_dataset=[sample],
        val_dataset=[sample],
        test_dataset=[sample],
    )

    monkeypatch.setattr("FastForecaster.trainer.build_data_bundle", lambda config: bundle)
    monkeypatch.setattr("FastForecaster.trainer.FastForecasterModel", DummyModel)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    cfg = FastForecasterConfig(
        data_dir=tmp_path,
        output_dir=tmp_path / "out",
        symbols=("NVDA",),
        max_symbols=0,
        lookback=48,
        horizon=2,
        min_rows_per_symbol=128,
        max_train_windows_per_symbol=4,
        max_eval_windows_per_symbol=2,
        batch_size=1,
        epochs=1,
        num_workers=0,
        torch_compile=False,
        precision="fp32",
        use_cpp_kernels=False,
        log_interval=1000,
        pin_memory=True,
    )

    trainer = FastForecasterTrainer(cfg)

    assert trainer.device.type == "cpu"
    assert trainer.train_loader.pin_memory is False
    assert trainer.optimizer.defaults.get("fused") in (None, False)
    assert calls == ["cuda", "cpu"]
