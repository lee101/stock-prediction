from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from differentiable_market import (
    DataConfig,
    DifferentiableMarketTrainer,
    EnvironmentConfig,
    EvaluationConfig,
    TrainingConfig,
)
from differentiable_market.data import load_aligned_ohlc
from differentiable_market.marketsimulator import DifferentiableMarketBacktester


def _write_synthetic_ohlc(root: Path, symbols: tuple[str, ...] = ("AAA", "BBB", "CCC"), steps: int = 64) -> None:
    rng = np.random.default_rng(1234)
    dates = pd.date_range("2022-01-01", periods=steps, freq="D")
    for symbol in symbols:
        base = 100 + rng.standard_normal(steps).cumsum()
        open_prices = base
        close = base + rng.normal(0, 0.5, steps)
        high = np.maximum(open_prices, close) + rng.uniform(0.1, 0.5, steps)
        low = np.minimum(open_prices, close) - rng.uniform(0.1, 0.5, steps)
        volume = rng.uniform(1e5, 2e5, steps)
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": open_prices,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
        df.to_csv(root / f"{symbol}.csv", index=False)


def test_load_aligned_ohlc(tmp_path: Path) -> None:
    _write_synthetic_ohlc(tmp_path)
    cfg = DataConfig(root=tmp_path, glob="*.csv")
    cfg.min_timesteps = 32
    ohlc, symbols, index = load_aligned_ohlc(cfg)
    assert ohlc.shape[-1] == 4
    assert len(symbols) == 3
    assert ohlc.shape[0] == len(index)


def test_trainer_fit_creates_checkpoints(tmp_path: Path) -> None:
    _write_synthetic_ohlc(tmp_path, steps=80)
    data_cfg = DataConfig(root=tmp_path, glob="*.csv")
    data_cfg.min_timesteps = 32
    env_cfg = EnvironmentConfig(transaction_cost=1e-4, risk_aversion=0.0)
    train_cfg = TrainingConfig(
        lookback=16,
        rollout_groups=2,
        batch_windows=4,
        microbatch_windows=2,
        epochs=3,
        eval_interval=1,
        save_dir=tmp_path / "runs",
        device="cpu",
        dtype="float32",
        use_muon=False,
        use_compile=False,
        bf16_autocast=False,
    )
    eval_cfg = EvaluationConfig(report_dir=tmp_path / "evals", store_trades=False)

    train_cfg.include_cash = True
    data_cfg.include_cash = True
    trainer = DifferentiableMarketTrainer(data_cfg, env_cfg, train_cfg, eval_cfg)
    trainer.fit()

    run_dirs = sorted((tmp_path / "runs").glob("*"))
    assert run_dirs, "Expected at least one training run directory"
    ckpt_dir = run_dirs[0] / "checkpoints"
    assert (ckpt_dir / "latest.pt").exists()
    assert (ckpt_dir / "best.pt").exists()
    metrics_path = run_dirs[0] / "metrics.jsonl"
    with metrics_path.open() as handle:
        records = [json.loads(line) for line in handle]
    assert any(rec["phase"] == "eval" for rec in records)
    train_records = [rec for rec in records if rec["phase"] == "train"]
    assert train_records, "Expected at least one train metric row"
    assert train_records[0]["microbatch"] == 2
    assert "peak_mem_gb" in train_records[0]


def test_backtester_generates_reports(tmp_path: Path) -> None:
    _write_synthetic_ohlc(tmp_path, steps=80)
    data_cfg = DataConfig(root=tmp_path, glob="*.csv")
    data_cfg.min_timesteps = 32
    env_cfg = EnvironmentConfig(transaction_cost=1e-4, risk_aversion=0.0)
    train_cfg = TrainingConfig(
        lookback=16,
        rollout_groups=2,
        batch_windows=4,
        microbatch_windows=2,
        epochs=2,
        eval_interval=1,
        save_dir=tmp_path / "runs",
        device="cpu",
        dtype="float32",
        use_muon=False,
        use_compile=False,
        bf16_autocast=False,
    )
    eval_cfg = EvaluationConfig(report_dir=tmp_path / "evals", store_trades=False, window_length=32, stride=16)

    trainer = DifferentiableMarketTrainer(data_cfg, env_cfg, train_cfg, eval_cfg)
    trainer.fit()
    run_dir = sorted((tmp_path / "runs").glob("*"))[0]
    best_ckpt = run_dir / "checkpoints" / "best.pt"
    backtester = DifferentiableMarketBacktester(data_cfg, env_cfg, eval_cfg)
    metrics = backtester.run(best_ckpt)
    report = eval_cfg.report_dir / "report.json"
    windows = eval_cfg.report_dir / "windows.json"
    assert report.exists()
    assert windows.exists()
    assert metrics["windows"] >= 1


def test_backtester_respects_include_cash(tmp_path: Path) -> None:
    _write_synthetic_ohlc(tmp_path, steps=96)
    data_cfg = DataConfig(root=tmp_path, glob="*.csv")
    data_cfg.min_timesteps = 32
    env_cfg = EnvironmentConfig(transaction_cost=1e-4, risk_aversion=0.0)
    train_cfg = TrainingConfig(
        lookback=16,
        rollout_groups=2,
        batch_windows=4,
        microbatch_windows=2,
        epochs=3,
        eval_interval=1,
        save_dir=tmp_path / "runs",
        device="cpu",
        dtype="float32",
        use_muon=False,
        use_compile=False,
        bf16_autocast=False,
        include_cash=True,
    )
    eval_cfg = EvaluationConfig(report_dir=tmp_path / "evals", store_trades=False, window_length=32, stride=16)

    trainer = DifferentiableMarketTrainer(data_cfg, env_cfg, train_cfg, eval_cfg)
    trainer.fit()

    run_dir = sorted((tmp_path / "runs").glob("*"))[0]
    best_ckpt = run_dir / "checkpoints" / "best.pt"

    backtester = DifferentiableMarketBacktester(data_cfg, env_cfg, eval_cfg)
    metrics = backtester.run(best_ckpt)

    assert metrics["windows"] >= 1
    assert backtester.eval_features.shape[1] == len(backtester.symbols) + 1


def test_backtester_trade_timestamps_use_eval_offset(tmp_path: Path) -> None:
    _write_synthetic_ohlc(tmp_path, steps=10)
    data_cfg = DataConfig(root=tmp_path, glob="*.csv")
    data_cfg.min_timesteps = 1
    env_cfg = EnvironmentConfig(transaction_cost=0.0, risk_aversion=0.0)
    eval_cfg = EvaluationConfig(report_dir=tmp_path / "evals", store_trades=True, window_length=1, stride=1)

    backtester = DifferentiableMarketBacktester(data_cfg, env_cfg, eval_cfg)
    eval_cfg.report_dir.mkdir(parents=True, exist_ok=True)
    trade_path = eval_cfg.report_dir / "trades.jsonl"

    returns = backtester.eval_returns[:1]
    weights = torch.full(
        (1, returns.shape[1]),
        1.0 / returns.shape[1],
        dtype=returns.dtype,
        device=returns.device,
    )

    with trade_path.open("w", encoding="utf-8") as handle:
        backtester._simulate_window(weights, returns, start=0, end=1, trade_handle=handle)

    records = [json.loads(line) for line in trade_path.read_text(encoding="utf-8").splitlines() if line]
    assert records, "Expected at least one logged trade"
    first_timestamp = records[0]["timestamp"]
    expected_timestamp = str(backtester.index[backtester.eval_start_idx + 1])
    assert first_timestamp == expected_timestamp


def test_trainer_supports_augmented_losses(tmp_path: Path) -> None:
    _write_synthetic_ohlc(tmp_path, steps=72)
    data_cfg = DataConfig(root=tmp_path, glob="*.csv")
    data_cfg.min_timesteps = 32
    env_cfg = EnvironmentConfig(transaction_cost=1e-4, risk_aversion=0.0)
    train_cfg = TrainingConfig(
        lookback=16,
        rollout_groups=2,
        batch_windows=4,
        microbatch_windows=2,
        epochs=2,
        eval_interval=1,
        save_dir=tmp_path / "runs",
        device="cpu",
        dtype="float32",
        use_muon=False,
        use_compile=False,
        bf16_autocast=False,
        soft_drawdown_lambda=0.1,
        risk_budget_lambda=0.05,
        risk_budget_target=(1.0, 1.0, 1.0),
        trade_memory_lambda=0.2,
        use_taylor_features=True,
        taylor_order=2,
        taylor_scale=8.0,
        use_wavelet_features=True,
        wavelet_levels=1,
    )
    eval_cfg = EvaluationConfig(report_dir=tmp_path / "evals", store_trades=False)

    trainer = DifferentiableMarketTrainer(data_cfg, env_cfg, train_cfg, eval_cfg)
    state = trainer.fit()
    assert state.step == train_cfg.epochs
    metrics = list((tmp_path / "runs").glob("*/metrics.jsonl"))
    assert metrics, "Expected metrics to be written"
    assert trainer.train_features.shape[-1] == 8
