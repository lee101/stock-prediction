from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("stable_baselines3")

from training import run_fastppo


def _write_ohlc(path: Path, symbol: str, close: np.ndarray) -> None:
    ts = pd.date_range("2026-02-01", periods=len(close), freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": symbol,
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 1_000.0,
        }
    )
    frame.to_csv(path, index=False)


def test_select_correlated_symbols_prefers_high_corr_peer(tmp_path: Path) -> None:
    base = np.linspace(100.0, 140.0, 128, dtype=np.float64)
    _write_ohlc(tmp_path / "AAPL.csv", "AAPL", base)
    _write_ohlc(tmp_path / "MSFT.csv", "MSFT", base * 1.01)
    _write_ohlc(tmp_path / "TSLA.csv", "TSLA", np.cos(np.linspace(0.0, 8.0, 128)))

    cfg = run_fastppo.TrainingConfig(
        symbol="AAPL",
        data_root=str(tmp_path),
        auto_correlated_count=1,
        correlation_lookback=96,
        correlation_min_abs=0.2,
    )

    peers = run_fastppo._select_correlated_symbols(cfg, Path(cfg.data_root))
    assert peers
    assert peers[0] == "MSFT"


def test_load_price_tensor_merges_forecast_and_peer_features(tmp_path: Path) -> None:
    close = np.linspace(100.0, 110.0, 64, dtype=np.float64)
    _write_ohlc(tmp_path / "AAPL.csv", "AAPL", close)
    _write_ohlc(tmp_path / "MSFT.csv", "MSFT", close * 1.02)

    ts = pd.date_range("2026-02-01", periods=64, freq="h", tz="UTC")
    forecast = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": "AAPL",
            "predicted_close_p50": close + 0.3,
            "predicted_high_p50": close + 0.8,
            "predicted_low_p50": close - 0.6,
            "predicted_close_p10": close - 0.4,
            "predicted_close_p90": close + 0.7,
        }
    )
    cache_dir = tmp_path / "cache" / "h1"
    cache_dir.mkdir(parents=True)
    forecast.to_parquet(cache_dir / "AAPL.parquet", index=False)

    cfg = run_fastppo.TrainingConfig(
        symbol="AAPL",
        data_root=str(tmp_path),
        forecast_cache_root=str(tmp_path / "cache"),
        forecast_horizons=(1,),
        correlated_symbols=("MSFT",),
    )
    prices, columns = run_fastppo._load_price_tensor(cfg)

    assert prices.shape[0] == 64
    assert "chronos_ret_h1" in columns
    assert "chronos_spread_h1" in columns
    assert "peer_msft_ret1" in columns
    assert "peer_msft_ret6" in columns


def test_validation_snapshots_summary_is_json_serializable(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyVecEnv:
        def __init__(self, env_fns):
            self.env_fns = env_fns

        def close(self) -> None:
            return None

    class _DummyPPO:
        def __init__(self, *_args, **_kwargs):
            self.logger = SimpleNamespace(
                name_to_value={
                    "train/loss": 0.01,
                    "train/entropy_loss": 0.02,
                    "train/value_loss": 0.03,
                    "train/policy_gradient_loss": 0.04,
                    "train/approx_kl": 0.001,
                    "train/clip_fraction": 0.1,
                    "train/explained_variance": 0.5,
                }
            )

        def learn(self, *_, **__):
            return self

    monkeypatch.setattr(run_fastppo, "DummyVecEnv", _DummyVecEnv)
    monkeypatch.setattr(run_fastppo, "PPO", _DummyPPO)
    monkeypatch.setattr(
        run_fastppo,
        "_load_price_tensor",
        lambda _cfg: (
            np.ones((256, 4), dtype=np.float32),
            ["open", "high", "low", "close"],
        ),
    )
    monkeypatch.setattr(
        run_fastppo,
        "_evaluate_policy",
        lambda *_args, **_kwargs: {
            "total_reward": 1.0,
            "gross_pnl": 2.0,
            "reward_trace": [0.1, -0.1, 0.2],
            "gross_trace": [0.0, 0.05, 0.02],
            "equity_trace": [1.0, 1.05, 1.02],
            "reward_stats": {"mean": 0.0666, "stdev": 0.1527, "sma": 0.0, "ema": 0.0},
        },
    )

    cfg = run_fastppo.TrainingConfig(
        symbol="AAPL",
        data_root="unused",
        context_len=32,
        total_timesteps=128,
        validation_interval_timesteps=64,
        evaluate=True,
    )
    _model, metrics = run_fastppo.run_training(cfg)
    summary = {
        **metrics,
        "symbol": cfg.symbol.upper(),
        "total_timesteps": cfg.total_timesteps,
        "learning_rate": cfg.learning_rate,
        "gamma": cfg.gamma,
        "context_len": cfg.context_len,
        "horizon": cfg.horizon,
        "reward_stats": metrics.get("reward_stats", {}),
        "evaluation_skipped": not cfg.evaluate,
        "forecast_horizons": list(cfg.forecast_horizons),
        "correlated_symbols": list(cfg.correlated_symbols),
        "auto_correlated_count": cfg.auto_correlated_count,
        "validation_interval_timesteps": cfg.validation_interval_timesteps,
    }
    payload = json.dumps(summary, default=run_fastppo._json_default)
    assert payload
