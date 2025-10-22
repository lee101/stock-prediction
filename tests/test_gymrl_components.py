import csv
from datetime import datetime, timedelta

import numpy as np
import pytest

from gymrl.cache_utils import load_feature_cache, save_feature_cache
from gymrl.config import FeatureBuilderConfig, PortfolioEnvConfig
from gymrl.feature_pipeline import FeatureBuilder
from gymrl.portfolio_env import PortfolioEnv
from loss_utils import CRYPTO_TRADING_FEE, TRADING_FEE


def _write_daily_csv(path, start_price=100.0, drift=0.01):
    start_time = datetime(2024, 1, 1)
    price = start_price
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for day in range(40):
            timestamp = start_time + timedelta(days=day)
            open_price = price
            close_price = price * (1.0 + drift * 0.1)
            high_price = max(open_price, close_price) * 1.01
            low_price = min(open_price, close_price) * 0.99
            volume = 1_000_000 + 1000 * day
            writer.writerow([
                timestamp.isoformat(),
                f"{open_price:.4f}",
                f"{high_price:.4f}",
                f"{low_price:.4f}",
                f"{close_price:.4f}",
                volume,
            ])
            price = close_price


def test_feature_builder_bootstrap_daily(tmp_path):
    data_dir = tmp_path / "daily"
    data_dir.mkdir()
    _write_daily_csv(data_dir / "AAPL.csv", start_price=150.0, drift=0.02)
    _write_daily_csv(data_dir / "BTCUSD.csv", start_price=30000.0, drift=0.05)

    config = FeatureBuilderConfig(
        forecast_backend="bootstrap",
        context_window=8,
        min_history=8,
        num_samples=64,
        realized_horizon=1,
        prediction_length=1,
        enforce_common_index=False,
        fill_method="ffill",
    )

    builder = FeatureBuilder(config=config)
    cube = builder.build_from_directory(data_dir)

    assert cube.features.shape[0] > 0
    assert cube.features.shape[1] == 2
    assert "forecast_mu" in cube.feature_names
    assert "forecast_sigma" in cube.feature_names
    # Ensure realized returns are not accidentally replaced by forecast means
    fidx = cube.feature_names.index("forecast_mean_return")
    assert not np.allclose(cube.realized_returns[:, 0], cube.features[:, 0, fidx])
    assert len(cube.timestamps) == cube.features.shape[0]


def test_portfolio_env_cost_vector_handles_crypto_and_cash():
    T, N, F = 12, 2, 4
    features = np.zeros((T, N, F), dtype=np.float32)
    realized_returns = np.zeros((T, N), dtype=np.float32)
    config = PortfolioEnvConfig(costs_bps=5.0, include_cash=True, leverage_head=False, weight_cap=None)

    env = PortfolioEnv(
        features,
        realized_returns,
        config=config,
        symbols=["AAPL", "BTCUSD"],
    )

    assert env.costs_vector.shape[0] == 3  # includes cash asset
    expected_stock_cost = TRADING_FEE + (config.costs_bps / 1e4)
    expected_crypto_cost = CRYPTO_TRADING_FEE + (config.costs_bps / 1e4)

    assert env.costs_vector[0] == pytest.approx(expected_stock_cost, rel=1e-4)
    assert env.costs_vector[1] == pytest.approx(expected_crypto_cost, rel=1e-4)
    assert env.costs_vector[2] == pytest.approx(0.0, abs=1e-6)


def test_feature_cache_round_trip(tmp_path):
    data_dir = tmp_path / "daily"
    data_dir.mkdir()
    _write_daily_csv(data_dir / "AAPL.csv", start_price=120.0)
    _write_daily_csv(data_dir / "MSFT.csv", start_price=310.0)

    config = FeatureBuilderConfig(
        forecast_backend="bootstrap",
        context_window=8,
        min_history=8,
        num_samples=32,
        realized_horizon=1,
        prediction_length=1,
    )

    builder = FeatureBuilder(config=config)
    cube = builder.build_from_directory(data_dir)

    cache_path = tmp_path / "features.npz"
    save_feature_cache(cache_path, cube, extra_metadata={"note": "unit_test"})
    loaded_cube, meta = load_feature_cache(cache_path)

    assert loaded_cube.features.shape == cube.features.shape
    assert loaded_cube.realized_returns.shape == cube.realized_returns.shape
    assert loaded_cube.feature_names == cube.feature_names
    assert meta.get("note") == "unit_test"


def test_portfolio_env_info_crypto_breakdown():
    T, N, F = 5, 2, 3
    features = np.zeros((T, N, F), dtype=np.float32)
    realized_returns = np.zeros((T, N), dtype=np.float32)
    realized_returns[:, 0] = 0.01
    realized_returns[:, 1] = 0.05

    env = PortfolioEnv(
        features,
        realized_returns,
        config=PortfolioEnvConfig(include_cash=False, leverage_head=False, weight_cap=None),
        symbols=["AAPL", "BTCUSD"],
    )

    obs, _ = env.reset()
    assert obs.shape[0] == env.observation_space.shape[0]
    action = np.zeros(env.action_space.shape)
    _, _, terminated, _, info = env.step(action)
    assert not terminated
    assert "step_return_crypto" in info
    assert "step_return_non_crypto" in info
    assert "net_return_crypto" in info
    assert "weight_crypto" in info
    assert info["weight_crypto"] == pytest.approx(0.5, rel=1e-3)
    assert info["weight_non_crypto"] == pytest.approx(0.5, rel=1e-3)
    assert info["step_return_crypto"] >= 0.0
    assert info["step_return_non_crypto"] >= 0.0
    assert info["loss_shutdown_penalty"] == pytest.approx(0.0)
    assert info["loss_shutdown_active_long"] == pytest.approx(0.0)
    assert info["loss_shutdown_active_short"] == pytest.approx(0.0)
    assert info["loss_shutdown_clipped"] == pytest.approx(0.0)
    assert info["interest_cost"] == pytest.approx(0.0)
    assert info["gross_exposure_intraday"] == pytest.approx(1.0)
    assert info["gross_exposure_close"] == pytest.approx(1.0)
    assert info["closing_turnover"] == pytest.approx(0.0)
    assert info["closing_trading_cost"] == pytest.approx(0.0)


def test_portfolio_leverage_closing_interest(tmp_path):
    T, N, F = 3, 2, 1
    features = np.zeros((T, N, F), dtype=np.float32)
    realized_returns = np.zeros((T, N), dtype=np.float32)
    realized_returns[:, 0] = 0.01
    config = PortfolioEnvConfig(
        include_cash=False,
        intraday_leverage_cap=4.0,
        closing_leverage_cap=2.0,
        leverage_interest_rate=0.0675,
        trading_days_per_year=252,
        weight_cap=None,
    )

    env = PortfolioEnv(features, realized_returns, config=config, symbols=["AAPL", "MSFT"])
    env.reset()

    _, _, _, _, info = env.step_with_weights(np.array([3.0, 1.0], dtype=np.float32))

    assert info["gross_exposure_intraday"] == pytest.approx(4.0, rel=1e-6)
    assert info["gross_exposure_close"] == pytest.approx(2.0, rel=1e-6)
    assert info["closing_turnover"] == pytest.approx(2.0, rel=1e-6)
    expected_cost = (4.0 + 2.0) * (TRADING_FEE + (config.costs_bps / 1e4))
    assert info["trading_cost"] == pytest.approx(expected_cost, rel=1e-6)
    assert info["closing_trading_cost"] == pytest.approx(2.0 * (TRADING_FEE + (config.costs_bps / 1e4)), rel=1e-6)
    assert info["turnover"] == pytest.approx(6.0, rel=1e-6)
    daily_rate = (1.0 + config.leverage_interest_rate) ** (1.0 / config.trading_days_per_year) - 1.0
    assert info["interest_cost"] == pytest.approx(daily_rate, rel=1e-6)
    assert env.current_weights.sum() == pytest.approx(2.0, rel=1e-6)
