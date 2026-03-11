from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


RL_BINANCE_DIR = Path(__file__).resolve().parents[1] / "rl-trainingbinance"


def _load_module(name: str, relative_path: str):
    if str(RL_BINANCE_DIR) not in sys.path:
        sys.path.insert(0, str(RL_BINANCE_DIR))
    module_path = RL_BINANCE_DIR / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


presets = _load_module("rl_trainingbinance_presets", "presets.py")
data_mod = _load_module("rl_trainingbinance_data", "data.py")
env_mod = _load_module("rl_trainingbinance_env", "env.py")
model_mod = _load_module("rl_trainingbinance_model", "model.py")
validate_mod = _load_module("rl_trainingbinance_validate", "validate.py")
sweep_mod = _load_module("rl_trainingbinance_sweep", "sweep.py")


def _write_hourly_csv(path: Path, *, symbol: str, start: str, periods: int, price_start: float) -> None:
    index = pd.date_range(start=start, periods=periods, freq="h", tz="UTC")
    base = np.linspace(price_start, price_start * 1.05, periods, dtype=np.float64)
    frame = pd.DataFrame(
        {
            "timestamp": index,
            "open": base,
            "high": base * 1.01,
            "low": base * 0.99,
            "close": base * (1.0 + 0.001 * np.sin(np.arange(periods))),
            "volume": np.linspace(10.0, 20.0, periods),
            "symbol": symbol,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_build_symbol_metadata_marks_free_fee_and_shortable() -> None:
    metadata = presets.build_symbol_metadata(
        ["BTCFDUSD", "DOGEUSD"],
        shortable_symbols=["DOGEUSD"],
        default_trade_fee_bps=3.0,
    )
    assert [item.symbol for item in metadata] == ["BTCFDUSD", "DOGEUSD"]
    assert metadata[0].trade_fee_bps == 0.0
    assert metadata[0].shortable is False
    assert metadata[1].trade_fee_bps == 3.0
    assert metadata[1].shortable is True


def test_load_hourly_market_data_aligns_and_computes_features(tmp_path: Path) -> None:
    _write_hourly_csv(tmp_path / "BTCFDUSD.csv", symbol="BTCFDUSD", start="2024-01-01", periods=240, price_start=100.0)
    _write_hourly_csv(tmp_path / "ETHFDUSD.csv", symbol="ETHFDUSD", start="2024-01-01", periods=240, price_start=50.0)

    market = data_mod.load_hourly_market_data(
        data_root=tmp_path,
        symbols=["BTCFDUSD", "ETHFDUSD"],
        shortable_symbols=["ETHFDUSD"],
        min_history_hours=200,
    )

    assert market.num_assets == 2
    assert market.feature_dim == len(data_mod.FEATURE_NAMES)
    assert len(market) == 240
    assert market.features.shape == (240, 2, len(data_mod.FEATURE_NAMES))
    assert market.closes.shape == (240, 2)
    assert market.shortable_mask.tolist() == [0.0, 1.0]
    assert market.trade_fee_bps.tolist() == [0.0, 0.0]


def test_feature_normalizer_zero_centers_training_slice() -> None:
    timestamps = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    features = np.asarray(
        [
            [[1.0, 10.0], [3.0, 30.0]],
            [[2.0, 20.0], [4.0, 40.0]],
            [[3.0, 30.0], [5.0, 50.0]],
            [[4.0, 40.0], [6.0, 60.0]],
            [[5.0, 50.0], [7.0, 70.0]],
        ],
        dtype=np.float32,
    )
    closes = np.ones((5, 2), dtype=np.float32)
    returns = np.zeros((5, 2), dtype=np.float32)
    market = data_mod.HourlyMarketData(
        symbols=["A", "B"],
        timestamps=timestamps,
        features=features,
        closes=closes,
        returns=returns,
        shortable_mask=np.ones(2, dtype=np.float32),
        trade_fee_bps=np.zeros(2, dtype=np.float32),
    )

    normalizer = data_mod.fit_feature_normalizer(market)
    normalized = data_mod.apply_feature_normalizer(market, normalizer)

    np.testing.assert_allclose(normalized.features.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(normalized.features.std(axis=0), 1.0, atol=1e-6)


def test_env_penalizes_downside_and_respects_shortable_mask() -> None:
    timestamps = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    features = np.zeros((8, 2, 3), dtype=np.float32)
    closes = np.asarray(
        [
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
            [90.0, 110.0],
            [90.0, 110.0],
            [90.0, 110.0],
        ],
        dtype=np.float32,
    )
    returns = np.zeros_like(closes)
    returns[5] = np.asarray([-0.10, 0.10], dtype=np.float32)
    market = data_mod.HourlyMarketData(
        symbols=["LONGONLY", "SHORTABLE"],
        timestamps=timestamps,
        features=features,
        closes=closes,
        returns=returns,
        shortable_mask=np.asarray([0.0, 1.0], dtype=np.float32),
        trade_fee_bps=np.asarray([2.0, 2.0], dtype=np.float32),
    )
    cfg = env_mod.EnvConfig(lookback=4, episode_steps=2, random_reset=False, reward_scale=1.0)
    env = env_mod.BinanceHourlyEnv(market, cfg, slice_start=4, slice_end=6, seed=7)
    env.reset(start_index=4)

    _, reward, _, info = env.step(np.asarray([-1.0, 0.5], dtype=np.float32))

    assert env.current_weights[0] == 0.0
    assert env.current_weights[1] > 0.0
    assert reward < info["step_return"]
    assert info["drawdown"] >= 0.0


def test_transformer_policy_shapes() -> None:
    cfg = model_mod.PolicyConfig(
        lookback=4,
        num_assets=3,
        feature_dim=5,
        portfolio_dim=7,
        d_model=32,
        n_heads=4,
        n_layers=2,
        mlp_hidden=64,
    )
    model = model_mod.RiskAwareActorCritic(cfg)
    obs = torch.zeros((6, 4 * 3 * 5 + 7), dtype=torch.float32)
    action, log_prob, value = model.act(obs, deterministic=False)
    det_action, det_value = model.predict_deterministic(obs)
    assert action.shape == (6, 3)
    assert log_prob.shape == (6,)
    assert value.shape == (6,)
    assert det_action.shape == (6, 3)
    assert det_value.shape == (6,)


def test_build_walk_forward_windows_applies_purge_gap() -> None:
    windows = validate_mod.build_walk_forward_windows(
        num_steps=500,
        lookback=48,
        window_hours=72,
        purge_hours=24,
    )
    assert windows
    for left, right in zip(windows, windows[1:]):
        assert right[0] - left[1] >= 24


def test_aggregate_window_summaries_supports_multi_horizon_scoring() -> None:
    aggregate = validate_mod.aggregate_window_summaries(
        {
            "7d": {
                "median_total_return": 0.02,
                "p10_total_return": -0.01,
                "median_sortino": 1.0,
                "p90_max_drawdown": 0.05,
                "median_volatility": 0.20,
                "mean_turnover": 0.10,
                "score": 0.0,
            },
            "30d": {
                "median_total_return": 0.04,
                "p10_total_return": 0.00,
                "median_sortino": 2.0,
                "p90_max_drawdown": 0.04,
                "median_volatility": 0.15,
                "mean_turnover": 0.08,
                "score": 0.0,
            },
        },
        {"7d": 0.4, "30d": 0.6},
    )

    assert aggregate["median_total_return"] > 0.03
    assert aggregate["p90_max_drawdown"] < 0.05
    assert "score" in aggregate


def test_resolve_stride_hours_accepts_checkpoint_none_list() -> None:
    strides = validate_mod.resolve_stride_hours([168, 720], [None, None])
    assert strides == [None, None]


def test_batched_window_validation_matches_single_window_path() -> None:
    timestamps = pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC")
    features = np.zeros((20, 2, 3), dtype=np.float32)
    closes = np.ones((20, 2), dtype=np.float32)
    returns = np.zeros((20, 2), dtype=np.float32)
    returns[:, 0] = 0.001
    returns[:, 1] = -0.0005
    market = data_mod.HourlyMarketData(
        symbols=["A", "B"],
        timestamps=timestamps,
        features=features,
        closes=closes,
        returns=returns,
        shortable_mask=np.ones(2, dtype=np.float32),
        trade_fee_bps=np.zeros(2, dtype=np.float32),
    )
    cfg = env_mod.EnvConfig(
        lookback=4,
        episode_steps=6,
        random_reset=False,
        spread_bps=0.0,
        slippage_bps=0.0,
        downside_penalty=0.0,
        drawdown_penalty=0.0,
        turnover_penalty=0.0,
        concentration_penalty=0.0,
        leverage_penalty=0.0,
        smoothness_penalty=0.0,
        volatility_penalty=0.0,
    )
    policy_cfg = model_mod.PolicyConfig(
        lookback=4,
        num_assets=2,
        feature_dim=3,
        portfolio_dim=6,
        d_model=16,
        n_heads=4,
        n_layers=1,
        mlp_hidden=16,
    )
    model = model_mod.RiskAwareActorCritic(policy_cfg)
    for parameter in model.parameters():
        parameter.data.zero_()
    device = torch.device("cpu")
    windows = [(4, 10), (6, 12)]

    single = [
        validate_mod.run_window_validation(
            market=market,
            model=model,
            env_config=cfg,
            window=window,
            device=device,
        )
        for window in windows
    ]
    batched = validate_mod.run_window_validation_batch(
        market=market,
        model=model,
        env_config=cfg,
        windows=windows,
        device=device,
        batch_size=2,
    )

    for left, right in zip(single, batched):
        assert left.start_index == right.start_index
        assert left.end_index == right.end_index
        np.testing.assert_allclose(left.total_return, right.total_return, atol=1e-9)
        np.testing.assert_allclose(left.max_drawdown, right.max_drawdown, atol=1e-9)
        np.testing.assert_allclose(left.mean_turnover, right.mean_turnover, atol=1e-9)


def test_build_sweep_configs_cartesian_product() -> None:
    configs = sweep_mod.build_sweep_configs(
        {
            "max_gross_leverage": [3.0, 5.0],
            "downside_penalty": [2.0],
            "smoothness_penalty": [2.0, 4.0],
        }
    )
    assert len(configs) == 4
    assert configs[0]["max_gross_leverage"] == 3.0
    assert configs[-1]["smoothness_penalty"] == 4.0
