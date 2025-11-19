import json
from types import SimpleNamespace

import pandas as pd

from strategytrainingneural.forecast_cache import ChronosForecastGenerator, ForecastGenerationConfig
from strategytrainingneural.trade_windows import load_trade_window_metrics, FORECAST_COLUMNS


class StubChronosWrapper:
    def __init__(self) -> None:
        self.timestamp_column = "timestamp"
        self.id_column = "symbol"

    def predict_ohlc(
        self,
        context_df: pd.DataFrame,
        *,
        symbol: str,
        prediction_length: int,
        context_length: int,
        future_covariates: pd.DataFrame | None = None,
        batch_size: int,
        quantile_levels=None,
        predict_kwargs=None,
    ) -> SimpleNamespace:
        if future_covariates is not None:
            next_ts = pd.to_datetime(future_covariates["timestamp"].iloc[0], utc=True)
        else:
            last_ts = pd.to_datetime(context_df["timestamp"].iloc[-1], utc=True)
            next_ts = last_ts + pd.Timedelta(days=1)
        last_close = float(context_df["close"].iloc[-1])

        quantile_frames = {}
        for level, delta in zip((0.1, 0.5, 0.9), (-0.5, 1.0, 2.0)):
            frame = pd.DataFrame(
                {
                    "close": [last_close + delta],
                    "high": [last_close + delta + 0.25],
                    "low": [last_close + delta - 0.25],
                },
                index=[next_ts],
            )
            quantile_frames[level] = frame
        return SimpleNamespace(quantile_frames=quantile_frames)


def test_forecast_generator_creates_cache(tmp_path):
    data_dir = tmp_path / "trainingdata" / "train"
    data_dir.mkdir(parents=True)
    cache_dir = tmp_path / "forecast_cache"
    symbol = "TEST"
    rows = [
        {"timestamp": "2024-01-01", "Open": 10, "High": 11, "Low": 9, "Close": 10.5, "Volume": 100},
        {"timestamp": "2024-01-02", "Open": 10.6, "High": 11.2, "Low": 10.2, "Close": 10.9, "Volume": 120},
        {"timestamp": "2024-01-03", "Open": 10.8, "High": 11.5, "Low": 10.5, "Close": 11.3, "Volume": 140},
    ]
    pd.DataFrame(rows).to_csv(data_dir / f"{symbol}.csv", index=False)

    config = ForecastGenerationConfig(context_length=2, batch_size=4, quantile_levels=(0.1, 0.5, 0.9))
    generator = ChronosForecastGenerator(
        data_dir=data_dir,
        cache_dir=cache_dir,
        config=config,
        wrapper_factory=StubChronosWrapper,
    )
    generator.generate([symbol], start_date="2024-01-02", end_date="2024-01-03")

    cache_file = cache_dir / f"{symbol}.parquet"
    assert cache_file.exists()
    cached = pd.read_parquet(cache_file)
    assert not cached.empty
    assert set(["timestamp", "symbol", "forecast_move_pct"]).issubset(cached.columns)
    assert cached["symbol"].iloc[0] == symbol
    assert cached["forecast_move_pct"].iloc[0] != 0.0
    quantiles = json.loads(cached["quantile_levels"].iloc[0])
    assert all(level in quantiles for level in (0.1, 0.5, 0.9))


def test_trade_window_join_with_forecasts(tmp_path):
    # create trade parquet
    trades = pd.DataFrame(
        {
            "symbol": ["FOO"] * 4,
            "strategy": ["simple"] * 4,
            "exit_timestamp": pd.date_range("2024-01-01", periods=4, freq="D"),
            "pnl": [1.0, -0.5, 0.2, 0.3],
            "pnl_pct": [0.01, -0.005, 0.002, 0.003],
        }
    )
    trade_path = tmp_path / "trades.parquet"
    trades.to_parquet(trade_path)

    # forecast cache file
    forecast_dir = tmp_path / "forecast_cache"
    forecast_dir.mkdir()
    forecast_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": ["FOO", "FOO"],
            "forecast_move_pct": [0.02, 0.03],
            "forecast_volatility_pct": [0.01, 0.015],
            "predicted_close": [10.5, 10.7],
            "predicted_close_p10": [10.1, 10.3],
            "predicted_close_p90": [10.9, 11.1],
            "predicted_high": [11.0, 11.2],
            "predicted_low": [10.0, 10.2],
            "context_close": [10.3, 10.5],
        }
    )
    forecast_df.to_parquet(forecast_dir / "FOO.parquet", index=False)

    dataset = load_trade_window_metrics(
        [str(trade_path)],
        window_days=2,
        min_trades=1,
        forecast_cache_dir=str(forecast_dir),
    )
    assert "forecast_move_pct" in dataset.columns
    assert dataset["forecast_move_pct"].max() > 0
    for column in FORECAST_COLUMNS:
        assert column in dataset.columns


def test_forecast_config_injects_required_quantiles():
    cfg = ForecastGenerationConfig(quantile_levels=(0.2, 0.8), frequency="daily")
    assert all(level in cfg.quantile_levels for level in (0.1, 0.5, 0.9))
    assert 0.2 in cfg.quantile_levels and 0.8 in cfg.quantile_levels
