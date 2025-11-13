import json

import pandas as pd

from marketsimulator import backtest_test3_inline as ms_backtest


def _write_preaug_config(directory, symbol: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbol": symbol,
        "best_strategy": "detrending",
        "selection_metric": "mae_percent",
        "selection_value": 0.42,
        "config": {"name": "detrending", "params": {"window": 16}},
        "comparison": {
            "detrending": {"mae_percent": 0.42},
            "baseline": {"mae_percent": 0.9},
        },
    }
    path = directory / f"{symbol}.json"
    path.write_text(json.dumps(payload))
    return str(path)


def test_fallback_includes_chronos_metadata(monkeypatch, tmp_path):
    symbol = "SIMSYM"
    preaug_dir = tmp_path / "chronos2" / "hourly"
    preaug_path = _write_preaug_config(preaug_dir, symbol)

    monkeypatch.setattr(ms_backtest, "_default_preaug_dirs", lambda freq: (preaug_dir,))
    ms_backtest._reset_chronos2_metadata_cache()

    fake_params = {
        "model_id": "chronos/mock-1",
        "device_map": "cpu",
        "context_length": 64,
        "prediction_length": 5,
        "quantile_levels": (0.2, 0.5, 0.8),
        "batch_size": 16,
        "predict_kwargs": {"num_samples": 3},
        "_config_path": "/tmp/hparams.json",
    }

    def _fake_resolve(symbol_arg: str, **kwargs):
        assert symbol_arg == symbol
        return dict(fake_params)

    monkeypatch.setattr(ms_backtest, "resolve_chronos2_params", _fake_resolve)
    monkeypatch.setattr(ms_backtest, "_REAL_BACKTEST_MODULE", None)
    monkeypatch.setattr(ms_backtest, "_REAL_BACKTEST_ERROR", None)

    sample = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=16, freq="D"),
            "Open": pd.Series(range(100, 116), dtype=float),
            "High": pd.Series(range(101, 117), dtype=float),
            "Low": pd.Series(range(99, 115), dtype=float),
            "Close": pd.Series(range(100, 116), dtype=float),
            "Volume": pd.Series(range(1, 17), dtype=float),
        }
    )
    monkeypatch.setattr(ms_backtest, "_load_live_price_history", lambda *_args, **_kwargs: sample.copy())

    frame = ms_backtest.backtest_forecasts(symbol, num_simulations=6)
    assert isinstance(frame, pd.DataFrame) and not frame.empty

    assert frame["chronos2_preaug_strategy"].unique().tolist() == ["detrending"]
    assert frame["chronos2_preaug_source"].iloc[0] == preaug_path
    assert (frame["chronos2_context_length"] == fake_params["context_length"]).all()
    assert (frame["chronos2_batch_size"] == fake_params["batch_size"]).all()
    assert (frame["chronos2_prediction_length"] == fake_params["prediction_length"]).all()

    quantiles = frame["chronos2_quantile_levels"].iloc[0]
    assert quantiles == list(fake_params["quantile_levels"])

    predict_kwargs = frame["chronos2_predict_kwargs"].iloc[0]
    assert predict_kwargs == fake_params["predict_kwargs"]

    assert frame["chronos2_hparams_config_path"].iloc[0] == fake_params["_config_path"]
    assert frame["chronos2_model_id"].iloc[0] == fake_params["model_id"]

    ms_backtest._reset_chronos2_metadata_cache()
