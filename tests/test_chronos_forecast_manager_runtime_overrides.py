from __future__ import annotations

from pathlib import Path

from binanceneural.config import ForecastConfig
from binanceneural.forecasts import ChronosForecastManager


def _build_cfg(tmp_path: Path, *, force_multivariate: bool | None, force_cross_learning: bool | None) -> ForecastConfig:
    return ForecastConfig(
        symbol="ETHUSD",
        data_root=tmp_path,
        context_hours=128,
        prediction_horizon_hours=6,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=8,
        cache_dir=tmp_path / "cache",
        force_multivariate=force_multivariate,
        force_cross_learning=force_cross_learning,
    )


def test_manager_runtime_overrides_can_force_multivariate_and_cross_learning(tmp_path: Path, monkeypatch) -> None:
    def _fake_resolve(symbol: str, *, frequency: str | None = None, default_prediction_length: int = 7):  # type: ignore[no-untyped-def]
        del symbol, frequency, default_prediction_length
        return {
            "predict_kwargs": {},
            "use_multivariate": False,
            "use_cross_learning": False,
        }

    monkeypatch.setattr("binanceneural.forecasts.resolve_chronos2_params", _fake_resolve)

    manager = ChronosForecastManager(
        _build_cfg(tmp_path, force_multivariate=True, force_cross_learning=True),
        wrapper_factory=lambda: object(),
    )

    assert manager._use_multivariate is True
    assert manager._use_cross_learning is True
    assert manager._predict_kwargs.get("predict_batches_jointly") is True


def test_manager_runtime_overrides_can_disable_cross_learning_predict_flag(tmp_path: Path, monkeypatch) -> None:
    def _fake_resolve(symbol: str, *, frequency: str | None = None, default_prediction_length: int = 7):  # type: ignore[no-untyped-def]
        del symbol, frequency, default_prediction_length
        return {
            "predict_kwargs": {"predict_batches_jointly": True},
            "use_multivariate": True,
            "use_cross_learning": True,
        }

    monkeypatch.setattr("binanceneural.forecasts.resolve_chronos2_params", _fake_resolve)

    manager = ChronosForecastManager(
        _build_cfg(tmp_path, force_multivariate=True, force_cross_learning=False),
        wrapper_factory=lambda: object(),
    )

    assert manager._use_multivariate is True
    assert manager._use_cross_learning is False
    assert "predict_batches_jointly" not in manager._predict_kwargs
