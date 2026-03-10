from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _write_hourly_history_csv(path: Path, symbol: str, timestamps: pd.DatetimeIndex) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000.0,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_chronos_sol_data_module_windows_forecasts_when_max_history_days_set(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import binancechronossolexperiment.data as data_mod
    from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig

    symbol = "TEST"
    timestamps = pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC")
    _write_hourly_history_csv(tmp_path / f"{symbol}.csv", symbol, timestamps)

    calls: dict[str, pd.Timestamp | None] = {"start": None, "end": None}

    def _fake_build_feature_frame(frame: pd.DataFrame, *, horizons, max_lookback: int) -> pd.DataFrame:
        work = frame.copy()
        work["reference_close"] = work["close"]
        return work

    def _fake_build_forecast_bundle(
        *,
        symbol: str,
        data_root: Path,
        cache_root: Path,
        horizons,
        context_hours: int,
        quantile_levels,
        batch_size: int,
        model_id: str,
        device_map: str = "cuda",
        preaugmentation_dirs=None,
        cache_only: bool = False,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        wrapper_factory=None,
    ) -> pd.DataFrame:
        calls["start"] = start
        calls["end"] = end
        window = timestamps
        if start is not None:
            window = window[window > start]
        if end is not None:
            window = window[window <= end]
        return pd.DataFrame(
            {
                "timestamp": window,
                "symbol": symbol,
                "predicted_close_p50_h1": 100.0,
                "predicted_high_p50_h1": 101.0,
                "predicted_low_p50_h1": 99.0,
            }
        )

    monkeypatch.setattr(data_mod, "build_feature_frame", _fake_build_feature_frame)
    monkeypatch.setattr(data_mod, "build_forecast_bundle", _fake_build_forecast_bundle)

    split_config = SplitConfig(val_days=1, test_days=1)
    module = ChronosSolDataModule(
        symbol=symbol,
        data_root=tmp_path,
        forecast_cache_root=tmp_path / "cache",
        forecast_horizons=(1,),
        context_hours=16,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=8,
        model_id="dummy",
        sequence_length=4,
        split_config=split_config,
        max_feature_lookback_hours=10,
        min_history_hours=5,
        max_history_days=3,
        feature_columns=(
            "close",
            "predicted_close_p50_h1",
            "predicted_high_p50_h1",
            "predicted_low_p50_h1",
        ),
        cache_only=True,
    )

    expected_end = timestamps.max()
    expected_start = expected_end - pd.Timedelta(hours=float(3 * 24 + 10))

    assert calls["end"] == expected_end
    assert calls["start"] == expected_start
    assert pd.to_datetime(module.full_frame["timestamp"], utc=True).min() > expected_start


def test_chronos_sol_data_module_propagates_directional_constraints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import binancechronossolexperiment.data as data_mod
    from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig

    symbol = "TEST"
    timestamps = pd.date_range("2026-01-01", periods=120, freq="h", tz="UTC")
    _write_hourly_history_csv(tmp_path / f"{symbol}.csv", symbol, timestamps)

    def _fake_build_feature_frame(frame: pd.DataFrame, *, horizons, max_lookback: int) -> pd.DataFrame:
        work = frame.copy()
        work["reference_close"] = work["close"]
        return work

    def _fake_build_forecast_bundle(
        *,
        symbol: str,
        data_root: Path,
        cache_root: Path,
        horizons,
        context_hours: int,
        quantile_levels,
        batch_size: int,
        model_id: str,
        device_map: str = "cuda",
        preaugmentation_dirs=None,
        cache_only: bool = False,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        wrapper_factory=None,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": symbol,
                "predicted_close_p50_h1": 100.0,
                "predicted_high_p50_h1": 101.0,
                "predicted_low_p50_h1": 99.0,
            }
        )

    monkeypatch.setattr(data_mod, "build_feature_frame", _fake_build_feature_frame)
    monkeypatch.setattr(data_mod, "build_forecast_bundle", _fake_build_forecast_bundle)

    module = ChronosSolDataModule(
        symbol=symbol,
        data_root=tmp_path,
        forecast_cache_root=tmp_path / "cache",
        forecast_horizons=(1,),
        context_hours=16,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=8,
        model_id="dummy",
        sequence_length=4,
        split_config=SplitConfig(val_days=1, test_days=1),
        min_history_hours=5,
        feature_columns=(
            "close",
            "predicted_close_p50_h1",
            "predicted_high_p50_h1",
            "predicted_low_p50_h1",
        ),
        cache_only=True,
        can_long=0.0,
        can_short=1.0,
    )

    assert module.train_dataset._can_long == 0.0
    assert module.train_dataset._can_short == 1.0
    assert module.val_dataset._can_long == 0.0
    assert module.val_dataset._can_short == 1.0
    assert module.test_dataset._can_long == 0.0
    assert module.test_dataset._can_short == 1.0
    sample = module.train_dataset[0]
    assert float(sample["can_long"].item()) == 0.0
    assert float(sample["can_short"].item()) == 1.0


def test_recover_dataset_artifacts_uses_override_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import binancechronossolexperiment.inference as inference_mod

    captured = {}

    class FakeDataModule:
        def __init__(self, config) -> None:
            captured["config"] = config
            self.normalizer = inference_mod.FeatureNormalizer(
                mean=np.array([1.0, 2.0], dtype=np.float32),
                std=np.array([3.0, 4.0], dtype=np.float32),
            )
            self.feature_columns = ("feat_a", "feat_b")

    monkeypatch.setattr(inference_mod, "BinanceHourlyDataModule", FakeDataModule)

    normalizer, feature_columns = inference_mod._recover_dataset_artifacts(
        {
            "sequence_length": 96,
            "dataset": {
                "symbol": "BTCUSD",
                "data_root": "trainingdatahourly/crypto",
                "forecast_cache_root": "binanceneural/forecast_cache",
                "forecast_horizons": (1, 24),
                "sequence_length": 96,
                "val_fraction": 0.2,
                "min_history_hours": 720,
                "max_feature_lookback_hours": 168,
                "validation_days": 30,
                "cache_only": False,
            },
        },
        data_root=tmp_path / "hourly",
        forecast_cache_root=tmp_path / "cache",
    )

    assert feature_columns == ["feat_a", "feat_b"]
    assert normalizer.mean.tolist() == [1.0, 2.0]
    assert captured["config"].symbol == "BTCUSD"
    assert captured["config"].data_root == tmp_path / "hourly"
    assert captured["config"].forecast_cache_root == tmp_path / "cache"
    assert captured["config"].forecast_horizons == (1, 24)
    assert captured["config"].sequence_length == 96


def test_load_policy_checkpoint_recovers_lean_checkpoint_metadata(monkeypatch) -> None:
    import binancechronossolexperiment.inference as inference_mod

    recovered_normalizer = inference_mod.FeatureNormalizer(
        mean=np.array([0.0, 1.0], dtype=np.float32),
        std=np.array([1.0, 2.0], dtype=np.float32),
    )
    recovered_features = ["feat_a", "feat_b"]
    payload = {
        "state_dict": {"dummy": torch.tensor([1.0])},
        "config": {"dataset": {"symbol": "BTCUSD"}},
    }

    class DummyModel:
        pass

    monkeypatch.setattr(inference_mod.torch, "load", lambda *args, **kwargs: payload)
    monkeypatch.setattr(
        inference_mod,
        "_recover_dataset_artifacts",
        lambda cfg, **kwargs: (recovered_normalizer, recovered_features),
    )
    monkeypatch.setattr(
        inference_mod,
        "_build_policy",
        lambda state_dict, cfg, input_dim: ("built", state_dict, cfg, input_dim),
    )

    model, normalizer, feature_columns, cfg = inference_mod.load_policy_checkpoint("dummy.ckpt")

    assert model == ("built", payload["state_dict"], payload["config"], 2)
    assert normalizer is recovered_normalizer
    assert list(feature_columns) == recovered_features
    assert cfg == payload["config"]


def test_build_policy_aligns_state_dict_input_dim(monkeypatch) -> None:
    import binancechronossolexperiment.inference as inference_mod

    calls = {}

    def _fake_align(state_dict, *, input_dim):
        calls["align_input_dim"] = input_dim
        patched = dict(state_dict)
        patched["embed.weight"] = torch.zeros((256, input_dim))
        return patched

    def _fake_policy_config(payload, *, input_dim, state_dict):
        calls["policy_input_dim"] = input_dim
        calls["policy_state_shape"] = tuple(state_dict["embed.weight"].shape)
        return {"input_dim": input_dim}

    class DummyModel:
        def __init__(self) -> None:
            self.loaded = None

        def load_state_dict(self, state_dict, strict=False):
            self.loaded = (state_dict, strict)

        def eval(self):
            return self

    dummy_model = DummyModel()

    monkeypatch.setattr(inference_mod, "align_state_dict_input_dim", _fake_align)
    monkeypatch.setattr(inference_mod, "policy_config_from_payload", _fake_policy_config)
    monkeypatch.setattr(inference_mod, "build_policy", lambda cfg: dummy_model)

    model = inference_mod._build_policy({"embed.weight": torch.zeros((256, 13))}, {}, 16)

    assert model is dummy_model
    assert calls["align_input_dim"] == 16
    assert calls["policy_input_dim"] == 16
    assert calls["policy_state_shape"] == (256, 16)
    assert dummy_model.loaded[1] is False
