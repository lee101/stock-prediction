import numpy as np
import pandas as pd
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gymrl import FeatureBuilder, FeatureBuilderConfig
from gymrl.cache_utils import load_feature_cache, save_feature_cache
from gymrl.train_ppo_allocator import optional_float
from src.models.kronos_wrapper import KronosForecastResult


def _write_symbol_csv(path: Path, symbol: str, *, periods: int = 12) -> None:
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="D")
    base = np.linspace(100.0, 110.0, periods)
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": base,
            "high": base * 1.01,
            "low": base * 0.99,
            "close": base,
            "volume": np.linspace(1_000_000, 1_200_000, periods),
        }
    )
    df.to_csv(path / f"{symbol}.csv", index=False)


class GymRLTrainingTests(unittest.TestCase):
    def test_optional_float_parses_none_and_values(self) -> None:
        self.assertIsNone(optional_float("none"))
        self.assertIsNone(optional_float("NaN"))
        self.assertEqual(optional_float("0.25"), 0.25)

    def test_feature_builder_backend_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "train"
            data_dir.mkdir()
            _write_symbol_csv(data_dir, "AAA")
            _write_symbol_csv(data_dir, "BBB")

            config = FeatureBuilderConfig(
                forecast_backend="bootstrap",
                num_samples=16,
                context_window=3,
                prediction_length=1,
                realized_horizon=1,
                min_history=3,
                enforce_common_index=False,
                fill_method="ffill",
                bootstrap_block_size=2,
            )
            builder = FeatureBuilder(config=config)
            cube = builder.build_from_directory(data_dir)

            self.assertEqual(builder.backend_name, "bootstrap")
            self.assertEqual(builder.backend_errors, [])
            self.assertGreater(cube.features.shape[0], 0)

            cache_path = root / "features_bootstrap.npz"
            save_feature_cache(
                cache_path,
                cube,
                extra_metadata={
                    "backend_name": builder.backend_name,
                    "backend_errors": builder.backend_errors,
                },
            )
            _, meta = load_feature_cache(cache_path)
        self.assertEqual(meta["backend_name"], "bootstrap")
        self.assertEqual(meta["backend_errors"], [])

    @mock.patch("src.models.kronos_wrapper.KronosForecastingWrapper")
    def test_feature_builder_kronos_backend_with_stub(self, kronos_mock: mock.MagicMock) -> None:
        class _StubKronos:
            def __init__(self, **_kwargs) -> None:  # noqa: D401 - simple stub
                self.calls = 0

            def predict_series(self, data, timestamp_col, columns, pred_len, lookback, **_kwargs):
                self.calls += 1
                horizon = int(pred_len)
                timestamps = pd.Index(pd.to_datetime(data[timestamp_col].iloc[-horizon:]))
                absolute = np.linspace(120.0, 120.0 + horizon - 1, horizon, dtype=float)
                percent = np.full(horizon, 0.01, dtype=np.float32)
                return {
                    columns[0]: KronosForecastResult(
                        absolute=absolute,
                        percent=percent,
                        timestamps=timestamps,
                    )
                }

            def unload(self) -> None:  # pragma: no cover - interface parity only
                pass

        kronos_mock.side_effect = _StubKronos

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "train"
            data_dir.mkdir()
            _write_symbol_csv(data_dir, "AAA", periods=18)
            _write_symbol_csv(data_dir, "BBB", periods=18)

            config = FeatureBuilderConfig(
                forecast_backend="kronos",
                num_samples=8,
                context_window=6,
                prediction_length=2,
                realized_horizon=1,
                min_history=8,
                enforce_common_index=True,
                fill_method="ffill",
            )
            builder = FeatureBuilder(config=config, backend_kwargs={"kronos_device": "cpu"})
            cube = builder.build_from_directory(data_dir)

            self.assertEqual(builder.backend_name, "kronos")
            self.assertEqual(builder.backend_errors, [])
            self.assertGreater(cube.features.shape[0], 0)
            self.assertEqual(kronos_mock.call_count, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
