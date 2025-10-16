import numpy as np
import pytest

from hftraining.toto_features import TotoFeatureGenerator, TotoOptions


def _make_price_matrix():
    rng = np.random.default_rng(42)
    # 32 timesteps, 5 columns (open/high/low/close/volume)
    base = rng.normal(loc=100.0, scale=1.0, size=(32, 5)).astype(np.float32)
    # Ensure volume strictly positive
    base[:, 4] = np.abs(base[:, 4]) + 1.0
    return base


def test_toto_feature_generator_caches_to_disk(tmp_path, monkeypatch):
    price_matrix = _make_price_matrix()
    columns = ["open", "high", "low", "close", "volume"]
    options = TotoOptions(
        horizon=4,
        context_length=8,
        num_samples=64,
        use_toto=False,
        cache_dir=str(tmp_path),
        enable_cache=True,
    )

    call_counter = {"count": 0}

    def fake_statistical(self, matrix, column_index):
        call_counter["count"] += 1
        feats = np.full((matrix.shape[0], options.horizon * 2), float(column_index), dtype=np.float32)
        return feats, feats.shape[1]

    monkeypatch.setattr(TotoFeatureGenerator, "_compute_statistical_forecasts", fake_statistical, raising=False)

    generator = TotoFeatureGenerator(options)
    features_first, names_first = generator.compute_features(price_matrix, columns, symbol_prefix="TEST")

    # Expect four target columns -> four invocations
    assert call_counter["count"] == 4
    assert features_first.shape[0] == price_matrix.shape[0]
    assert len(names_first) == features_first.shape[1]

    # Second pass should hit the cache and avoid recomputation
    def explode(*args, **kwargs):  # pragma: no cover - should not fire
        raise AssertionError("Cache miss when cache should be used")

    monkeypatch.setattr(TotoFeatureGenerator, "_compute_statistical_forecasts", explode, raising=False)
    generator_cached = TotoFeatureGenerator(options)
    features_cached, names_cached = generator_cached.compute_features(price_matrix, columns, symbol_prefix="TEST")

    np.testing.assert_array_equal(features_first, features_cached)
    assert names_first == names_cached

    cache_files = list(tmp_path.glob("*.npz"))
    assert cache_files, "Expected Toto feature cache artifacts on disk"
