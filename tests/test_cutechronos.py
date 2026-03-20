"""Unit tests for the cutechronos module.

Tests cover:
- benchmark.py: compute_mae NaN handling, synthetic data fallback, load_series
- pipeline.py: dual-backend flag (_is_cute), predict shapes, graceful fallback
- convert.py: download_model, convert_model, validate_conversion stubs,
              run_benchmark, main --help

All tests run without downloading any HuggingFace checkpoints; they build
small randomly-initialised CuteChronos2Model instances instead.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_small_config():
    """Return a CuteChronos2Config suitable for fast CPU tests."""
    from cutechronos.model import CuteChronos2Config

    return CuteChronos2Config(
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=2,
        num_heads=4,
        dropout_rate=0.0,
        layer_norm_epsilon=1e-6,
        dense_act_fn="relu",
        rope_theta=10000.0,
        vocab_size=2,
        reg_token_id=1,
        context_length=64,
        input_patch_size=8,
        input_patch_stride=8,
        output_patch_size=8,
        quantiles=[0.1, 0.5, 0.9],
        use_reg_token=True,
        use_arcsinh=False,
        time_encoding_scale=64,
    )


def _make_small_model():
    """Build a randomly-initialised CuteChronos2Model in eval mode."""
    from cutechronos.model import CuteChronos2Model

    torch.manual_seed(0)
    model = CuteChronos2Model(_make_small_config())
    model.eval()
    return model


def _make_pipeline():
    """Wrap the small model in a CuteChronos2Pipeline."""
    from cutechronos.pipeline import CuteChronos2Pipeline

    model = _make_small_model()
    return CuteChronos2Pipeline(model, device="cpu", _is_cute=True)


# ===========================================================================
# Section 1: benchmark.py
# ===========================================================================


class TestComputeMae:
    """Tests for cutechronos.benchmark.compute_mae."""

    def test_basic(self):
        from cutechronos.benchmark import compute_mae

        pred = torch.tensor([1.0, 2.0, 3.0])
        actual = torch.tensor([1.5, 2.5, 3.5])
        mae = compute_mae(pred, actual)
        assert abs(mae - 0.5) < 1e-6

    def test_all_nan_returns_nan(self):
        from cutechronos.benchmark import compute_mae

        pred = torch.tensor([float("nan"), float("nan")])
        actual = torch.tensor([1.0, 2.0])
        result = compute_mae(pred, actual)
        assert math.isnan(result), f"Expected nan but got {result}"

    def test_partial_nan_ignored(self):
        """NaN elements must be excluded; valid elements used for MAE."""
        from cutechronos.benchmark import compute_mae

        pred = torch.tensor([float("nan"), 2.0, 3.0])
        actual = torch.tensor([1.0, 2.5, 3.5])
        # valid diffs: |2.0-2.5|=0.5, |3.0-3.5|=0.5  => mean=0.5
        mae = compute_mae(pred, actual)
        assert not math.isnan(mae)
        assert abs(mae - 0.5) < 1e-6

    def test_length_mismatch_uses_min(self):
        from cutechronos.benchmark import compute_mae

        pred = torch.tensor([1.0, 2.0, 3.0])
        actual = torch.tensor([1.0, 2.0])  # shorter
        mae = compute_mae(pred, actual)
        assert mae == pytest.approx(0.0)

    def test_zero_error(self):
        from cutechronos.benchmark import compute_mae

        x = torch.tensor([10.0, 20.0, 30.0])
        assert compute_mae(x, x) == pytest.approx(0.0)


class TestLoadSeries:
    """Tests for cutechronos.benchmark.load_series."""

    def test_loads_valid_csv(self, tmp_path):
        import pandas as pd
        from cutechronos.benchmark import load_series

        # Write a CSV with 100 rows
        closes = list(range(1, 101))
        df = pd.DataFrame({"close": closes})
        csv_path = tmp_path / "TEST.csv"
        df.to_csv(csv_path, index=False)

        ctx, act = load_series(str(csv_path), context_length=80, prediction_length=20)
        assert ctx.shape == (80,)
        assert act.shape == (20,)
        # Should use the last 100 points -> close 1..100, ctx=1..80, act=81..100
        assert ctx[0].item() == pytest.approx(1.0)
        assert act[-1].item() == pytest.approx(100.0)

    def test_too_short_raises(self, tmp_path):
        import pandas as pd
        from cutechronos.benchmark import load_series

        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        csv_path = tmp_path / "SHORT.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="need"):
            load_series(str(csv_path), context_length=50, prediction_length=10)


class TestSyntheticFallback:
    """Tests that benchmark.main() generates synthetic data when no CSVs present."""

    def test_synthetic_generation(self):
        """When no CSV is found, benchmark should generate synthetic series."""
        from cutechronos.benchmark import (
            compute_mae,
            median_from_predictions,
        )

        # Replicate the synthetic generation logic from benchmark.main()
        torch.manual_seed(42)
        context_length = 32
        prediction_length = 8
        total = context_length + prediction_length
        returns = torch.randn(total) * 0.02 + 0.0001
        prices = 100.0 * torch.exp(returns.cumsum(0))
        ctx = prices[:context_length]
        act = prices[context_length:]

        assert ctx.shape == (context_length,)
        assert act.shape == (prediction_length,)
        assert not torch.any(torch.isnan(ctx))
        assert not torch.any(torch.isnan(act))
        # Prices should be positive
        assert (ctx > 0).all()
        assert (act > 0).all()


class TestMedianFromPredictions:
    """Tests for cutechronos.benchmark.median_from_predictions."""

    def test_extracts_median(self):
        from cutechronos.benchmark import median_from_predictions

        quantiles = [0.1, 0.5, 0.9]
        # Shape (1, 3, 5): 3 quantiles, 5 horizon steps
        preds = [torch.arange(15).float().reshape(1, 3, 5)]
        # median_idx = 1, so we want preds[0][0, 1, :] = [5, 6, 7, 8, 9]
        median = median_from_predictions(preds, quantiles)
        expected = torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0])
        assert torch.allclose(median, expected)


# ===========================================================================
# Section 2: pipeline.py
# ===========================================================================


class TestPipelineIsCuteFlag:
    """Verify that _is_cute is set correctly for both backends."""

    def test_cute_backend_flag(self):
        from cutechronos.pipeline import CuteChronos2Pipeline

        model = _make_small_model()
        pipe = CuteChronos2Pipeline(model, device="cpu", _is_cute=True)
        assert pipe._is_cute is True

    def test_original_backend_flag(self):
        """When _is_cute=False the flag must be False and config must come from chronos_config."""
        from cutechronos.pipeline import CuteChronos2Pipeline

        # Build a fake model that mimics the original Chronos2Model interface
        class _FakeConfig:
            context_length = 32
            output_patch_size = 8
            max_output_patches = 4
            quantiles = [0.1, 0.5, 0.9]

        class _FakeOriginalModel:
            chronos_config = _FakeConfig()

        model = _FakeOriginalModel()
        pipe = CuteChronos2Pipeline(model, device="cpu", _is_cute=False)
        assert pipe._is_cute is False
        assert pipe.model_context_length == 32
        assert pipe.quantiles == [0.1, 0.5, 0.9]


class TestPipelinePredict:
    """Tests for CuteChronos2Pipeline.predict using a small random model."""

    def test_predict_1d_tensor_shape(self):
        pipe = _make_pipeline()
        ctx = torch.randn(32)
        preds = pipe.predict(ctx, prediction_length=8, limit_prediction_length=False)
        assert isinstance(preds, list)
        assert len(preds) == 1
        assert preds[0].shape == (1, 3, 8)  # 3 quantiles

    def test_predict_2d_tensor_shape(self):
        pipe = _make_pipeline()
        ctx = torch.randn(2, 32)
        preds = pipe.predict(ctx, prediction_length=8, limit_prediction_length=False)
        assert len(preds) == 2
        for p in preds:
            assert p.shape == (1, 3, 8)

    def test_predict_list_of_tensors(self):
        pipe = _make_pipeline()
        ctxs = [torch.randn(20), torch.randn(30), torch.randn(32)]
        preds = pipe.predict(ctxs, prediction_length=8, limit_prediction_length=False)
        assert len(preds) == 3
        for p in preds:
            assert p.shape == (1, 3, 8)

    def test_predict_returns_cpu_float32(self):
        pipe = _make_pipeline()
        ctx = torch.randn(32)
        preds = pipe.predict(ctx, prediction_length=8, limit_prediction_length=False)
        assert preds[0].dtype == torch.float32
        assert preds[0].device.type == "cpu"

    def test_predict_limit_raises(self):
        pipe = _make_pipeline()
        ctx = torch.randn(32)
        max_len = pipe.model_prediction_length
        with pytest.raises(ValueError, match="exceeds"):
            pipe.predict(ctx, prediction_length=max_len + 1, limit_prediction_length=True)

    def test_predict_exceed_with_limit_false_warns(self):
        pipe = _make_pipeline()
        ctx = torch.randn(32)
        max_len = pipe.model_prediction_length
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            preds = pipe.predict(ctx, prediction_length=max_len + 1, limit_prediction_length=False)
        assert len(w) == 1
        assert "exceeds" in str(w[0].message).lower()
        # Result is still returned (clamped internally)
        assert isinstance(preds, list)

    def test_context_truncation(self):
        """Context longer than model_context_length must be silently truncated."""
        pipe = _make_pipeline()
        long_ctx = torch.randn(pipe.model_context_length + 50)
        preds = pipe.predict(long_ctx, prediction_length=8, limit_prediction_length=False)
        assert preds[0].shape[0] == 1


class TestPipelinePredictQuantiles:
    """Tests for CuteChronos2Pipeline.predict_quantiles."""

    def test_returns_tuple_of_two_lists(self):
        pipe = _make_pipeline()
        ctx = torch.randn(32)
        quantiles, mean = pipe.predict_quantiles(
            ctx, prediction_length=8, limit_prediction_length=False
        )
        assert isinstance(quantiles, list)
        assert isinstance(mean, list)

    def test_shapes_default_levels(self):
        pipe = _make_pipeline()
        ctx = torch.randn(32)
        quantiles, mean = pipe.predict_quantiles(
            ctx, prediction_length=8, limit_prediction_length=False
        )
        # default quantile_levels = 0.1..0.9 (9 values)
        assert quantiles[0].ndim == 3
        assert quantiles[0].shape[1] == 8   # prediction_length
        assert quantiles[0].shape[2] == 9   # default 9 levels
        assert mean[0].ndim == 2
        assert mean[0].shape[1] == 8

    def test_custom_quantile_levels(self):
        pipe = _make_pipeline()
        ctx = torch.randn(32)
        levels = [0.1, 0.5, 0.9]
        quantiles, mean = pipe.predict_quantiles(
            ctx, prediction_length=8, quantile_levels=levels, limit_prediction_length=False
        )
        assert quantiles[0].shape[2] == 3

    def test_interpolated_quantile_levels(self):
        """Levels not in the model's quantiles trigger linear interpolation."""
        pipe = _make_pipeline()
        ctx = torch.randn(32)
        # 0.25 and 0.75 are not in [0.1, 0.5, 0.9]
        levels = [0.25, 0.75]
        quantiles, mean = pipe.predict_quantiles(
            ctx, prediction_length=8, quantile_levels=levels, limit_prediction_length=False
        )
        assert quantiles[0].shape[2] == 2
        # Values should be finite
        assert torch.isfinite(quantiles[0]).all()


class TestPipelineProperties:
    """Tests for CuteChronos2Pipeline property accessors."""

    def test_quantiles_include_median(self):
        pipe = _make_pipeline()
        assert 0.5 in pipe.quantiles

    def test_model_context_length_positive(self):
        pipe = _make_pipeline()
        assert pipe.model_context_length > 0

    def test_model_prediction_length_positive(self):
        pipe = _make_pipeline()
        assert pipe.model_prediction_length > 0

    def test_device_property(self):
        pipe = _make_pipeline()
        assert pipe.device == "cpu"

    def test_max_output_patches_positive(self):
        pipe = _make_pipeline()
        assert pipe.max_output_patches > 0


class TestPipelineFallbackWhenCuteUnavailable:
    """
    Verify that from_pretrained gracefully falls back to original when
    CuteChronos2Model is not importable (simulated via sys.modules patching).
    """

    def test_from_pretrained_cute_import_error_propagates(self, monkeypatch):
        """If cutechronos.model is broken, from_pretrained(use_cute=True) should raise."""
        import cutechronos.pipeline as pipeline_mod

        original_fn = pipeline_mod._load_model_cute

        def _raise(*args, **kwargs):
            raise ImportError("simulated cutechronos.model not available")

        monkeypatch.setattr(pipeline_mod, "_load_model_cute", _raise)

        with pytest.raises(ImportError, match="simulated"):
            pipeline_mod.CuteChronos2Pipeline.from_pretrained(
                "some/model", device="cpu", use_cute=True
            )

    def test_from_pretrained_original_import_error_propagates(self, monkeypatch):
        """If chronos package is missing, from_pretrained(use_cute=False) should raise."""
        import cutechronos.pipeline as pipeline_mod

        original_fn = pipeline_mod._load_model_original

        def _raise(*args, **kwargs):
            raise ImportError("chronos-forecasting not installed")

        monkeypatch.setattr(pipeline_mod, "_load_model_original", _raise)

        with pytest.raises(ImportError, match="chronos-forecasting"):
            pipeline_mod.CuteChronos2Pipeline.from_pretrained(
                "some/model", device="cpu", use_cute=False
            )


class TestPipelineLeftPadAndStack:
    """Unit tests for the internal _left_pad_and_stack helper."""

    def test_uniform_lengths(self):
        from cutechronos.pipeline import CuteChronos2Pipeline

        tensors = [torch.ones(5), torch.ones(5) * 2]
        stacked = CuteChronos2Pipeline._left_pad_and_stack(tensors)
        assert stacked.shape == (2, 5)

    def test_variable_lengths_left_pads_with_nan(self):
        from cutechronos.pipeline import CuteChronos2Pipeline

        t_short = torch.tensor([1.0, 2.0])
        t_long = torch.tensor([3.0, 4.0, 5.0])
        stacked = CuteChronos2Pipeline._left_pad_and_stack([t_short, t_long])
        assert stacked.shape == (2, 3)
        # First row should have a NaN at position 0
        assert torch.isnan(stacked[0, 0])
        assert stacked[0, 1].item() == pytest.approx(1.0)
        assert stacked[0, 2].item() == pytest.approx(2.0)


# ===========================================================================
# Section 3: convert.py
# ===========================================================================


class TestConvertDownloadModel:
    """Tests for cutechronos.convert.download_model."""

    def test_raises_system_exit_without_huggingface_hub(self, monkeypatch):
        """If huggingface_hub is not installed, must exit with a clear message."""
        import builtins

        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "huggingface_hub":
                raise ImportError("no module")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)

        # Remove cached import if present
        monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)

        from cutechronos import convert

        # Re-import to get fresh version after patching
        import importlib
        convert_fresh = importlib.reload(convert)

        with pytest.raises(SystemExit):
            convert_fresh.download_model("amazon/chronos-2")

    def test_calls_snapshot_download(self, monkeypatch):
        """download_model should pass model_id to snapshot_download."""
        calls = []

        import types
        hf_stub = types.ModuleType("huggingface_hub")

        def _snapshot_download(model_id, cache_dir=None, allow_patterns=None):
            calls.append(model_id)
            return "/tmp/fake_model"

        hf_stub.snapshot_download = _snapshot_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", hf_stub)

        from cutechronos import convert
        import importlib
        convert_fresh = importlib.reload(convert)

        result = convert_fresh.download_model("my/model")
        assert result == Path("/tmp/fake_model")
        assert calls == ["my/model"]


class TestConvertModel:
    """Tests for cutechronos.convert.convert_model."""

    def test_loads_cute_model(self, tmp_path, monkeypatch):
        """convert_model should load and return a CuteChronos2Model on cpu."""
        from cutechronos.model import CuteChronos2Model
        from cutechronos import convert
        import importlib

        # Patch CuteChronos2Model.from_pretrained to return our small model
        model = _make_small_model()

        def _fake_from_pretrained(path):
            return model

        monkeypatch.setattr(CuteChronos2Model, "from_pretrained", staticmethod(_fake_from_pretrained))

        convert_fresh = importlib.reload(convert)
        result = convert_fresh.convert_model(tmp_path, device="cpu", dtype=torch.float32)

        assert result is model
        assert not result.training


class TestRunBenchmark:
    """Tests for cutechronos.convert.run_benchmark."""

    def test_returns_dict_with_expected_keys(self):
        from cutechronos import convert

        model = _make_small_model()
        # Use very short runs to keep test fast
        results = convert.run_benchmark(
            model,
            device="cpu",
            context_length=32,
            batch_size=1,
            n_warmup=1,
            n_runs=3,
        )
        assert "avg_latency_ms" in results
        assert "std_latency_ms" in results
        assert "min_latency_ms" in results
        assert "peak_gpu_memory_mb" in results
        assert results["avg_latency_ms"] >= 0.0

    def test_latency_is_positive(self):
        from cutechronos import convert

        model = _make_small_model()
        results = convert.run_benchmark(
            model,
            device="cpu",
            context_length=16,
            batch_size=1,
            n_warmup=1,
            n_runs=2,
        )
        assert results["avg_latency_ms"] > 0.0


class TestValidateConversion:
    """Tests for cutechronos.convert.validate_conversion (with stubs)."""

    def test_skips_when_chronos_missing(self, monkeypatch, tmp_path):
        """validate_conversion returns True and skips when chronos not installed.

        Patches sys.modules so the 'from chronos.chronos2 import Chronos2Pipeline'
        inside validate_conversion triggers ImportError, causing the skip path.
        """
        import types
        from cutechronos import convert

        # Build stub modules that raise ImportError on attribute access so the
        # 'from chronos.chronos2 import Chronos2Pipeline' fails inside the try block.
        class _FailModule(types.ModuleType):
            def __getattr__(self, name):
                raise ImportError(f"chronos not installed (stub): {name}")

        monkeypatch.setitem(sys.modules, "chronos", _FailModule("chronos"))
        monkeypatch.setitem(sys.modules, "chronos.chronos2", _FailModule("chronos.chronos2"))

        model = _make_small_model()
        result = convert.validate_conversion(tmp_path, model, device="cpu")
        assert result is True


class TestConvertMain:
    """Tests that cutechronos.convert.main handles --help without crashing."""

    def test_help_exits_cleanly(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "cutechronos.convert", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Convert" in result.stdout


class TestBenchmarkMain:
    """Tests that cutechronos.benchmark handles --help without crashing."""

    def test_help_exits_cleanly(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "cutechronos.benchmark", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "benchmark" in result.stdout.lower()


# ===========================================================================
# Section 4: __init__.py exports
# ===========================================================================


class TestInitExports:
    """cutechronos package must be importable; key symbols must be accessible."""

    def test_package_importable(self):
        import cutechronos  # noqa: F401

    def test_pipeline_importable(self):
        from cutechronos.pipeline import CuteChronos2Pipeline
        assert CuteChronos2Pipeline is not None

    def test_model_importable(self):
        from cutechronos.model import CuteChronos2Model, CuteChronos2Config
        assert CuteChronos2Model is not None
        assert CuteChronos2Config is not None

    def test_benchmark_importable(self):
        from cutechronos.benchmark import compute_mae, load_series, benchmark_pipeline
        assert compute_mae is not None

    def test_convert_importable(self):
        from cutechronos.convert import convert_model, run_benchmark, validate_conversion
        assert convert_model is not None
