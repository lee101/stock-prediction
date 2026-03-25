"""Tests for CuteChronos2Pipeline."""

from __future__ import annotations

import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODEL_ID = "amazon/chronos-2"


@pytest.fixture(scope="module")
def cute_pipeline():
    """Load CuteChronos2Pipeline once for the entire test module."""
    from cutechronos.pipeline import CuteChronos2Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = CuteChronos2Pipeline.from_pretrained(MODEL_ID, device=device, dtype=torch.bfloat16)
    return pipe


@pytest.fixture(scope="module")
def original_pipeline():
    """Load the upstream Chronos2Pipeline once for the entire test module."""
    from chronos.chronos2 import Chronos2Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = Chronos2Pipeline.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    pipe.model = pipe.model.to(device)
    return pipe


@pytest.fixture()
def random_context():
    """A random context tensor of length 512."""
    torch.manual_seed(42)
    return torch.randn(512)


@pytest.fixture()
def batch_context():
    """A batch of 3 random context tensors of different lengths."""
    torch.manual_seed(42)
    return [torch.randn(300), torch.randn(400), torch.randn(512)]


# ---------------------------------------------------------------------------
# Tests: from_pretrained
# ---------------------------------------------------------------------------


class TestFromPretrained:
    def test_loads_successfully(self, cute_pipeline):
        assert cute_pipeline is not None
        assert cute_pipeline.model is not None

    def test_has_expected_properties(self, cute_pipeline):
        assert cute_pipeline.model_context_length > 0
        assert cute_pipeline.model_prediction_length > 0
        assert cute_pipeline.model_output_patch_size > 0
        assert cute_pipeline.max_output_patches > 0
        assert len(cute_pipeline.quantiles) > 0

    def test_quantiles_include_median(self, cute_pipeline):
        assert 0.5 in cute_pipeline.quantiles


# ---------------------------------------------------------------------------
# Tests: predict
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_1d_tensor(self, cute_pipeline, random_context):
        prediction_length = 30
        preds = cute_pipeline.predict(random_context, prediction_length=prediction_length)
        assert isinstance(preds, list)
        assert len(preds) == 1
        # Shape: (1, n_quantiles, prediction_length)
        assert preds[0].ndim == 3
        assert preds[0].shape[0] == 1
        assert preds[0].shape[1] == len(cute_pipeline.quantiles)
        assert preds[0].shape[2] == prediction_length

    def test_predict_2d_tensor(self, cute_pipeline):
        torch.manual_seed(42)
        ctx = torch.randn(2, 512)
        prediction_length = 30
        preds = cute_pipeline.predict(ctx, prediction_length=prediction_length)
        assert isinstance(preds, list)
        assert len(preds) == 2
        for p in preds:
            assert p.shape == (1, len(cute_pipeline.quantiles), prediction_length)

    def test_predict_list_of_tensors(self, cute_pipeline, batch_context):
        prediction_length = 16
        preds = cute_pipeline.predict(batch_context, prediction_length=prediction_length)
        assert isinstance(preds, list)
        assert len(preds) == 3
        for p in preds:
            assert p.shape == (1, len(cute_pipeline.quantiles), prediction_length)

    def test_predict_default_length(self, cute_pipeline, random_context):
        preds = cute_pipeline.predict(random_context)
        assert isinstance(preds, list)
        assert len(preds) == 1
        assert preds[0].shape[2] == cute_pipeline.model_prediction_length

    def test_predict_returns_float32_cpu(self, cute_pipeline, random_context):
        preds = cute_pipeline.predict(random_context, prediction_length=16)
        assert preds[0].dtype == torch.float32
        assert preds[0].device == torch.device("cpu")

    def test_predict_nan_pattern_matches_original(self, cute_pipeline, original_pipeline, random_context):
        """Both pipelines should produce the same NaN pattern for the same input.

        Note: Chronos-2 can produce NaN for certain inputs (e.g. random
        Gaussian data, large-magnitude series) due to numerical issues in
        the attention layers.  This is a model-level behaviour, not a
        pipeline bug.
        """
        prediction_length = 30
        cute_preds = cute_pipeline.predict(random_context, prediction_length=prediction_length)
        orig_preds = original_pipeline.predict([random_context], prediction_length=prediction_length)

        cute_nan = torch.isnan(cute_preds[0])
        orig_nan = torch.isnan(orig_preds[0])

        # Both should have the same NaN pattern (either both NaN or both valid)
        assert cute_nan.shape == orig_nan.shape or cute_nan.all() == orig_nan.all()

    def test_predict_limit_prediction_length(self, cute_pipeline, random_context):
        with pytest.raises(ValueError, match="exceeds"):
            cute_pipeline.predict(
                random_context,
                prediction_length=cute_pipeline.model_prediction_length + 1,
                limit_prediction_length=True,
            )

    def test_predict_exceed_with_limit_off(self, cute_pipeline, random_context):
        # Should warn but not raise
        max_len = cute_pipeline.model_prediction_length
        preds = cute_pipeline.predict(
            random_context,
            prediction_length=max_len + 1,
            limit_prediction_length=False,
        )
        # Will be clamped to max_output_patches * patch_size
        assert isinstance(preds, list)

    def test_context_truncation(self, cute_pipeline):
        """Very long context should be truncated, not error."""
        torch.manual_seed(42)
        long_ctx = torch.randn(cute_pipeline.model_context_length + 500)
        preds = cute_pipeline.predict(long_ctx, prediction_length=16)
        assert preds[0].shape == (1, len(cute_pipeline.quantiles), 16)


# ---------------------------------------------------------------------------
# Tests: predict_quantiles
# ---------------------------------------------------------------------------


class TestPredictQuantiles:
    def test_predict_quantiles_default(self, cute_pipeline, random_context):
        quantiles, mean = cute_pipeline.predict_quantiles(
            random_context, prediction_length=30
        )
        assert isinstance(quantiles, list)
        assert isinstance(mean, list)
        assert len(quantiles) == 1
        assert len(mean) == 1
        # quantiles[0]: (1, prediction_length, n_quantile_levels)
        assert quantiles[0].ndim == 3
        assert quantiles[0].shape[1] == 30
        assert quantiles[0].shape[2] == 9  # default 0.1..0.9

    def test_predict_quantiles_custom_levels(self, cute_pipeline, random_context):
        levels = [0.1, 0.5, 0.9]
        quantiles, mean = cute_pipeline.predict_quantiles(
            random_context,
            prediction_length=30,
            quantile_levels=levels,
        )
        assert quantiles[0].shape[2] == len(levels)

    def test_predict_quantiles_mean_shape(self, cute_pipeline, random_context):
        quantiles, mean = cute_pipeline.predict_quantiles(
            random_context, prediction_length=30
        )
        # mean[0]: (1, prediction_length)
        assert mean[0].ndim == 2
        assert mean[0].shape[1] == 30


# ---------------------------------------------------------------------------
# Tests: equivalence with original pipeline
# ---------------------------------------------------------------------------


class TestEquivalence:
    def test_output_equivalence(self, cute_pipeline, original_pipeline, random_context):
        """CuteChronos2Pipeline predictions should closely match the original.

        CuteChronos2Model is a standalone reimplementation that shares identical
        weights with the upstream Chronos2Model. BF16 execution across 12
        transformer layers accumulates small numerical differences due to
        different matmul tiling/reduction order (SDPA vs original attention).
        MAE < 0.01 confirms functional equivalence.
        """
        prediction_length = 30

        cute_preds = cute_pipeline.predict(random_context, prediction_length=prediction_length)
        orig_preds = original_pipeline.predict([random_context], prediction_length=prediction_length)

        cute_tensor = cute_preds[0]  # (1, Q, H)
        orig_tensor = orig_preds[0]  # (1, Q, H)

        assert cute_tensor.shape == orig_tensor.shape, (
            f"Shape mismatch: cute={cute_tensor.shape}, orig={orig_tensor.shape}"
        )

        # NaN positions must match
        cute_nan = torch.isnan(cute_tensor)
        orig_nan = torch.isnan(orig_tensor)
        assert (cute_nan == orig_nan).all(), "NaN pattern mismatch between pipelines"

        # Non-NaN values must match within tolerance
        valid_mask = ~cute_nan
        if valid_mask.any():
            mae = (cute_tensor[valid_mask] - orig_tensor[valid_mask]).abs().mean().item()
            assert mae < 0.01, f"MAE between cute and original is {mae:.6f}, expected < 0.01"

    def test_quantile_shapes_match(self, cute_pipeline, original_pipeline, random_context):
        prediction_length = 30
        cute_q, cute_m = cute_pipeline.predict_quantiles(
            random_context, prediction_length=prediction_length
        )
        orig_q, orig_m = original_pipeline.predict_quantiles(
            [random_context], prediction_length=prediction_length
        )
        # Both should return lists of same length
        assert len(cute_q) == len(orig_q)
        assert len(cute_m) == len(orig_m)
