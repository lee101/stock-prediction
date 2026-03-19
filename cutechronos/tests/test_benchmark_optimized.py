"""Tests for cutechronos.benchmark_optimized module.

Validates the helper functions and benchmark_config logic using
small synthetic data (no GPU or real model required for unit tests).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cutechronos.benchmark_optimized import (
    compute_mae,
    load_series,
    reset_gpu_stats,
    get_peak_memory_mb,
    benchmark_config,
    timed_call_cuda,
)


class TestComputeMAE:
    def test_identical_tensors(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        assert compute_mae(a, a) == 0.0

    def test_known_mae(self):
        pred = torch.tensor([1.0, 2.0, 3.0])
        actual = torch.tensor([2.0, 3.0, 4.0])
        assert abs(compute_mae(pred, actual) - 1.0) < 1e-6

    def test_different_lengths(self):
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        actual = torch.tensor([2.0, 3.0, 4.0])
        # Should use min length = 3
        assert abs(compute_mae(pred, actual) - 1.0) < 1e-6

    def test_float_precision(self):
        pred = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float16)
        actual = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        # Should convert to float32 internally
        mae = compute_mae(pred, actual)
        assert mae < 0.01  # Allow small float16 rounding


class TestLoadSeries:
    def test_loads_correct_shapes(self, tmp_path):
        # Create a minimal CSV
        csv_path = tmp_path / "TEST.csv"
        n = 600
        data = {
            "timestamp": list(range(n)),
            "open": list(range(n)),
            "high": list(range(n)),
            "low": list(range(n)),
            "close": [float(i) * 0.1 for i in range(n)],
            "volume": list(range(n)),
        }
        import pandas as pd

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        ctx, act = load_series(str(csv_path), context_length=512, prediction_length=30)
        assert ctx.shape == (512,)
        assert act.shape == (30,)

    def test_raises_on_insufficient_data(self, tmp_path):
        csv_path = tmp_path / "SHORT.csv"
        n = 100
        import pandas as pd

        df = pd.DataFrame({"close": list(range(n))})
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="need"):
            load_series(str(csv_path), context_length=512, prediction_length=30)


class TestTimedCall:
    def test_returns_result_and_time(self):
        result, elapsed = timed_call_cuda(lambda: torch.tensor(42))
        assert result.item() == 42
        assert elapsed > 0


class TestBenchmarkConfig:
    def test_basic_benchmark(self):
        # Create simple synthetic data
        ctx_batch = torch.randn(2, 64)
        act_batch = torch.randn(2, 30)

        call_count = {"n": 0}

        def dummy_predict(cb):
            call_count["n"] += 1
            # Return "predictions" of the right shape
            return torch.randn(cb.shape[0], 30)

        result = benchmark_config(
            label="test",
            predict_fn=dummy_predict,
            context_batches=[ctx_batch],
            actuals=[act_batch],
            prediction_length=30,
            n_warmup=1,
            n_runs=2,
        )

        assert result["label"] == "test"
        assert result["avg_latency_ms"] > 0
        assert result["min_latency_ms"] > 0
        assert result["std_latency_ms"] >= 0
        assert result["throughput_samples_per_sec"] > 0
        assert isinstance(result["avg_mae"], float)
        # warmup: 1 call, runs: 2 calls
        assert call_count["n"] == 3

    def test_multi_batch_benchmark(self):
        ctx1 = torch.randn(2, 64)
        ctx2 = torch.randn(3, 64)
        act1 = torch.randn(2, 30)
        act2 = torch.randn(3, 30)

        def dummy_predict(cb):
            return torch.randn(cb.shape[0], 30)

        result = benchmark_config(
            label="multi",
            predict_fn=dummy_predict,
            context_batches=[ctx1, ctx2],
            actuals=[act1, act2],
            prediction_length=30,
            n_warmup=1,
            n_runs=3,
        )

        assert result["label"] == "multi"
        assert result["avg_latency_ms"] > 0


class TestResetGpuStats:
    def test_no_crash_without_cuda(self):
        # Should not raise even without CUDA
        reset_gpu_stats()

    def test_get_peak_memory(self):
        # Should return 0 or a valid number
        mem = get_peak_memory_mb()
        assert mem >= 0
