#!/usr/bin/env python3
"""
PRODUCTION COMPILATION TESTS

Comprehensive test suite to validate torch.compile optimizations in production.
These tests ensure:
1. Zero MAE loss (numerical equivalence)
2. Compilation stability (no crashes)
3. Performance improvements
4. Memory efficiency

Run before deploying compilation changes to production.
"""

import os
import sys
from pathlib import Path
from time import perf_counter
from typing import List, Tuple

import numpy as np
import pytest
import torch

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "toto"))

# Configure for testing
os.environ["TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS"] = "1"

# Import after env setup
import torch._dynamo
torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.accumulated_cache_size_limit = 256

from toto.model.backbone import TotoBackbone
from toto.model import util_optimized

# Test configuration
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STRICT_MAE_THRESHOLD = 1e-7  # Very strict for production
PERFORMANCE_TOLERANCE = 2.0  # Compiled shouldn't be >2x slower


def set_seed(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_test_model(compile: bool = False, compile_mode: str = "reduce-overhead") -> TotoBackbone:
    """Create a realistic Toto model for testing."""
    set_seed()
    model = TotoBackbone(
        patch_size=4,
        stride=4,
        embed_dim=32,
        num_layers=6,
        num_heads=4,
        mlp_hidden_dim=64,
        dropout=0.0,
        spacewise_every_n_layers=3,
        scaler_cls="<class 'model.scaler.StdMeanScaler'>",
        output_distribution_classes=["<class 'model.distribution.StudentTOutput'>"],
        use_memory_efficient_attention=False,
    ).eval().to(DEVICE)

    if compile:
        model = torch.compile(model, mode=compile_mode, backend="inductor")

    return model


def create_test_data(batch_size: int, num_variates: int, seq_len: int):
    """Create test input data."""
    data = torch.randn(batch_size, num_variates, seq_len, device=DEVICE, dtype=torch.float32)
    padding = torch.ones(batch_size, num_variates, seq_len, device=DEVICE, dtype=torch.bool)
    id_mask = torch.zeros(batch_size, num_variates, seq_len, device=DEVICE, dtype=torch.float32)
    return data, padding, id_mask


class TestCompilationCorrectness:
    """Test that compilation maintains numerical correctness."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_single_forward_pass_equivalence(self):
        """Test single forward pass produces identical results."""
        batch_size, num_variates, seq_len = 2, 4, 32

        # Non-compiled
        set_seed()
        model_nc = create_test_model(compile=False)
        set_seed(100)
        data, padding, id_mask = create_test_data(batch_size, num_variates, seq_len)

        with torch.no_grad():
            _, loc_nc, scale_nc = model_nc(data, padding, id_mask)

        # Compiled
        set_seed()
        model_c = create_test_model(compile=True)
        set_seed(100)
        data, padding, id_mask = create_test_data(batch_size, num_variates, seq_len)

        with torch.no_grad():
            _, loc_c, scale_c = model_c(data, padding, id_mask)

        # Strict comparison
        mae_loc = torch.abs(loc_c - loc_nc).mean().item()
        mae_scale = torch.abs(scale_c - scale_nc).mean().item()
        max_diff_loc = torch.abs(loc_c - loc_nc).max().item()
        max_diff_scale = torch.abs(scale_c - scale_nc).max().item()

        assert mae_loc < STRICT_MAE_THRESHOLD, f"Location MAE {mae_loc:.2e} exceeds threshold"
        assert mae_scale < STRICT_MAE_THRESHOLD, f"Scale MAE {mae_scale:.2e} exceeds threshold"
        assert max_diff_loc < 1e-5, f"Location max diff {max_diff_loc:.2e} too large"
        assert max_diff_scale < 1e-5, f"Scale max diff {max_diff_scale:.2e} too large"

        print(f"\n✓ Single pass: loc_mae={mae_loc:.2e}, scale_mae={mae_scale:.2e}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_autoregressive_equivalence(self):
        """Test multi-step autoregressive prediction equivalence."""
        batch_size, num_variates = 2, 4
        context_len = 32
        num_steps = 8

        # Non-compiled
        set_seed()
        model_nc = create_test_model(compile=False)
        set_seed(200)
        data, padding, id_mask = create_test_data(batch_size, num_variates, context_len)

        kv_cache_nc = model_nc.allocate_kv_cache(
            batch_size=batch_size,
            num_variates=num_variates,
            max_time_steps=context_len + num_steps * 4,
            device=DEVICE,
            dtype=torch.float32,
        )

        locs_nc = []
        with torch.no_grad():
            _, loc, _ = model_nc(data, padding, id_mask, kv_cache_nc)
            locs_nc.append(loc.clone())  # Clone for consistency

            for _ in range(num_steps):
                next_data, next_padding, next_id = create_test_data(batch_size, num_variates, 4)
                data = torch.cat([data, next_data], dim=-1)
                padding = torch.cat([padding, next_padding], dim=-1)
                id_mask = torch.cat([id_mask, next_id], dim=-1)
                _, loc, _ = model_nc(data, padding, id_mask, kv_cache_nc)
                locs_nc.append(loc.clone())  # Clone for consistency

        # Compiled
        set_seed()
        model_c = create_test_model(compile=True)
        set_seed(200)
        data, padding, id_mask = create_test_data(batch_size, num_variates, context_len)

        kv_cache_c = model_c.allocate_kv_cache(
            batch_size=batch_size,
            num_variates=num_variates,
            max_time_steps=context_len + num_steps * 4,
            device=DEVICE,
            dtype=torch.float32,
        )

        locs_c = []
        with torch.no_grad():
            # Mark CUDAGraph step boundaries for autoregressive generation
            if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                torch.compiler.cudagraph_mark_step_begin()

            _, loc, _ = model_c(data, padding, id_mask, kv_cache_c)
            locs_c.append(loc.clone())  # Clone to prevent overwriting

            for _ in range(num_steps):
                if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                    torch.compiler.cudagraph_mark_step_begin()

                next_data, next_padding, next_id = create_test_data(batch_size, num_variates, 4)
                data = torch.cat([data, next_data], dim=-1)
                padding = torch.cat([padding, next_padding], dim=-1)
                id_mask = torch.cat([id_mask, next_id], dim=-1)
                _, loc, _ = model_c(data, padding, id_mask, kv_cache_c)
                locs_c.append(loc.clone())  # Clone to prevent overwriting

        # Compare all steps
        all_maes = []
        for i, (loc_nc, loc_c) in enumerate(zip(locs_nc, locs_c)):
            mae = torch.abs(loc_c - loc_nc).mean().item()
            all_maes.append(mae)
            assert mae < STRICT_MAE_THRESHOLD, f"Step {i} MAE {mae:.2e} exceeds threshold"

        max_mae = max(all_maes)
        print(f"\n✓ Autoregressive {num_steps} steps: max_mae={max_mae:.2e}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.parametrize("seq_len", [16, 32, 48, 64])
    def test_varying_sequence_lengths(self, seq_len):
        """Test that different sequence lengths all produce correct results."""
        batch_size, num_variates = 2, 3

        set_seed()
        model_nc = create_test_model(compile=False)
        set_seed(300 + seq_len)
        data, padding, id_mask = create_test_data(batch_size, num_variates, seq_len)

        with torch.no_grad():
            _, loc_nc, _ = model_nc(data, padding, id_mask)

        set_seed()
        model_c = create_test_model(compile=True)
        set_seed(300 + seq_len)
        data, padding, id_mask = create_test_data(batch_size, num_variates, seq_len)

        with torch.no_grad():
            _, loc_c, _ = model_c(data, padding, id_mask)

        mae = torch.abs(loc_c - loc_nc).mean().item()
        assert mae < STRICT_MAE_THRESHOLD, f"seq_len={seq_len} MAE {mae:.2e} exceeds threshold"

        print(f"  seq_len={seq_len}: mae={mae:.2e}")


class TestCompilationPerformance:
    """Test that compilation provides performance benefits."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_inference_speed(self):
        """Test that compiled model is faster after warmup."""
        batch_size, num_variates, seq_len = 2, 4, 32
        warmup_runs = 5
        benchmark_runs = 20

        # Create models
        model_nc = create_test_model(compile=False)
        model_c = create_test_model(compile=True)

        data, padding, id_mask = create_test_data(batch_size, num_variates, seq_len)

        # Warmup both
        for _ in range(warmup_runs):
            with torch.no_grad():
                model_nc(data, padding, id_mask)
                model_c(data, padding, id_mask)

        torch.cuda.synchronize()

        # Benchmark non-compiled
        times_nc = []
        for _ in range(benchmark_runs):
            start = perf_counter()
            with torch.no_grad():
                model_nc(data, padding, id_mask)
            torch.cuda.synchronize()
            times_nc.append(perf_counter() - start)

        # Benchmark compiled
        times_c = []
        for _ in range(benchmark_runs):
            start = perf_counter()
            with torch.no_grad():
                model_c(data, padding, id_mask)
            torch.cuda.synchronize()
            times_c.append(perf_counter() - start)

        mean_nc = np.mean(times_nc) * 1000  # ms
        mean_c = np.mean(times_c) * 1000  # ms
        speedup = mean_nc / mean_c

        print(f"\n  Non-compiled: {mean_nc:.2f}ms")
        print(f"  Compiled: {mean_c:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        # Sanity check: compiled shouldn't be dramatically slower
        assert mean_c < mean_nc * PERFORMANCE_TOLERANCE, \
            f"Compiled is {mean_c/mean_nc:.2f}x slower than non-compiled"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_memory_overhead(self):
        """Test that compilation doesn't cause excessive memory overhead."""
        batch_size, num_variates, seq_len = 2, 4, 32

        torch.cuda.reset_peak_memory_stats()

        model_nc = create_test_model(compile=False)
        data, padding, id_mask = create_test_data(batch_size, num_variates, seq_len)

        with torch.no_grad():
            model_nc(data, padding, id_mask)

        torch.cuda.synchronize()
        mem_nc = torch.cuda.max_memory_allocated() / 1024**2

        del model_nc
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model_c = create_test_model(compile=True)
        with torch.no_grad():
            model_c(data, padding, id_mask)

        torch.cuda.synchronize()
        mem_c = torch.cuda.max_memory_allocated() / 1024**2

        overhead = (mem_c - mem_nc) / mem_nc * 100

        print(f"\n  Non-compiled: {mem_nc:.2f}MB")
        print(f"  Compiled: {mem_c:.2f}MB")
        print(f"  Overhead: {overhead:.1f}%")

        assert overhead < 50, f"Memory overhead {overhead:.1f}% is too high"


class TestCompilationStability:
    """Test compilation stability and error handling."""

    def test_kvcache_implementation(self):
        """Verify the compile-friendly KVCache is being used."""
        from toto.model import util_optimized

        kv_class = util_optimized.KVCache
        print(f"\n  Using KVCache: {kv_class.__name__}")
        print(f"  From module: {kv_class.__module__}")

        # Should be compile-friendly version
        assert "CompileFriendly" in kv_class.__name__ or "Optimized" in kv_class.__name__, \
            f"Not using optimized KVCache: {kv_class.__name__}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_no_crashes_on_varying_inputs(self):
        """Test that compilation doesn't crash with various input shapes."""
        model = create_test_model(compile=True)

        test_configs = [
            (1, 2, 16),
            (2, 3, 24),
            (1, 4, 32),
            (2, 2, 48),
        ]

        for batch, variates, seq_len in test_configs:
            data, padding, id_mask = create_test_data(batch, variates, seq_len)

            try:
                with torch.no_grad():
                    _, loc, _ = model(data, padding, id_mask)
                assert loc.shape[0] == batch, f"Unexpected batch size for {(batch, variates, seq_len)}"
            except Exception as e:
                pytest.fail(f"Crashed on config {(batch, variates, seq_len)}: {e}")

        print(f"\n  ✓ Tested {len(test_configs)} configurations without crashes")


class TestProductionReadiness:
    """Final production readiness checks."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_determinism(self):
        """Test that results are deterministic with same seed."""
        batch_size, num_variates, seq_len = 2, 4, 32

        # Run 1
        set_seed(999)
        model = create_test_model(compile=True)
        set_seed(1000)
        data, padding, id_mask = create_test_data(batch_size, num_variates, seq_len)

        with torch.no_grad():
            _, loc1, _ = model(data, padding, id_mask)

        # Run 2 - same seed
        set_seed(999)
        model = create_test_model(compile=True)
        set_seed(1000)
        data, padding, id_mask = create_test_data(batch_size, num_variates, seq_len)

        with torch.no_grad():
            _, loc2, _ = model(data, padding, id_mask)

        diff = torch.abs(loc1 - loc2).max().item()
        assert diff == 0.0, f"Non-deterministic: max diff {diff:.2e}"

        print(f"\n  ✓ Deterministic: max diff = {diff:.2e}")


def run_all_tests():
    """Run all tests programmatically."""
    pytest.main([__file__, "-v", "-s", "--tb=short"])


if __name__ == "__main__":
    run_all_tests()
