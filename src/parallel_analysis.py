"""
Parallel analysis optimizations for trade_stock_e2e.py

This module provides:
1. Model warmup to pre-compile torch kernels
2. Parallel symbol analysis using ThreadPoolExecutor (GPU-safe)
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Callable, Any
import torch

logger = logging.getLogger(__name__)


def warmup_models(
    load_toto: Callable,
    load_kronos: Optional[Callable] = None,
    toto_context_length: int = 512,
) -> None:
    """
    Warm up models by running dummy inference to pre-compile torch kernels.

    This eliminates the ~40 second first-inference penalty by triggering
    kernel compilation upfront with dummy data.

    Args:
        load_toto: Function to load Toto pipeline (returns TotoPipeline)
        load_kronos: Optional function to load Kronos (returns KronosWrapper)
        toto_context_length: Context length for Toto warmup inference
    """
    logger.info("=" * 80)
    logger.info("MODEL WARMUP: Pre-compiling torch kernels...")
    logger.info("=" * 80)

    warmup_start = time.time()

    # Warmup Toto
    if load_toto:
        try:
            logger.info("Warming up Toto pipeline...")
            toto_start = time.time()

            pipeline = load_toto()

            # Run dummy inference to trigger compilation
            with torch.no_grad():
                # Create dummy context tensor
                dummy_context = torch.randn(
                    1, toto_context_length,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )

                # Trigger inference (this compiles kernels)
                _ = pipeline(
                    context=dummy_context,
                    prediction_length=96,
                    num_samples=2,  # Minimal samples for warmup
                )

            toto_elapsed = time.time() - toto_start
            logger.info(f"✓ Toto warmup complete: {toto_elapsed:.1f}s")

        except Exception as e:
            logger.warning(f"Toto warmup failed (non-fatal): {e}")

    # Warmup Kronos (if provided)
    if load_kronos:
        try:
            logger.info("Warming up Kronos model...")
            kronos_start = time.time()

            # Load Kronos with default params
            default_params = {
                "temperature": 1.0,
                "top_p": 0.9,
                "top_k": 50,
                "sample_count": 100,
                "max_context": 512,
                "clip": 100.0,
            }
            wrapper = load_kronos(default_params)

            # Run dummy forecast
            dummy_data = torch.randn(100, device='cuda' if torch.cuda.is_available() else 'cpu')
            _ = wrapper.forecast(dummy_data, prediction_length=24)

            kronos_elapsed = time.time() - kronos_start
            logger.info(f"✓ Kronos warmup complete: {kronos_elapsed:.1f}s")

        except Exception as e:
            logger.warning(f"Kronos warmup failed (non-fatal): {e}")

    total_elapsed = time.time() - warmup_start
    logger.info("=" * 80)
    logger.info(f"✓ MODEL WARMUP COMPLETE: {total_elapsed:.1f}s")
    logger.info("  Torch kernels pre-compiled - subsequent inference will be fast")
    logger.info("=" * 80)


def analyze_symbols_parallel(
    symbols: List[str],
    analyze_func: Callable[[str], Dict[str, Any]],
    max_workers: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze symbols in parallel using ThreadPoolExecutor.

    Uses threads (not processes) to safely share GPU models across workers
    while parallelizing I/O and CPU-bound operations.

    Args:
        symbols: List of symbols to analyze
        analyze_func: Function that takes a symbol and returns analysis dict
        max_workers: Number of threads (default: min(32, cpu_count + 4))

    Returns:
        Dict mapping symbol -> analysis results
    """
    if not symbols:
        return {}

    # Determine worker count
    if max_workers is None:
        # ThreadPoolExecutor default formula
        max_workers = min(32, (os.cpu_count() or 1) + 4)

    # For very large CPU counts, cap at a reasonable number
    # Too many threads can cause contention
    max_workers = min(max_workers, 32)

    logger.info(f"Analyzing {len(symbols)} symbols in parallel with {max_workers} workers")

    results = {}
    failed_symbols = []

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_symbol = {
            executor.submit(analyze_func, symbol): symbol
            for symbol in symbols
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            completed += 1

            try:
                result = future.result()
                if result:  # Only store non-empty results
                    results[symbol] = result
                    logger.info(f"[{completed}/{len(symbols)}] ✓ {symbol}")
                else:
                    logger.warning(f"[{completed}/{len(symbols)}] ✗ {symbol} (empty result)")
                    failed_symbols.append(symbol)
            except Exception as e:
                logger.error(f"[{completed}/{len(symbols)}] ✗ {symbol} failed: {e}")
                failed_symbols.append(symbol)

    elapsed = time.time() - start_time
    success_count = len(results)

    logger.info("=" * 80)
    logger.info(f"Parallel analysis complete: {elapsed:.1f}s")
    logger.info(f"  Success: {success_count}/{len(symbols)} symbols")
    logger.info(f"  Failed: {len(failed_symbols)} symbols")
    if failed_symbols:
        logger.info(f"  Failed symbols: {', '.join(failed_symbols[:10])}")
    logger.info(f"  Avg time per symbol: {elapsed/len(symbols):.2f}s")
    logger.info("=" * 80)

    return results


def analyze_single_symbol_wrapper(
    symbol: str,
    analyze_impl: Callable,
    num_simulations: int = 70,
    model_override: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Wrapper for analyzing a single symbol (used by parallel executor).

    This wrapper handles exceptions and ensures proper error logging
    without crashing the entire parallel analysis.

    Args:
        symbol: Symbol to analyze
        analyze_impl: The actual analysis implementation function
        num_simulations: Number of backtest simulations
        model_override: Optional model override

    Returns:
        Analysis result dict or None if failed
    """
    try:
        logger.debug(f"Starting analysis for {symbol}")

        # Call the actual implementation
        # This should call backtest_forecasts internally
        result = analyze_impl(symbol, num_simulations, model_override)

        return result

    except Exception as e:
        logger.error(f"Symbol {symbol} analysis failed: {e}", exc_info=True)
        return None


# Environment variable to control parallelization
def should_use_parallel() -> bool:
    """Check if parallel analysis is enabled via environment variable."""
    env_value = os.getenv("MARKETSIM_PARALLEL_ANALYSIS", "1").strip().lower()
    return env_value in {"1", "true", "yes", "on"}


def get_parallel_workers() -> int:
    """Get number of parallel workers from environment or auto-detect."""
    env_value = os.getenv("MARKETSIM_PARALLEL_WORKERS")
    if env_value:
        try:
            return max(1, int(env_value))
        except ValueError:
            logger.warning(f"Invalid MARKETSIM_PARALLEL_WORKERS={env_value}, using auto")

    # Auto: Use min(32, cpu_count + 4) - ThreadPoolExecutor default
    return min(32, (os.cpu_count() or 1) + 4)
