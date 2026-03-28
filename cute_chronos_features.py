"""CuteChronos2 integration for fast forecast feature generation.

Drop-in replacement for chronos_forecasting.ChronosPipeline using
CuteChronos2's Triton kernels for 1.4x-24x speedup.

Usage:
    from pufferlib_market.cute_chronos_features import build_pipeline, predict_quantiles

    pipeline = build_pipeline(model_path="amazon/chronos-t5-small")
    # or: pipeline = build_pipeline(model_path="/path/to/local/model")

    # predictions: [batch, num_quantiles]
    predictions = predict_quantiles(pipeline, context_tensor, prediction_length=24)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Path resolution — find cutedsl in priority order
# ---------------------------------------------------------------------------


def _find_cutedsl() -> Path | None:
    """Find cutedsl in priority order:

    1. Symlink at repo_root/cutedsl (canonical dev setup via ``make setup-cutedsl``)
    2. ../cutedsl relative to repo root (adjacent directory)
    3. Sibling of grandparent directories

    The repo root is identified by the presence of ``pyproject.toml``.
    """
    here = Path(__file__).resolve()
    # Walk up candidate parents: pufferlib_market/, repo_root/, one above
    for parent in (here.parent, here.parent.parent, here.parent.parent.parent):
        candidate = parent / "cutedsl"
        if candidate.exists() and (candidate / "pyproject.toml").exists():
            return candidate
    # Also check one level above the deepest parent tried
    candidate = here.parent.parent.parent.parent / "cutedsl"
    if candidate.exists() and (candidate / "pyproject.toml").exists():
        return candidate
    return None


_CUTEDSL_PATH = _find_cutedsl()


def _add_cutedsl_to_path() -> bool:
    """Insert cutedsl into sys.path if it exists and is not already there.

    Returns True if the path was added (or already present and valid).
    """
    if _CUTEDSL_PATH is not None and str(_CUTEDSL_PATH) not in sys.path:
        sys.path.insert(0, str(_CUTEDSL_PATH))
        return True
    return _CUTEDSL_PATH is not None


# Eagerly probe availability so callers can do:
#   from pufferlib_market.cute_chronos_features import CUTE_AVAILABLE
_add_cutedsl_to_path()
try:
    from cutechronos.pipeline import CuteChronos2Pipeline as _CCP  # type: ignore[import]  # noqa: F401

    CUTE_AVAILABLE: bool = True
except Exception:
    CUTE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def build_pipeline(model_path: str, device: str = "auto") -> Any:
    """Build a CuteChronos2 pipeline, falling back to standard ChronosPipeline.

    Tries to import from cutedsl/cutechronos first.  Falls back to
    chronos_forecasting if CuteChronos2 is not available (e.g., on pods
    without cutedsl installed).

    Parameters
    ----------
    model_path:
        HuggingFace model id (e.g. ``"amazon/chronos-t5-small"``) or a local
        directory path containing a compatible Chronos2 checkpoint.
    device:
        Target device.  ``"auto"`` resolves to ``"cuda"`` when a GPU is
        available, otherwise ``"cpu"``.

    Returns
    -------
    A pipeline object that exposes a ``predict`` / ``predict_quantiles``
    interface compatible with both CuteChronos2Pipeline and ChronosPipeline.
    """
    resolved_device = _resolve_device(device)

    try:
        from cutechronos.pipeline import CuteChronos2Pipeline  # type: ignore[import]

        pipeline = CuteChronos2Pipeline.from_pretrained(
            model_path,
            device=resolved_device,
        )
        return pipeline
    except Exception:
        pass

    # Fallback: upstream chronos_forecasting / chronos package
    try:
        from chronos import ChronosPipeline  # type: ignore[import]

        pipeline = ChronosPipeline.from_pretrained(
            model_path,
            device_map=resolved_device,
        )
        return pipeline
    except Exception:
        pass

    raise ImportError(
        "Neither cutechronos nor chronos_forecasting could be imported. "
        "Install one of them:\n"
        "  uv pip install git+https://github.com/amazon-science/chronos-forecasting\n"
        "  (or ensure cutedsl exists: run 'make setup-cutedsl' in the repo root)"
    )


def _resolve_device(device: str) -> str:
    """Resolve ``'auto'`` to a concrete device string."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------


def predict_quantiles(
    pipeline: Any,
    context: torch.Tensor,
    prediction_length: int = 24,
    num_samples: int = 20,
    quantile_levels: list[float] | None = None,
) -> torch.Tensor:
    """Run *pipeline* and return quantile predictions shaped ``[batch, n_quantiles]``.

    Handles both CuteChronos2Pipeline and the original ChronosPipeline from
    chronos_forecasting.  The returned tensor always has shape
    ``[batch, len(quantile_levels)]``, aggregating over the prediction horizon
    by taking the median across time steps.

    Parameters
    ----------
    pipeline:
        Pipeline object returned by :func:`build_pipeline`.
    context:
        Float tensor of shape ``[batch, seq_len]`` or ``[seq_len]`` (1-D is
        treated as a single batch item).
    prediction_length:
        Forecast horizon in time steps.
    num_samples:
        Number of Monte-Carlo samples (only used by the original
        ChronosPipeline; CuteChronos2Pipeline is deterministic).
    quantile_levels:
        Quantile levels to return.  Defaults to ``[0.1, 0.5, 0.9]``.

    Returns
    -------
    torch.Tensor of shape ``[batch, len(quantile_levels)]``.
    """
    if quantile_levels is None:
        quantile_levels = [0.1, 0.5, 0.9]

    if context.ndim == 1:
        context = context.unsqueeze(0)

    if is_cute_chronos(pipeline):
        # CuteChronos2Pipeline.predict_quantiles returns
        # (list_of_quantile_tensors, list_of_mean_tensors)
        # each element in the list: (1, H, Q)
        quantile_list, _ = pipeline.predict_quantiles(
            context,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            limit_prediction_length=False,
        )
        # Stack list of (1, H, Q) → (B, H, Q), then median over H
        stacked = torch.cat(quantile_list, dim=0)  # (B, H, Q)
        result = stacked.median(dim=1).values        # (B, Q)
        return result
    else:
        # Original ChronosPipeline.predict returns list of (num_samples, H) tensors
        # We compute empirical quantiles from the samples.
        predictions = pipeline.predict(
            context,
            prediction_length=prediction_length,
            num_samples=num_samples,
            limit_prediction_length=False,
        )
        # predictions: list of length B, each (num_samples, H)
        rows = []
        q_tensor = torch.tensor(quantile_levels, dtype=torch.float32)
        for pred in predictions:
            # pred: (num_samples, H) — aggregate over H then compute quantiles
            per_sample = pred.float().mean(dim=-1)  # (num_samples,)
            q_vals = torch.quantile(per_sample, q_tensor)  # (Q,)
            rows.append(q_vals)
        result = torch.stack(rows, dim=0)  # (B, Q)
        return result


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def is_cute_chronos(pipeline: Any) -> bool:
    """Return True if *pipeline* is a CuteChronos2Pipeline instance."""
    return type(pipeline).__module__.startswith("cutechronos")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def benchmark(model_path: str = "amazon/chronos-t5-small", n_iter: int = 10) -> dict[str, Any]:
    """Compare CuteChronos2 vs original ChronosPipeline latency.

    Loads both pipelines (if available) and measures wall-clock time for
    ``n_iter`` forward passes with a fixed synthetic context.

    Returns a dict with keys:
        cute_ms     — median latency per call for CuteChronos2 (ms), or None
        original_ms — median latency per call for ChronosPipeline (ms), or None
        speedup     — cute_ms / original_ms ratio, or None
    """
    context = torch.randn(4, 512)
    results: dict[str, Any] = {"cute_ms": None, "original_ms": None, "speedup": None}

    # --- CuteChronos2 ---
    try:
        from cutechronos.pipeline import CuteChronos2Pipeline  # type: ignore[import]

        cute_pipe = CuteChronos2Pipeline.from_pretrained(model_path)
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            cute_pipe.predict(context, prediction_length=24, limit_prediction_length=False)
            times.append((time.perf_counter() - t0) * 1000)
        results["cute_ms"] = float(sorted(times)[n_iter // 2])
    except Exception as exc:
        results["cute_error"] = str(exc)

    # --- Original ChronosPipeline ---
    try:
        from chronos import ChronosPipeline  # type: ignore[import]

        orig_pipe = ChronosPipeline.from_pretrained(model_path, device_map="auto")
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            orig_pipe.predict(context, prediction_length=24, num_samples=20, limit_prediction_length=False)
            times.append((time.perf_counter() - t0) * 1000)
        results["original_ms"] = float(sorted(times)[n_iter // 2])
    except Exception as exc:
        results["original_error"] = str(exc)

    if results["cute_ms"] is not None and results["original_ms"] is not None:
        results["speedup"] = results["original_ms"] / results["cute_ms"]

    return results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="CuteChronos2 feature pipeline — benchmark and diagnostics"
    )
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    parser.add_argument(
        "--model",
        default="amazon/chronos-t5-small",
        help="Model id or local path (default: amazon/chronos-t5-small)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print diagnostics without loading any model",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=10,
        help="Number of benchmark iterations (default: 10)",
    )
    args = parser.parse_args()

    avail_str = f"Yes (path: {_CUTEDSL_PATH})" if CUTE_AVAILABLE else f"No (searched from {Path(__file__).resolve().parent})"
    print(f"CuteChronos2 available: {avail_str}")

    if args.dry_run:
        print("Dry run: skipping model load")
        print(f"Would benchmark: {args.model}")
        print("  CuteChronos2 vs ChronosPipeline latency comparison")
        print("  Expected speedup: 1.4x-24x (GPU-dependent)")
        return

    if args.benchmark:
        print(f"Benchmarking {args.model} ({args.n_iter} iterations)...")
        results = benchmark(model_path=args.model, n_iter=args.n_iter)
        if results["cute_ms"] is not None:
            print(f"  CuteChronos2:    {results['cute_ms']:.1f} ms/call")
        else:
            print(f"  CuteChronos2:    unavailable ({results.get('cute_error', 'unknown error')})")
        if results["original_ms"] is not None:
            print(f"  ChronosPipeline: {results['original_ms']:.1f} ms/call")
        else:
            print(f"  ChronosPipeline: unavailable ({results.get('original_error', 'unknown error')})")
        if results["speedup"] is not None:
            print(f"  Speedup:         {results['speedup']:.2f}x")
        return

    parser.print_help()


if __name__ == "__main__":
    _main()
