"""Batch size auto-tuning utilities for faltrain sweeps."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

LOG = logging.getLogger(__name__)

_CACHE: Dict[str, int] = {}
_PERSISTED: Dict[str, Dict[str, Any]] = {}
_PERSIST_PATH = Path(__file__).resolve().parents[1] / "hyperparamstore" / "best_hyper_params.json"


def _load_persisted() -> Dict[str, Dict[str, Any]]:
    if not _PERSIST_PATH.exists():
        return {}
    try:
        with _PERSIST_PATH.open("r") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        LOG.warning("Failed to load persisted batch sizes: %s", exc)
        return {}
    if not isinstance(data, dict):
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for key, value in data.items():
        if isinstance(value, dict) and "batch_size" in value:
            result[key] = value
    return result


def _persist_signature(
    signature: str,
    *,
    batch_size: int,
    context_length: int,
    horizon: int,
) -> None:
    _PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(_PERSISTED)
    payload[signature] = {
        "batch_size": int(batch_size),
        "context_length": int(context_length),
        "horizon": int(horizon),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp_path = _PERSIST_PATH.with_suffix(".tmp")
    with tmp_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(_PERSIST_PATH)
    _PERSISTED.update(payload)


_PERSISTED.update(_load_persisted())


def auto_tune_batch_sizes(
    *,
    candidates: Sequence[int],
    context_lengths: Sequence[int],
    horizons: Sequence[int],
    auto_tune: bool = True,
    safety_margin: float = 0.8,
) -> List[int]:
    """Return a batch-size list tailored to the current CUDA device.

    When ``auto_tune`` is ``True`` and multiple candidates are provided, the tuner
    estimates memory requirements using a monotonic heuristic and applies a
    binary search to select the largest feasible batch size. Results are cached
    per device signature so repeat sweeps become free.
    """

    uniq = sorted(set(int(value) for value in candidates))
    if not uniq:
        raise ValueError("SweepSpace.batch_sizes must contain at least one value")
    if len(uniq) == 1 or not auto_tune:
        return uniq

    torch_mod = _load_torch()
    if torch_mod is None:
        LOG.debug("PyTorch is not installed; skipping batch-size auto-tuning")
        return uniq
    try:
        if not torch_mod.cuda.is_available():
            LOG.debug("CUDA is not available; skipping batch-size auto-tuning")
            return uniq
    except AttributeError:
        LOG.debug("torch.cuda.is not usable; skipping batch-size auto-tuning")
        return uniq

    try:
        device_index = torch_mod.cuda.current_device()
    except Exception:
        device_index = 0

    try:
        device_name = torch_mod.cuda.get_device_name(device_index)
    except Exception:
        device_name = f"cuda:{device_index}"

    signature = _device_signature(torch_mod, device_index, device_name)
    max_context = _max_or_default(context_lengths)
    max_horizon = _max_or_default(horizons)

    persisted = _PERSISTED.get(signature)
    if persisted:
        persisted_bs = persisted.get("batch_size")
        persisted_context = int(persisted.get("context_length", 1))
        persisted_horizon = int(persisted.get("horizon", 1))
        if (
            isinstance(persisted_bs, int)
            and persisted_bs in uniq
            and persisted_context >= max_context
            and persisted_horizon >= max_horizon
        ):
            LOG.info("Using persisted batch size %s for %s", persisted_bs, signature)
            _CACHE[signature] = persisted_bs
            return [persisted_bs]

    cached = _CACHE.get(signature)
    if cached is not None and cached in uniq:
        LOG.debug("Using cached batch size %s for %s", cached, signature)
        return [cached]

    try:
        tester = _HeuristicBatchSizeTester(
            torch_mod=torch_mod,
            device_index=device_index,
            context_length=max_context,
            horizon=max_horizon,
            safety_margin=safety_margin,
        )
    except Exception as exc:
        LOG.debug("Batch-size heuristic unavailable (%s); using provided grid", exc)
        return uniq

    best = _binary_search(uniq, tester.supports)
    _CACHE[signature] = best
    try:
        _persist_signature(
            signature,
            batch_size=best,
            context_length=max_context,
            horizon=max_horizon,
        )
    except Exception as exc:
        LOG.warning("Failed to persist batch size %s for %s: %s", best, signature, exc)
    LOG.info("Auto-selected batch size %s for %s", best, signature)
    return [best]


def _load_torch():
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    return torch


def _device_signature(torch_mod, device_index: int, name: str) -> str:
    try:
        props = torch_mod.cuda.get_device_properties(device_index)
        total_memory = getattr(props, "total_memory", None)
    except Exception:
        total_memory = None
    return f"{name}:{total_memory}"


def _max_or_default(values: Iterable[int], default: int = 1) -> int:
    iterator = iter(values)
    try:
        first = next(iterator)
    except StopIteration:
        return max(1, default)
    maximum = first
    for value in iterator:
        if value > maximum:
            maximum = value
    return max(1, maximum)


def _binary_search(candidates: Sequence[int], predicate: Callable[[int], bool]) -> int:
    lo, hi = 0, len(candidates) - 1
    best = candidates[0]
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = candidates[mid]
        try:
            ok = predicate(candidate)
        except Exception as exc:
            LOG.debug("Batch-size predicate raised %s for %s", exc, candidate)
            ok = False
        if ok:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


class _HeuristicBatchSizeTester:
    """Monotonic GPU memory estimator for batch-size feasibility checks."""

    MODEL_WIDTH = 8192
    DTYPE_BYTES = 2  # assume bf16/FP16 activations

    def __init__(
        self,
        *,
        torch_mod,
        device_index: int,
        context_length: int,
        horizon: int,
        safety_margin: float,
    ) -> None:
        self._torch = torch_mod
        self._device_index = device_index
        self._context_length = max(1, context_length)
        self._horizon = max(1, horizon)
        margin = max(0.1, min(0.99, safety_margin))

        props = torch_mod.cuda.get_device_properties(device_index)
        total_memory = getattr(props, "total_memory", None)
        if total_memory is None:
            raise RuntimeError("Unable to determine CUDA total memory")
        self._budget_bytes = int(total_memory * margin)

    def supports(self, batch_size: int) -> bool:
        required = self._estimate_bytes(batch_size)
        return required <= self._budget_bytes

    def _estimate_bytes(self, batch_size: int) -> int:
        sequence = self._context_length + self._horizon
        activation = batch_size * sequence * self.MODEL_WIDTH * self.DTYPE_BYTES
        gradients = activation
        optimizer = activation // 2
        return activation + gradients + optimizer


__all__ = ["auto_tune_batch_sizes"]
