"""Batch size auto-tuning utilities for faltrain sweeps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from faltrain.logger_utils import std_logger

LOG = std_logger(__name__)

_CACHE: Dict[str, int] = {}
_CACHE_LOCK = Lock()
_PERSISTED: Dict[str, Dict[str, Any]] = {}
_PERSIST_LOCK = Lock()
_PERSIST_PATHS: Tuple[Path, ...] = tuple(
    dict.fromkeys(
        (
            Path("/data/params/best_hyperparams.json"),
            Path(__file__).resolve().parents[1]
            / "hyperparamstore"
            / "best_hyper_params.json",
        )
    )
)
_TORCH: Optional[ModuleType] = None
_NUMPY: Optional[ModuleType] = None


@dataclass(frozen=True)
class BatchSizeSelection:
    signature: Optional[str]
    selected: int
    descending_candidates: Tuple[int, ...]
    user_candidates: Tuple[int, ...]
    context_length: int
    horizon: int
    exhaustive: bool

    def sweep_values(self) -> Tuple[int, ...]:
        if self.exhaustive:
            return self.descending_candidates
        return (self.selected,)

    def fallback_values(self, start: Optional[int] = None) -> List[int]:
        """Return descending batch sizes starting at ``start`` or the selected value."""
        reference = self.selected if start is None else int(start)
        values: List[int] = []
        for value in self.descending_candidates:
            if value <= reference:
                values.append(value)
        return values

    def meta(self) -> Dict[str, Any]:
        return {
            "signature": self.signature,
            "candidates_desc": list(self.descending_candidates),
            "candidates_user": list(self.user_candidates),
            "context_length": self.context_length,
            "horizon": self.horizon,
            "exhaustive": self.exhaustive,
        }


def _load_from_path(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with path.open("r") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        LOG.warning("Failed to load persisted batch sizes from %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for key, value in data.items():
        if isinstance(value, dict) and "batch_size" in value:
            result[key] = value
    return result


def _load_persisted() -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for path in reversed(_PERSIST_PATHS):
        merged.update(_load_from_path(path))
    return merged


def _normalise_candidates(
    candidates: Sequence[int],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    seen: Dict[int, None] = {}
    user_sequence: List[int] = []
    for value in candidates:
        ivalue = int(value)
        if ivalue not in seen:
            seen[ivalue] = None
            user_sequence.append(ivalue)
    if not user_sequence:
        return (), ()
    descending_candidates = tuple(sorted(user_sequence, reverse=True))
    user_candidates = tuple(user_sequence)
    return descending_candidates, user_candidates


def _persist_signature(
    signature: str,
    *,
    batch_size: int,
    context_length: int,
    horizon: int,
) -> None:
    with _PERSIST_LOCK:
        payload = dict(_PERSISTED)
        payload[signature] = {
            "batch_size": int(batch_size),
            "context_length": int(context_length),
            "horizon": int(horizon),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        for path in _PERSIST_PATHS:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix:
                    tmp_path = path.with_suffix(path.suffix + ".tmp")
                else:
                    tmp_path = path.with_name(path.name + ".tmp")
                with tmp_path.open("w") as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True)
                tmp_path.replace(path)
            except Exception as exc:
                LOG.warning("Failed to persist batch size to %s: %s", path, exc)
        _PERSISTED.clear()
        _PERSISTED.update(payload)


with _PERSIST_LOCK:
    _PERSISTED.update(_load_persisted())


def setup_training_imports(torch_module: Optional[ModuleType], numpy_module: Optional[ModuleType]) -> None:
    """Allow external callers to inject torch/numpy modules before usage."""
    global _TORCH, _NUMPY
    if torch_module is not None:
        _TORCH = torch_module
    if numpy_module is not None:
        _NUMPY = numpy_module


def persist_batch_size(
    selection: BatchSizeSelection,
    *,
    batch_size: Optional[int] = None,
) -> None:
    """Persist the working ``batch_size`` for the selection."""
    if selection.signature is None:
        return
    chosen = selection.selected if batch_size is None else int(batch_size)
    _persist_signature(
        selection.signature,
        batch_size=chosen,
        context_length=selection.context_length,
        horizon=selection.horizon,
    )
    with _CACHE_LOCK:
        _CACHE[selection.signature] = chosen


def get_cached_batch_selection(
    *,
    candidates: Sequence[int],
    context_lengths: Sequence[int],
    horizons: Sequence[int],
) -> Optional[BatchSizeSelection]:
    descending_candidates, user_candidates = _normalise_candidates(candidates)
    if not descending_candidates:
        raise ValueError("SweepSpace.batch_sizes must contain at least one value")

    max_context = _max_or_default(context_lengths)
    max_horizon = _max_or_default(horizons)

    if len(descending_candidates) == 1:
        return BatchSizeSelection(
            signature=None,
            selected=descending_candidates[0],
            descending_candidates=descending_candidates,
            user_candidates=user_candidates,
            context_length=max_context,
            horizon=max_horizon,
            exhaustive=False,
        )

    torch_mod = _load_torch()
    if torch_mod is None:
        return None

    try:
        if not torch_mod.cuda.is_available():
            return None
    except AttributeError:
        return None

    try:
        device_index = torch_mod.cuda.current_device()
    except Exception:
        device_index = 0

    try:
        device_name = torch_mod.cuda.get_device_name(device_index)
    except Exception:
        device_name = f"cuda:{device_index}"

    signature = _device_signature(torch_mod, device_index, device_name)
    with _PERSIST_LOCK:
        persisted = _PERSISTED.get(signature)

    if not persisted:
        return None

    persisted_bs = persisted.get("batch_size")
    persisted_context = int(persisted.get("context_length", 1))
    persisted_horizon = int(persisted.get("horizon", 1))
    if not (
        isinstance(persisted_bs, int)
        and persisted_bs in descending_candidates
        and persisted_context >= max_context
        and persisted_horizon >= max_horizon
    ):
        return None

    with _CACHE_LOCK:
        _CACHE[signature] = persisted_bs

    return BatchSizeSelection(
        signature=signature,
        selected=persisted_bs,
        descending_candidates=descending_candidates,
        user_candidates=user_candidates,
        context_length=max_context,
        horizon=max_horizon,
        exhaustive=False,
    )


def auto_tune_batch_sizes(
    *,
    candidates: Sequence[int],
    context_lengths: Sequence[int],
    horizons: Sequence[int],
    auto_tune: bool = True,
    safety_margin: float = 0.8,
) -> BatchSizeSelection:
    """Return batch-size selection metadata tailored to the current CUDA device.

    When ``auto_tune`` is ``True`` and multiple candidates are provided, the tuner
    estimates memory requirements using a monotonic heuristic and applies a
    binary search to select the largest feasible batch size. Results are cached
    per device signature so repeat sweeps become free. The returned
    :class:`BatchSizeSelection` exposes both the chosen batch size and the
    ordered candidate list for fallback handling.
    """

    descending_candidates, user_candidates = _normalise_candidates(candidates)
    if not descending_candidates:
        raise ValueError("SweepSpace.batch_sizes must contain at least one value")
    ascending_candidates = tuple(sorted(descending_candidates))
    max_context = _max_or_default(context_lengths)
    max_horizon = _max_or_default(horizons)

    if len(descending_candidates) == 1:
        selected = descending_candidates[0]
        return BatchSizeSelection(
            signature=None,
            selected=selected,
            descending_candidates=descending_candidates,
            user_candidates=user_candidates,
            context_length=max_context,
            horizon=max_horizon,
            exhaustive=False,
        )

    if not auto_tune:
        return BatchSizeSelection(
            signature=None,
            selected=descending_candidates[0],
            descending_candidates=descending_candidates,
            user_candidates=user_candidates,
            context_length=max_context,
            horizon=max_horizon,
            exhaustive=True,
        )

    torch_mod = _load_torch()
    if torch_mod is None:
        LOG.debug("PyTorch is not installed; skipping batch-size auto-tuning")
        return BatchSizeSelection(
            signature=None,
            selected=descending_candidates[0],
            descending_candidates=descending_candidates,
            user_candidates=user_candidates,
            context_length=max_context,
            horizon=max_horizon,
            exhaustive=True,
        )

    try:
        if not torch_mod.cuda.is_available():
            LOG.debug("CUDA is not available; skipping batch-size auto-tuning")
            return BatchSizeSelection(
                signature=None,
                selected=descending_candidates[0],
                descending_candidates=descending_candidates,
                user_candidates=user_candidates,
                context_length=max_context,
                horizon=max_horizon,
                exhaustive=True,
            )
    except AttributeError:
        LOG.debug("torch.cuda is not usable; skipping batch-size auto-tuning")
        return BatchSizeSelection(
            signature=None,
            selected=descending_candidates[0],
            descending_candidates=descending_candidates,
            user_candidates=user_candidates,
            context_length=max_context,
            horizon=max_horizon,
            exhaustive=True,
        )

    try:
        device_index = torch_mod.cuda.current_device()
    except Exception:
        device_index = 0

    try:
        device_name = torch_mod.cuda.get_device_name(device_index)
    except Exception:
        device_name = f"cuda:{device_index}"

    signature = _device_signature(torch_mod, device_index, device_name)

    with _PERSIST_LOCK:
        persisted = _PERSISTED.get(signature)
    if persisted:
        persisted_bs = persisted.get("batch_size")
        persisted_context = int(persisted.get("context_length", 1))
        persisted_horizon = int(persisted.get("horizon", 1))
        if (
            isinstance(persisted_bs, int)
            and persisted_bs in descending_candidates
            and persisted_context >= max_context
            and persisted_horizon >= max_horizon
        ):
            LOG.info("Using persisted batch size %s for %s", persisted_bs, signature)
            with _CACHE_LOCK:
                _CACHE[signature] = persisted_bs
            return BatchSizeSelection(
                signature=signature,
                selected=persisted_bs,
                descending_candidates=descending_candidates,
                user_candidates=user_candidates,
                context_length=max_context,
                horizon=max_horizon,
                exhaustive=False,
            )

    with _CACHE_LOCK:
        cached = _CACHE.get(signature)
    if cached is not None and cached in descending_candidates:
        LOG.debug("Using cached batch size %s for %s", cached, signature)
        return BatchSizeSelection(
            signature=signature,
            selected=cached,
            descending_candidates=descending_candidates,
            user_candidates=user_candidates,
            context_length=max_context,
            horizon=max_horizon,
            exhaustive=False,
        )

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
        return BatchSizeSelection(
            signature=signature,
            selected=descending_candidates[0],
            descending_candidates=descending_candidates,
            user_candidates=user_candidates,
            context_length=max_context,
            horizon=max_horizon,
            exhaustive=True,
        )

    best = _binary_search(ascending_candidates, tester.supports)
    with _CACHE_LOCK:
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
    return BatchSizeSelection(
        signature=signature,
        selected=best,
        descending_candidates=descending_candidates,
        user_candidates=user_candidates,
        context_length=max_context,
        horizon=max_horizon,
        exhaustive=False,
    )


def _load_torch():
    global _TORCH
    if _TORCH is not None:
        return _TORCH
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    _TORCH = torch
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

        try:
            mem_info = torch_mod.cuda.mem_get_info(device_index)
        except TypeError:
            mem_info = torch_mod.cuda.mem_get_info()
        except Exception:
            mem_info = None

        if isinstance(mem_info, (tuple, list)) and len(mem_info) == 2:
            free_bytes, total_bytes = mem_info
            try:
                free_bytes_int = int(free_bytes)
                if free_bytes_int > 0:
                    self._budget_bytes = min(
                        self._budget_bytes, int(free_bytes_int * margin)
                    )
            except (TypeError, ValueError):
                pass
            try:
                total_bytes_int = int(total_bytes)
                if total_bytes_int > 0:
                    self._budget_bytes = min(
                        self._budget_bytes, int(total_bytes_int * margin)
                    )
            except (TypeError, ValueError):
                pass

    def supports(self, batch_size: int) -> bool:
        required = self._estimate_bytes(batch_size)
        return required <= self._budget_bytes

    def _estimate_bytes(self, batch_size: int) -> int:
        sequence = self._context_length + self._horizon
        activation = batch_size * sequence * self.MODEL_WIDTH * self.DTYPE_BYTES
        gradients = activation
        optimizer = activation // 2
        return activation + gradients + optimizer


__all__ = [
    "BatchSizeSelection",
    "auto_tune_batch_sizes",
    "persist_batch_size",
    "get_cached_batch_selection",
    "setup_training_imports",
]
