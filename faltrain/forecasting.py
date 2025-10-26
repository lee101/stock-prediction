"""Wrapper helpers for Kronos and Toto forecasting within FAL."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch

from src.models.kronos_wrapper import KronosForecastingWrapper
from src.models.toto_wrapper import TotoPipeline

from .hyperparams import HyperparamResolver, HyperparamResult


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return float(default)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return int(default)


def _default_kronos_device(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _default_toto_device(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class KronosWrapperBundle:
    wrapper: KronosForecastingWrapper
    hyperparams: HyperparamResult
    temperature: float
    top_p: float
    top_k: int
    sample_count: int
    max_context: int
    clip: float


@dataclass(frozen=True)
class TotoWrapperBundle:
    pipeline: TotoPipeline
    hyperparams: HyperparamResult
    aggregate: Optional[str]
    num_samples: int
    samples_per_batch: int


def create_kronos_wrapper(
    symbol: str,
    *,
    resolver: Optional[HyperparamResolver] = None,
    device: Optional[str] = None,
    prefer_best: bool = True,
    wrapper_ctor: Optional[Callable[..., KronosForecastingWrapper]] = None,
    **overrides: Any,
) -> KronosWrapperBundle:
    """Instantiate a Kronos wrapper configured with resolved hyperparameters."""

    resolver_obj = resolver or HyperparamResolver()
    result = resolver_obj.load(symbol, "kronos", prefer_best=prefer_best)
    if result is None:
        raise FileNotFoundError(f"Kronos hyperparameters not found for symbol '{symbol}'.")

    config = result.config
    kronos_kwargs: Dict[str, Any] = {
        "model_name": config.get("model_name") or os.getenv("KRONOS_MODEL_NAME", "NeoQuasar/Kronos-base"),
        "tokenizer_name": config.get("tokenizer_name")
        or os.getenv("KRONOS_TOKENIZER_NAME", "NeoQuasar/Kronos-Tokenizer-base"),
        "device": _default_kronos_device(config.get("device") or device),
        "max_context": _coerce_int(config.get("max_context"), 512),
        "clip": _coerce_float(config.get("clip"), 5.0),
        "temperature": _coerce_float(config.get("temperature"), 0.75),
        "top_p": _coerce_float(config.get("top_p"), 0.9),
        "top_k": _coerce_int(config.get("top_k"), 0),
        "sample_count": _coerce_int(config.get("sample_count"), 8),
    }

    if config.get("cache_dir") and "cache_dir" not in overrides:
        kronos_kwargs["cache_dir"] = config.get("cache_dir")

    kronos_kwargs.update(overrides)

    ctor = wrapper_ctor or KronosForecastingWrapper
    wrapper = ctor(**kronos_kwargs)

    return KronosWrapperBundle(
        wrapper=wrapper,
        hyperparams=result,
        temperature=float(kronos_kwargs["temperature"]),
        top_p=float(kronos_kwargs["top_p"]),
        top_k=int(kronos_kwargs["top_k"]),
        sample_count=int(kronos_kwargs["sample_count"]),
        max_context=int(kronos_kwargs["max_context"]),
        clip=float(kronos_kwargs["clip"]),
    )


def create_toto_pipeline(
    symbol: str,
    *,
    resolver: Optional[HyperparamResolver] = None,
    device_map: Optional[str] = None,
    prefer_best: bool = True,
    pipeline_factory: Optional[Callable[..., TotoPipeline]] = None,
    **factory_kwargs: Any,
) -> TotoWrapperBundle:
    """Instantiate a Toto pipeline configured with resolved hyperparameters."""

    resolver_obj = resolver or HyperparamResolver()
    result = resolver_obj.load(symbol, "toto", prefer_best=prefer_best)
    if result is None:
        raise FileNotFoundError(f"Toto hyperparameters not found for symbol '{symbol}'.")

    config = result.config
    aggregate_value = config.get("aggregate")
    aggregate = None
    if isinstance(aggregate_value, str):
        aggregate = aggregate_value.strip() or None
    num_samples = _coerce_int(config.get("num_samples"), 4096)
    samples_per_batch = _coerce_int(config.get("samples_per_batch"), min(512, num_samples))
    if samples_per_batch > num_samples:
        samples_per_batch = num_samples

    effective_device = _default_toto_device(config.get("device") or device_map)
    factory = pipeline_factory or TotoPipeline.from_pretrained
    payload_kwargs = dict(factory_kwargs)
    payload_kwargs.setdefault("device_map", effective_device)
    payload_kwargs.setdefault("torch_dtype", torch.float32)
    payload_kwargs.setdefault("amp_dtype", None)
    payload_kwargs.setdefault("torch_compile", True)
    payload_kwargs.setdefault("compile_mode", "max-autotune")
    payload_kwargs.setdefault("cache_policy", "prefer")

    pipeline = factory(**payload_kwargs)

    return TotoWrapperBundle(
        pipeline=pipeline,
        hyperparams=result,
        aggregate=aggregate,
        num_samples=num_samples,
        samples_per_batch=samples_per_batch,
    )


__all__ = [
    "create_kronos_wrapper",
    "create_toto_pipeline",
    "KronosWrapperBundle",
    "TotoWrapperBundle",
]
