"""
Toto forecasting wrapper that mirrors the Chronos interface while adding
torch.compile options, AMP controls, and GPU-aware retry logic.
"""

from __future__ import annotations

import logging
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, ContextManager, Dict, List, Optional, Union, cast

from src.torch_backend import configure_tf32_backends

from .model_cache import ModelCacheError, ModelCacheManager, dtype_to_token

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANDIDATE_PATHS = [
    _REPO_ROOT / "toto",
    _REPO_ROOT / "toto" / "src",
    _REPO_ROOT / "toto" / "build" / "lib",
    _REPO_ROOT / "toto" / "toto",
    _REPO_ROOT / "totoembedding",
]
_LEGACY_PATH = Path("/mnt/fast/code/chronos-forecasting/toto")
if _LEGACY_PATH.exists():
    _CANDIDATE_PATHS.append(_LEGACY_PATH)

for _path in reversed(_CANDIDATE_PATHS):
    if _path.exists():
        path_str = str(_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

_IMPORT_ERROR: Optional[Exception] = None


def _optional_import(module_name: str) -> ModuleType | None:
    try:
        return import_module(module_name)
    except ModuleNotFoundError:
        return None


torch: ModuleType | None = _optional_import("torch")
np: ModuleType | None = _optional_import("numpy")

if TYPE_CHECKING:
    from numpy import ndarray as NDArray
    import torch as torch_types

    TorchDType = torch_types.dtype
    TorchTensor = torch_types.Tensor
else:  # pragma: no cover - typing fallback when optional deps missing
    NDArray = Any
    TorchDType = Any
    TorchTensor = Any


def setup_toto_wrapper_imports(
    *,
    torch_module: ModuleType | None = None,
    numpy_module: ModuleType | None = None,
    **_: Any,
) -> None:
    global torch, np
    if torch_module is not None:
        torch = torch_module
    if numpy_module is not None:
        np = numpy_module


def _require_torch() -> ModuleType:
    global torch
    if torch is not None:
        return torch
    try:
        torch = import_module("torch")  # type: ignore[assignment]
    except ModuleNotFoundError as exc:
        raise RuntimeError("Torch is unavailable. Call setup_toto_wrapper_imports before use.") from exc
    return torch


def _require_numpy() -> ModuleType:
    global np
    if np is not None:
        return np
    try:
        np = import_module("numpy")  # type: ignore[assignment]
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is unavailable. Call setup_toto_wrapper_imports before use.") from exc
    return np


if TYPE_CHECKING:
    from toto.data.util.dataset import MaskedTimeseries as MaskedTimeseriesType
    from toto.inference.forecaster import TotoForecaster as TotoForecasterType
    from toto.model.toto import Toto as TotoModelType
else:
    MaskedTimeseriesType = Any
    TotoForecasterType = Any
    TotoModelType = Any

try:
    from toto.data.util.dataset import MaskedTimeseries
    from toto.inference.forecaster import TotoForecaster
    from toto.model.toto import Toto
except ModuleNotFoundError:  # pragma: no cover - compatibility with namespace installs
    from toto.toto.data.util.dataset import MaskedTimeseries  # type: ignore
    from toto.toto.inference.forecaster import TotoForecaster  # type: ignore
    from toto.toto.model.toto import Toto  # type: ignore
except Exception as exc:  # pragma: no cover - allow graceful degradation when deps missing
    _IMPORT_ERROR = exc
    MaskedTimeseries = None  # type: ignore
    TotoForecaster = None  # type: ignore
    Toto = None  # type: ignore
else:  # pragma: no cover - executed when imports succeed
    _IMPORT_ERROR = None


logger = logging.getLogger(__name__)

# Enable tensor-core friendly defaults when possible.
if torch is not None:
    configure_tf32_backends(torch, logger=logging.getLogger(__name__))


@dataclass
class TotoForecast:
    """Container for Toto forecast results compatible with Chronos outputs."""

    samples: NDArray

    def numpy(self) -> NDArray:
        """Return samples in Chronos-compatible layout."""
        samples = self.samples

        if samples.ndim == 4 and samples.shape[0] == 1:
            samples = samples.squeeze(0)
        if samples.ndim == 3 and samples.shape[0] == 1:
            samples = samples.squeeze(0)
        if samples.ndim == 2 and samples.shape[0] == 1:
            return samples.squeeze(0)
        if samples.ndim == 2:
            return samples.T
        return samples


def _is_cuda_oom(exc: BaseException) -> bool:
    """Return True if the exception represents a CUDA OOM condition."""
    cuda_mod = getattr(torch, "cuda", None)
    oom_error = getattr(cuda_mod, "OutOfMemoryError", None)
    if oom_error is not None and isinstance(exc, oom_error):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "busy or unavailable" in message or "cuda error" in message


def _maybe_empty_cuda_cache(device: str) -> None:
    cuda_mod = getattr(torch, "cuda", None)
    if (
        device.startswith("cuda")
        and cuda_mod is not None
        and callable(getattr(cuda_mod, "is_available", None))
        and cuda_mod.is_available()
    ):
        try:
            cuda_mod.empty_cache()
        except Exception as cache_exc:  # pragma: no cover - best effort
            logger.debug("Failed to empty CUDA cache after OOM: %s", cache_exc)


def _inference_context() -> ContextManager[None]:
    """Return the best available inference context manager (inference_mode or no_grad)."""
    torch_module = _require_torch()
    context_ctor = getattr(torch_module, "inference_mode", None)
    if callable(context_ctor):
        return cast(ContextManager[None], context_ctor())
    return cast(ContextManager[None], torch_module.no_grad())


def _autocast_context(device: str, dtype: Optional[TorchDType]) -> ContextManager[None]:
    torch_module = _require_torch()
    if dtype is None:
        return cast(ContextManager[None], nullcontext())
    if device.startswith("cuda"):
        autocast_fn = getattr(torch_module, "autocast", None)
        if callable(autocast_fn):
            return cast(ContextManager[None], autocast_fn(device_type="cuda", dtype=dtype))
        cuda_amp = getattr(torch_module, "cuda", None)
        amp_mod = getattr(cuda_amp, "amp", None)
        autocast_ctor = getattr(amp_mod, "autocast", None)
        if callable(autocast_ctor):
            return cast(ContextManager[None], autocast_ctor(dtype=dtype))
        return cast(ContextManager[None], nullcontext())
    return cast(ContextManager[None], nullcontext())


def _forecast_with_retries(
    forecaster,
    *,
    inputs,
    prediction_length: int,
    num_samples: int,
    samples_per_batch: int,
    device: str,
    autocast_dtype: Optional[TorchDType],
    max_retries: int,
    min_samples_per_batch: int,
    min_num_samples: int,
    forecast_kwargs: Optional[dict] = None,
):
    """
    Execute Toto forecasting with basic CUDA OOM recovery.

    Returns the forecast together with the effective (num_samples, samples_per_batch).
    """
    effective_kwargs = dict(forecast_kwargs or {})
    attempt = 0
    current_samples_per_batch = max(1, min(samples_per_batch, num_samples))
    current_num_samples = max(1, num_samples)
    last_error: Optional[Exception] = None

    while attempt <= max_retries:
        try:
            with _inference_context():
                with _autocast_context(device, autocast_dtype):
                    forecast = forecaster.forecast(
                        inputs,
                        prediction_length=prediction_length,
                        num_samples=current_num_samples,
                        samples_per_batch=current_samples_per_batch,
                        **effective_kwargs,
                    )
            return forecast, current_num_samples, current_samples_per_batch
        except Exception as exc:
            if not _is_cuda_oom(exc):
                raise
            last_error = exc
            logger.warning(
                "Toto forecast OOM (attempt %d/%d) with num_samples=%d, samples_per_batch=%d: %s",
                attempt + 1,
                max_retries + 1,
                current_num_samples,
                current_samples_per_batch,
                exc,
            )
            _maybe_empty_cuda_cache(device)
            attempt += 1
            next_samples_per_batch = max(min_samples_per_batch, current_samples_per_batch // 2)
            next_num_samples = current_num_samples
            if next_samples_per_batch == current_samples_per_batch:
                if current_num_samples > min_num_samples:
                    next_num_samples = max(min_num_samples, current_num_samples // 2)
            else:
                next_num_samples = max(next_samples_per_batch, current_num_samples)

            if next_samples_per_batch == current_samples_per_batch and next_num_samples == current_num_samples:
                break

            current_samples_per_batch = next_samples_per_batch
            current_num_samples = next_num_samples

    raise RuntimeError(
        f"Toto forecasting failed after {max_retries + 1} attempts due to GPU OOM "
        f"(last settings: num_samples={current_num_samples}, "
        f"samples_per_batch={current_samples_per_batch})."
    ) from last_error


class TotoPipeline:
    """
    Wrapper class that mimics ChronosPipeline behaviour for Toto.
    """

    def __init__(
        self,
        model: TotoModelType,
        device: str = "cuda",
        *,
        torch_dtype: Optional[TorchDType] = None,
        amp_dtype: Optional[TorchDType] = None,
        amp_autocast: bool = True,
        max_oom_retries: int = 2,
        min_samples_per_batch: int = 32,
        min_num_samples: int = 256,
        compile_model: bool = True,
        torch_compile: bool = False,
        compile_mode: Optional[str] = "max-autotune",
        compile_backend: Optional[str] = None,
    ):
        if _IMPORT_ERROR is not None or MaskedTimeseries is None or TotoForecaster is None:
            raise RuntimeError(
                "Toto dependencies are not available; ensure toto and its requirements are installed"
            ) from _IMPORT_ERROR

        if torch is None or np is None:
            raise RuntimeError(
                "Torch and NumPy must be configured via setup_toto_wrapper_imports before instantiating TotoPipeline."
            )

        normalised = device.lower()
        is_cuda_request = normalised.startswith("cuda")
        is_cpu_request = normalised == "cpu" or normalised.startswith("cpu:")
        if not (is_cuda_request or is_cpu_request):
            raise RuntimeError(
                f"TotoPipeline requires a CUDA or CPU device; received {device!r}."
            )
        if is_cuda_request:
            cuda_mod = getattr(torch, "cuda", None)
            is_available = bool(getattr(cuda_mod, "is_available", lambda: False)()) if cuda_mod is not None else False
            if not is_available:
                raise RuntimeError("CUDA is unavailable. TotoPipeline requires a CUDA-capable PyTorch installation.")

        if not amp_autocast:
            amp_dtype = None
        elif amp_dtype is None:
            amp_dtype = getattr(torch, "float16", None)

        self.device = device
        self.max_oom_retries = max(0, int(max_oom_retries))
        self.min_samples_per_batch = max(1, int(min_samples_per_batch))
        self.min_num_samples = max(1, int(min_num_samples))

        target_kwargs: Dict[str, Any] = {"device": self.device}
        if torch_dtype is not None:
            target_kwargs["dtype"] = torch_dtype

        try:
            self.model = model.to(**target_kwargs)
        except Exception as exc:
            if device.startswith("cuda") and _is_cuda_oom(exc):
                logger.warning(
                    "Toto model initialisation OOM on %s; retrying on CPU. (%s)",
                    device,
                    exc,
                )
                try:
                    torch.cuda.empty_cache()
                except Exception:  # pragma: no cover - cache clearing best effort
                    pass
                self.device = "cpu"
                target_kwargs = {"device": "cpu"}
                if torch_dtype is not None:
                    target_kwargs["dtype"] = torch_dtype
                self.model = model.to(**target_kwargs)
            else:
                raise
        self.model.eval()

        device = self.device

        try:
            first_param = next(self.model.parameters())
            self.model_dtype = first_param.dtype
        except StopIteration:
            self.model_dtype = torch_dtype or torch.float32

        if device.startswith("cuda"):
            self.amp_dtype = amp_dtype
        else:
            self.amp_dtype = None

        if self.amp_dtype is not None and device.startswith("cuda"):
            self._autocast_dtype: Optional[TorchDType] = self.amp_dtype
        elif device.startswith("cuda") and torch_dtype in {torch.float16, torch.bfloat16}:
            self._autocast_dtype = torch_dtype
        else:
            self._autocast_dtype = None

        self._torch_compile_enabled = bool(torch_compile and hasattr(torch, "compile"))
        self._torch_compile_success = False
        self._compile_mode = compile_mode
        self._compile_backend = compile_backend
        self._compiled = False

        if self._torch_compile_enabled:
            if getattr(self.model, "model", None) is None:
                logger.warning("torch.compile requested but Toto model has no 'model' attribute.")
                self._torch_compile_enabled = False
            else:
                compile_kwargs = {}
                if compile_mode:
                    compile_kwargs["mode"] = compile_mode
                if compile_backend:
                    compile_kwargs["backend"] = compile_backend
                try:
                    compiled_core = torch.compile(self.model.model, **compile_kwargs)  # type: ignore[arg-type]
                    self.model.model = compiled_core  # type: ignore[attr-defined]
                    self._torch_compile_success = True
                    self._compiled = True
                    logger.info(
                        "Enabled torch.compile for Toto model (mode=%s, backend=%s).",
                        compile_mode,
                        compile_backend,
                    )
                except Exception as exc:
                    self._torch_compile_enabled = False
                    logger.warning("torch.compile failed for Toto model: %s", exc)

        if compile_model and not self._torch_compile_success:
            try:
                if compile_mode:
                    self.model.compile(mode=compile_mode)  # type: ignore[attr-defined]
                else:
                    self.model.compile()  # type: ignore[attr-defined]
                self._compiled = True
            except AttributeError:
                if hasattr(torch, "compile"):
                    compile_kwargs = {}
                    if compile_mode:
                        compile_kwargs["mode"] = compile_mode
                    if compile_backend:
                        compile_kwargs["backend"] = compile_backend
                    try:
                        self.model = torch.compile(self.model, **compile_kwargs)  # type: ignore[assignment]
                        self._compiled = True
                    except Exception as exc:
                        logger.debug("torch.compile fallback failed for Toto model: %s", exc)
            except Exception as exc:
                logger.debug("Could not compile Toto model: %s", exc)

        model_core = cast(Any, self.model)
        forecaster_ctor = cast(Any, TotoForecaster)
        self.forecaster = cast(TotoForecasterType, forecaster_ctor(model_core.model))
        self._last_run_metadata: Optional[dict] = None

    @property
    def compiled(self) -> bool:
        """Return True if any compile step succeeded."""
        return self._compiled or self._torch_compile_success

    # ------------------------------------------------------------------ #
    # Internal warm-up helpers
    # ------------------------------------------------------------------ #
    def _warmup(
        self,
        *,
        sequence_length: int,
        prediction_length: int = 8,
        num_samples: int = 64,
        samples_per_batch: Optional[int] = None,
    ) -> None:
        """
        Execute a lightweight forward pass to pre-populate torch.compile / inductor caches.
        """
        if sequence_length <= 0:
            return
        samples_per_batch = samples_per_batch or min(num_samples, 64)
        try:
            context = torch.zeros(sequence_length, dtype=self.model_dtype, device=self.device)
        except Exception as exc:  # pragma: no cover - defensive against device issues
            logger.debug("Skipping Toto warmup due to tensor allocation failure: %s", exc)
            return

        try:
            self.predict(
                context=context,
                prediction_length=prediction_length,
                num_samples=num_samples,
                samples_per_batch=samples_per_batch,
            )
        except Exception as exc:  # pragma: no cover - warmup best effort
            logger.debug("Toto warmup prediction failed (best effort): %s", exc)

    @property
    def last_run_metadata(self) -> Optional[dict]:
        """Return details captured during the most recent forecast execution."""
        return self._last_run_metadata

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "Datadog/Toto-Open-Base-1.0",
        device_map: str = "cuda",
        torch_dtype: Optional[TorchDType] = None,
        *,
        compile_model: bool = True,
        compile_mode: Optional[str] = "max-autotune",
        amp_dtype: Optional[TorchDType] = None,
        amp_autocast: bool = True,
        torch_compile: bool = False,
        compile_backend: Optional[str] = None,
        cache_policy: str = "prefer",
        warmup_sequence: int = 512,
        force_refresh: bool = False,
        cache_manager: Optional[ModelCacheManager] = None,
        **kwargs: Any,
    ) -> "TotoPipeline":
        """
        Load a pretrained Toto model and build a pipeline around it.
        """
        if _IMPORT_ERROR is not None or Toto is None:
            raise RuntimeError(
                "Toto dependencies are not available; ensure toto and its requirements are installed"
            ) from _IMPORT_ERROR

        torch_module = _require_torch()
        if not amp_autocast:
            effective_amp_dtype: Optional[TorchDType] = None
        elif amp_dtype is None:
            effective_amp_dtype = getattr(torch_module, "float16", None)
        else:
            effective_amp_dtype = amp_dtype

        policy = cache_policy.lower()
        if policy not in {"prefer", "never", "only"}:
            raise ValueError(f"Unrecognised cache policy '{cache_policy}'. Expected 'prefer', 'never', or 'only'.")

        manager = cache_manager or ModelCacheManager("toto")
        dtype_token = dtype_to_token(torch_dtype)
        amp_token = dtype_to_token(effective_amp_dtype)
        device_str = str(device_map) if not isinstance(device_map, str) else device_map
        device = device_str if device_str != "mps" else "cpu"
        normalised = device.lower()
        is_cuda_request = normalised.startswith("cuda")
        is_cpu_request = normalised == "cpu" or normalised.startswith("cpu:")
        if not (is_cuda_request or is_cpu_request):
            raise RuntimeError(
                "TotoPipeline requires a device_map of 'cuda' or 'cpu'; received "
                f"{device_map!r}."
            )

        if is_cuda_request:
            cuda_mod = getattr(torch_module, "cuda", None)
            is_available = bool(getattr(cuda_mod, "is_available", lambda: False)()) if cuda_mod is not None else False
            if not is_available:
                raise RuntimeError("CUDA is unavailable. TotoPipeline requires a CUDA-capable PyTorch installation.")

        extra_kwargs: Dict[str, Any] = dict(kwargs)
        pipeline_kwargs: Dict[str, Any] = {}
        for key in ("max_oom_retries", "min_samples_per_batch", "min_num_samples"):
            if key in extra_kwargs:
                pipeline_kwargs[key] = extra_kwargs.pop(key)

        model_kwargs: Dict[str, Any] = extra_kwargs
        metadata_requirements = {
            "model_id": model_id,
            "dtype": dtype_token,
            "amp_dtype": amp_token,
            "compile_mode": (compile_mode or "none"),
            "compile_backend": (compile_backend or "none"),
            "torch_version": torch.__version__,
        }

        use_cache = policy != "never"
        loaded_from_cache = False
        with manager.compilation_env(model_id, dtype_token):
            metadata = manager.load_metadata(model_id, dtype_token) if use_cache else None
            model: TotoModelType
            if (
                use_cache
                and not force_refresh
                and metadata
                and manager.metadata_matches(metadata, metadata_requirements)
            ):
                cache_path = manager.load_pretrained_path(model_id, dtype_token)
                if cache_path is not None:
                    try:
                        model = cast(
                            TotoModelType,
                            Toto.from_pretrained(str(cache_path), **model_kwargs),
                        )
                        loaded_from_cache = True
                        logger.info(
                            "Loaded Toto model '%s' (%s) from compiled cache.",
                            model_id,
                            dtype_token,
                        )
                    except Exception as exc:  # pragma: no cover - backstop for unexpected load failures
                        loaded_from_cache = False
                        logger.warning(
                            "Failed to load cached Toto weights from %s: %s",
                            cache_path,
                            exc,
                        )
            if policy == "only" and not loaded_from_cache:
                raise RuntimeError(
                    f"Compiled Toto cache unavailable for model '{model_id}' and dtype '{dtype_token}'. "
                    "Run the model pre-warming utilities to generate cached weights."
                )

            if not loaded_from_cache:
                model = cast(TotoModelType, Toto.from_pretrained(model_id, **model_kwargs))
                logger.info(
                    "Loaded Toto model '%s' from source (cache_policy=%s).",
                    model_id,
                    policy,
                )

            pipeline = cls(
                model,
                device=device,
                torch_dtype=torch_dtype,
                amp_dtype=effective_amp_dtype,
                amp_autocast=amp_autocast,
                max_oom_retries=int(pipeline_kwargs.get("max_oom_retries", 2)),
                min_samples_per_batch=int(pipeline_kwargs.get("min_samples_per_batch", 32)),
                min_num_samples=int(pipeline_kwargs.get("min_num_samples", 256)),
                compile_model=compile_model,
                torch_compile=torch_compile,
                compile_mode=compile_mode,
                compile_backend=compile_backend,
            )

            should_warmup = (
                warmup_sequence > 0 and (compile_model or torch_compile or pipeline.compiled) and not loaded_from_cache
            )
            if should_warmup:
                pipeline._warmup(sequence_length=warmup_sequence)

            if use_cache and (force_refresh or not loaded_from_cache):
                model_obj = getattr(pipeline, "model", None)
                if model_obj is not None:
                    metadata_payload = {
                        **metadata_requirements,
                        "device": device,
                        "compile_model": bool(pipeline._compiled),
                        "torch_compile": bool(pipeline._torch_compile_success),
                        "amp_autocast": bool(amp_autocast),
                        "warmup_sequence": int(warmup_sequence),
                    }
                    try:
                        manager.persist_model_state(
                            model_id=model_id,
                            dtype_token=dtype_token,
                            model=model_obj,
                            metadata=metadata_payload,
                            force=force_refresh,
                        )
                    except ModelCacheError as exc:
                        logger.warning(
                            "Failed to persist Toto cache for model '%s': %s",
                            model_id,
                            exc,
                        )
                else:
                    logger.debug("Toto pipeline model attribute missing; skipping cache persistence.")

        return pipeline

    def predict(
        self,
        context: Union[TorchTensor, NDArray, List[float]],
        prediction_length: int,
        num_samples: int = 4096,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> List[TotoForecast]:
        """
        Generate forecasts using Toto with Chronos-compatible semantics.
        """
        _ = temperature, top_k, top_p  # Compatibility placeholders.

        if MaskedTimeseries is None:
            raise RuntimeError("Toto dependencies are not available; cannot build MaskedTimeseries inputs.")

        torch_module = _require_torch()
        numpy_mod = _require_numpy()

        if isinstance(context, (list, numpy_mod.ndarray)):
            context = torch_module.tensor(context, dtype=torch_module.float32)

        context = context.to(self.device)
        if context.dtype != self.model_dtype:
            context = context.to(dtype=self.model_dtype)

        if context.dim() == 1:
            context = context.unsqueeze(0)

        batch_size = int(context.shape[0])
        seq_len = context.shape[-1]

        time_interval_seconds = int(kwargs.pop("time_interval_seconds", 60 * 15))
        timestamp_seconds = torch.zeros(
            context.shape[0],
            seq_len,
            device=self.device,
            dtype=torch.float32,
        )
        time_interval_tensor = torch.full(
            (context.shape[0],),
            time_interval_seconds,
            device=self.device,
            dtype=torch.float32,
        )

        inputs = MaskedTimeseries(
            series=context,
            padding_mask=torch.ones_like(context, dtype=torch.bool),
            id_mask=torch.zeros_like(context, dtype=torch.int),
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_tensor,
        )

        samples_per_batch = int(kwargs.pop("samples_per_batch", 512))
        samples_per_batch = max(1, min(samples_per_batch, num_samples))

        max_oom_retries = int(kwargs.pop("max_oom_retries", self.max_oom_retries))
        min_samples_per_batch = int(kwargs.pop("min_samples_per_batch", self.min_samples_per_batch))
        min_num_samples = int(kwargs.pop("min_num_samples", self.min_num_samples))

        forecast_kwargs = kwargs if kwargs else None

        forecast, effective_num_samples, effective_samples_per_batch = _forecast_with_retries(
            self.forecaster,
            inputs=inputs,
            prediction_length=prediction_length,
            num_samples=num_samples,
            samples_per_batch=samples_per_batch,
            device=self.device,
            autocast_dtype=self._autocast_dtype,
            max_retries=max_oom_retries,
            min_samples_per_batch=min_samples_per_batch,
            min_num_samples=min_num_samples,
            forecast_kwargs=forecast_kwargs,
        )

        if effective_num_samples != num_samples or effective_samples_per_batch != samples_per_batch:
            logger.info(
                "Toto forecast adjusted sampling from num_samples=%d, samples_per_batch=%d "
                "to num_samples=%d, samples_per_batch=%d due to OOM.",
                num_samples,
                samples_per_batch,
                effective_num_samples,
                effective_samples_per_batch,
            )

        self._last_run_metadata = {
            "num_samples_requested": num_samples,
            "num_samples_used": effective_num_samples,
            "samples_per_batch_requested": samples_per_batch,
            "samples_per_batch_used": effective_samples_per_batch,
            "torch_dtype": str(self.model_dtype),
            "torch_compile_requested": self._torch_compile_enabled,
            "torch_compile_success": self._torch_compile_success,
            "torch_compile_mode": self._compile_mode,
            "torch_compile_backend": self._compile_backend,
            "batch_size": batch_size,
        }

        if getattr(forecast, "samples", None) is None:
            raise RuntimeError("Toto forecaster returned no samples.")

        samples = forecast.samples.detach().cpu().numpy()

        primary_axis = samples.shape[0]
        if primary_axis != batch_size and samples.ndim > 1 and samples.shape[1] == batch_size:
            samples = numpy_mod.swapaxes(samples, 0, 1)
            primary_axis = samples.shape[0]

        if primary_axis != batch_size:
            raise RuntimeError("Toto forecast samples tensor does not match the requested batch size.")

        forecasts: List[TotoForecast] = []
        for idx in range(batch_size):
            series_samples = samples[idx : idx + 1]
            forecasts.append(TotoForecast(samples=series_samples))

        return forecasts

    def unload(self) -> None:
        """Release GPU resources held by the Toto pipeline."""
        try:
            model = getattr(self, "model", None)
            move_to_cpu = getattr(model, "to", None)
            if callable(move_to_cpu):
                move_to_cpu("cpu")
        except Exception as exc:  # pragma: no cover - defensive cleanup
            logger.debug("Failed to move Toto model to CPU during unload: %s", exc)
        self.model = None
        self.forecaster = None
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("Failed to empty CUDA cache after Toto unload: %s", exc)
