"""
Toto forecasting wrapper to replace Chronos
"""
from __future__ import annotations

import logging
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

# need to uv pip install -e . in here after dl toto
sys.path.insert(0, '/mnt/fast/code/chronos-forecasting/toto')

_IMPORT_ERROR: Optional[Exception] = None

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


@dataclass
class TotoForecast:
    """Container for Toto forecast results - compatible with Chronos format"""
    samples: np.ndarray
    
    def numpy(self):
        """Return numpy array of samples in Chronos-compatible format"""
        # Toto returns shape (1, num_variables, prediction_length, num_samples)
        # We need to reshape to match Chronos format
        samples = self.samples
        
        # Remove batch dimension if present (first dim = 1)
        if samples.ndim == 4 and samples.shape[0] == 1:
            samples = samples.squeeze(0)  # Now (num_variables, prediction_length, num_samples)
        
        # Remove variable dimension if single variable (first dim = 1)
        if samples.ndim == 3 and samples.shape[0] == 1:
            samples = samples.squeeze(0)  # Now (prediction_length, num_samples)
        
        # For single prediction step, return 1D array of samples
        if samples.ndim == 2 and samples.shape[0] == 1:
            return samples.squeeze(0)  # Shape (num_samples,)
        
        # For multiple prediction steps, transpose to (num_samples, prediction_length)
        if samples.ndim == 2:
            return samples.T
        
        return samples


def _is_cuda_oom(exc: BaseException) -> bool:
    """Return True if the exception represents a CUDA OOM condition."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message


def _maybe_empty_cuda_cache(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as cache_exc:  # pragma: no cover - best effort
            logger.debug("Failed to empty CUDA cache after OOM: %s", cache_exc)


def _inference_context():
    """Return the best available inference context manager (inference_mode or no_grad)."""
    context_ctor = getattr(torch, "inference_mode", None)
    if callable(context_ctor):
        return context_ctor()
    return torch.no_grad()


def _autocast_context(device: str, dtype: Optional[torch.dtype]):
    if dtype is None:
        return nullcontext()
    if device.startswith("cuda"):
        autocast_fn = getattr(torch, "autocast", None)
        if callable(autocast_fn):
            return autocast_fn(device_type="cuda", dtype=dtype)
        return torch.cuda.amp.autocast(dtype=dtype)
    return nullcontext()


def _forecast_with_retries(
    forecaster,
    *,
    inputs,
    prediction_length: int,
    num_samples: int,
    samples_per_batch: int,
    device: str,
    autocast_dtype: Optional[torch.dtype],
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
            next_samples_per_batch = max(
                min_samples_per_batch,
                current_samples_per_batch // 2,
            )
            next_num_samples = current_num_samples
            if next_samples_per_batch == current_samples_per_batch:
                if current_num_samples > min_num_samples:
                    next_num_samples = max(min_num_samples, current_num_samples // 2)
            else:
                next_num_samples = max(next_samples_per_batch, current_num_samples)

            if (
                next_samples_per_batch == current_samples_per_batch
                and next_num_samples == current_num_samples
            ):
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
    Wrapper class that mimics ChronosPipeline interface for Toto model
    """
    
    def __init__(
        self,
        model: Toto,
        device: str = 'cuda',
        *,
        torch_dtype: Optional[torch.dtype] = None,
        max_oom_retries: int = 2,
        min_samples_per_batch: int = 32,
        min_num_samples: int = 256,
        torch_compile: bool = False,
        compile_mode: Optional[str] = None,
        compile_backend: Optional[str] = None,
    ):
        if _IMPORT_ERROR is not None or MaskedTimeseries is None or TotoForecaster is None:
            raise RuntimeError(
                "Toto dependencies are not available; ensure toto and its requirements are installed"
            ) from _IMPORT_ERROR
        self.device = device
        self.max_oom_retries = max(0, int(max_oom_retries))
        self.min_samples_per_batch = max(1, int(min_samples_per_batch))
        self.min_num_samples = max(1, int(min_num_samples))

        target_kwargs = {"device": self.device}
        if torch_dtype is not None:
            target_kwargs["dtype"] = torch_dtype

        self.model = model.to(**target_kwargs)
        self.model.eval()

        try:
            first_param = next(self.model.parameters())
            self.model_dtype = first_param.dtype
        except StopIteration:
            self.model_dtype = torch_dtype or torch.float32

        self._torch_compile_enabled = bool(torch_compile and hasattr(torch, "compile"))
        self._torch_compile_success = False
        self._compile_mode = compile_mode
        self._compile_backend = compile_backend

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
                    logger.info(
                        "Enabled torch.compile for Toto model (mode=%s, backend=%s).",
                        compile_mode,
                        compile_backend,
                    )
                except Exception as exc:
                    self._torch_compile_enabled = False
                    logger.warning("torch.compile failed for Toto model: %s", exc)

        # Optionally run native Toto compile when torch.compile is not used
        if not self._torch_compile_success:
            try:
                self.model.compile()
            except Exception as e:
                logger.debug("Could not compile Toto model: %s", e)

        self.forecaster = TotoForecaster(self.model.model)
        if device.startswith("cuda") and torch_dtype in {torch.float16, torch.bfloat16}:
            self._autocast_dtype: Optional[torch.dtype] = torch_dtype
        else:
            self._autocast_dtype = None
        self._last_run_metadata: Optional[dict] = None
    
    @classmethod
    def from_pretrained(
        cls, 
        model_id: str = "Datadog/Toto-Open-Base-1.0",
        device_map: str = "cuda",
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs
    ):
        """
        Load pretrained Toto model
        
        Args:
            model_id: Model identifier (default: Datadog/Toto-Open-Base-1.0)
            device_map: Device to load model on
            torch_dtype: Data type for model (ignored for compatibility)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            TotoPipeline instance
        """
        if _IMPORT_ERROR is not None or Toto is None:
            raise RuntimeError(
                "Toto dependencies are not available; ensure toto and its requirements are installed"
            ) from _IMPORT_ERROR
        device = device_map if device_map != "mps" else "cpu"  # MPS not fully supported
        
        # Load pre-trained Toto model
        extra_kwargs = dict(kwargs)
        pipeline_kwargs = {}
        for key in (
            "max_oom_retries",
            "min_samples_per_batch",
            "min_num_samples",
            "torch_compile",
            "compile_mode",
            "compile_backend",
        ):
            if key in extra_kwargs:
                pipeline_kwargs[key] = extra_kwargs.pop(key)

        model_kwargs = extra_kwargs

        model = Toto.from_pretrained(model_id, **model_kwargs)
        
        return cls(
            model,
            device=device,
            torch_dtype=torch_dtype,
            **pipeline_kwargs,
        )
    
    def predict(
        self,
        context: Union[torch.Tensor, np.ndarray, List[float]],
        prediction_length: int,
        num_samples: int = 4096,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> List[TotoForecast]:
        """
        Generate forecasts using Toto model
        
        Args:
            context: Historical time series data
            prediction_length: Number of steps to forecast
            num_samples: Number of sample paths to generate
            temperature: Sampling temperature (ignored, for compatibility)
            top_k: Top-k sampling (ignored, for compatibility)
            top_p: Top-p sampling (ignored, for compatibility)
            
        Returns:
            List containing single TotoForecast object
        """
        # Convert context to tensor if needed
        if isinstance(context, (list, np.ndarray)):
            context = torch.tensor(context, dtype=torch.float32)
            
        # Move to device
        context = context.to(self.device)
        if context.dtype != self.model_dtype:
            context = context.to(dtype=self.model_dtype)
        
        # Ensure 2D shape (variables x timesteps)
        if context.dim() == 1:
            context = context.unsqueeze(0)  # Add variable dimension
            
        # Get context length
        seq_len = context.shape[-1]
        
        # Create timestamps (assuming regular intervals)
        # Using 15-minute intervals as default (can be adjusted)
        time_interval_seconds = 60 * 15  # 15 minutes
        timestamp_seconds = torch.zeros(1, seq_len, device=self.device, dtype=torch.float32)
        time_interval_tensor = torch.full((1,), time_interval_seconds, device=self.device, dtype=torch.float32)
        
        # Create MaskedTimeseries input
        inputs = MaskedTimeseries(
            series=context,
            padding_mask=torch.full_like(context, True, dtype=torch.bool),
            id_mask=torch.zeros_like(context),
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_tensor,
        )
        
        # Generate forecasts
        # Note: Toto generates multiple samples at once
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

        if (
            effective_num_samples != num_samples
            or effective_samples_per_batch != samples_per_batch
        ):
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
        }
        
        # Convert to numpy and reshape to match Chronos output format
        # Chronos returns shape: (num_samples, prediction_length)
        samples = forecast.samples.cpu().numpy()
        
        # If samples has shape (1, num_samples, prediction_length), squeeze first dim
        if samples.ndim == 3 and samples.shape[0] == 1:
            samples = samples.squeeze(0)
        
        # Return as list with single forecast (matching Chronos interface)
        return [TotoForecast(samples=samples)]

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
