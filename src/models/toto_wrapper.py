"""
Toto forecasting wrapper that mirrors the Chronos interface while enabling
TorchCompile and hardware-aware speed-ups.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch

from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto

# Enable tensor-core friendly matmul defaults once per process.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("medium")


@dataclass
class TotoForecast:
    """Container for Toto forecast results compatible with Chronos outputs."""

    samples: np.ndarray

    def numpy(self) -> np.ndarray:
        """
        Return samples in Chronos-compatible layout.
        """
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


class TotoPipeline:
    """
    Wrapper class that mimics ChronosPipeline behaviour for Toto.
    """

    def __init__(
        self,
        model: Toto,
        device: str = "cuda",
        *,
        compile_model: bool = True,
        compile_mode: str = "max-autotune",
        amp_dtype: Optional[torch.dtype] = torch.float16,
    ):
        self.device = device
        self.amp_dtype = amp_dtype if device.startswith("cuda") else None
        self.model = model.to(device)
        self.model.eval()
        self._compiled = False

        if compile_model:
            try:
                # Native module.compile introduced in recent PyTorch releases.
                self.model.compile(mode=compile_mode)
                self._compiled = True
            except AttributeError:
                # Fall back to torch.compile on older builds.
                self.model = torch.compile(self.model, mode=compile_mode)  # type: ignore[assignment]
                self._compiled = True
            except Exception as exc:  # pragma: no cover - informational
                print(f"Could not compile model: {exc}")

        self.forecaster = TotoForecaster(self.model.model)

    @property
    def compiled(self) -> bool:
        return self._compiled

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "Datadog/Toto-Open-Base-1.0",
        device_map: str = "cuda",
        torch_dtype: Optional[torch.dtype] = None,
        *,
        compile_model: bool = True,
        compile_mode: str = "max-autotune",
        amp_dtype: Optional[torch.dtype] = torch.float16,
        **_: object,
    ) -> "TotoPipeline":
        """
        Load a pretrained Toto model and build a pipeline around it.
        """
        device = device_map if device_map != "mps" else "cpu"
        model = Toto.from_pretrained(model_id)
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)

        return cls(
            model,
            device,
            compile_model=compile_model,
            compile_mode=compile_mode,
            amp_dtype=amp_dtype,
        )

    def predict(
        self,
        context: Union[torch.Tensor, np.ndarray, List[float]],
        prediction_length: int,
        num_samples: int = 2048,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs: object,
    ) -> List[TotoForecast]:
        """
        Generate forecasts using Toto with Chronos-compatible semantics.
        """
        _ = temperature, top_k, top_p  # Unused compatibility hooks.

        if isinstance(context, (list, np.ndarray)):
            context = torch.tensor(context, dtype=torch.float32)

        context = context.to(self.device)

        if context.dim() == 1:
            context = context.unsqueeze(0)

        seq_len = context.shape[-1]

        time_interval_seconds = kwargs.pop("time_interval_seconds", 60 * 15)
        timestamp_seconds = torch.zeros(context.shape[0], seq_len, device=self.device)
        time_interval_tensor = torch.full(
            (context.shape[0],),
            time_interval_seconds,
            device=self.device,
            dtype=torch.long,
        )

        inputs = MaskedTimeseries(
            series=context,
            padding_mask=torch.ones_like(context, dtype=torch.bool),
            id_mask=torch.zeros_like(context, dtype=torch.int),
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_tensor,
        )

        samples_per_batch = min(num_samples, int(kwargs.pop("samples_per_batch", 256)))
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.amp_dtype)
            if self.amp_dtype is not None
            else contextlib.nullcontext()
        )

        with torch.inference_mode(), autocast_ctx:
            forecast = self.forecaster.forecast(
                inputs,
                prediction_length=prediction_length,
                num_samples=num_samples,
                samples_per_batch=samples_per_batch,
            )

        if forecast.samples is None:
            raise RuntimeError("Toto forecaster returned no samples.")

        samples = forecast.samples.detach().cpu().numpy()

        if samples.ndim == 3 and samples.shape[0] == 1:
            samples = samples.squeeze(0)

        return [TotoForecast(samples=samples)]
