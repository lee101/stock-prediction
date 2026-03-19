"""CuteChronos2Pipeline -- a drop-in wrapper around Chronos2Pipeline.

For now this delegates to the original Chronos2Model (the real model
implementation).  Once the custom Triton/CUDA-backed CuteChronos2Model
lands, ``from_pretrained`` will load *that* instead while keeping the
same public API.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import List, Optional, Union

import torch
from einops import rearrange

logger = logging.getLogger(__name__)


def _load_model(model_path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Load model weights via the upstream Chronos2Pipeline loader, then
    extract the underlying model and move it to *device*.

    When ``cutechronos.model.CuteChronos2Model`` becomes available this
    function will load into that class instead.
    """
    from chronos.chronos2 import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(model_path, dtype=dtype)
    model = pipeline.model
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class CuteChronos2Pipeline:
    """Lightweight pipeline for Chronos-2 inference.

    Provides a simpler API than the upstream ``Chronos2Pipeline``: the
    caller passes a raw ``torch.Tensor`` (or list of tensors) and gets
    back quantile predictions without needing to deal with ``DataLoader``
    or ``Dataset`` wrappers.

    The pipeline handles:
    * Context truncation to ``model_context_length``.
    * Left-padding variable-length series.
    * Device management and dtype casting.
    * Single-step prediction (non-autoregressive) for
      ``prediction_length <= model_prediction_length``.
    """

    def __init__(self, model, *, device: str = "cuda"):
        self.model = model
        self._device = device

    # -- properties ----------------------------------------------------------

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_context_length(self) -> int:
        return self.model.chronos_config.context_length

    @property
    def model_output_patch_size(self) -> int:
        return self.model.chronos_config.output_patch_size

    @property
    def model_prediction_length(self) -> int:
        return self.model.chronos_config.max_output_patches * self.model.chronos_config.output_patch_size

    @property
    def quantiles(self) -> list[float]:
        return self.model.chronos_config.quantiles

    @property
    def max_output_patches(self) -> int:
        return self.model.chronos_config.max_output_patches

    # -- factory -------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "CuteChronos2Pipeline":
        """Load a Chronos-2 model and wrap it in a CuteChronos2Pipeline.

        Parameters
        ----------
        model_path
            HuggingFace model id (e.g. ``"amazon/chronos-2"``) or local path.
        device
            Target device, default ``"cuda"``.
        dtype
            Model dtype, default ``torch.bfloat16``.
        """
        model = _load_model(model_path, device=device, dtype=dtype)
        return cls(model, device=device)

    # -- helpers -------------------------------------------------------------

    def _prepare_context(self, context: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Normalise *context* to a 2-D float32 tensor on the right device."""
        if isinstance(context, list):
            from chronos.utils import left_pad_and_stack_1D

            context = left_pad_and_stack_1D(context)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if context.ndim == 3 and context.shape[1] == 1:
            # (B, 1, T) -> (B, T)  univariate convenience
            context = context.squeeze(1)
        assert context.ndim == 2, f"context must be 2-D, got shape {context.shape}"
        # Truncate to model context length
        if context.shape[-1] > self.model_context_length:
            context = context[..., -self.model_context_length:]
        return context.to(device=self._device, dtype=torch.float32)

    # -- prediction ----------------------------------------------------------

    @torch.inference_mode()
    def predict(
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        limit_prediction_length: bool = True,
    ) -> List[torch.Tensor]:
        """Generate quantile predictions.

        Parameters
        ----------
        context
            A 1-D tensor, list of 1-D tensors, or 2-D tensor (B, T).
        prediction_length
            Horizon.  Defaults to ``model_prediction_length``.
        limit_prediction_length
            If *True*, raise when ``prediction_length`` exceeds the model's
            default prediction length.

        Returns
        -------
        List of tensors, each of shape ``(1, n_quantiles, prediction_length)``
        (one element per batch item), matching the upstream API.
        """
        if prediction_length is None:
            prediction_length = self.model_prediction_length

        if prediction_length > self.model_prediction_length:
            msg = (
                f"prediction_length ({prediction_length}) exceeds "
                f"model_prediction_length ({self.model_prediction_length}). "
                "Quality may degrade."
            )
            if limit_prediction_length:
                msg += " Set limit_prediction_length=False to allow this."
                raise ValueError(msg)
            warnings.warn(msg)

        ctx = self._prepare_context(context)
        batch_size = ctx.shape[0]

        num_output_patches = math.ceil(prediction_length / self.model_output_patch_size)
        num_output_patches = min(num_output_patches, self.max_output_patches)

        # group_ids: each series independent
        group_ids = torch.arange(batch_size, dtype=torch.long, device=self._device)

        output = self.model(
            context=ctx,
            group_ids=group_ids,
            num_output_patches=num_output_patches,
        )
        # output.quantile_preds: (B, Q, H)
        preds = output.quantile_preds[..., :prediction_length]
        preds = preds.to(dtype=torch.float32, device="cpu")

        # Return as list of (1, Q, H) tensors for API compatibility
        return [preds[i : i + 1] for i in range(batch_size)]

    @torch.inference_mode()
    def predict_quantiles(
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        quantile_levels: Optional[List[float]] = None,
        limit_prediction_length: bool = True,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Generate quantile and mean forecasts.

        Parameters
        ----------
        context
            Same as ``predict``.
        prediction_length
            Same as ``predict``.
        quantile_levels
            Quantile levels to return.  Default uses the model's trained
            quantiles.
        limit_prediction_length
            Same as ``predict``.

        Returns
        -------
        quantiles
            List of tensors, each ``(1, prediction_length, len(quantile_levels))``.
        mean
            List of tensors, each ``(1, prediction_length)``.
        """
        if quantile_levels is None:
            quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        predictions = self.predict(
            context,
            prediction_length=prediction_length,
            limit_prediction_length=limit_prediction_length,
        )

        training_quantile_levels = self.quantiles
        # predictions: list of (1, Q, H)

        # Transpose each to (1, H, Q)
        predictions = [rearrange(p, "b q h -> b h q") for p in predictions]

        if set(quantile_levels).issubset(training_quantile_levels):
            indices = [training_quantile_levels.index(q) for q in quantile_levels]
            quantiles = [p[..., indices] for p in predictions]
        else:
            # Interpolate for non-standard quantile levels
            from chronos.utils import interpolate_quantiles

            quantiles = [
                interpolate_quantiles(quantile_levels, training_quantile_levels, p)
                for p in predictions
            ]

        # median as "mean"
        median_idx = training_quantile_levels.index(0.5)
        mean = [p[..., median_idx] for p in predictions]

        return quantiles, mean
