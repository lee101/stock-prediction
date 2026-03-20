"""CuteChronos2Pipeline -- drop-in pipeline for Chronos-2 inference.

Supports two backends:
- ``use_cute=True`` (default): Uses CuteChronos2Model with custom Triton
  kernels and optional torch.compile for maximum performance.
- ``use_cute=False``: Delegates to the original upstream Chronos2Model.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import List, Optional, Union

import torch
from einops import rearrange

logger = logging.getLogger(__name__)


def _load_model_original(model_path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Load via the upstream Chronos2Pipeline (requires chronos-forecasting)."""
    from chronos.chronos2 import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(model_path, dtype=dtype)
    model = pipeline.model
    model = model.to(device)
    model.eval()
    return model


def _load_model_cute(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    compile_mode: str | None = None,
):
    """Load via CuteChronos2Model (no upstream dependency for inference)."""
    from huggingface_hub import snapshot_download
    from cutechronos.model import CuteChronos2Model

    # Resolve HuggingFace model ID to local path
    local_path = snapshot_download(
        model_path,
        allow_patterns=["*.json", "*.safetensors", "*.bin"],
    )

    if compile_mode:
        model = CuteChronos2Model.from_pretrained_compiled(local_path, compile_mode=compile_mode)
    else:
        model = CuteChronos2Model.from_pretrained(local_path)

    model = model.to(device=device, dtype=dtype)
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

    def __init__(self, model, *, device: str = "cuda", _is_cute: bool = False):
        self.model = model
        self._device = device
        self._is_cute = _is_cute

    # -- properties ----------------------------------------------------------

    def _get_config(self):
        """Access the model config, handling both CuteChronos2Model and original."""
        if self._is_cute:
            return self.model.config
        return self.model.chronos_config

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_context_length(self) -> int:
        return self._get_config().context_length

    @property
    def model_output_patch_size(self) -> int:
        return self._get_config().output_patch_size

    @property
    def model_prediction_length(self) -> int:
        cfg = self._get_config()
        max_patches = getattr(cfg, "max_output_patches", 64)
        return max_patches * cfg.output_patch_size

    @property
    def quantiles(self) -> list[float]:
        return self._get_config().quantiles

    @property
    def max_output_patches(self) -> int:
        cfg = self._get_config()
        return getattr(cfg, "max_output_patches", 64)

    # -- factory -------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_cute: bool = True,
        compile_mode: str | None = None,
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
        use_cute
            If True (default), use CuteChronos2Model with custom kernels.
            If False, use the original upstream Chronos2Model.
        compile_mode
            If set (e.g. ``"reduce-overhead"``), apply torch.compile to the
            CuteChronos2Model. Only effective when ``use_cute=True``.
        """
        if use_cute:
            model = _load_model_cute(model_path, device=device, dtype=dtype, compile_mode=compile_mode)
            return cls(model, device=device, _is_cute=True)
        else:
            model = _load_model_original(model_path, device=device, dtype=dtype)
            return cls(model, device=device, _is_cute=False)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _left_pad_and_stack(tensors: List[torch.Tensor]) -> torch.Tensor:
        """Left-pad variable-length 1D tensors and stack into a 2D batch."""
        max_len = max(t.shape[-1] for t in tensors)
        padded = []
        for t in tensors:
            pad_len = max_len - t.shape[-1]
            if pad_len > 0:
                pad = torch.full((pad_len,), float("nan"), dtype=t.dtype, device=t.device)
                t = torch.cat([pad, t])
            padded.append(t)
        return torch.stack(padded)

    def _prepare_context(self, context: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Normalise *context* to a 2-D float32 tensor on the right device."""
        if isinstance(context, list):
            context = self._left_pad_and_stack(context)
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

        if self._is_cute:
            # CuteChronos2Model: direct tensor in/out
            preds = self.model(ctx, num_output_patches=num_output_patches)
        else:
            # Original Chronos2Model: named kwargs, ModelOutput return
            group_ids = torch.arange(batch_size, dtype=torch.long, device=self._device)
            output = self.model(
                context=ctx,
                group_ids=group_ids,
                num_output_patches=num_output_patches,
            )
            preds = output.quantile_preds

        # preds: (B, Q, H) - truncate to requested length
        preds = preds[..., :prediction_length]
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
            # Linear interpolation for non-standard quantile levels
            quantiles = []
            tq = torch.tensor(training_quantile_levels, dtype=torch.float32)
            for p in predictions:
                interp_results = []
                for ql in quantile_levels:
                    idx = torch.searchsorted(tq, ql).clamp(1, len(tq) - 1).item()
                    lo, hi = idx - 1, idx
                    frac = (ql - tq[lo].item()) / max(tq[hi].item() - tq[lo].item(), 1e-9)
                    val = p[..., lo] * (1 - frac) + p[..., hi] * frac
                    interp_results.append(val)
                quantiles.append(torch.stack(interp_results, dim=-1))


        # median as "mean"
        median_idx = training_quantile_levels.index(0.5)
        mean = [p[..., median_idx] for p in predictions]

        return quantiles, mean
