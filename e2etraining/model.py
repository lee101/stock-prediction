from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from .config import E2EModelConfig
from .data import momentum_feature, realized_volatility


@dataclass(slots=True)
class PolicyStepOutput:
    weights: torch.Tensor
    quantile_preds: torch.Tensor
    aux_forecast_loss: torch.Tensor
    asset_scores: torch.Tensor


def _parse_dtype(name: str) -> torch.dtype | None:
    normalized = str(name).strip().lower()
    if normalized in {"", "auto"}:
        return None
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[normalized]


def _default_model_loader(model_id: str, device_map: str, torch_dtype: str) -> Any:
    from chronos import Chronos2Pipeline

    dtype = _parse_dtype(torch_dtype)
    return Chronos2Pipeline.from_pretrained(model_id, device_map=device_map, dtype=dtype)


def _apply_lora(model: nn.Module, cfg: E2EModelConfig) -> nn.Module:
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_targets),
        bias="none",
    )
    return get_peft_model(model, lora_cfg)


class ChronosTradingPolicy(nn.Module):
    def __init__(
        self,
        cfg: E2EModelConfig,
        *,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        loader: Callable[[str, str, str], Any] = _default_model_loader,
        pipeline: Any | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.device_name = device
        self.device = torch.device(device)

        loaded_pipeline = pipeline if pipeline is not None else loader(cfg.model_id, device, torch_dtype)
        self.pipeline = loaded_pipeline
        base_model = loaded_pipeline.model
        if cfg.lora_enabled:
            base_model = _apply_lora(base_model, cfg)
        self.backbone = base_model

        quantiles = getattr(loaded_pipeline, "quantiles", None) or [0.1, 0.5, 0.9]
        self.quantiles = [float(level) for level in quantiles]
        self.q10_idx = min(range(len(self.quantiles)), key=lambda idx: abs(self.quantiles[idx] - 0.1))
        self.q50_idx = min(range(len(self.quantiles)), key=lambda idx: abs(self.quantiles[idx] - 0.5))
        self.q90_idx = min(range(len(self.quantiles)), key=lambda idx: abs(self.quantiles[idx] - 0.9))

        feature_dim = 7
        self.asset_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, cfg.policy_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.policy_hidden_dim, 1),
        )
        cash_feature_dim = feature_dim * 2
        self.cash_head = nn.Sequential(
            nn.LayerNorm(cash_feature_dim),
            nn.Linear(cash_feature_dim, cfg.policy_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.policy_hidden_dim, 1),
        )

    def _group_ids(self, asset_count: int, device: torch.device) -> torch.Tensor:
        if self.cfg.cross_learning:
            return torch.zeros(asset_count, device=device, dtype=torch.long)
        return torch.arange(asset_count, device=device, dtype=torch.long)

    def forward(
        self,
        context_close: torch.Tensor,
        actual_next_close: torch.Tensor,
        prev_weights: torch.Tensor | None = None,
    ) -> PolicyStepOutput:
        if context_close.ndim != 2:
            raise ValueError("context_close must have shape [assets, context_length]")
        asset_count = context_close.shape[0]
        device = next(self.parameters()).device
        context_close = context_close.to(device=device, dtype=torch.float32)
        actual_next_close = actual_next_close.to(device=device, dtype=torch.float32)

        outputs = self.backbone(
            context=context_close,
            group_ids=self._group_ids(asset_count, device),
            num_output_patches=max(1, int(self.cfg.prediction_length)),
        )
        quantile_preds = outputs.quantile_preds
        if quantile_preds is None:
            raise RuntimeError("Chronos backbone did not return quantile_preds")
        next_preds = quantile_preds[..., 0]
        current_close = context_close[:, -1].clamp_min(1e-8)
        pred_q10 = next_preds[:, self.q10_idx]
        pred_q50 = next_preds[:, self.q50_idx]
        pred_q90 = next_preds[:, self.q90_idx]

        expected_return = (pred_q50 / current_close) - 1.0
        upside_return = (pred_q90 / current_close) - 1.0
        downside_return = (pred_q10 / current_close) - 1.0
        interval_width = (pred_q90 - pred_q10) / current_close
        realized_vol = realized_volatility(context_close)
        recent_momentum = momentum_feature(context_close)

        expected_prev_size = asset_count + (1 if self.cfg.include_cash else 0)
        prev_weights_tensor = (
            prev_weights.to(device=device, dtype=torch.float32)
            if prev_weights is not None and prev_weights.numel() == expected_prev_size
            else None
        )
        if prev_weights_tensor is None:
            prev_asset_weights = torch.zeros(asset_count, device=device, dtype=torch.float32)
        elif self.cfg.include_cash:
            prev_asset_weights = prev_weights_tensor[:-1]
        else:
            prev_asset_weights = prev_weights_tensor
        asset_features = torch.stack(
            [
                expected_return,
                upside_return,
                downside_return,
                interval_width,
                realized_vol,
                recent_momentum,
                prev_asset_weights,
            ],
            dim=-1,
        )
        asset_scores = self.asset_head(asset_features).squeeze(-1)

        if self.cfg.include_cash:
            cash_features = torch.cat(
                [
                    asset_features.mean(dim=0),
                    asset_features.std(dim=0, correction=0),
                ],
                dim=0,
            )
            cash_score = self.cash_head(cash_features).reshape(1)
            logits = torch.cat([asset_scores, cash_score], dim=0)
        else:
            logits = asset_scores
        weights = torch.softmax(logits, dim=0)

        actual_next_close = actual_next_close.clamp_min(1e-8)
        actual_return = (actual_next_close / current_close) - 1.0
        pred_stack = torch.stack([pred_q10, pred_q50, pred_q90], dim=1)
        quantile_levels = torch.tensor([0.1, 0.5, 0.9], device=device, dtype=pred_stack.dtype).view(1, 3)
        future_target = actual_next_close.unsqueeze(1)
        pinball = torch.maximum(
            quantile_levels * (future_target - pred_stack),
            (quantile_levels - 1.0) * (future_target - pred_stack),
        )
        aux_forecast_loss = pinball.mean() + 0.1 * torch.nn.functional.smooth_l1_loss(pred_q50, actual_next_close)
        del actual_return  # constructed for debugging symmetry; objective uses caller-side returns.

        return PolicyStepOutput(
            weights=weights,
            quantile_preds=quantile_preds,
            aux_forecast_loss=aux_forecast_loss,
            asset_scores=asset_scores,
        )
