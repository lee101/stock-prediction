from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from src.alpaca_utils import INTRADAY_GROSS_EXPOSURE, MAX_GROSS_EXPOSURE


@dataclass(frozen=True)
class AllocationLeverageLimits:
    long_max_leverage: float
    short_max_leverage: float
    overnight_max_gross: float


@dataclass(frozen=True)
class AllocationRefinement:
    current_allocation: float
    target_allocation: float
    overnight_allocation: float
    forecast_alignment: float
    expected_move_pct: float
    confidence_multiplier: float
    forecast_multiplier: float
    reason: str


def leverage_limits_for_asset(asset_class: str) -> AllocationLeverageLimits:
    normalized = str(asset_class or "").strip().lower()
    if normalized == "stock":
        return AllocationLeverageLimits(
            long_max_leverage=float(INTRADAY_GROSS_EXPOSURE),
            short_max_leverage=1.0,
            overnight_max_gross=float(MAX_GROSS_EXPOSURE),
        )
    return AllocationLeverageLimits(
        long_max_leverage=1.0,
        short_max_leverage=0.0,
        overnight_max_gross=1.0,
    )


def forecast_move_pct(
    forecast: Optional[Mapping[str, float]],
    *,
    current_price: float,
) -> float:
    if not forecast or current_price <= 0.0:
        return 0.0
    predicted = forecast.get("predicted_close_p50", forecast.get("predicted_close", 0.0))
    try:
        predicted_value = float(predicted or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if predicted_value <= 0.0:
        return 0.0
    return (predicted_value / float(current_price)) - 1.0


def forecast_band_width_pct(
    forecast: Optional[Mapping[str, float]],
    *,
    current_price: float,
) -> float:
    if not forecast or current_price <= 0.0:
        return 0.0
    try:
        p10 = float(forecast.get("predicted_close_p10", 0.0) or 0.0)
        p90 = float(forecast.get("predicted_close_p90", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if p10 <= 0.0 or p90 <= 0.0 or p90 < p10:
        return 0.0
    return (p90 - p10) / float(current_price)


def refine_allocation(
    *,
    asset_class: str,
    rl_direction: str,
    rl_allocation_pct: float,
    rl_confidence: float,
    rl_logit_gap: float,
    current_allocation: float = 0.0,
    current_price: float,
    forecast_1h: Optional[Mapping[str, float]] = None,
    forecast_24h: Optional[Mapping[str, float]] = None,
    previous_forecast_error: Optional[float] = None,
    leverage_limits: Optional[AllocationLeverageLimits] = None,
) -> AllocationRefinement:
    limits = leverage_limits or leverage_limits_for_asset(asset_class)
    direction = str(rl_direction or "").strip().lower()
    signed_current = float(current_allocation)

    if direction not in {"long", "short"}:
        return AllocationRefinement(
            current_allocation=signed_current,
            target_allocation=0.0,
            overnight_allocation=0.0,
            forecast_alignment=0.0,
            expected_move_pct=0.0,
            confidence_multiplier=0.0,
            forecast_multiplier=0.0,
            reason="flat_or_invalid_direction",
        )

    if direction == "short" and limits.short_max_leverage <= 0.0:
        return AllocationRefinement(
            current_allocation=signed_current,
            target_allocation=0.0,
            overnight_allocation=0.0,
            forecast_alignment=-1.0,
            expected_move_pct=0.0,
            confidence_multiplier=0.0,
            forecast_multiplier=0.0,
            reason="shorts_disabled_for_asset",
        )

    direction_sign = 1.0 if direction == "long" else -1.0
    side_cap = limits.long_max_leverage if direction_sign > 0 else limits.short_max_leverage
    base_alloc_pct = float(np.clip(float(rl_allocation_pct), 0.0, 1.0))
    base_target = direction_sign * base_alloc_pct * max(side_cap, 0.0)

    blended_target = base_target
    if base_target != 0.0 and signed_current * base_target > 0.0:
        blended_target = 0.65 * base_target + 0.35 * signed_current

    confidence_component = float(np.clip(float(rl_confidence), 0.0, 1.0))
    gap_component = 0.5 * (math.tanh(float(rl_logit_gap)) + 1.0)
    rl_strength = float(np.clip(0.7 * confidence_component + 0.3 * gap_component, 0.0, 1.0))
    confidence_multiplier = 0.45 + 0.75 * rl_strength

    horizon_inputs = (
        (forecast_1h, 0.35, "1h"),
        (forecast_24h, 0.65, "24h"),
    )
    weighted_move = 0.0
    weighted_alignment = 0.0
    band_weight = 0.0
    used_weight = 0.0
    move_signs: list[int] = []
    reason_parts = [f"rl={direction} {base_alloc_pct:.2f}"]

    for forecast, weight, label in horizon_inputs:
        raw_move = forecast_move_pct(forecast, current_price=current_price)
        band_width = forecast_band_width_pct(forecast, current_price=current_price)
        if raw_move == 0.0 and band_width == 0.0:
            continue
        expected_signed_move = direction_sign * raw_move
        weighted_move += raw_move * weight
        weighted_alignment += math.tanh(expected_signed_move / 0.015) * weight
        band_weight += min(max(band_width / 0.06, 0.0), 1.0) * weight
        used_weight += weight
        if abs(raw_move) > 1e-6:
            move_signs.append(1 if raw_move > 0.0 else -1)
        reason_parts.append(f"{label}={raw_move:+.2%}")

    if used_weight > 0.0:
        expected_move = weighted_move / used_weight
        forecast_alignment = float(np.clip(weighted_alignment / used_weight, -1.0, 1.0))
        uncertainty_penalty = float(np.clip(band_weight / used_weight, 0.0, 1.0))
    else:
        expected_move = 0.0
        forecast_alignment = 0.0
        uncertainty_penalty = 0.0
        reason_parts.append("forecast=missing")

    disagreement_penalty = 0.0
    if len(move_signs) >= 2 and len(set(move_signs)) > 1:
        disagreement_penalty = 1.0
        reason_parts.append("forecast=conflict")

    if previous_forecast_error is None:
        error_penalty = 0.0
    else:
        error_penalty = float(np.clip(abs(float(previous_forecast_error)) / 6.0, 0.0, 1.0))
        reason_parts.append(f"prev_err={float(previous_forecast_error):+.2f}%")

    forecast_multiplier = (
        0.90
        + 0.35 * max(forecast_alignment, 0.0)
        - 0.60 * max(-forecast_alignment, 0.0)
        - 0.20 * disagreement_penalty
        - 0.15 * uncertainty_penalty
        - 0.15 * error_penalty
    )
    forecast_multiplier = float(np.clip(forecast_multiplier, 0.10, 1.25))

    target_allocation = blended_target * confidence_multiplier * forecast_multiplier
    if signed_current * base_target > 0.0 and forecast_alignment > 0.25:
        retention_floor = min(abs(signed_current), side_cap) * (0.40 + 0.20 * forecast_alignment)
        if abs(target_allocation) < retention_floor:
            target_allocation = math.copysign(retention_floor, base_target)
            reason_parts.append("retain_current_alloc")

    target_abs = float(np.clip(abs(target_allocation), 0.0, side_cap))
    target_allocation = math.copysign(target_abs, direction_sign)

    overnight_cap = min(side_cap, max(float(limits.overnight_max_gross), 0.0))
    overnight_abs = min(abs(target_allocation), overnight_cap)
    overnight_allocation = math.copysign(overnight_abs, target_allocation)
    if abs(target_allocation) > abs(overnight_allocation) + 1e-9:
        reason_parts.append(f"overnight_cap={overnight_cap:.2f}x")

    return AllocationRefinement(
        current_allocation=signed_current,
        target_allocation=float(target_allocation),
        overnight_allocation=float(overnight_allocation),
        forecast_alignment=float(forecast_alignment),
        expected_move_pct=float(expected_move),
        confidence_multiplier=float(confidence_multiplier),
        forecast_multiplier=float(forecast_multiplier),
        reason="; ".join(reason_parts),
    )


def scale_allocations_to_gross_limit(
    allocations: Mapping[str, float],
    *,
    max_gross: float,
) -> dict[str, float]:
    keys = list(allocations.keys())
    if not keys:
        return {}
    weights = np.asarray([float(allocations[key]) for key in keys], dtype=np.float32)
    cap = max(float(max_gross), 0.0)
    gross = float(np.sum(np.abs(weights)))
    if cap <= 0.0:
        return {key: 0.0 for key in keys}
    if gross <= cap + 1e-9:
        scaled = weights
    else:
        scaled = weights * (cap / max(gross, 1e-8))
    return {key: float(value) for key, value in zip(keys, scaled.tolist())}


__all__ = [
    "AllocationLeverageLimits",
    "AllocationRefinement",
    "forecast_band_width_pct",
    "forecast_move_pct",
    "leverage_limits_for_asset",
    "refine_allocation",
    "scale_allocations_to_gross_limit",
]
