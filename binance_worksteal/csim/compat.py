"""Compatibility helpers for deciding when the C simulator is safe to use."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def get_csim_incompatibility_reasons(config) -> list[str]:
    """Return Python-only strategy features that the C sim would ignore or mis-model."""
    reasons: list[str] = []

    if tuple(getattr(config, "dip_pct_fallback", ()) or ()):
        reasons.append("dip_pct_fallback")
    if bool(getattr(config, "realistic_fill", False)):
        reasons.append("realistic_fill")
    if bool(getattr(config, "daily_checkpoint_only", False)):
        reasons.append("daily_checkpoint_only")
    if str(getattr(config, "ref_price_method", "high")) != "high":
        reasons.append(f"ref_price_method={getattr(config, 'ref_price_method', 'high')}")
    if float(getattr(config, "market_breadth_filter", 0.0)) > 0.0:
        reasons.append("market_breadth_filter")
    if int(getattr(config, "sma_filter_period", 0)) > 0 and str(
        getattr(config, "sma_check_method", "current")
    ) != "current":
        reasons.append(f"sma_check_method={getattr(config, 'sma_check_method', 'current')}")
    if int(getattr(config, "rsi_filter", 0)) > 0:
        reasons.append("rsi_filter")
    if float(getattr(config, "volume_spike_filter", 0.0)) > 0.0:
        reasons.append("volume_spike_filter")
    if bool(getattr(config, "adaptive_dip", False)):
        reasons.append("adaptive_dip")
    if (
        int(getattr(config, "risk_off_trigger_sma_period", 0)) > 0
        or int(getattr(config, "risk_off_trigger_momentum_period", 0)) > 0
    ):
        reasons.append("risk_off regime switching")
    if float(getattr(config, "deleverage_threshold", 0.0)) > 0.0:
        reasons.append("deleverage_threshold")
    if getattr(config, "initial_holdings", None):
        reasons.append("initial_holdings")
    if str(getattr(config, "base_asset_symbol", "")).strip():
        reasons.append("base_asset_symbol")
    if float(getattr(config, "forecast_bias_weight", 0.0)) > 0.0:
        reasons.append("forecast_bias_weight")

    return reasons


def config_supports_csim(config) -> bool:
    return not get_csim_incompatibility_reasons(config)


def summarize_csim_incompatibility(config, max_items: int = 4) -> str:
    reasons = get_csim_incompatibility_reasons(config)
    if not reasons:
        return ""
    shown = reasons[:max_items]
    remaining = len(reasons) - len(shown)
    suffix = f" (+{remaining} more)" if remaining > 0 else ""
    return ", ".join(shown) + suffix


def find_first_csim_incompatible_config(configs: Iterable[Any]) -> tuple[int, Any, str] | None:
    for index, config in enumerate(configs):
        if not config_supports_csim(config):
            return index, config, summarize_csim_incompatibility(config)
    return None


def assert_csim_compatible_configs(configs: Iterable[Any], *, context: str) -> None:
    incompatible = find_first_csim_incompatible_config(configs)
    if incompatible is None:
        return
    index, _config, summary = incompatible
    raise ValueError(f"{context} generated C-sim-incompatible config #{index}: {summary}")
