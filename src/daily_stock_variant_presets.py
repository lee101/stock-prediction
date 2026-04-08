"""Named daily-stock backtest variant presets."""

from __future__ import annotations

from dataclasses import dataclass

import trade_daily_stock_prod as daily_stock


VariantSpec = daily_stock.BacktestVariantSpec


@dataclass(frozen=True)
class VariantPreset:
    name: str
    description: str
    variants: tuple[VariantSpec, ...]


CURRENT_LIVE_VARIANT = VariantSpec(name="current_live_12p5", allocation_pct=12.5)
SINGLE_STATIC_25_VARIANT = VariantSpec(name="single_static_25", allocation_pct=25.0)
PORTFOLIO2_STATIC_50_VARIANT = VariantSpec(name="portfolio2_static_50", allocation_pct=50.0, multi_position=2)
PORTFOLIO3_STATIC_50_VARIANT = VariantSpec(name="portfolio3_static_50", allocation_pct=50.0, multi_position=3)


PRESET_VARIANTS: dict[str, VariantPreset] = {
    "current_vs_candidates": VariantPreset(
        name="current_vs_candidates",
        description="Current live-equivalent config plus the strongest server-aware static candidates.",
        variants=(
            CURRENT_LIVE_VARIANT,
            SINGLE_STATIC_25_VARIANT,
            PORTFOLIO2_STATIC_50_VARIANT,
            PORTFOLIO3_STATIC_50_VARIANT,
        ),
    ),
    "current_only": VariantPreset(
        name="current_only",
        description="Only the current live-equivalent single-position configuration.",
        variants=(CURRENT_LIVE_VARIANT,),
    ),
    "promising_only": VariantPreset(
        name="promising_only",
        description="Only the short-window variants that beat the current live-equivalent baseline.",
        variants=(
            SINGLE_STATIC_25_VARIANT,
            PORTFOLIO2_STATIC_50_VARIANT,
            PORTFOLIO3_STATIC_50_VARIANT,
        ),
    ),
}


def preset_choices() -> list[str]:
    return sorted(PRESET_VARIANTS)


def resolve_variant_preset(name: str) -> VariantPreset:
    return PRESET_VARIANTS[str(name).strip()]
