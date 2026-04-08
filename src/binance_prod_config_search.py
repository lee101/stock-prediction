from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from src.binance_hybrid_launch import (
    BinanceHybridLaunchConfig,
    parse_launch_script,
    resolve_target_launch_config,
)


DEFAULT_MAX_VARIANTS = 64
DEFAULT_MAX_CHECKPOINT_CONFIG_EVALS = DEFAULT_MAX_VARIANTS * 3
T = TypeVar("T")


@dataclass(frozen=True)
class ConfigVariant:
    symbols: tuple[str, ...]
    leverage: float
    output_dir: str
    slug: str


def _parse_symbol_set(raw: str) -> tuple[str, ...]:
    tokens = [token.strip().upper() for token in raw.replace(",", " ").split() if token.strip()]
    if not tokens:
        raise ValueError("symbol set must contain at least one symbol")
    return tuple(tokens)


def _slugify_symbols(symbols: tuple[str, ...]) -> str:
    return "_".join(re.sub(r"[^A-Za-z0-9]+", "", symbol).lower() for symbol in symbols)


def _slugify_leverage(leverage: float) -> str:
    return str(float(leverage)).replace("-", "m").replace(".", "p")


def _is_launch_variant(variant: ConfigVariant, launch_config: BinanceHybridLaunchConfig) -> bool:
    return tuple(variant.symbols) == tuple(launch_config.symbols) and float(variant.leverage) == float(launch_config.leverage)


def build_config_variants(
    *,
    launch_script: str | Path,
    symbol_set_specs: list[str],
    symbol_subset_sizes: list[int],
    leverage_options: list[float],
    output_root: str | Path,
    include_launch_variant: bool = False,
) -> list[ConfigVariant]:
    launch_cfg = parse_launch_script(launch_script, require_rl_checkpoint=False)
    symbol_sets = [_parse_symbol_set(spec) for spec in symbol_set_specs]
    launch_symbols = tuple(launch_cfg.symbols)
    for subset_size in symbol_subset_sizes:
        if subset_size < 1:
            raise ValueError("symbol subset size must be at least 1")
        if subset_size > len(launch_symbols):
            raise ValueError(
                f"symbol subset size {subset_size} exceeds launch symbol count {len(launch_symbols)}"
            )
        symbol_sets.extend(itertools.combinations(launch_symbols, subset_size))
    if not symbol_sets:
        symbol_sets = [launch_symbols]
    leverages = [float(leverage) for leverage in leverage_options] or [float(launch_cfg.leverage)]

    output_root_path = Path(output_root)
    variants: list[ConfigVariant] = []
    seen: set[tuple[tuple[str, ...], float]] = set()
    for symbols in symbol_sets:
        for leverage in leverages:
            target_cfg = resolve_target_launch_config(
                launch_script,
                symbols_override=",".join(symbols),
                leverage_override=leverage,
            )
            key = (tuple(target_cfg.symbols), float(target_cfg.leverage))
            if key in seen:
                continue
            seen.add(key)
            slug = f"{_slugify_symbols(tuple(target_cfg.symbols))}__lev_{_slugify_leverage(float(target_cfg.leverage))}"
            variants.append(
                ConfigVariant(
                    symbols=tuple(target_cfg.symbols),
                    leverage=float(target_cfg.leverage),
                    output_dir=str((output_root_path / slug).resolve()),
                    slug=slug,
                )
            )
    if include_launch_variant:
        launch_key = (tuple(launch_cfg.symbols), float(launch_cfg.leverage))
        if launch_key not in seen:
            slug = f"{_slugify_symbols(tuple(launch_cfg.symbols))}__lev_{_slugify_leverage(float(launch_cfg.leverage))}"
            variants.append(
                ConfigVariant(
                    symbols=tuple(launch_cfg.symbols),
                    leverage=float(launch_cfg.leverage),
                    output_dir=str((output_root_path / slug).resolve()),
                    slug=slug,
                )
            )
    return variants


def limit_config_variants(
    variants: list[ConfigVariant],
    *,
    launch_config: BinanceHybridLaunchConfig,
    max_variants: int | None,
    variant_offset: int = 0,
) -> list[ConfigVariant]:
    if variant_offset < 0:
        raise ValueError("variant_offset must be >= 0")
    if max_variants is None or max_variants == 0 or len(variants) <= max_variants:
        return list(variants)
    if max_variants < 0:
        raise ValueError("max_variants must be >= 0")
    start = variant_offset % len(variants)
    rotated = list(variants[start:]) + list(variants[:start])
    limited = rotated[:max_variants]
    if any(_is_launch_variant(variant, launch_config) for variant in limited):
        return limited
    launch_variant = next((variant for variant in variants if _is_launch_variant(variant, launch_config)), None)
    if launch_variant is not None and limited:
        limited[-1] = launch_variant
    return limited


def estimate_checkpoint_evals_per_variant(
    launch_config: BinanceHybridLaunchConfig,
    candidate_checkpoints: list[str | Path],
) -> int:
    count = len(candidate_checkpoints)
    if launch_config.rl_checkpoint:
        count += 1
    return max(1, count)


def max_variants_for_checkpoint_eval_budget(
    launch_config: BinanceHybridLaunchConfig,
    *,
    candidate_checkpoints: list[str | Path],
    max_checkpoint_config_evals: int | None,
) -> int | None:
    if max_checkpoint_config_evals is None:
        return None
    if max_checkpoint_config_evals < 0:
        raise ValueError("max_checkpoint_config_evals must be >= 0")
    if max_checkpoint_config_evals == 0:
        return None
    per_variant_evals = estimate_checkpoint_evals_per_variant(launch_config, candidate_checkpoints)
    return max_checkpoint_config_evals // per_variant_evals


def max_candidate_checkpoints_for_budget(
    launch_config: BinanceHybridLaunchConfig,
    *,
    variant_count: int,
    max_checkpoint_config_evals: int | None,
) -> int | None:
    if max_checkpoint_config_evals is None:
        return None
    if max_checkpoint_config_evals < 0:
        raise ValueError("max_checkpoint_config_evals must be >= 0")
    if max_checkpoint_config_evals == 0:
        return None
    if variant_count < 1:
        return 0
    per_variant_budget = max_checkpoint_config_evals // variant_count
    baseline_eval_slots = 1 if launch_config.rl_checkpoint else 0
    return max(0, per_variant_budget - baseline_eval_slots)


def limit_ranked_candidates(  # noqa: UP047
    ranked_items: list[T],
    *,
    max_candidates: int | None,
    offset: int = 0,
    preserve_first: bool = True,
) -> list[T]:
    if offset < 0:
        raise ValueError("candidate offset must be >= 0")
    if max_candidates is None or max_candidates == 0 or len(ranked_items) <= max_candidates:
        return list(ranked_items)
    if max_candidates < 0:
        raise ValueError("max_candidates must be >= 0")
    if max_candidates == 1 or not ranked_items:
        return list(ranked_items[:1])
    if not preserve_first:
        start = offset % len(ranked_items)
        rotated = list(ranked_items[start:]) + list(ranked_items[:start])
        return rotated[:max_candidates]

    leader = ranked_items[0]
    tail = list(ranked_items[1:])
    if not tail:
        return [leader]
    start = offset % len(tail)
    rotated_tail = tail[start:] + tail[:start]
    return [leader, *rotated_tail[: max_candidates - 1]]
