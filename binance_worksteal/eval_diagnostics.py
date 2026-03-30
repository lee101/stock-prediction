"""Helpers for reporting evaluation failures in worksteal tooling."""
from __future__ import annotations

from binance_worksteal.strategy import WorkStealConfig


def summarize_config(config: WorkStealConfig) -> str:
    parts = [
        f"dip={config.dip_pct:.0%}",
        f"tp={config.profit_target_pct:.0%}",
        f"sl={config.stop_loss_pct:.0%}",
        f"pos={config.max_positions}",
        f"hold={config.max_hold_days}",
        f"look={config.lookback_days}",
        f"ref={config.ref_price_method}",
        f"lev={config.max_leverage:.1f}x",
    ]
    if config.enable_shorts:
        parts.append("shorts=on")
    if config.sma_filter_period:
        parts.append(f"sma={config.sma_filter_period}")
    if config.realistic_fill:
        parts.append("realistic_fill=on")
    if config.daily_checkpoint_only:
        parts.append("daily_checkpoint_only=on")
    return " ".join(parts)


def format_eval_failure(
    context: str,
    engine: str,
    config: WorkStealConfig,
    start_date: str | None,
    end_date: str | None,
    exc: Exception,
) -> str:
    start = start_date or "start"
    end = end_date or "end"
    return (
        f"{context} {engine} evaluation failed for {start}..{end} "
        f"({exc.__class__.__name__}: {exc}); config {summarize_config(config)}"
    )
