from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


MetricPart = Tuple[str, Optional[float], int]


def format_metric_parts(parts: Sequence[MetricPart]) -> str:
    """
    Convert a list of (name, value, digits) tuples into a compact metric string.

    Values that are None or non-numeric are skipped.
    """
    formatted = []
    for name, value, digits in parts:
        if value is None:
            continue
        try:
            formatted.append(f"{name}={value:.{digits}f}")
        except (TypeError, ValueError):
            continue
    return " ".join(formatted)


def _build_probe_summary(data: Dict[str, Any]) -> Optional[str]:
    if data.get("trade_mode") != "probe":
        return None

    probe_notes: list[str] = []
    if data.get("pending_probe"):
        probe_notes.append("pending")
    if data.get("probe_active"):
        probe_notes.append("active")
    if data.get("probe_transition_ready"):
        probe_notes.append("transition-ready")
    if data.get("probe_expired"):
        probe_notes.append("expired")

    probe_age = data.get("probe_age_seconds")
    if probe_age is not None:
        try:
            probe_notes.append(f"age={int(probe_age)}s")
        except (TypeError, ValueError):
            probe_notes.append(f"age={probe_age}")

    probe_time_info = []
    if data.get("probe_started_at"):
        probe_time_info.append(f"start={data['probe_started_at']}")
    if data.get("probe_expires_at"):
        probe_time_info.append(f"expires={data['probe_expires_at']}")
    probe_notes.extend(probe_time_info)

    if not probe_notes:
        return None
    return "probe=" + ",".join(str(note) for note in probe_notes)


def build_analysis_summary_messages(symbol: str, data: Dict[str, Any]) -> Tuple[str, str]:
    """
    Build the compact and detailed analysis summary strings for logging.

    Returns:
        compact_message: Single-line summary for compact logging.
        detailed_message: Multi-line summary for verbose logging.
    """
    status_parts = [
        f"{symbol} analysis",
        f"strategy={data.get('strategy')}",
        f"side={data.get('side')}",
        f"mode={data.get('trade_mode', 'normal')}",
        f"blocked={data.get('trade_blocked', False)}",
    ]

    strategy_returns = data.get("strategy_returns", {}) or {}
    returns_metrics = format_metric_parts(
        [
            ("avg", data.get("avg_return"), 3),
            ("annual", data.get("annual_return"), 3),
            ("simple", data.get("simple_return"), 3),
            ("all", strategy_returns.get("all_signals"), 3),
            ("takeprofit", strategy_returns.get("takeprofit"), 3),
            ("highlow", strategy_returns.get("highlow"), 3),
            ("maxdiff", strategy_returns.get("maxdiff"), 3),
            ("maxdiffalwayson", strategy_returns.get("maxdiffalwayson"), 3),
            ("unprofit", data.get("unprofit_shutdown_return"), 3),
            ("composite", data.get("composite_score"), 3),
        ]
    )

    edges_metrics = format_metric_parts(
        [
            ("move", data.get("predicted_movement"), 3),
            ("expected_pct", data.get("expected_move_pct"), 5),
            ("price_skill", data.get("price_skill"), 5),
            ("edge_strength", data.get("edge_strength"), 5),
            ("directional", data.get("directional_edge"), 5),
        ]
    )

    prices_metrics = format_metric_parts(
        [
            ("pred_close", data.get("predicted_close"), 3),
            ("pred_high", data.get("predicted_high"), 3),
            ("pred_low", data.get("predicted_low"), 3),
            ("last_close", data.get("last_close"), 3),
        ]
    )

    walk_forward_notes = data.get("walk_forward_notes")

    summary_parts = [
        " ".join(status_parts),
        f"returns[{returns_metrics or '-'}]",
        f"edges[{edges_metrics or '-'}]",
        f"prices[{prices_metrics or '-'}]",
    ]

    block_reason = data.get("block_reason")
    if data.get("trade_blocked") and block_reason:
        summary_parts.append(f"block_reason={block_reason}")
    if walk_forward_notes:
        summary_parts.append("walk_forward_notes=" + "; ".join(str(note) for note in walk_forward_notes))

    probe_summary = _build_probe_summary(data)
    if probe_summary:
        summary_parts.append(probe_summary)

    compact_message = " | ".join(summary_parts)

    detail_lines = [" ".join(status_parts)]
    detail_lines.append(f"  returns: {returns_metrics or '-'}")
    detail_lines.append(f"  edges: {edges_metrics or '-'}")
    detail_lines.append(f"  prices: {prices_metrics or '-'}")

    walk_forward_metrics = format_metric_parts(
        [
            ("oos", data.get("walk_forward_oos_sharpe"), 2),
            ("turnover", data.get("walk_forward_turnover"), 2),
            ("highlow", data.get("walk_forward_highlow_sharpe"), 2),
            ("takeprofit", data.get("walk_forward_takeprofit_sharpe"), 2),
            ("maxdiff", data.get("walk_forward_maxdiff_sharpe"), 2),
        ]
    )
    if walk_forward_metrics:
        detail_lines.append(f"  walk_forward: {walk_forward_metrics}")

    if data.get("trade_blocked") and block_reason:
        detail_lines.append(f"  block_reason: {block_reason}")

    if walk_forward_notes:
        detail_lines.append("  walk_forward_notes: " + "; ".join(str(note) for note in walk_forward_notes))

    if probe_summary:
        detail_lines.append("  " + probe_summary.replace("=", ": ", 1))

    detailed_message = "\n".join(detail_lines)
    return compact_message, detailed_message


__all__ = [
    "MetricPart",
    "format_metric_parts",
    "build_analysis_summary_messages",
]
