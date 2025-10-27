#!/usr/bin/env python3
"""Render guard compile history into markdown summaries.

This script maintains two artefacts:

* ``guard_compile_history.md`` – a chronological view of every compile vs
  baseline comparison.
* ``guard_compile_stats.md`` – aggregated statistics per symbol so it is easy
  to spot trends (e.g., whether ``torch.compile`` consistently helps).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = REPO_ROOT / "evaltests" / "guard_compile_history.json"
HISTORY_MD_PATH = REPO_ROOT / "evaltests" / "guard_compile_history.md"
STATS_MD_PATH = REPO_ROOT / "evaltests" / "guard_compile_stats.md"

NUMBER_METRICS: tuple[str, ...] = (
    "maxdiff_return",
    "simple_return",
    "maxdiff_sharpe",
    "close_val_loss",
    "maxdiff_turnover",
)

METRIC_LABELS: dict[str, str] = {
    "maxdiff_return": "Δ MaxDiff Return",
    "simple_return": "Δ Simple Return",
    "maxdiff_sharpe": "Δ MaxDiff Sharpe",
    "close_val_loss": "Δ Val Loss",
    "maxdiff_turnover": "Δ MaxDiff Turnover",
}


def load_history() -> list[dict[str, object]]:
    if not HISTORY_PATH.exists():
        return []
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def fmt(value: object, precision: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}"
    return "n/a"


def collect_by_symbol(history: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for entry in history:
        symbol = str(entry.get("symbol", "")).strip()
        delta = entry.get("delta")
        if not symbol or not isinstance(delta, dict):
            continue
        grouped.setdefault(symbol, []).append(
            {
                "timestamp": entry.get("timestamp"),
                "delta": delta,
            }
        )
    return grouped


def summarise_metric(entries: list[dict[str, object]], metric: str) -> dict[str, object]:
    values: list[float] = []
    for item in entries:
        raw_delta = item.get("delta", {})
        value = raw_delta.get(metric)
        if isinstance(value, (int, float)):
            values.append(float(value))

    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "last": None,
            "rolling_mean": None,
            "positive": 0,
            "negative": 0,
            "zero": 0,
        }

    rolling_window = values[-5:]
    positive = sum(1 for v in values if v > 0)
    negative = sum(1 for v in values if v < 0)
    zero = len(values) - positive - negative
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "last": values[-1],
        "rolling_mean": statistics.mean(rolling_window),
        "positive": positive,
        "negative": negative,
        "zero": zero,
    }


def render_stats(history: list[dict[str, object]]) -> None:
    grouped = collect_by_symbol(history)
    lines = ["# Guard Compile Stats", ""]

    if not grouped:
        lines.append("_No compile runs recorded yet._")
        STATS_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {STATS_MD_PATH}")
        return

    symbols = sorted(grouped)

    lines.extend(
        [
            "## Entry Counts",
            "| Symbol | Entries | First Timestamp | Latest Timestamp |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for symbol in symbols:
        entries = grouped[symbol]
        first_ts = str(entries[0].get("timestamp", "")) if entries else ""
        last_ts = str(entries[-1].get("timestamp", "")) if entries else ""
        lines.append(
            "| {symbol} | {count} | {first} | {last} |".format(
                symbol=symbol,
                count=len(entries),
                first=first_ts,
                last=last_ts,
            )
        )
    lines.append("")

    for metric in NUMBER_METRICS:
        label = METRIC_LABELS.get(metric, metric)
        lines.extend(
            [
                f"## {label}",
                "| Symbol | Mean | Std Dev | Last | Rolling Mean (last 5) | Samples | Pos | Neg | Zero | Pos Ratio | Alert |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for symbol in symbols:
            stats = summarise_metric(grouped[symbol], metric)
            positive = stats["positive"]
            negative = stats["negative"]
            zero = stats["zero"]
            count = stats["count"] or 0
            pos_ratio = positive / count if count else 0.0
            mean_val = stats["mean"] or 0.0
            high_confidence = count >= 5 and abs(mean_val) > 0.01
            if high_confidence:
                if mean_val > 0:
                    alert = "promote"
                elif mean_val < 0:
                    alert = "regress"
                else:
                    alert = "watch"
            elif count >= 5 and abs(mean_val) > 0.005:
                alert = "watch"
            elif count >= 5 and pos_ratio <= 0.3:
                alert = "regress"
            else:
                alert = ""
            lines.append(
                "| {symbol} | {mean} | {std} | {last} | {rolling} | {count} | {pos} | {neg} | {zero} | {ratio} | {alert} |".format(
                    symbol=symbol,
                    mean=fmt(stats["mean"]),
                    std=fmt(stats["std"]),
                    last=fmt(stats["last"]),
                    rolling=fmt(stats["rolling_mean"]),
                    count=count,
                    pos=positive,
                    neg=negative,
                    zero=zero,
                    ratio=fmt(pos_ratio),
                    alert=alert,
                )
            )
        lines.append("")

    STATS_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {STATS_MD_PATH}")


def render_history(history: list[dict[str, object]]) -> None:
    lines = ["# Guard Compile History", ""]
    if not history:
        lines.append("_No compile runs recorded yet._")
    else:
        lines.append(
            "| Timestamp (UTC) | Symbol | Variant | Δ MaxDiff Return | Δ Simple Return | Δ MaxDiff Sharpe | Δ Val Loss |"
        )
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
        for entry in history:
            timestamp = str(entry.get("timestamp", ""))
            symbol = str(entry.get("symbol", ""))
            variant = str(entry.get("variant", ""))
            delta = entry.get("delta")
            maxdiff_delta = delta.get("maxdiff_return") if isinstance(delta, dict) else None
            simple_delta = delta.get("simple_return") if isinstance(delta, dict) else None
            sharpe_delta = delta.get("maxdiff_sharpe") if isinstance(delta, dict) else None
            loss_delta = delta.get("close_val_loss") if isinstance(delta, dict) else None
            lines.append(
                "| {ts} | {sym} | {variant} | {md} | {sd} | {sh} | {ld} |".format(
                    ts=timestamp,
                    sym=symbol,
                    variant=variant if variant else "compile",
                    md=fmt(maxdiff_delta),
                    sd=fmt(simple_delta),
                    sh=fmt(sharpe_delta),
                    ld=fmt(loss_delta, precision=5),
                )
            )

    HISTORY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {HISTORY_MD_PATH}")


def main() -> None:
    history = load_history()
    render_history(history)
    render_stats(history)


if __name__ == "__main__":
    main()
