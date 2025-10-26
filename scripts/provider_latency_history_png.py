#!/usr/bin/env python3
"""Render provider latency history to a PNG thumbnail."""

from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFgwJ/l0uYxgAAAABJRU5ErkJggg=="
)


def load_history(path: Path, window: int) -> Dict[str, Dict[str, List[float]]]:
    entries: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    entries.sort(key=lambda item: item["timestamp"])
    tail = entries[-window:] if window > 0 else entries
    providers: Dict[str, Dict[str, List[float]]] = {}
    for snap in tail:
        ts = snap["timestamp"]
        for provider, stats in snap.get("aggregates", {}).items():
            bucket = providers.setdefault(provider, {"timestamps": [], "avg_ms": []})
            avg = stats.get("avg_ms")
            if avg is None:
                continue
            bucket["timestamps"].append(ts)
            bucket["avg_ms"].append(avg)
    return providers


def write_placeholder(path: Path) -> None:
    path.write_bytes(PLACEHOLDER_PNG)
    print(f"[warn] Plotly/kaleido not available; wrote placeholder PNG to {path}")


def render_with_plotly(path: Path, history: Dict[str, Dict[str, List[float]]], threshold: float) -> None:
    import plotly.graph_objects as go  # type: ignore

    fig = go.Figure()
    for provider, series in sorted(history.items()):
        timestamps = series["timestamps"]
        avgs = series["avg_ms"]
        fig.add_trace(
            go.Scatter(x=timestamps, y=avgs, mode="lines+markers", name=f"{provider} avg")
        )
        if avgs:
            baseline = sum(avgs) / len(avgs)
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[baseline] * len(timestamps),
                    mode="lines",
                    line=dict(color="gray", dash="dot"),
                    name=f"{provider} mean",
                    showlegend=True,
                )
            )
            if threshold > 0:
                upper = baseline + threshold
                lower = baseline - threshold
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=[upper] * len(timestamps),
                        mode="lines",
                        line=dict(color="orange", dash="dash"),
                        name=f"{provider} mean+{threshold}ms",
                        showlegend=True,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=[lower] * len(timestamps),
                        mode="lines",
                        line=dict(color="orange", dash="dash"),
                        name=f"{provider} mean-{threshold}ms",
                        showlegend=True,
                    )
                )
        if len(avgs) >= 2 and threshold > 0:
            jumps_x = []
            jumps_y = []
            prev = avgs[0]
            for ts, current in zip(timestamps[1:], avgs[1:]):
                if abs(current - prev) >= threshold:
                    jumps_x.append(ts)
                    jumps_y.append(current)
                prev = current
            if jumps_x:
                fig.add_trace(
                    go.Scatter(
                        x=jumps_x,
                        y=jumps_y,
                        mode="markers",
                        marker=dict(color="red", size=10, symbol="x"),
                        name=f"{provider} Δ≥{threshold}ms",
                        showlegend=True,
                    )
                )
    fig.update_layout(
        title="Rolling Provider Latency (avg)",
        xaxis_title="Timestamp",
        yaxis_title="Latency (ms)",
        margin=dict(t=40, l=40, r=20, b=40),
        width=640,
        height=360,
    )
    fig.write_image(str(path))
    print(f"[info] Wrote latency history PNG to {path} (plotly)")


def render_with_matplotlib(path: Path, history: Dict[str, Dict[str, List[float]]], threshold: float) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    plt.figure(figsize=(6.4, 3.6))
    for provider, series in sorted(history.items()):
        timestamps = [datetime.fromisoformat(ts) for ts in series["timestamps"]]
        avgs = series["avg_ms"]
        plt.plot(timestamps, avgs, marker="o", label=f"{provider} avg")
        if avgs:
            baseline = sum(avgs) / len(avgs)
            plt.plot(timestamps, [baseline] * len(timestamps), linestyle="--", color="gray")
            if threshold > 0:
                upper = baseline + threshold
                lower = baseline - threshold
                plt.plot(timestamps, [upper] * len(timestamps), linestyle=":", color="orange")
                plt.plot(timestamps, [lower] * len(timestamps), linestyle=":", color="orange")
        if len(avgs) >= 2 and threshold > 0:
            prev = avgs[0]
            for ts, current in zip(timestamps[1:], avgs[1:]):
                if abs(current - prev) >= threshold:
                    plt.scatter(ts, current, color="red", marker="x")
                prev = current
    plt.title("Rolling Provider Latency (avg)")
    plt.xlabel("Timestamp")
    plt.ylabel("Latency (ms)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"[info] Wrote latency history PNG to {path} (matplotlib)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PNG plot for latency history.")
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_rolling_history.jsonl"),
        help="JSONL history from provider_latency_rolling.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_history.png"),
        help="PNG output path.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Number of snapshots to include (0 = all).",
    )
    parser.add_argument(
        "--warning-threshold",
        type=float,
        default=40.0,
        help="Highlight points where avg latency jumps by this many ms between snapshots.",
    )
    args = parser.parse_args()

    if not args.history.exists():
        raise FileNotFoundError(f"Latency history not found: {args.history}")

    history = load_history(args.history, args.window)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        render_with_plotly(args.output, history, args.warning_threshold)
        return
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Failed to render PNG with Plotly ({exc}); trying matplotlib fallback.")
    try:
        render_with_matplotlib(args.output, history, args.warning_threshold)
        return
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Matplotlib fallback failed ({exc}); writing placeholder image.")
        write_placeholder(args.output)


if __name__ == "__main__":
    main()
