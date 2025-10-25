#!/usr/bin/env python3
"""Generate an HTML plot for provider latency history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Provider Latency History</title>
  <script src=\"https://cdn.plot.ly/plotly-2.27.0.min.js\"></script>
  <style>
    body { font-family: system-ui, sans-serif; margin: 24px; }
    #chart { width: 100%; max-width: 960px; height: 480px; }
  </style>
</head>
<body>
  <h1>Provider Latency History</h1>
  <div id=\"chart\"></div>
  <script>
    const data = __DATA__;
    const traces = data.map(provider => ({
      x: provider.timestamps,
      y: provider.avg_ms,
      name: provider.name + ' avg',
      mode: 'lines+markers'
    })).concat(data.map(provider => ({
      x: provider.timestamps,
      y: provider.p95_ms,
      name: provider.name + ' p95',
      mode: 'lines',
      line: { dash: 'dash' }
    })));
    Plotly.newPlot('chart', traces, {
      margin: { t: 40 },
      xaxis: { title: 'Timestamp' },
      yaxis: { title: 'Latency (ms)' },
      legend: { orientation: 'h' }
    });
  </script>
</body>
</html>
"""


def load_history(path: Path, window: int) -> Dict[str, List[Dict[str, float]]]:
    if not path.exists():
        raise FileNotFoundError(f"latency history not found: {path}")
    entries: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    entries.sort(key=lambda item: item["timestamp"])
    tail = entries[-window:] if window > 0 else entries
    providers: Dict[str, Dict[str, List[float]]] = {}
    for snap in tail:
        timestamp = snap["timestamp"]
        for provider, stats in snap.get("aggregates", {}).items():
            bucket = providers.setdefault(
                provider,
                {"timestamps": [], "avg_ms": [], "p95_ms": []},
            )
            avg = stats.get("avg_ms")
            p95 = stats.get("p95_ms")
            if avg is None or p95 is None:
                continue
            bucket["timestamps"].append(timestamp)
            bucket["avg_ms"].append(avg)
            bucket["p95_ms"].append(p95)
    return providers


def serialize_data(providers: Dict[str, Dict[str, List[float]]]) -> str:
    dataset = []
    for name, series in sorted(providers.items()):
        dataset.append(
            {
                "name": name,
                "timestamps": series["timestamps"],
                "avg_ms": series["avg_ms"],
                "p95_ms": series["p95_ms"],
            }
        )
    return json.dumps(dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create HTML plot for latency history.")
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_rolling_history.jsonl"),
        help="JSONL history created by provider_latency_rolling.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_history.html"),
        help="HTML output path.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Number of snapshots to include (0 = all).",
    )
    args = parser.parse_args()

    providers = load_history(args.history, args.window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    html = HTML_TEMPLATE.replace("__DATA__", serialize_data(providers))
    args.output.write_text(html, encoding="utf-8")
    print(f"[info] Wrote latency history plot to {args.output}")


if __name__ == "__main__":
    main()
