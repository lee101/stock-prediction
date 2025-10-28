#!/usr/bin/env python3
"""
Aggregate per-variant Kronos/Toto summaries into a combined overlay plot.

This script expects individual summary JSON files produced by
``tests/tools/kronos_toto_btc_overlay.py`` under
``testresults/btc_kronos_toto_overlay/<variant>/`` and writes the merged
overlay image plus summary JSON back to ``testresults/btc_kronos_toto_overlay``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class VariantEntry:
    label: str
    model_type: str
    description: str
    config: dict
    env_overrides: dict
    price_mae: float
    pct_return_mae: float
    latency_s: float
    predicted_prices: np.ndarray
    metadata: dict


def _load_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("variants"):
        raise ValueError(f"Summary {path} contains no variants.")
    return data


def _build_variant_entries(summary: dict) -> List[VariantEntry]:
    entries: List[VariantEntry] = []
    for payload in summary["variants"]:
        entries.append(
            VariantEntry(
                label=str(payload["label"]),
                model_type=str(payload["model_type"]),
                description=str(payload.get("description", "")),
                config=dict(payload.get("config") or {}),
                env_overrides=dict(payload.get("env_overrides") or {}),
                price_mae=float(payload["price_mae"]),
                pct_return_mae=float(payload["pct_return_mae"]),
                latency_s=float(payload["latency_s"]),
                predicted_prices=np.asarray(payload["predicted_prices"], dtype=np.float64),
                metadata=dict(payload.get("metadata") or {}),
            )
        )
    return entries


def _plot_overlay(
    timestamps: Sequence[pd.Timestamp],
    actual_prices: Sequence[float],
    variants: Sequence[VariantEntry],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(timestamps, actual_prices, label="Actual close", color="#111827", linewidth=2.2)

    palette = plt.get_cmap("tab10")
    for idx, variant in enumerate(variants):
        color = palette(idx % palette.N)
        linestyle = "--" if variant.model_type.lower() == "toto" else "-"
        ax.plot(
            timestamps,
            variant.predicted_prices,
            label=variant.label,
            color=color,
            linewidth=1.7,
            linestyle=linestyle,
        )
        ax.scatter(
            timestamps,
            variant.predicted_prices,
            color=color,
            s=28,
            marker="s" if variant.model_type.lower() == "toto" else "o",
            alpha=0.85,
        )

    ax.set_title("BTCUSD Close vs. Kronos/Toto Forecast Variants")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Close Price (USD)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _to_serialisable(value):
    if isinstance(value, np.ndarray):
        return value.astype(np.float64).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _to_serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serialisable(item) for item in value]
    return value


def _save_summary(
    symbol: str,
    window: int,
    timestamps: Sequence[pd.Timestamp],
    actual_prices: Sequence[float],
    variants: Sequence[VariantEntry],
    output_path: Path,
) -> None:
    payload = {
        "symbol": symbol,
        "window": window,
        "timestamps": [ts.isoformat() for ts in timestamps],
        "actual_close": [float(price) for price in actual_prices],
        "variants": [],
    }
    for variant in variants:
        payload["variants"].append(
            {
                "label": variant.label,
                "model_type": variant.model_type,
                "description": variant.description,
                "config": _to_serialisable(variant.config),
                "env_overrides": _to_serialisable(variant.env_overrides),
                "price_mae": variant.price_mae,
                "pct_return_mae": variant.pct_return_mae,
                "latency_s": variant.latency_s,
                "predicted_prices": _to_serialisable(variant.predicted_prices),
                "metadata": _to_serialisable(variant.metadata),
            }
        )
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-variant Kronos/Toto summaries.")
    parser.add_argument("--symbol", default="BTCUSD", help="Target symbol (default: %(default)s).")
    parser.add_argument(
        "--source-root",
        default=REPO_ROOT / "testresults" / "btc_kronos_toto_overlay",
        help="Directory containing per-variant subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        default=REPO_ROOT / "testresults" / "btc_kronos_toto_overlay",
        help="Directory to store combined artefacts.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Evaluation window length (used for validation metadata).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbol = args.symbol.upper()
    window = int(args.window)

    source_root = Path(args.source_root)
    if not source_root.exists():
        raise FileNotFoundError(f"Source directory {source_root} does not exist.")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_suffix = f"{symbol.lower()}_overlay_summary.json"
    summary_paths = sorted(source_root.glob(f"*/{summary_suffix}"))
    if not summary_paths:
        raise FileNotFoundError(f"No per-variant summaries found under {source_root}.")

    base_summary = None
    combined_variants: List[VariantEntry] = []

    for path in summary_paths:
        summary = _load_summary(path)
        timestamps = pd.to_datetime(summary["timestamps"])
        actual_prices = np.asarray(summary["actual_close"], dtype=np.float64)

        if base_summary is None:
            base_summary = (timestamps, actual_prices)
        else:
            base_ts, base_prices = base_summary
            if len(timestamps) != len(base_ts) or not np.allclose(actual_prices, base_prices):
                raise ValueError(f"Actual price series mismatch in {path}.")

        combined_variants.extend(_build_variant_entries(summary))

    combined_variants.sort(key=lambda item: item.label.lower())
    timestamps, actual_prices = base_summary  # type: ignore[misc]

    plot_path = output_dir / f"{symbol.lower()}_overlay.png"
    summary_path = output_dir / summary_suffix

    _plot_overlay(timestamps, actual_prices, combined_variants, plot_path)
    _save_summary(symbol, window, timestamps, actual_prices, combined_variants, summary_path)

    print(f"[INFO] Wrote combined overlay -> {plot_path}")
    print(f"[INFO] Wrote combined summary -> {summary_path}")


if __name__ == "__main__":
    main()
