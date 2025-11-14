from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt

GATE_STRATEGIES = [
    "VolAdjusted_10pct",
    "VolAdjusted_10pct_StockDirShutdown",
    "VolAdjusted_10pct_UnprofitShutdown",
    "VolAdjusted_10pct_UnprofitShutdown_StockDirShutdown",
    "VolAdjusted_15pct",
    "VolAdjusted_15pct_StockDirShutdown",
    "VolAdjusted_15pct_UnprofitShutdown",
    "VolAdjusted_15pct_UnprofitShutdown_StockDirShutdown",
    "CorrAware_Moderate",
    "CorrAware_Moderate_StockDirShutdown",
    "CorrAware_Moderate_UnprofitShutdown",
    "CorrAware_Moderate_UnprofitShutdown_StockDirShutdown",
]


def load_gate_metrics(path: Path) -> List[Dict[str, float]]:
    data = json.loads(path.read_text())
    results = []
    for entry in data["results"]:
        if entry["strategy"] not in GATE_STRATEGIES:
            continue
        result = {
            "label": entry["strategy"],
            "sortino": entry.get("sortino_ratio", 0.0),
            "annual_return": entry.get("annualized_return_pct", 0.0),
            "max_dd": entry.get("max_dd_pct", 0.0),
            "family": "Gate",
        }
        results.append(result)
    return results


def load_model_metrics(model_dir: Path, label: str) -> List[Dict[str, float]]:
    metrics: List[Dict[str, float]] = []
    neural_file = model_dir / "neural_metrics.json"
    xgb_file = model_dir / "xgboost_metrics.json"

    if neural_file.exists():
        neural_data = json.loads(neural_file.read_text())
        metrics.append(
            {
                "label": f"{label}_Neural",
                "sortino": neural_data.get("sortino", 0.0),
                "annual_return": neural_data.get("annual_return", 0.0) * 100.0,
                "max_dd": 0.0,
                "family": "Neural",
            }
        )
    if xgb_file.exists():
        xgb_data = json.loads(xgb_file.read_text())
        metrics.append(
            {
                "label": f"{label}_XGB",
                "sortino": xgb_data.get("sortino", 0.0),
                "annual_return": xgb_data.get("annual_return", 0.0),
                "max_dd": 0.0,
                "family": "XGBoost",
            }
        )
    return metrics


def plot_metrics(entries: Sequence[Dict[str, float]], output_path: Path) -> None:
    labels = [entry["label"] for entry in entries]
    sortinos = [entry["sortino"] for entry in entries]
    ann_returns = [entry["annual_return"] for entry in entries]
    colors = {
        "Gate": "#1b9e77",
        "Neural": "#d95f02",
        "XGBoost": "#7570b3",
    }
    bar_colors = [colors.get(entry["family"], "#333333") for entry in entries]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    axes[0].bar(labels, sortinos, color=bar_colors)
    axes[0].set_ylabel("Sortino Ratio")
    axes[0].set_title("Strategy Sortino (Gate Variants vs Neural/XGBoost)")
    axes[0].tick_params(axis="x", rotation=60)

    axes[1].bar(labels, ann_returns, color=bar_colors)
    axes[1].set_ylabel("Annualized Return (%)")
    axes[1].set_title("Annualized Return")
    axes[1].tick_params(axis="x", rotation=60)

    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="w", label=name, markerfacecolor=color, markersize=10)
        for name, color in colors.items()
    ]
    axes[0].legend(handles=legend_handles, loc="upper left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot gate strategies vs neural/xgboost baselines.")
    parser.add_argument(
        "--fast-results",
        default="strategytraining/sizing_strategy_fast_test_results.json",
        help="Path to sizing_strategy_fast_test_results.json",
    )
    parser.add_argument(
        "--model-report",
        action="append",
        default=[],
        help="Model report directories (each should contain neural/xgboost metrics). Can repeat.",
    )
    parser.add_argument(
        "--output",
        default="strategytrainingneural/reports/strategy_comparison.png",
        help="Output path for the PNG chart.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries: List[Dict[str, float]] = []
    entries.extend(load_gate_metrics(Path(args.fast_results)))
    for report_dir in args.model_report:
        path = Path(report_dir)
        label = path.name
        entries.extend(load_model_metrics(path, label))
    if not entries:
        raise SystemExit("No metrics found; ensure --fast-results and --model-report inputs are correct.")
    entries.sort(key=lambda row: row["sortino"], reverse=True)
    plot_metrics(entries, Path(args.output))


if __name__ == "__main__":
    main()
