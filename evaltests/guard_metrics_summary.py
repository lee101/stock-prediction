#!/usr/bin/env python3
"""
Aggregate GymRL guard telemetry across evaluation artefacts.

This helper reads:
  - evaltests/gymrl_guard_analysis.json   (hold-out A/B records)
  - evaltests/rl_benchmark_results.json   (latest validation runs)

and emits a concise summary highlighting guard hit rates, turnover deltas,
and leverage impacts. The goal is to quickly sanity check whether the guards
behave as intended before/after running new sweeps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, MutableMapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
GUARD_ANALYSIS_PATH = REPO_ROOT / "evaltests" / "gymrl_guard_analysis.json"
BENCHMARK_RESULTS_PATH = REPO_ROOT / "evaltests" / "rl_benchmark_results.json"
SCOREBOARD_HISTORY_PATH = REPO_ROOT / "evaltests" / "scoreboard_history.json"
BACKTEST_DIR = REPO_ROOT / "evaltests" / "backtests"
BASELINE_PATH = REPO_ROOT / "evaltests" / "baseline_pnl_summary.json"


def _load_json(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse {path}: {exc}") from exc


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%" if isinstance(value, (int, float)) else "n/a"


def summarise_holdout() -> list[str]:
    analysis = _load_json(GUARD_ANALYSIS_PATH)
    lines: list[str] = []
    if not analysis:
        lines.append("No hold-out guard analysis found.")
        return lines

    def _summary_block(label: str, payload: Mapping[str, object]) -> list[str]:
        baseline = payload.get("baseline", {})
        guarded = payload.get("guarded_calibrated", {})
        delta = payload.get("delta_guard_minus_baseline", {})
        if not delta and "delta_guard_calibrated_minus_baseline" in analysis:
            delta = analysis["delta_guard_calibrated_minus_baseline"]
        lines = [
            f"- **{label}**",
            f"  - Baseline cumulative return: {baseline.get('cumulative_return', 'n/a'):.6f}"
            if isinstance(baseline, Mapping) and isinstance(baseline.get("cumulative_return"), (int, float))
            else f"  - Baseline cumulative return: {baseline.get('cumulative_return', 'n/a')}",
            f"  - Guarded cumulative return: {guarded.get('cumulative_return', 'n/a')}",
            f"  - Turnover delta: {delta.get('average_turnover', 'n/a')}",
            f"  - Guard hit rates (neg/turn/draw): {_pct(guarded.get('guard_negative_return_hit_rate', 0.0))} / {_pct(guarded.get('guard_turnover_hit_rate', 0.0))} / {_pct(guarded.get('guard_drawdown_hit_rate', 0.0))}",
        ]
        return lines

    lines.append("### Hold-Out Windows")
    lines.append("")
    lines.extend(_summary_block("Start index 3781 (stress slice)", analysis))
    additional = analysis.get("additional_windows", {})
    if isinstance(additional, Mapping):
        for key, payload in additional.items():
            if isinstance(payload, MutableMapping):
                label = f"Start index {key.split('_')[-1]}"
                lines.extend(_summary_block(label, payload))
    latest = analysis.get("latest_run")
    if isinstance(latest, Mapping):
        holdout_latest = latest.get("holdout_start_3781")
        if isinstance(holdout_latest, Mapping):
            lines.append("- **Latest guard confirm (start index 3781)**")
            lines.append(f"  - Cumulative return: {holdout_latest.get('cumulative_return', 'n/a')}")
            lines.append(
                "  - Guard hit rates (neg/turn/draw): "
                f"{_pct(holdout_latest.get('guard_negative_return_hit_rate', 0.0))} / "
                f"{_pct(holdout_latest.get('guard_turnover_hit_rate', 0.0))} / "
                f"{_pct(holdout_latest.get('guard_drawdown_hit_rate', 0.0))}"
            )
            lines.append(
                f"  - Avg turnover: {holdout_latest.get('average_turnover', 'n/a')} (avg leverage scale {holdout_latest.get('guard_average_leverage_scale', 'n/a')})"
            )
        extra_windows = latest.get("additional_windows")
        if isinstance(extra_windows, Mapping):
            lines.append("")
            lines.append("Additional guard confirm windows:")
            for start, metrics in sorted(extra_windows.items(), key=lambda x: int(x[0])):
                if not isinstance(metrics, Mapping):
                    continue
                lines.append(
                    f"- start {start}: return {metrics.get('cumulative_return', 'n/a')}, "
                    f"turn={metrics.get('average_turnover', 'n/a')}, "
                    f"guards neg/turn/draw = "
                    f"{_pct(metrics.get('guard_negative_return_hit_rate', 0.0))}/"
                    f"{_pct(metrics.get('guard_turnover_hit_rate', 0.0))}/"
                    f"{_pct(metrics.get('guard_drawdown_hit_rate', 0.0))}"
                )
    lines.append("")
    return lines


def summarise_validation() -> list[str]:
    results = _load_json(BENCHMARK_RESULTS_PATH)
    scoreboard = results.get("scoreboard", [])
    lines: list[str] = []
    lines.append("### Latest GymRL Validation Runs")
    lines.append("")
    found = False
    if isinstance(scoreboard, list):
        for entry in scoreboard:
            if not isinstance(entry, Mapping):
                continue
            if entry.get("module") != "gymrl":
                continue
            details = entry.get("details", {})
            if not isinstance(details, Mapping):
                details = {}
            guard_config = entry.get("extra", {}).get("regime_config") if isinstance(entry.get("extra"), Mapping) else {}
            guard_cfg_str: Optional[str] = None
            if isinstance(guard_config, Mapping) and guard_config:
                parts = []
                if "regime_drawdown_threshold" in guard_config:
                    parts.append(f"draw={guard_config['regime_drawdown_threshold']}")
                if "regime_negative_return_threshold" in guard_config:
                    parts.append(f"neg={guard_config['regime_negative_return_threshold']}")
                if "regime_turnover_threshold" in guard_config:
                    parts.append(f"turn={guard_config['regime_turnover_threshold']}")
                if parts:
                    guard_cfg_str = ", ".join(parts)
            lines.append(f"- **{entry.get('name')}**")
            lines.append(f"  - Cumulative return: {details.get('cumulative_return', 'n/a')}")
            lines.append(f"  - Avg daily return: {details.get('average_daily_return', 'n/a')}")
            lines.append(
                "  - Guard hit rates (neg/turn/draw): "
                f"{details.get('guard_negative_hit_rate', 'n/a')} / "
                f"{details.get('guard_turnover_hit_rate', 'n/a')} / "
                f"{details.get('guard_drawdown_hit_rate', 'n/a')}"
            )
            if guard_cfg_str:
                lines.append(f"  - Guard config: {guard_cfg_str}")
            found = True
    if not found:
        lines.append("No GymRL entries found in the current scoreboard.")
    lines.append("")
    return lines


def summarise_scoreboard_history(limit: int = 5) -> list[str]:
    raw_history = _load_json(SCOREBOARD_HISTORY_PATH)
    if not isinstance(raw_history, list) or not raw_history:
        return []
    lines = ["### Guard Metrics Trend (latest history)", ""]
    count = 0
    for snapshot in reversed(raw_history):
        if count >= limit:
            break
        if not isinstance(snapshot, Mapping):
            continue
        timestamp = snapshot.get("timestamp", "unknown")
        scoreboard = snapshot.get("scoreboard")
        if not isinstance(scoreboard, list):
            continue
        for entry in scoreboard:
            if not isinstance(entry, Mapping) or entry.get("module") != "gymrl":
                continue
            name = entry.get("name", "gymrl")
            details = entry.get("details", {})
            if not isinstance(details, Mapping):
                details = {}
            lines.append(f"- {timestamp} – {name}")
            lines.append(
                "  - Guard hit rates (neg/turn/draw): "
                f"{details.get('guard_negative_hit_rate', 'n/a')} / "
                f"{details.get('guard_turnover_hit_rate', 'n/a')} / "
                f"{details.get('guard_drawdown_hit_rate', 'n/a')}"
            )
            lines.append(
                f"  - Avg daily return: {details.get('average_daily_return', 'n/a')}, Turnover: {details.get('turnover', 'n/a')}"
            )
            count += 1
    if count == 0:
        return []
    lines.append("")
    return lines


def summarise_backtests() -> list[str]:
    if not BACKTEST_DIR.exists():
        return []
    rows = []
    for path in sorted(BACKTEST_DIR.glob("gymrl_guard_confirm_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        symbol = data.get("symbol", path.stem)
        strategies = data.get("strategies", {})
        maxdiff = strategies.get("maxdiff", {}) if isinstance(strategies, Mapping) else {}
        rows.append(
            {
                "symbol": symbol,
                "maxdiff_return": maxdiff.get("return"),
                "maxdiff_sharpe": maxdiff.get("sharpe"),
                "simple_return": strategies.get("simple", {}).get("return") if isinstance(strategies, Mapping) else None,
            }
        )
    if not rows:
        return []
    lines = ["### Mock Backtest Results", "", "| Symbol | MaxDiff Return | MaxDiff Sharpe | Simple Return |", "| --- | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(
            f"| {row['symbol']} | "
            f"{row['maxdiff_return'] if row['maxdiff_return'] is not None else 'n/a'} | "
            f"{row['maxdiff_sharpe'] if row['maxdiff_sharpe'] is not None else 'n/a'} | "
            f"{row['simple_return'] if row['simple_return'] is not None else 'n/a'} |"
        )
    lines.append("")
    return lines


def main() -> None:
    lines = ["# GymRL Guard Telemetry Summary", ""]
    baseline = _load_json(BASELINE_PATH)
    if isinstance(baseline, Mapping):
        trade_history = baseline.get("trade_history")
        realised = None
        if isinstance(trade_history, Mapping):
            realised = trade_history.get("total_realized_pnl")
        if isinstance(realised, (int, float)):
            lines.append(f"- Production baseline realised PnL (latest snapshot): {realised:,.2f}")
            lines.append("")
    lines.extend(summarise_holdout())
    lines.extend(summarise_validation())
    lines.extend(summarise_scoreboard_history())
    backtest_section = summarise_backtests()
    if backtest_section:
        lines.extend(backtest_section)
    output_path = REPO_ROOT / "evaltests" / "guard_metrics_summary.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Guard summary written to {output_path}")


if __name__ == "__main__":
    main()
