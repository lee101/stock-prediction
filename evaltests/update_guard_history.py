#!/usr/bin/env python3
"""Append the latest baseline vs compile metrics to the guard history log."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
import argparse

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKTEST_DIR = REPO_ROOT / "evaltests" / "backtests"
DEFAULT_CONFIG_PATH = REPO_ROOT / "evaltests" / "guard_backtest_targets_compile.json"
HISTORY_PATH = REPO_ROOT / "evaltests" / "guard_compile_history.json"


def _load_json(path: Path) -> Mapping[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _extract_metrics(payload: Mapping[str, object] | None) -> dict[str, float]:
    if not payload or not isinstance(payload, Mapping):
        return {}
    strategies = payload.get("strategies")
    metrics: dict[str, float] = {}
    if isinstance(strategies, Mapping):
        for name in ("maxdiff", "simple"):
            strat = strategies.get(name)
            if isinstance(strat, Mapping):
                for key in ("return", "sharpe", "turnover"):
                    value = strat.get(key)
                    if isinstance(value, (int, float)):
                        metrics[f"{name}_{key}"] = float(value)
    extra = payload.get("metrics")
    if isinstance(extra, Mapping):
        value = extra.get("close_val_loss")
        if isinstance(value, (int, float)):
            metrics["close_val_loss"] = float(value)
    return metrics


def _diff(base: dict[str, float], compiled: dict[str, float]) -> dict[str, float]:
    keys = set(base.keys()) | set(compiled.keys())
    return {key: compiled.get(key, 0.0) - base.get(key, 0.0) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Append baseline vs compile metrics to history.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to compile target configuration (default: guard_backtest_targets_compile.json)",
    )
    parser.add_argument(
        "--variant",
        default="compile",
        help="Optional label for the compile variant (stored in the history log).",
    )
    args = parser.parse_args()

    config = _load_json(args.config)
    if not isinstance(config, list):
        raise RuntimeError(f"Invalid compile config: {args.config}")

    history = []
    if HISTORY_PATH.exists():
        try:
            history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = []

    timestamp = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    updates = []

    for entry in config:
        if not isinstance(entry, Mapping):
            continue
        symbol = str(entry.get("symbol", "")).upper()
        if not symbol:
            continue
        baseline_path = BACKTEST_DIR / f"gymrl_guard_confirm_{symbol.lower()}_real_full.json"
        compile_path = None
        if isinstance(entry.get("output_json"), str) and entry["output_json"]:
            compile_path = REPO_ROOT / entry["output_json"]
        if not compile_path:
            compile_path = BACKTEST_DIR / f"gymrl_guard_confirm_{symbol.lower()}_real_full_compile.json"
        compile_path = Path(compile_path)
        baseline = _extract_metrics(_load_json(baseline_path))
        compiled = _extract_metrics(_load_json(compile_path))
        if not baseline or not compiled:
            continue
        updates.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "baseline": baseline,
                "compile": compiled,
                "delta": _diff(baseline, compiled),
                "variant": args.variant,
            }
        )

    if not updates:
        print("No updates written; missing baseline or compile metrics.")
        return

    history.extend(updates)
    HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Appended {len(updates)} entries to {HISTORY_PATH}")


if __name__ == "__main__":
    main()
