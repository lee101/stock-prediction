#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from newnanoalpacahourlyexp.data import _effective_min_history_bars
from src.alpaca_stock_expansion import (
    StockExpansionCandidate,
    candidate_lora_command,
    count_candidate_history_rows,
    default_stock_expansion_candidates,
    load_stock_expansion_manifest,
    split_candidates_by_history,
    stock_expansion_sort_key,
    summarize_reforecast_result,
    write_stock_expansion_manifest,
)


DEFAULT_CHECKPOINT = (
    "binanceneural/checkpoints/"
    "alpaca_cross_global_mixed14_robust_short_seq128_lb4000_20260205_2319/epoch_004.pt"
)


def _parse_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [token.strip().upper() for token in str(raw).split(",") if token.strip()]


def _candidate_args(candidate: StockExpansionCandidate) -> tuple[list[str], list[str]]:
    if candidate.side == "short":
        return [], [candidate.symbol]
    if candidate.side == "both":
        return [], []
    return [candidate.symbol], []


def _run_command(cmd: Sequence[str]) -> None:
    completed = subprocess.run(list(cmd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}")


def _build_forecast_caches(
    *,
    symbols: Iterable[str],
    data_root: Path,
    forecast_cache_root: Path,
    lookback_hours: float,
    output_json: Path,
    force_rebuild: bool,
) -> None:
    symbol_csv = ",".join(sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}))
    if not symbol_csv:
        return
    cmd = [
        sys.executable,
        "scripts/build_hourly_forecast_caches.py",
        "--symbols",
        symbol_csv,
        "--data-root",
        str(data_root),
        "--forecast-cache-root",
        str(forecast_cache_root),
        "--horizons",
        "1,24",
        "--lookback-hours",
        str(float(lookback_hours)),
        "--output-json",
        str(output_json),
    ]
    if force_rebuild:
        cmd.append("--force-rebuild")
    _run_command(cmd)


def _run_candidate_eval(
    *,
    candidate: StockExpansionCandidate,
    base_stock_universe: str,
    default_checkpoint: Path,
    stock_data_root: Path,
    forecast_cache_root: Path,
    output_dir: Path,
    moving_average_windows: str,
    min_history_hours: int,
) -> dict[str, object]:
    long_only_symbols, short_only_symbols = _candidate_args(candidate)
    cmd = [
        sys.executable,
        "-m",
        "newnanoalpacahourlyexp.run_hourly_trader_sim",
        "--stock-universe",
        base_stock_universe,
        "--symbols",
        candidate.symbol,
        "--default-checkpoint",
        str(default_checkpoint),
        "--sequence-length",
        "128",
        "--horizon",
        "1",
        "--forecast-horizons",
        "1,24",
        "--context-lengths",
        "64,96,192",
        "--forecast-cache-root",
        str(forecast_cache_root),
        "--stock-data-root",
        str(stock_data_root),
        "--allocation-pct",
        "0.2",
        "--allocation-mode",
        "portfolio",
        "--cache-only",
        "--robust-60d",
        "--allow-short",
        "--entry-near-book-bps",
        "25",
        "--moving-average-windows",
        moving_average_windows,
        "--min-history-hours",
        str(int(min_history_hours)),
        "--output-dir",
        str(output_dir),
    ]
    if long_only_symbols:
        cmd.extend(["--long-only-symbols", ",".join(long_only_symbols)])
    if short_only_symbols:
        cmd.extend(["--short-only-symbols", ",".join(short_only_symbols)])
    _run_command(cmd)

    summary_path = output_dir / "reforecast_summary.json"
    summary = json.loads(summary_path.read_text())
    row = summarize_reforecast_result(summary)
    row.update(
        {
            "symbol": candidate.symbol,
            "side": candidate.side,
            "sector": candidate.sector,
            "priority": candidate.priority,
            "thesis": candidate.thesis,
            "summary_path": str(summary_path),
            "lora_command": candidate_lora_command(candidate.symbol),
        }
    )
    return row


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "symbol",
        "side",
        "sector",
        "priority",
        "best_mode",
        "best_scenario",
        "best_total_return",
        "best_sortino",
        "best_max_drawdown",
        "flat_sortino",
        "flat_total_return",
        "flat_max_drawdown",
        "flat_pnl_abs",
        "summary_path",
        "lora_command",
        "thesis",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = row.get("flat") or {}
            writer.writerow(
                {
                    "symbol": row.get("symbol"),
                    "side": row.get("side"),
                    "sector": row.get("sector"),
                    "priority": row.get("priority"),
                    "best_mode": row.get("best_mode"),
                    "best_scenario": row.get("best_scenario"),
                    "best_total_return": row.get("best_total_return"),
                    "best_sortino": row.get("best_sortino"),
                    "best_max_drawdown": row.get("best_max_drawdown"),
                    "flat_sortino": flat.get("sortino"),
                    "flat_total_return": flat.get("total_return"),
                    "flat_max_drawdown": flat.get("max_drawdown"),
                    "flat_pnl_abs": flat.get("pnl_abs"),
                    "summary_path": row.get("summary_path"),
                    "lora_command": row.get("lora_command"),
                    "thesis": row.get("thesis"),
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate stock-universe expansion candidates by adding one symbol at a time to the Alpaca hourly stock base universe.",
    )
    parser.add_argument("--base-stock-universe", default="stock19")
    parser.add_argument("--default-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--stock-data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--forecast-cache-root", type=Path, default=Path("binanceneural/forecast_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/alpaca_stock_expansion_20260318"))
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--candidate-symbols", default=None, help="Optional comma-separated subset of candidate symbols.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--moving-average-windows", default="168,600,720")
    parser.add_argument("--min-history-hours", type=int, default=480)
    parser.add_argument("--cache-lookback-hours", type=float, default=1800.0)
    parser.add_argument("--skip-cache-build", action="store_true")
    parser.add_argument("--force-cache-rebuild", action="store_true")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest_path or output_dir / "candidate_manifest.json"
    if manifest_path.exists():
        base_stock_universe, default_checkpoint, candidates = load_stock_expansion_manifest(manifest_path)
        if not base_stock_universe:
            base_stock_universe = str(args.base_stock_universe)
        if not default_checkpoint:
            default_checkpoint = str(args.default_checkpoint)
    else:
        base_stock_universe = str(args.base_stock_universe)
        default_checkpoint = str(args.default_checkpoint)
        candidates = default_stock_expansion_candidates()
        write_stock_expansion_manifest(
            manifest_path,
            base_stock_universe=base_stock_universe,
            default_checkpoint=default_checkpoint,
            candidates=candidates,
        )

    requested = set(_parse_csv(args.candidate_symbols))
    if requested:
        candidates = [candidate for candidate in candidates if candidate.symbol in requested]
    if args.limit and args.limit > 0:
        candidates = list(candidates)[: int(args.limit)]

    required_history_rows = _effective_min_history_bars("AAPL", int(args.min_history_hours))
    ready, insufficient_history, missing = split_candidates_by_history(
        candidates,
        data_root=args.stock_data_root,
        min_history_rows=required_history_rows,
    )
    if missing:
        missing_payload = [{"symbol": candidate.symbol, "side": candidate.side} for candidate in missing]
        (output_dir / "missing_candidates.json").write_text(json.dumps(missing_payload, indent=2) + "\n")
    if insufficient_history:
        insufficient_payload = [
            {
                "symbol": candidate.symbol,
                "side": candidate.side,
                "history_rows": count_candidate_history_rows(candidate.symbol, data_root=args.stock_data_root),
                "required_history_rows": required_history_rows,
            }
            for candidate in insufficient_history
        ]
        (output_dir / "insufficient_history_candidates.json").write_text(
            json.dumps(insufficient_payload, indent=2) + "\n"
        )
    if not ready:
        raise SystemExit("No candidates have hourly stock data.")

    if not args.skip_cache_build:
        _build_forecast_caches(
            symbols=[candidate.symbol for candidate in ready],
            data_root=args.stock_data_root,
            forecast_cache_root=args.forecast_cache_root,
            lookback_hours=float(args.cache_lookback_hours),
            output_json=output_dir / "forecast_cache_mae.json",
            force_rebuild=bool(args.force_cache_rebuild),
        )

    baseline_dir = output_dir / "baseline"
    baseline_cmd = [
        sys.executable,
        "-m",
        "newnanoalpacahourlyexp.run_hourly_trader_sim",
        "--stock-universe",
        base_stock_universe,
        "--stock-universe-only",
        "--default-checkpoint",
        default_checkpoint,
        "--sequence-length",
        "128",
        "--horizon",
        "1",
        "--forecast-horizons",
        "1,24",
        "--context-lengths",
        "64,96,192",
        "--forecast-cache-root",
        str(args.forecast_cache_root),
        "--stock-data-root",
        str(args.stock_data_root),
        "--allocation-pct",
        "0.2",
        "--allocation-mode",
        "portfolio",
        "--cache-only",
        "--robust-60d",
        "--allow-short",
        "--entry-near-book-bps",
        "25",
        "--moving-average-windows",
        str(args.moving_average_windows),
        "--min-history-hours",
        str(int(args.min_history_hours)),
        "--output-dir",
        str(baseline_dir),
    ]
    _run_command(baseline_cmd)
    baseline_summary = json.loads((baseline_dir / "reforecast_summary.json").read_text())
    rows: list[dict[str, object]] = [
        {
            "symbol": base_stock_universe,
            "side": "mixed",
            "sector": "base",
            "priority": 999,
            "thesis": "Base stock universe baseline.",
            "summary_path": str(baseline_dir / "reforecast_summary.json"),
            "lora_command": "",
            **summarize_reforecast_result(baseline_summary),
        }
    ]

    checkpoint_path = Path(default_checkpoint).expanduser().resolve()
    failed_candidates: list[dict[str, object]] = []
    for candidate in ready:
        try:
            row = _run_candidate_eval(
                candidate=candidate,
                base_stock_universe=base_stock_universe,
                default_checkpoint=checkpoint_path,
                stock_data_root=args.stock_data_root,
                forecast_cache_root=args.forecast_cache_root,
                output_dir=output_dir / candidate.symbol,
                moving_average_windows=str(args.moving_average_windows),
                min_history_hours=int(args.min_history_hours),
            )
        except RuntimeError as exc:
            failed_candidates.append(
                {
                    "symbol": candidate.symbol,
                    "side": candidate.side,
                    "error": str(exc),
                }
            )
            continue
        rows.append(row)

    sorted_rows = [rows[0]] + sorted(rows[1:], key=stock_expansion_sort_key, reverse=True)
    (output_dir / "expansion_results.json").write_text(json.dumps(sorted_rows, indent=2) + "\n")
    _write_csv(output_dir / "expansion_results.csv", sorted_rows)
    if failed_candidates:
        (output_dir / "failed_candidates.json").write_text(json.dumps(failed_candidates, indent=2) + "\n")

    top_lora_commands = [
        {
            "symbol": row["symbol"],
            "flat": row.get("flat"),
            "best_scenario": row.get("best_scenario"),
            "lora_command": row.get("lora_command"),
        }
        for row in sorted_rows[1:6]
    ]
    (output_dir / "top_lora_commands.json").write_text(json.dumps(top_lora_commands, indent=2) + "\n")

    for row in sorted_rows:
        flat = row.get("flat") or {}
        print(
            f"{row['symbol']:>18} "
            f"flat_sortino={float(flat.get('sortino', 0.0)):.4f} "
            f"flat_return={float(flat.get('total_return', 0.0)):.6f} "
            f"best_scenario={row.get('best_scenario')} "
            f"best_sortino={float(row.get('best_sortino', 0.0)):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
