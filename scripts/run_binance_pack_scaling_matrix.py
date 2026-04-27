#!/usr/bin/env python3
"""Run short Binance pack scaling-law sweeps over model/data axes.

The portfolio-pack sweep has many execution knobs.  This wrapper keeps a
small, fixed pack grid and varies only train window, boosting rounds, and label
horizon so we can tell whether better PnL is coming from model scale or from
execution-rule luck.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]

DEFAULT_PACK_ARGS = (
    "--risk-penalties 0.8 "
    "--cvar-weights 1.0 "
    "--entry-gap-bps-grid 75 "
    "--entry-alpha-grid 0.5 "
    "--exit-alpha-grid 0.8 "
    "--edge-threshold-grid 0.01 "
    "--edge-to-full-size-grid 0.02,0.035 "
    "--min-close-ret-grid=0.0,0.002 "
    "--close-edge-weight-grid 0.0,0.5 "
    "--min-upside-downside-ratio-grid 0.0 "
    "--max-positions-grid 5,8 "
    "--max-pending-entries-grid 12,24 "
    "--entry-ttl-hours-grid 6 "
    "--max-hold-hours-grid 24 "
    "--max-leverage-grid 1.0 "
    "--entry-selection-modes edge_rank "
    "--entry-allocator-modes concentrated "
    "--entry-allocator-edge-power-grid 2.0 "
    "--top-candidates-per-hour 15 "
    "--min-result-trades 20"
)


def _parse_int_list(value: str) -> list[int]:
    items = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not items:
        raise ValueError(f"empty integer list: {value!r}")
    return items


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _best_row(rows: list[dict[str, str]], key: str) -> dict[str, str] | None:
    scored = [(row, _float_or_none(row.get(key))) for row in rows]
    scored = [(row, value) for row, value in scored if value is not None]
    if not scored:
        return None
    return max(scored, key=lambda item: item[1])[0]


def read_best_rows(path: Path) -> tuple[dict[str, str] | None, dict[str, str] | None, int]:
    if not path.exists() or path.stat().st_size <= 0:
        return None, None, 0
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return _best_row(rows, "selection_score"), _best_row(rows, "monthly_return_pct"), len(rows)


def build_sweep_command(
    *,
    args: argparse.Namespace,
    out_csv: Path,
    seed: int,
    train_days: int,
    rounds: int,
    label_horizon: int,
) -> list[str]:
    command = [
        str(args.python),
        str(args.sweep_script),
        "--hourly-root",
        str(args.hourly_root),
        "--min-bars",
        str(args.min_bars),
        "--min-symbols-per-hour",
        str(args.min_symbols_per_hour),
        "--train-days",
        str(train_days),
        "--eval-days",
        str(args.eval_days),
        "--label-horizon",
        str(label_horizon),
        "--rounds",
        str(rounds),
        "--device",
        str(args.device),
        "--max-configs",
        str(args.max_configs_per_cell),
        "--config-sample-seed",
        str(seed),
        "--out",
        str(out_csv),
        "--html-out",
        "",
        "--trace-json-out",
        "",
        "--mp4-out",
        "",
    ]
    if args.symbols:
        command.extend(["--symbols", str(args.symbols)])
    if args.pack_args:
        command.extend(shlex.split(str(args.pack_args)))
    return command


def _config_json(row: dict[str, str] | None) -> str:
    if not row:
        return "{}"
    keys = [
        "risk_penalty",
        "cvar_weight",
        "entry_gap_bps",
        "edge_threshold",
        "edge_to_full_size",
        "min_close_ret",
        "close_edge_weight",
        "min_upside_downside_ratio",
        "min_recent_ret_24h",
        "min_recent_ret_72h",
        "max_recent_vol_72h",
        "max_positions",
        "max_pending_entries",
        "entry_ttl_hours",
        "max_hold_hours",
        "max_leverage",
    ]
    return json.dumps({key: row[key] for key in keys if key in row}, sort_keys=True)


def _summary_row(
    *,
    cell_id: str,
    train_days: int,
    rounds: int,
    label_horizon: int,
    elapsed_sec: float,
    returncode: int,
    result_csv: Path,
    result_rows: int,
    best_score: dict[str, str] | None,
    best_monthly: dict[str, str] | None,
) -> dict[str, Any]:
    return {
        "cell_id": cell_id,
        "train_days": int(train_days),
        "rounds": int(rounds),
        "label_horizon": int(label_horizon),
        "returncode": int(returncode),
        "status": "ok" if returncode == 0 else "failed",
        "elapsed_sec": round(float(elapsed_sec), 3),
        "result_rows": int(result_rows),
        "result_csv": str(result_csv),
        "best_selection_score": _float_or_none((best_score or {}).get("selection_score")),
        "best_score_monthly_return_pct": _float_or_none((best_score or {}).get("monthly_return_pct")),
        "best_score_total_return_pct": _float_or_none((best_score or {}).get("total_return_pct")),
        "best_score_max_drawdown_pct": _float_or_none((best_score or {}).get("max_drawdown_pct")),
        "best_score_sortino": _float_or_none((best_score or {}).get("sortino")),
        "best_score_num_sells": _int_or_none((best_score or {}).get("num_sells")),
        "best_score_config": _config_json(best_score),
        "best_monthly_return_pct": _float_or_none((best_monthly or {}).get("monthly_return_pct")),
        "best_monthly_selection_score": _float_or_none((best_monthly or {}).get("selection_score")),
        "best_monthly_max_drawdown_pct": _float_or_none((best_monthly or {}).get("max_drawdown_pct")),
        "best_monthly_num_sells": _int_or_none((best_monthly or {}).get("num_sells")),
        "best_monthly_config": _config_json(best_monthly),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Binance pack scaling-law matrix.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sweep-script", type=Path, default=Path("scripts/sweep_binance_hourly_portfolio_pack.py"))
    parser.add_argument("--hourly-root", type=Path, default=Path("binance_spot_hourly"))
    parser.add_argument("--symbols", default="")
    parser.add_argument("--min-bars", type=int, default=5000)
    parser.add_argument("--min-symbols-per-hour", type=int, default=20)
    parser.add_argument("--train-days-grid", default="360,720,1080")
    parser.add_argument("--rounds-grid", default="80,160,320")
    parser.add_argument("--label-horizon-grid", default="12,24,48")
    parser.add_argument("--eval-days", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-configs-per-cell", type=int, default=32)
    parser.add_argument("--config-sample-seed", type=int, default=2026042706)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--pack-args", default=DEFAULT_PACK_ARGS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    train_days_values = _parse_int_list(args.train_days_grid)
    rounds_values = _parse_int_list(args.rounds_grid)
    horizon_values = _parse_int_list(args.label_horizon_grid)
    out_dir = args.out_dir or Path("analysis") / f"binance_pack_scaling_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "args": {key: str(value) for key, value in vars(args).items()},
                "train_days_grid": train_days_values,
                "rounds_grid": rounds_values,
                "label_horizon_grid": horizon_values,
            },
            indent=2,
        )
        + "\n"
    )

    fieldnames: list[str] | None = None
    rows: list[dict[str, Any]] = []
    cell_idx = 0
    for train_days in train_days_values:
        for rounds in rounds_values:
            for label_horizon in horizon_values:
                cell_idx += 1
                cell_id = f"train{train_days}_rounds{rounds}_h{label_horizon}"
                cell_dir = out_dir / cell_id
                cell_dir.mkdir(parents=True, exist_ok=True)
                result_csv = cell_dir / "portfolio_pack.csv"
                log_path = cell_dir / "run.log"
                command = build_sweep_command(
                    args=args,
                    out_csv=result_csv,
                    seed=int(args.config_sample_seed) + cell_idx,
                    train_days=train_days,
                    rounds=rounds,
                    label_horizon=label_horizon,
                )
                (cell_dir / "command.txt").write_text(" ".join(shlex.quote(part) for part in command) + "\n")
                print(f"[{cell_idx}] {cell_id}: {' '.join(shlex.quote(part) for part in command)}", flush=True)
                start = time.time()
                returncode = 0
                if not args.dry_run:
                    with log_path.open("w") as log:
                        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
                    returncode = int(completed.returncode)
                elapsed = time.time() - start
                best_score, best_monthly, result_rows = read_best_rows(result_csv)
                summary = _summary_row(
                    cell_id=cell_id,
                    train_days=train_days,
                    rounds=rounds,
                    label_horizon=label_horizon,
                    elapsed_sec=elapsed,
                    returncode=returncode,
                    result_csv=result_csv,
                    result_rows=result_rows,
                    best_score=best_score,
                    best_monthly=best_monthly,
                )
                rows.append(summary)
                with summary_path.open("w", newline="") as fh:
                    if fieldnames is None:
                        fieldnames = list(summary.keys())
                    writer = csv.DictWriter(fh, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                print(
                    f"  status={summary['status']} rows={result_rows} "
                    f"best_score_monthly={summary['best_score_monthly_return_pct']} "
                    f"best_monthly={summary['best_monthly_return_pct']} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                if returncode != 0 and args.stop_on_failure:
                    print(f"stopping after failed cell; see {log_path}", file=sys.stderr)
                    return returncode

    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
