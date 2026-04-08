#!/usr/bin/env python3
"""Convenience launcher for stock trade-log replay audits."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

replay_mod = importlib.import_module("unified_hourly_experiment.replay_stock_trade_log_sim")


DEFAULT_REPLAY_AUDIT_OUTPUT_DIR = Path("artifacts/replay_audit")


def _default_artifact_name(*, now: datetime | None = None) -> str:
    resolved_now = now or datetime.now(UTC)
    return f"stock_replay_audit_{resolved_now.strftime('%Y%m%d_%H%M%S')}"


def _resolve_artifact_paths(
    *,
    output_dir: Path,
    name: str | None,
    now: datetime | None = None,
) -> tuple[Path, Path]:
    stem = str(name).strip() if name is not None else ""
    if not stem:
        stem = _default_artifact_name(now=now)
    stem_path = Path(stem)
    if stem_path.is_absolute() or stem_path.parent != Path():
        raise ValueError("--name must be a simple artifact stem, not a path")
    normalized_stem = stem_path.stem
    base = output_dir / normalized_stem
    return base.with_suffix(".json"), base.with_suffix(".html")


def _latest_alias_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "report": output_dir / "latest.json",
        "visualization": output_dir / "latest.html",
        "trace": output_dir / "latest.trace.json",
        "manifest": output_dir / "latest_manifest.json",
    }


def build_replay_args(args: argparse.Namespace, *, report_path: Path, html_path: Path) -> list[str]:
    def _fmt_float(value: float) -> str:
        return f"{float(value):g}"

    replay_args: list[str] = [
        "--trade-log",
        str(args.trade_log),
        "--event-log",
        str(args.event_log),
        "--data-root",
        str(args.data_root),
        "--output",
        str(report_path),
        "--visualize-html",
        str(html_path),
        "--visualize-num-pairs",
        str(args.visualize_num_pairs),
        "--max-hold-hours",
        str(args.max_hold_hours),
        "--min-edge",
        _fmt_float(args.min_edge),
        "--fee-rate",
        _fmt_float(args.fee_rate),
        "--leverage",
        _fmt_float(args.leverage),
        "--decision-lag-bars",
        str(args.decision_lag_bars),
        "--bar-margins",
        args.bar_margins,
        "--entry-order-ttls",
        args.entry_order_ttls,
        "--market-order-entries",
        args.market_order_entries,
        "--sim-backend",
        args.sim_backend,
        "--cancel-ack-delays",
        args.cancel_ack_delays,
        "--partial-fill-on-touch",
        args.partial_fill_on_touch,
    ]
    if args.symbols:
        replay_args.extend(["--symbols", args.symbols])
    if args.start:
        replay_args.extend(["--start", args.start])
    if args.end:
        replay_args.extend(["--end", args.end])
    if args.initial_cash is not None:
        replay_args.extend(["--initial-cash", f"{float(args.initial_cash):g}"])
    if args.max_positions is not None:
        replay_args.extend(["--max-positions", str(int(args.max_positions))])
    return replay_args


def _replace_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def _write_latest_aliases(*, output_dir: Path, report_path: Path) -> dict[str, str | None]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    alias_paths = _latest_alias_paths(output_dir)
    _replace_file(report_path, alias_paths["report"])

    visualization = payload.get("visualization") or {}
    generated_html = visualization.get("generated_html_path")
    trace_json = visualization.get("trace_json_path")

    html_src = Path(generated_html) if generated_html else None
    trace_src = Path(trace_json) if trace_json else None

    if html_src is not None and html_src.exists():
        _replace_file(html_src, alias_paths["visualization"])
        html_alias = str(alias_paths["visualization"])
    else:
        alias_paths["visualization"].unlink(missing_ok=True)
        html_alias = None

    if trace_src is not None and trace_src.exists():
        _replace_file(trace_src, alias_paths["trace"])
        trace_alias = str(alias_paths["trace"])
    else:
        alias_paths["trace"].unlink(missing_ok=True)
        trace_alias = None

    manifest = {
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "report_path": str(report_path),
        "latest_report_path": str(alias_paths["report"]),
        "visualization_path": generated_html,
        "latest_visualization_path": html_alias,
        "trace_json_path": trace_json,
        "latest_trace_json_path": trace_alias,
    }
    alias_paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "report": str(alias_paths["report"]),
        "visualization": html_alias,
        "trace": trace_alias,
        "manifest": str(alias_paths["manifest"]),
    }


def _write_latest_aliases_best_effort(*, output_dir: Path, report_path: Path) -> dict[str, str | None] | None:
    try:
        return _write_latest_aliases(output_dir=output_dir, report_path=report_path)
    except Exception as exc:  # pragma: no cover - exercised via tests with monkeypatch
        print(f"Warning: failed to update latest replay aliases: {exc}", file=sys.stderr)
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a stock trade-log replay audit with auto-named JSON and HTML artifacts."
    )
    parser.add_argument("--trade-log", type=Path, default=Path("strategy_state/stock_trade_log.jsonl"))
    parser.add_argument("--event-log", type=Path, default=Path("strategy_state/stock_event_log.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--symbols", default="")
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--initial-cash", type=float, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--max-hold-hours", type=int, default=6)
    parser.add_argument("--min-edge", type=float, default=-1.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--decision-lag-bars", type=int, default=0)
    parser.add_argument("--bar-margins", default="0.0005,0.001,0.002")
    parser.add_argument("--entry-order-ttls", default="0,1,2")
    parser.add_argument("--market-order-entries", default="0,1")
    parser.add_argument(
        "--sim-backend",
        choices=["python", "native", "auto", "hourly_trader"],
        default="python",
    )
    parser.add_argument("--cancel-ack-delays", default="1")
    parser.add_argument("--partial-fill-on-touch", default="0,1")
    parser.add_argument("--visualize-num-pairs", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPLAY_AUDIT_OUTPUT_DIR)
    parser.add_argument(
        "--name",
        default="",
        help="Artifact stem without extension. Defaults to a UTC timestamped stock_replay_audit_* name.",
    )
    parser.add_argument(
        "--no-latest-alias",
        action="store_true",
        help="Skip updating stable latest.json/latest.html alias artifacts in the output directory.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved artifact paths and delegated replay args.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report_path, html_path = _resolve_artifact_paths(
        output_dir=args.output_dir,
        name=args.name,
    )
    replay_args = build_replay_args(args, report_path=report_path, html_path=html_path)

    if args.dry_run:
        payload = {
            "report_path": str(report_path),
            "visualization_path": str(html_path),
            "latest_alias_paths": (
                None
                if args.no_latest_alias
                else {key: str(path) for key, path in _latest_alias_paths(args.output_dir).items()}
            ),
            "replay_args": replay_args,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    replay_mod.main(replay_args)
    if not args.no_latest_alias and report_path.exists():
        _write_latest_aliases_best_effort(output_dir=args.output_dir, report_path=report_path)


if __name__ == "__main__":
    main()
