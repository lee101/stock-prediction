#!/usr/bin/env python3
"""Run multihorizon proof evals for top screened32 leaderboard candidates."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_multihorizon_module():
    script_path = REPO / "scripts" / "eval_multihorizon_candidate.py"
    spec = importlib.util.spec_from_file_location("eval_multihorizon_candidate", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_ranked_rows(
    *,
    leaderboard_path: Path,
    sort_field: str = "rank_score",
    require_blank_error: bool = True,
) -> list[dict[str, str]]:
    with leaderboard_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if require_blank_error:
        rows = [row for row in rows if not str(row.get("error", "")).strip()]

    def _sort_key(row: dict[str, str]) -> tuple[float, float]:
        primary = _safe_float(row.get(sort_field))
        if primary is None and sort_field != "rank_score":
            primary = _safe_float(row.get("rank_score"))
        fallback = _safe_float(row.get("holdout_robust_score"))
        return (
            primary if primary is not None else -float("inf"),
            fallback if fallback is not None else -float("inf"),
        )

    rows.sort(key=_sort_key, reverse=True)
    return rows


def resolve_candidate_checkpoint(checkpoint_root: Path, description: str) -> Path:
    trial_dir = checkpoint_root / str(description)
    candidates = [
        trial_dir / "val_best.pt",
        trial_dir / "best.pt",
        trial_dir / "final.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    pts = sorted(trial_dir.glob("*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if pts:
        return pts[0]
    raise FileNotFoundError(f"No checkpoint found under {trial_dir}")


def _run_candidate_eval(
    *,
    candidate_checkpoint: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "eval_multihorizon_candidate.py"),
        "--data-path",
        str(args.data_path),
        "--baseline-checkpoint",
        str(args.baseline_checkpoint),
        "--baseline-extra-checkpoints",
        *[str(path) for path in args.baseline_extra_checkpoints],
        "--candidate-checkpoint",
        str(candidate_checkpoint),
        "--horizons-days",
        str(args.horizons_days),
        "--slippage-bps",
        str(args.slippage_bps),
        "--n-windows",
        str(int(args.n_windows)),
        "--seed",
        str(int(args.seed)),
        "--recent-within-days",
        str(int(args.recent_within_days)),
        "--fee-rate",
        str(float(args.fee_rate)),
        "--fill-buffer-bps",
        str(float(args.fill_buffer_bps)),
        "--decision-lag",
        str(int(args.decision_lag)),
        "--out",
        str(output_path),
    ]
    if args.exhaustive:
        cmd.append("--exhaustive")
    if not args.disable_shorts:
        cmd.append("--allow-shorts")

    subprocess.run(cmd, cwd=REPO, check=True)
    return json.loads(output_path.read_text())


def build_parser() -> argparse.ArgumentParser:
    module = _load_multihorizon_module()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaderboard", required=True, help="CSV leaderboard produced by autoresearch")
    parser.add_argument("--checkpoint-root", required=True, help="Checkpoint root used for the leaderboard")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--sort-field", default="rank_score")
    parser.add_argument("--require-blank-error", action="store_true", default=True)
    parser.add_argument("--no-require-blank-error", action="store_false", dest="require_blank_error")
    parser.add_argument("--data-path", default="pufferlib_market/data/screened32_single_offset_val_full.bin")
    parser.add_argument("--baseline-checkpoint", default=module.DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--baseline-extra-checkpoints",
        nargs="*",
        default=list(module.DEFAULT_EXTRA_CHECKPOINTS),
    )
    parser.add_argument("--horizons-days", default="30,60,100,120")
    parser.add_argument("--slippage-bps", default="0,5,10,20")
    parser.add_argument("--n-windows", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--recent-within-days", type=int, default=140)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--disable-shorts", action="store_true", default=True)
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--out-dir", default="reports/screened32_candidate_evals")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    leaderboard_path = Path(args.leaderboard)
    if not leaderboard_path.is_absolute():
        leaderboard_path = REPO / leaderboard_path
    checkpoint_root = Path(args.checkpoint_root)
    if not checkpoint_root.is_absolute():
        checkpoint_root = REPO / checkpoint_root
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked_rows = load_ranked_rows(
        leaderboard_path=leaderboard_path,
        sort_field=str(args.sort_field),
        require_blank_error=bool(args.require_blank_error),
    )
    selected_rows = ranked_rows[: max(int(args.top_k), 0)]
    summary: dict[str, Any] = {
        "leaderboard": str(leaderboard_path),
        "checkpoint_root": str(checkpoint_root),
        "top_k": int(args.top_k),
        "sort_field": str(args.sort_field),
        "candidates": [],
    }

    for index, row in enumerate(selected_rows, start=1):
        description = str(row.get("description", "")).strip()
        if not description:
            continue
        checkpoint_path = resolve_candidate_checkpoint(checkpoint_root, description)
        output_path = out_dir / f"{index:02d}_{description}.json"
        report = _run_candidate_eval(
            candidate_checkpoint=checkpoint_path,
            output_path=output_path,
            args=args,
        )
        summary["candidates"].append(
            {
                "rank": index,
                "description": description,
                "checkpoint": str(checkpoint_path),
                "leaderboard_row": row,
                "recommendation": report.get("recommendation", {}),
                "combo_vs_baseline": report.get("comparisons", {}).get("baseline_plus_candidate_vs_baseline", {}),
                "candidate_vs_baseline": report.get("comparisons", {}).get("candidate_vs_baseline", {}),
                "report_path": str(output_path),
            }
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"Screened32 candidate evals: {len(summary['candidates'])}")
    for item in summary["candidates"]:
        rec = item.get("recommendation", {})
        combo = item.get("combo_vs_baseline", {})
        print(
            f"{item['rank']:2d}. {item['description']}: "
            f"{rec.get('status', 'unknown')} "
            f"(mean_delta_monthly={float(combo.get('mean_delta_median_monthly_return', 0.0))*100:+.2f}%)"
        )
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
