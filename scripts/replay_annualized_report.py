from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pufferlib_market.autoresearch_rl import annualize_total_return_pct, compute_eval_window_years


def _infer_window_years(leaderboard_path: Path) -> float | None:
    candidates = [leaderboard_path.parent / "README.md"]

    run_id = leaderboard_path.name
    if run_id.endswith("_leaderboard.csv"):
        run_id = run_id[: -len("_leaderboard.csv")]
        remote_dir = REPO / "analysis" / "remote_runs" / run_id
        candidates.extend(
            [
                remote_dir / "README.md",
                remote_dir / "pipeline.sh",
                remote_dir / "launch_manifest.json",
            ]
        )

    for candidate in candidates:
        if not candidate.exists():
            continue
        text = candidate.read_text(errors="ignore")
        match = re.search(
            r"--replay-eval-start-date\s+([^\s]+)\s+--replay-eval-end-date\s+([^\s]+)",
            text,
        )
        if not match:
            continue
        years = compute_eval_window_years(match.group(1), match.group(2))
        if years is not None:
            return years
    return None


def _ensure_annualized_columns(df: pd.DataFrame, years: float | None) -> pd.DataFrame:
    if years is None or years <= 0.0:
        return df

    result = df.copy()
    for source_col, target_col in (
        ("replay_daily_return_pct", "replay_daily_annualized_return_pct"),
        ("replay_hourly_return_pct", "replay_hourly_annualized_return_pct"),
        ("replay_hourly_policy_return_pct", "replay_hourly_policy_annualized_return_pct"),
        ("replay_daily_robust_worst_return_pct", "replay_daily_robust_worst_annualized_return_pct"),
        ("replay_hourly_robust_worst_return_pct", "replay_hourly_robust_worst_annualized_return_pct"),
        (
            "replay_hourly_policy_robust_worst_return_pct",
            "replay_hourly_policy_robust_worst_annualized_return_pct",
        ),
    ):
        if source_col not in result.columns or target_col in result.columns:
            continue
        result[target_col] = result[source_col].map(lambda x: annualize_total_return_pct(x, years))
    return result


def _load_report_frame(leaderboard_path: Path, years: float | None) -> pd.DataFrame:
    df = pd.read_csv(leaderboard_path)
    df = _ensure_annualized_columns(df, years)
    df["leaderboard"] = str(leaderboard_path)
    df["eval_years"] = years
    return df


def _print_best(frame: pd.DataFrame, metric: str, label: str) -> None:
    if metric not in frame.columns:
        print(f"{label}: metric `{metric}` missing")
        return

    values = pd.to_numeric(frame[metric], errors="coerce")
    if values.notna().sum() == 0:
        print(f"{label}: metric `{metric}` has no populated rows")
        return

    best_idx = values.idxmax()
    row = frame.loc[best_idx]
    print(
        f"{label}: {row.get('description')}  "
        f"{metric}={values.loc[best_idx]:.2f}  "
        f"leaderboard={row.get('leaderboard')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare daily and hourly replay annualized returns")
    parser.add_argument("leaderboards", nargs="+", help="Leaderboard CSV paths")
    parser.add_argument("--start-date", default="", help="Override replay eval start date")
    parser.add_argument("--end-date", default="", help="Override replay eval end date")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    override_years = compute_eval_window_years(args.start_date, args.end_date)

    frames: list[pd.DataFrame] = []
    for raw_path in args.leaderboards:
        path = Path(raw_path)
        years = override_years if override_years is not None else _infer_window_years(path)
        frame = _load_report_frame(path, years)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    _print_best(combined, "replay_daily_annualized_return_pct", "Best daily annualized replay")
    _print_best(
        combined,
        "replay_hourly_policy_robust_worst_annualized_return_pct",
        "Best hourly robust annualized replay",
    )
    _print_best(combined, "replay_hourly_annualized_return_pct", "Best hourly annualized replay")

    preferred_cols = [
        "description",
        "leaderboard",
        "rank_metric",
        "rank_score",
        "replay_daily_annualized_return_pct",
        "replay_hourly_annualized_return_pct",
        "replay_hourly_policy_annualized_return_pct",
        "replay_hourly_policy_robust_worst_annualized_return_pct",
        "replay_combo_score",
        "holdout_robust_score",
        "generalization_score",
    ]
    cols = [c for c in preferred_cols if c in combined.columns]
    ranked = combined.copy()
    ranked["_sort_metric"] = pd.to_numeric(
        ranked.get("replay_hourly_policy_robust_worst_annualized_return_pct"),
        errors="coerce",
    )
    if ranked["_sort_metric"].notna().sum() == 0:
        ranked["_sort_metric"] = pd.to_numeric(
            ranked.get("replay_daily_annualized_return_pct"),
            errors="coerce",
        )
    ranked = ranked.sort_values("_sort_metric", ascending=False).head(args.top_k)
    print("\nTop rows")
    print(ranked[cols].to_string(index=False))

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
