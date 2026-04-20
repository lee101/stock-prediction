#!/usr/bin/env python3
"""Full-fit in-sample OVERFIT upper-bound report for the live XGB ensemble.

The deployed ensemble is trained on 100% of available data
(``train_end = date.today()``). There is no honest OOS window left over.
That is fine — max signal for live trading — but it means we can't just
re-run ``sweep_ensemble_grid`` to see how the ensemble is doing, because
every window is in-sample.

This script runs a sweep over the last N weeks of the TRAINING window
and labels the output file so aggressively that nobody can mistake it
for OOS:

    analysis/xgbnew_daily/full_fit_insample_overfit_YYYYMMDD/
        insample_OVERFIT_UPPER_BOUND.json

The numbers are an UPPER BOUND on live performance: if live PnL ever
exceeds these, something is wrong (likely a data leak into live, or a
benchmark-selection bug in live itself).

Usage
-----
    python scripts/xgb_full_fit_insample_report.py
        [--ensemble analysis/xgbnew_daily/alltrain_ensemble_gpu]
        [--weeks 12]

The script reads the ensemble's manifest to find train_start / train_end
and sweeps a suffix of the training window at the deploy config
(lev=2.0, ms=0.85, top_n=1, hold_through=Y, min_dollar_vol=50M,
min_vol_20d=0.10, fee_regimes=deploy+stress36x).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ensemble", type=Path,
                   default=REPO / "analysis/xgbnew_daily/alltrain_ensemble_gpu",
                   help="Ensemble dir with alltrain_ensemble.json + per-seed pkls")
    p.add_argument("--symbols-file", type=Path,
                   default=REPO / "symbol_lists/stocks_wide_1000_v1.txt")
    p.add_argument("--data-root", type=Path,
                   default=REPO / "trainingdata")
    p.add_argument("--weeks", type=int, default=12,
                   help="How many weeks of the tail of the training window to "
                        "evaluate (default 12, i.e. ~3 months)")
    p.add_argument("--leverage", type=float, default=2.0)
    p.add_argument("--min-score", type=float, default=0.85)
    p.add_argument("--top-n", type=int, default=1)
    p.add_argument("--hold-through", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--min-dollar-vol", type=float, default=50_000_000.0)
    p.add_argument("--min-vol-20d", type=float, default=0.10)
    p.add_argument("--fee-regimes", default="deploy,stress36x")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override output dir. Default: "
                        "analysis/xgbnew_daily/full_fit_insample_overfit_<today>/")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    manifest_path = args.ensemble / "alltrain_ensemble.json"
    if not manifest_path.exists():
        print(f"ERROR: no manifest at {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text())
    train_start = manifest.get("train_start")
    train_end = manifest.get("train_end")
    if not train_start or not train_end:
        print("ERROR: manifest missing train_start/train_end", file=sys.stderr)
        return 3

    # Seed pkls — take exactly the list in the manifest to avoid name drift.
    models = manifest.get("models") or []
    pkl_paths = [m["path"] for m in models if "path" in m]
    if not pkl_paths:
        print("ERROR: manifest has no models", file=sys.stderr)
        return 4

    te = date.fromisoformat(train_end)
    # Eval window: last `weeks` weeks of the training range.
    window_start = te - timedelta(days=args.weeks * 7)
    window_end = te
    # sweep_ensemble_grid uses a rolling (30d, stride 7d) inside [oos_start,
    # oos_end], so we get roughly `weeks - 3` rolling windows at stride 7.

    today = date.today().isoformat().replace("-", "")
    out_dir = args.output_dir or (
        REPO / f"analysis/xgbnew_daily/full_fit_insample_overfit_{today}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # We deliberately do not use "oos" anywhere in the filename — the
    # period being evaluated IS part of the training data.
    report_path = out_dir / "insample_OVERFIT_UPPER_BOUND.json"

    cmd = [
        sys.executable, "-m", "xgbnew.sweep_ensemble_grid",
        "--symbols-file", str(args.symbols_file),
        "--data-root", str(args.data_root),
        "--model-paths", ",".join(pkl_paths),
        "--train-start", train_start,
        "--train-end", train_end,
        "--oos-start", window_start.isoformat(),
        "--oos-end", window_end.isoformat(),
        "--window-days", "30",
        "--stride-days", "7",
        "--leverage-grid", f"{args.leverage}",
        "--min-score-grid", f"{args.min_score}",
        "--top-n-grid", f"{args.top_n}",
        "--fee-regimes", args.fee_regimes,
        "--inference-min-dolvol-grid", f"{int(args.min_dollar_vol)}",
        "--inference-min-vol-grid", f"{args.min_vol_20d}",
        "--output-dir", str(out_dir),
        "--verbose",
    ]
    if args.hold_through:
        cmd.append("--hold-through")
    else:
        cmd.append("--no-hold-through")

    # IMPORTANT: train_start and train_end here match the ensemble so the
    # sweep reports "oos" but we know it's in-sample. sweep's naming is
    # confusing — that's exactly why this helper renames the artifact.
    print(f"[insample-report] ensemble={args.ensemble}")
    print(f"[insample-report] train range: {train_start} → {train_end}")
    print(f"[insample-report] eval window: {window_start} → {window_end} "
          f"(last {args.weeks} weeks of training, IN-SAMPLE)")
    print(f"[insample-report] running sweep...")
    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=REPO)
    dt = time.perf_counter() - t0
    if rc != 0:
        print(f"[insample-report] sweep failed rc={rc}", file=sys.stderr)
        return rc

    # Find the latest sweep json and renames it to the honest-label path.
    sweeps = sorted(out_dir.glob("sweep_*.json"), key=lambda p: p.stat().st_mtime)
    if not sweeps:
        print(f"[insample-report] no sweep_*.json produced in {out_dir}",
              file=sys.stderr)
        return 5
    raw = sweeps[-1]
    payload = json.loads(raw.read_text())
    annotated = {
        "artifact_type": "full_fit_insample_overfit_upper_bound",
        "warning": "THESE METRICS ARE IN-SAMPLE. They are the best-case upper "
                   "bound on live realised performance, not an OOS estimate.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ensemble_dir": str(args.ensemble),
        "ensemble_manifest_trained_at": manifest.get("trained_at"),
        "ensemble_train_start": train_start,
        "ensemble_train_end": train_end,
        "eval_window_start": window_start.isoformat(),
        "eval_window_end": window_end.isoformat(),
        "eval_weeks": args.weeks,
        "sweep_duration_seconds": round(dt, 1),
        "config_note": "This eval window is a SUFFIX of the training window — "
                       "every row inside it was visible to the model at fit "
                       "time. Compare against OOS (oos2024_ensemble_gpu) to "
                       "see the overfit gap.",
        "deploy_config": {
            "leverage": args.leverage,
            "min_score": args.min_score,
            "top_n": args.top_n,
            "hold_through": bool(args.hold_through),
            "min_dollar_vol": args.min_dollar_vol,
            "min_vol_20d": args.min_vol_20d,
            "fee_regimes": args.fee_regimes.split(","),
        },
        "sweep_result": payload,
    }
    report_path.write_text(json.dumps(annotated, indent=2))
    print(f"\n[insample-report] → {report_path}")
    print(f"[insample-report] this is the FULL-FIT IN-SAMPLE UPPER BOUND — "
          f"live PnL that exceeds these numbers is a RED FLAG, not a win.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
