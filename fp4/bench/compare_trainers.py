"""Orchestrate trainer comparison: pufferlib_bf16 / hf_trainer / trl / fp4.

Writes:
  - one JSON per (trainer, seed) via bench_trading.run_one
  - aggregated CSV   fp4/bench/results/summary_<date>.csv
  - markdown report  fp4/bench/results/fp4_vs_baselines_<date>.md  (with
    a `## Recommendation` section that picks the winner by p10 then Sortino).

Smoke mode:
    python fp4/bench/compare_trainers.py --smoke

Full run:
    python fp4/bench/compare_trainers.py --seeds 3 --steps 100000000
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import bench_trading  # type: ignore  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CFG = Path(__file__).resolve().parents[1] / "experiments" / "fp4_ppo_stocks12.yaml"
ALL_TRAINERS = ["pufferlib_bf16", "hf_trainer", "trl", "fp4"]


def _flatten_eval(rec: dict[str, Any]) -> dict[str, Any]:
    """Pull p10/median/maxdd/sortino out of the per-slippage eval blob."""
    out: dict[str, Any] = {}
    ev = rec.get("eval") or {}
    by = ev.get("by_slippage") or {}
    for bps_key, blob in by.items():
        # evaluate_fast writes a JSON with various keys; try common names.
        for src, dst in (
            ("p10_return", f"p10_{bps_key}bps"),
            ("median_return", f"median_{bps_key}bps"),
            ("max_drawdown", f"maxdd_{bps_key}bps"),
            ("sortino", f"sortino_{bps_key}bps"),
            ("p10", f"p10_{bps_key}bps"),
            ("median", f"median_{bps_key}bps"),
            ("maxdd", f"maxdd_{bps_key}bps"),
        ):
            if src in blob and dst not in out:
                out[dst] = blob[src]
    return out


def _row(rec: dict[str, Any]) -> dict[str, Any]:
    train = rec.get("train") or {}
    row: dict[str, Any] = {
        "trainer": rec["trainer"],
        "seed": rec["seed"],
        "status": rec.get("status"),
        "reason": rec.get("reason") or train.get("reason", ""),
        "steps": rec.get("steps"),
        "wall_sec": train.get("wall_sec", ""),
        "steps_per_sec": rec.get("steps_per_sec", ""),
        "gpu_peak_mb": train.get("gpu_peak_mb", ""),
    }
    row.update(_flatten_eval(rec))
    return row


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _pick_winner(rows: list[dict[str, Any]]) -> tuple[str, str]:
    """Winner = highest p10_5bps then highest sortino_5bps. Returns (name, why)."""
    scored: list[tuple[float, float, str]] = []
    for r in rows:
        if r["status"] != "ok":
            continue
        try:
            p10 = float(r.get("p10_5bps", "nan"))
        except (TypeError, ValueError):
            p10 = float("nan")
        try:
            sortino = float(r.get("sortino_5bps", "nan"))
        except (TypeError, ValueError):
            sortino = float("nan")
        scored.append((p10, sortino, r["trainer"]))
    if not scored:
        return ("none", "no successful runs to score")
    scored.sort(key=lambda t: (t[0] if t[0] == t[0] else -1e9, t[1] if t[1] == t[1] else -1e9), reverse=True)
    p10, sortino, name = scored[0]
    return (name, f"p10_5bps={p10:.4f}, sortino_5bps={sortino:.4f}")


def _write_md(rows: list[dict[str, Any]], path: Path, cfg_path: Path, smoke: bool) -> None:
    lines: list[str] = []
    lines.append(f"# fp4 vs baseline trainer comparison")
    lines.append("")
    lines.append(f"- date: {_dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- config: `{cfg_path}`")
    lines.append(f"- smoke: {smoke}")
    lines.append("")
    if not rows:
        lines.append("_no rows_")
    else:
        keys = ["trainer", "seed", "status", "wall_sec", "steps_per_sec", "gpu_peak_mb",
                "p10_5bps", "median_5bps", "sortino_5bps", "maxdd_5bps", "reason"]
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("|" + "|".join(["---"] * len(keys)) + "|")
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    lines.append("")
    name, why = _pick_winner(rows)
    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"**Winner: `{name}`** — {why}")
    lines.append("")
    lines.append("Tie-break order: p10 @ 5bps slippage > Sortino @ 5bps slippage. "
                 "Only successful runs are eligible. Skipped trainers (missing deps "
                 "or unimplemented adapters) are reported above for transparency.")
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CFG))
    p.add_argument("--trainers", default=",".join(ALL_TRAINERS),
                   help="Comma-separated subset of " + ",".join(ALL_TRAINERS))
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=100_000_000)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny steps, 1 seed, fast for CI / sanity checks")
    args = p.parse_args(argv)

    if args.smoke:
        args.seeds = 1
        args.steps = max(2048, min(args.steps, 8192))

    trainers = [t.strip() for t in args.trainers.split(",") if t.strip()]
    cfg_path = Path(args.config)
    rows: list[dict[str, Any]] = []
    for trainer in trainers:
        for seed in range(args.seeds):
            print(f"[compare_trainers] {trainer} seed={seed} steps={args.steps}", flush=True)
            rec = bench_trading.run_one(trainer, cfg_path, args.steps, seed, smoke=args.smoke)
            print(f"  -> status={rec['status']} reason={rec.get('reason','')}", flush=True)
            rows.append(_row(rec))

    date = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"summary_{date}.csv"
    md_path = RESULTS_DIR / f"fp4_vs_baselines_{date}.md"
    _write_csv(rows, csv_path)
    _write_md(rows, md_path, cfg_path, args.smoke)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
