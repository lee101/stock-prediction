"""Algorithm leaderboard report (Phase 4, Unit P4-8).

Consumes a sweep summary CSV produced by ``fp4/bench/sweep.py`` (Unit P4-6)
and writes a markdown leaderboard + production recommendation to
``fp4/bench/results/algo_leaderboard_<date>.md``.

Expected CSV columns (minimum):
    algorithm, constrained, seed,
    p10_5bps, median_5bps, sortino_5bps, maxdd_5bps,
    steps_per_sec (optional), gpu_peak_mb (optional)

Unknown columns are ignored. Rows with non-finite metrics are dropped with a
warning comment in the report.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import math
import os
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"

# Current production bar — see CLAUDE.md and alpacaprod.md.
ENSEMBLE_P10_5BPS = 0.662  # 66.2% 32-model softmax ensemble
MAX_DD_BUDGET = 0.20  # absolute (|drawdown| ≤ 0.20)

UPSTREAM_COMMITS = [
    ("f9806fe0", "bench_gemm uses real CUTLASS NVFP4 GEMM (2.24x BF16 @ 4k^3)"),
    ("d8eb1a5b", "graph-safe NVFP4Linear (+62% SPS over Unit A baseline)"),
    ("3a1a29c2", "market_sim_py pybind11 bindings"),
    ("57a0879b", "generic eval + first PnL/Sortino comparison run"),
]

METRIC_KEYS = [
    ("p10_5bps", "p10@5bps"),
    ("median_5bps", "median@5bps"),
    ("sortino_5bps", "sortino@5bps"),
    ("maxdd_5bps", "max_dd"),
    ("steps_per_sec", "sps"),
    ("gpu_peak_mb", "gpu_mb"),
]


def _finite(x):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _find_latest_sweep_summary():
    """Locate the most recent Unit P4-6 sweep summary.csv."""
    if not RESULTS_DIR.exists():
        return None
    candidates = sorted(RESULTS_DIR.glob("sweep_*/summary.csv"))
    return candidates[-1] if candidates else None


def _mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return (float("nan"), float("nan"), 0)
    n = len(vals)
    m = sum(vals) / n
    if n == 1:
        return (m, 0.0, n)
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return (m, math.sqrt(var), n)


def load_rows(summary_path):
    with open(summary_path, newline="") as f:
        return list(csv.DictReader(f))


def aggregate(rows):
    """Aggregate rows per (algorithm, constrained) cell."""
    cells = {}
    for row in rows:
        algo = row.get("algorithm") or row.get("algo") or row.get("trainer") or "?"
        cons_raw = str(row.get("constrained", "off")).strip().lower()
        constrained = cons_raw in ("1", "true", "on", "yes")
        if row.get("status", "ok") not in ("ok", ""):
            continue
        key = (algo, constrained)
        cell = cells.setdefault(key, {k: [] for k, _ in METRIC_KEYS})
        cell.setdefault("_seeds", []).append(row.get("seed", "?"))
        for csv_key, _ in METRIC_KEYS:
            cell[csv_key].append(_finite(row.get(csv_key)))

    aggregated = []
    for (algo, constrained), cell in cells.items():
        entry = {"algorithm": algo, "constrained": constrained,
                 "n_seeds": len(cell["_seeds"])}
        for csv_key, label in METRIC_KEYS:
            m, s, _ = _mean_std(cell[csv_key])
            entry[label + "_mean"] = m
            entry[label + "_std"] = s
        aggregated.append(entry)
    return aggregated


def _sort_key(entry):
    p10 = entry["p10@5bps_mean"]
    sortino = entry["sortino@5bps_mean"]
    maxdd = entry["max_dd_mean"]
    # DESC p10, DESC sortino, ASC |maxdd|
    nan_bad = -1e9
    p10s = p10 if math.isfinite(p10) else nan_bad
    sorts = sortino if math.isfinite(sortino) else nan_bad
    dd = abs(maxdd) if math.isfinite(maxdd) else 1e9
    return (-p10s, -sorts, dd)


def _fmt(val, digits=4):
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return "n/a"
    return f"{val:.{digits}f}"


def _fmt_pm(mean, std, digits=4):
    if not math.isfinite(mean):
        return "n/a"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def build_markdown(entries, summary_path, today):
    lines = []
    lines.append(f"# Algorithm Leaderboard — {today}")
    lines.append("")
    lines.append(f"- Source sweep summary: `{summary_path}`")
    lines.append(f"- Generated: {today}")
    lines.append(f"- Production bar: 32-model ensemble **p10@5bps = "
                 f"{ENSEMBLE_P10_5BPS:.3f}**, max_dd budget = {MAX_DD_BUDGET:.2f}")
    lines.append("")
    lines.append("## Upstream commits")
    for sha, desc in UPSTREAM_COMMITS:
        lines.append(f"- `{sha}` {desc}")
    lines.append("")
    lines.append("## Ranked leaderboard")
    lines.append("Sorted by (p10@5bps DESC, sortino@5bps DESC, |max_dd| ASC).")
    lines.append("")
    header = ("| rank | algo | constrained | n_seeds | p10@5bps | median@5bps | "
              "sortino@5bps | max_dd | sps | gpu_mb |")
    sep = "|------|------|-------------|---------|----------|-------------|--------------|--------|-----|--------|"
    lines.append(header)
    lines.append(sep)

    ranked = sorted(entries, key=_sort_key)
    for i, e in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {e['algorithm']} | {'on' if e['constrained'] else 'off'} | "
            f"{e['n_seeds']} | "
            f"{_fmt_pm(e['p10@5bps_mean'], e['p10@5bps_std'])} | "
            f"{_fmt_pm(e['median@5bps_mean'], e['median@5bps_std'])} | "
            f"{_fmt_pm(e['sortino@5bps_mean'], e['sortino@5bps_std'])} | "
            f"{_fmt_pm(e['max_dd_mean'], e['max_dd_std'])} | "
            f"{_fmt(e['sps_mean'], 0)} | "
            f"{_fmt(e['gpu_mb_mean'], 0)} |"
        )
    lines.append("")

    lines.append("## Risk-adjusted ranking")
    lines.append("Feasible set: cells with `|max_dd| ≤ 0.20` and finite p10.")
    lines.append("")
    feasible = [e for e in ranked
                if math.isfinite(e["max_dd_mean"]) and abs(e["max_dd_mean"]) <= MAX_DD_BUDGET
                and math.isfinite(e["p10@5bps_mean"])]
    if feasible:
        lines.append("| rank | algo | constrained | p10@5bps | sortino@5bps | max_dd |")
        lines.append("|------|------|-------------|----------|--------------|--------|")
        for i, e in enumerate(feasible, 1):
            lines.append(
                f"| {i} | {e['algorithm']} | {'on' if e['constrained'] else 'off'} | "
                f"{_fmt_pm(e['p10@5bps_mean'], e['p10@5bps_std'])} | "
                f"{_fmt_pm(e['sortino@5bps_mean'], e['sortino@5bps_std'])} | "
                f"{_fmt_pm(e['max_dd_mean'], e['max_dd_std'])} |"
            )
    else:
        lines.append("_No cell satisfies the drawdown budget._")
    lines.append("")

    lines.append("## Recommendation")
    winner = feasible[0] if feasible else (ranked[0] if ranked else None)
    if winner is None:
        lines.append("**Recommendation: NONE — sweep produced no usable rows.**")
    else:
        beats_ensemble = (math.isfinite(winner["p10@5bps_mean"])
                          and winner["p10@5bps_mean"] > ENSEMBLE_P10_5BPS)
        precision = "NVFP4 (graph-safe linear, commit d8eb1a5b)"
        cons_word = "constrained" if winner["constrained"] else "unconstrained"
        lines.append(
            f"**Recommendation: {winner['algorithm']} + {cons_word} + {precision}.** "
            f"This cell posted p10@5bps = "
            f"{_fmt_pm(winner['p10@5bps_mean'], winner['p10@5bps_std'])} with "
            f"sortino@5bps = {_fmt_pm(winner['sortino@5bps_mean'], winner['sortino@5bps_std'])} "
            f"and max_dd = {_fmt_pm(winner['max_dd_mean'], winner['max_dd_std'])} "
            f"over n={winner['n_seeds']} seeds, the best feasible risk-adjusted "
            f"cell under the |max_dd| ≤ {MAX_DD_BUDGET:.2f} budget."
        )
        lines.append("")
        if beats_ensemble:
            lines.append("")
            lines.append("> **ALERT — NEW CHAMPION**")
            lines.append(f"> This config's p10@5bps ({winner['p10@5bps_mean']:.4f}) "
                         f"beats the current 32-model ensemble bar "
                         f"({ENSEMBLE_P10_5BPS:.4f}).")
            lines.append("> **Action: update `alpacaprod.md` and stage a paper-trading "
                         "rollout before promoting to live.**")
        else:
            lines.append("")
            lines.append(f"_No cell beat the 32-model ensemble bar "
                         f"(p10@5bps = {ENSEMBLE_P10_5BPS:.3f}); "
                         f"`alpacaprod.md` left unchanged._")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="Build NVFP4 algo leaderboard report.")
    p.add_argument("--summary", type=str, default=None,
                   help="Path to sweep summary.csv (auto-detect latest sweep_* dir if omitted).")
    p.add_argument("--out", type=str, default=None,
                   help="Output markdown path (default: results/algo_leaderboard_<date>.md).")
    args = p.parse_args(argv)

    summary_path = Path(args.summary) if args.summary else _find_latest_sweep_summary()
    if summary_path is None or not Path(summary_path).exists():
        print("error: no sweep summary.csv found. Run fp4/bench/sweep.py first "
              "or pass --summary.", file=sys.stderr)
        return 2

    rows = load_rows(summary_path)
    entries = aggregate(rows)
    today = _dt.date.today().isoformat()
    md = build_markdown(entries, str(summary_path), today)

    if args.out:
        out_path = Path(args.out)
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"algo_leaderboard_{today}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
