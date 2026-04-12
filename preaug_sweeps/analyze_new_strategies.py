#!/usr/bin/env python3
"""
Analyze results from new preaug strategy sweeps (log_diff, diff_norm vs differencing).
Run this after evaluate_preaug_chronos.py sweeps complete.

Usage:
    python preaug_sweeps/analyze_new_strategies.py preaug_sweeps/reports/new_strategies/*.json
"""
import json
import sys
import statistics
from pathlib import Path


def load_reports(*paths: str) -> dict:
    """Load and merge multiple report JSONs."""
    merged = {}
    for p in paths:
        data = json.loads(Path(p).read_text())
        merged.update(data)
    return merged


def summarize(data: dict, strategies: list[str] = None):
    if strategies is None:
        # Infer from first symbol
        first = next(iter(data.values()))
        strategies = list(first.get("comparison", {}).keys())

    print(f"\n{'='*60}")
    print(f"SYMBOLS: {len(data)}   STRATEGIES: {strategies}")
    print(f"{'='*60}")

    # Per-strategy avg test MAE%
    print("\n--- Average test MAE% by strategy ---")
    for strat in strategies:
        vals = []
        for item in data.values():
            v = item.get("comparison", {}).get(strat, {}).get("test", {}).get("mae_percent")
            if v is not None:
                vals.append(v)
        if vals:
            print(f"  {strat:20s}: n={len(vals):3d}  mean={statistics.mean(vals):.3f}%  "
                  f"median={statistics.median(vals):.3f}%  "
                  f"p10={sorted(vals)[len(vals)//10]:.3f}%  "
                  f"max={max(vals):.3f}%")

    # Win rate vs differencing
    print("\n--- Win rate vs differencing (test MAE%) ---")
    for strat in strategies:
        if strat == "differencing":
            continue
        wins = beats = total = 0
        ratios = []
        for item in data.values():
            comp = item.get("comparison", {})
            diff_v = comp.get("differencing", {}).get("test", {}).get("mae_percent")
            strat_v = comp.get(strat, {}).get("test", {}).get("mae_percent")
            if diff_v is None or strat_v is None:
                continue
            total += 1
            if strat_v < diff_v:
                wins += 1
            ratios.append(strat_v / diff_v)
        if total:
            print(f"  {strat:20s}: beats_diff={wins}/{total} ({wins/total*100:.0f}%)  "
                  f"avg_ratio={statistics.mean(ratios):.2f}x  "
                  f"median_ratio={statistics.median(ratios):.2f}x")

    # Best strategy winner counts
    print("\n--- Best strategy per symbol ---")
    winner_counts: dict[str, int] = {}
    for sym, item in data.items():
        best = item.get("best_strategy", "?")
        winner_counts[best] = winner_counts.get(best, 0) + 1
    for strat, cnt in sorted(winner_counts.items(), key=lambda x: -x[1]):
        print(f"  {strat:20s}: {cnt} ({cnt/len(data)*100:.0f}%)")

    # Per-symbol detail
    print("\n--- Per-symbol detail (sorted by differencing MAE%) ---")
    rows = []
    for sym, item in data.items():
        comp = item.get("comparison", {})
        diff_v = comp.get("differencing", {}).get("test", {}).get("mae_percent")
        ld_v = comp.get("log_diff", {}).get("test", {}).get("mae_percent")
        dn_v = comp.get("diff_norm", {}).get("test", {}).get("mae_percent") or comp.get("diff_norm_w20", {}).get("test", {}).get("mae_percent")
        bl_v = comp.get("baseline", {}).get("test", {}).get("mae_percent")
        rows.append((sym, diff_v or 999, ld_v, dn_v, bl_v, item.get("best_strategy")))
    rows.sort(key=lambda x: x[1])
    header = f"  {'SYM':8s} {'diff':>8s} {'log_diff':>10s} {'diff_norm':>10s} {'baseline':>10s} {'best':>12s}"
    print(header)
    for sym, dv, lv, nv, bv, best in rows:
        ld_str = f"{lv:.3f}%" if lv else "  -    "
        dn_str = f"{nv:.3f}%" if nv else "  -    "
        bv_str = f"{bv:.2f}%" if bv else "  -    "
        print(f"  {sym:8s} {dv:8.3f}% {ld_str:>10s} {dn_str:>10s} {bv_str:>10s} {best:>12s}")


if __name__ == "__main__":
    paths = sys.argv[1:] or list(Path("preaug_sweeps/reports/new_strategies").glob("*.json"))
    if not paths:
        print("No report files found.")
        sys.exit(1)
    data = load_reports(*paths)
    summarize(data)
