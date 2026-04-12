#!/usr/bin/env python3
"""
Cross-seed stability analysis for preaug strategy sweeps.
Compares winners across seeds to identify stable choices.

Usage:
    python preaug_sweeps/compare_seeds.py \
        preaugstrategies/new_strat_test/ \
        preaugstrategies/new_strat_s42/ \
        preaugstrategies/new_strat_s7777/
"""
import json
import sys
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_dir(directory: Path) -> Dict[str, dict]:
    """Load all symbol JSON files from a directory."""
    results = {}
    for p in sorted(directory.glob("*.json")):
        try:
            results[p.stem] = json.loads(p.read_text())
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")
    return results


def cross_seed_analysis(seed_dirs: List[Path]):
    seed_data = {}
    for d in seed_dirs:
        label = d.name
        data = load_dir(d)
        if data:
            seed_data[label] = data
        else:
            print(f"[WARN] No data in {d}")

    if not seed_data:
        print("No data found.")
        return

    # Find common symbols
    all_syms = set.intersection(*[set(v.keys()) for v in seed_data.values()])
    print(f"\nCommon symbols across {len(seed_data)} seeds: {len(all_syms)}")
    seed_labels = list(seed_data.keys())

    # Get all strategies
    first_sym = next(iter(all_syms))
    first_seed = seed_labels[0]
    strategies = list(seed_data[first_seed][first_sym].get("comparison", {}).keys())
    print(f"Strategies: {strategies}")

    print(f"\n{'='*80}")
    print(f"STABILITY ANALYSIS — winner agreement across seeds")
    print(f"{'='*80}")

    stable_wins: Dict[str, int] = {s: 0 for s in strategies}
    unstable_syms: List[str] = []
    rows = []

    for sym in sorted(all_syms):
        winners = [seed_data[sl][sym]["best_strategy"] for sl in seed_labels]
        all_same = len(set(winners)) == 1
        stable_winner = winners[0] if all_same else None

        # Avg MAE% per strategy across seeds
        strat_avgs = {}
        for s in strategies:
            vals = []
            for sl in seed_labels:
                v = seed_data[sl][sym].get("comparison", {}).get(s, {}).get("mae_percent")
                if v is not None:
                    vals.append(v)
            strat_avgs[s] = statistics.mean(vals) if vals else None

        # Seed-avg best
        valid = {s: v for s, v in strat_avgs.items() if v is not None}
        avg_best = min(valid, key=lambda s: valid[s]) if valid else None

        rows.append((sym, winners, all_same, stable_winner, strat_avgs, avg_best))
        if all_same:
            stable_wins[winners[0]] += 1
        else:
            unstable_syms.append(sym)

    # Print detail table
    header_strats = strategies[:4]  # truncate for display
    header = f"  {'SYM':8s} " + "  ".join(f"{sl[:6]:>6s}" for sl in seed_labels) + f"  {'avg_best':>12s}  stable"
    print(header)
    for sym, winners, all_same, sw, strat_avgs, avg_best in rows:
        winner_str = "  ".join(f"{w[:6]:>6s}" for w in winners)
        avg_best_mae = min(v for v in strat_avgs.values() if v) if strat_avgs else 0
        stable_str = "Y" if all_same else f"N({','.join(set(winners))})"
        print(f"  {sym:8s} {winner_str}  {avg_best_mae:10.4f}%  {stable_str}")

    print(f"\n--- Stable winners ({len(all_syms)-len(unstable_syms)}/{len(all_syms)} symbols) ---")
    for s, cnt in sorted(stable_wins.items(), key=lambda x: -x[1]):
        if cnt > 0:
            print(f"  {s:20s}: {cnt} symbols always win")

    print(f"\n--- Unstable symbols ({len(unstable_syms)}) ---")
    for sym in unstable_syms:
        row = next(r for r in rows if r[0] == sym)
        print(f"  {sym}: seed winners = {row[1]}")

    # Strategy MAE% averages across all seeds and symbols
    print(f"\n--- Cross-seed avg MAE% per strategy ---")
    for s in strategies:
        all_vals = []
        for sym in all_syms:
            for sl in seed_labels:
                v = seed_data[sl][sym].get("comparison", {}).get(s, {}).get("mae_percent")
                if v is not None:
                    all_vals.append(v)
        if all_vals:
            print(f"  {s:20s}: mean={statistics.mean(all_vals):.4f}%  median={statistics.median(all_vals):.4f}%  n={len(all_vals)}")

    # Recommendation
    print(f"\n--- RECOMMENDATION ---")
    strat_cross_avgs = {}
    for s in strategies:
        all_vals = [seed_data[sl][sym].get("comparison", {}).get(s, {}).get("mae_percent")
                    for sym in all_syms for sl in seed_labels]
        all_vals = [v for v in all_vals if v is not None]
        strat_cross_avgs[s] = statistics.mean(all_vals) if all_vals else float("inf")

    best_universal = min(strat_cross_avgs, key=lambda s: strat_cross_avgs[s])
    print(f"  Best universal default: {best_universal} ({strat_cross_avgs[best_universal]:.4f}%)")
    print(f"  Per-symbol sweep gain: {(strat_cross_avgs['differencing'] - min(strat_cross_avgs.values())) / strat_cross_avgs['differencing'] * 100:.1f}% vs differencing")


if __name__ == "__main__":
    dirs = [Path(d) for d in sys.argv[1:]]
    if not dirs:
        # default
        dirs = [
            Path("preaugstrategies/new_strat_test"),
            Path("preaugstrategies/new_strat_s42"),
            Path("preaugstrategies/new_strat_s7777"),
        ]
    dirs = [d for d in dirs if d.exists()]
    cross_seed_analysis(dirs)
