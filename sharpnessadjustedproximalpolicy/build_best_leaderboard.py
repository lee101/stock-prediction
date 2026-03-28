#!/usr/bin/env python3
"""Build unified leaderboard: pick best checkpoint per symbol across all sweeps.

Scans all checkpoint directories for sap_history.json, finds best epoch per run,
then picks the overall best per symbol. Outputs a leaderboard CSV compatible
with portfolio_eval.py.

Usage:
    python -m sharpnessadjustedproximalpolicy.build_best_leaderboard
"""
import csv
import json
import sys
import time
from pathlib import Path

EXCLUDE = {"AVAXUSD", "AAVEUSD", "BTCFDUSD", "ETHFDUSD"}


def main():
    ckpt_root = Path("sharpnessadjustedproximalpolicy/checkpoints")
    if not ckpt_root.exists():
        print("No checkpoint directory found")
        return

    # Scan all checkpoint dirs
    all_results = []
    for d in sorted(ckpt_root.iterdir()):
        if not d.is_dir():
            continue
        history_path = d / "sap_history.json"
        if not history_path.exists():
            continue

        # Extract symbol and config from dir name: sap_{config}_{symbol}_{timestamp}
        name = d.name
        parts = name.split("_")
        # Find the symbol (ends with USD)
        symbol = None
        config_parts = []
        for i, p in enumerate(parts[1:], 1):  # skip "sap_"
            if p.endswith("USD"):
                symbol = p
                config_parts = parts[1:i]
                break
        if not symbol:
            continue
        config = "_".join(config_parts)
        if symbol in EXCLUDE:
            continue

        history = json.loads(history_path.read_text())
        if not history:
            continue

        # Find best epoch by val_sortino (or ms_sortino if available)
        best = None
        for h in history:
            ms = h.get("ms_sortino", 0)
            vs = h.get("val_sortino", 0)
            score = ms if ms > 0 else vs
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "epoch": h["epoch"],
                    "val_sortino": vs,
                    "val_return": h.get("val_return", 0),
                    "ms_sortino": ms,
                    "sharpness_ema": h.get("sharpness_ema", 0),
                    "lr_scale": h.get("lr_scale", 1.0),
                }

        if best is None or best["score"] <= 0:
            continue

        # Verify checkpoint exists
        ep = best["epoch"]
        ckpt_path = d / f"epoch_{ep:03d}.pt"
        if not ckpt_path.exists():
            continue

        all_results.append({
            "symbol": symbol,
            "config": config,
            "best_epoch": ep,
            "val_sortino": round(best["val_sortino"], 3),
            "val_return": round(best["val_return"], 4),
            "sharpness": round(best["sharpness_ema"], 1),
            "wd_scale": round(best["lr_scale"], 2),
            "wall_s": 0,
            "checkpoint_dir": str(d),
            "error": "",
        })

    if not all_results:
        print("No valid results found")
        return

    # Pick best per symbol
    best_per_sym = {}
    for r in all_results:
        sym = r["symbol"]
        if sym not in best_per_sym or r["val_sortino"] > best_per_sym[sym]["val_sortino"]:
            best_per_sym[sym] = r

    # Sort by val_sortino descending
    ranked = sorted(best_per_sym.values(), key=lambda x: x["val_sortino"], reverse=True)

    # Write leaderboard CSV
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"sharpnessadjustedproximalpolicy/best_leaderboard_{ts}.csv")
    fields = ["symbol", "config", "best_epoch", "val_sortino", "val_return", "sharpness", "wd_scale", "wall_s", "error"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in ranked:
            w.writerow(r)

    print(f"Best-of-best leaderboard: {out_path}", flush=True)
    print(f"\n{'Symbol':<12} {'Config':<30} {'Sort':>8} {'Ret':>8} {'Ep':>3}", flush=True)
    print("-" * 65, flush=True)
    for r in ranked:
        print(f"{r['symbol']:<12} {r['config']:<30} {r['val_sortino']:>8.1f} {r['val_return']:>8.3f} {r['best_epoch']:>3}", flush=True)

    # Also write full results for reference
    all_ranked = sorted(all_results, key=lambda x: x["val_sortino"], reverse=True)
    full_path = Path(f"sharpnessadjustedproximalpolicy/full_leaderboard_{ts}.csv")
    with open(full_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in all_ranked:
            w.writerow(r)
    print(f"\nFull leaderboard ({len(all_results)} entries): {full_path}", flush=True)

    # Summary stats
    total_syms = len(ranked)
    avg_sort = sum(r["val_sortino"] for r in ranked) / total_syms
    min_sort = min(r["val_sortino"] for r in ranked)
    print(f"\n{total_syms} symbols, avg Sort={avg_sort:.1f}, min Sort={min_sort:.1f}", flush=True)


if __name__ == "__main__":
    main()
