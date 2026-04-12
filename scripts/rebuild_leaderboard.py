#!/usr/bin/env python3
"""
Rebuild a sweep leaderboard CSV from eval_lag2.json files.
Usage:
    python scripts/rebuild_leaderboard.py <checkpoint_root> [variant_label]

Example:
    python scripts/rebuild_leaderboard.py pufferlib_market/checkpoints/stocks17_sweep/C_low_tp C
    python scripts/rebuild_leaderboard.py pufferlib_market/checkpoints/wide73_sweep/C_low_tp C
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone


def main():
    if len(sys.argv) < 2:
        print("Usage: rebuild_leaderboard.py <checkpoint_root> [variant_label]", file=sys.stderr)
        sys.exit(1)

    root = Path(sys.argv[1])
    variant = sys.argv[2] if len(sys.argv) > 2 else root.name

    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        sys.exit(1)

    rows = []
    for seed_dir in sorted(root.glob("s*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 9999):
        seed = seed_dir.name[1:]
        eval_path = seed_dir / "eval_lag2.json"
        if not eval_path.exists():
            continue
        try:
            d = json.loads(eval_path.read_text())
        except Exception as e:
            print(f"  s{seed}: JSON parse error: {e}", file=sys.stderr)
            continue

        med   = d.get("median_total_return", 0) * 100
        p10   = d.get("p10_total_return", 0) * 100
        worst = (d.get("worst_window") or {}).get("total_return", 0) * 100
        neg   = d.get("negative_windows", 0)
        sort  = d.get("median_sortino", 0)

        # Prefer val_best checkpoint
        ckpt = seed_dir / "val_best.pt"
        if not ckpt.exists():
            ckpt = seed_dir / "best.pt"
        ckpt_str = str(ckpt) if ckpt.exists() else "missing"

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows.append({
            "ts": ts, "variant": variant, "seed": int(seed),
            "med": med, "p10": p10, "worst": worst, "neg": neg, "sort": sort, "ckpt": ckpt_str,
        })

    if not rows:
        print(f"No eval_lag2.json files found under {root}", file=sys.stderr)
        sys.exit(1)

    # Sort by median descending
    rows.sort(key=lambda r: r["med"], reverse=True)

    out_path = root / "leaderboard_rebuilt.csv"
    header = "timestamp,variant,seed,med_pct,p10_pct,worst_pct,neg_count,med_sortino,checkpoint\n"
    lines = [header]
    for r in rows:
        lines.append(
            f"{r['ts']},{r['variant']},{r['seed']},"
            f"{r['med']:.2f},{r['p10']:.2f},{r['worst']:.2f},"
            f"{r['neg']},{r['sort']:.2f},{r['ckpt']}\n"
        )
    out_path.write_text("".join(lines))
    print(f"Wrote {out_path} ({len(rows)} seeds)")

    # Print top 10 and positive-p10 summary
    print(f"\n=== TOP 10 [{variant}] ===")
    print(f"{'seed':>6} {'med%':>8} {'p10%':>8} {'worst%':>8} {'neg':>5} {'sortino':>8}")
    for r in rows[:10]:
        flag = " *** ALL-POS" if r["neg"] == 0 and r["p10"] > 0 else (" *** 0-NEG" if r["neg"] == 0 else "")
        print(f"s{r['seed']:>5} {r['med']:>8.2f} {r['p10']:>8.2f} {r['worst']:>8.2f} {r['neg']:>5} {r['sort']:>8.2f}{flag}")

    pos_p10 = [r for r in rows if r["p10"] > 0]
    print(f"\n=== POSITIVE-P10 seeds ({len(pos_p10)} total) ===")
    for r in sorted(pos_p10, key=lambda r: r["p10"], reverse=True):
        flag = " *** ALL-POS" if r["neg"] == 0 else ""
        print(f"s{r['seed']:>5} med={r['med']:>8.2f}% p10={r['p10']:>7.2f}% neg={r['neg']}/50{flag}")


if __name__ == "__main__":
    main()
