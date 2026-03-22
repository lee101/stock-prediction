#!/usr/bin/env python3
"""Compare training results across multiple seeds for statistical significance."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CKPT_ROOT = REPO / "binanceneural" / "checkpoints"

def load_manifest(run_dir: Path) -> list[dict]:
    manifest = run_dir / ".topk_manifest.json"
    if not manifest.exists():
        return []
    return json.loads(manifest.read_text())

def main():
    seeds = [1337, 42, 7]
    patterns = ["crypto_portfolio_seed{}", "crypto_portfolio_6sym"]

    print(f"{'Run':<35} {'Epochs':>6} {'Best Score':>10} {'Best Ep':>7}")
    print("-" * 65)

    all_best = []
    for seed in seeds:
        for pat in patterns:
            name = pat.format(seed)
            d = CKPT_ROOT / name
            if not d.exists():
                continue
            entries = load_manifest(d)
            if not entries:
                continue
            best = max(entries, key=lambda e: e["metric"])
            print(f"{name:<35} {len(entries):>6} {best['metric']:>10.2f} {best.get('epoch','?'):>7}")
            all_best.append(best["metric"])

    if len(all_best) >= 2:
        arr = np.array(all_best)
        print(f"\nAcross {len(all_best)} runs:")
        print(f"  mean={arr.mean():.2f}  std={arr.std():.2f}  min={arr.min():.2f}  max={arr.max():.2f}")
        print(f"  95% CI: [{arr.mean() - 1.96*arr.std()/np.sqrt(len(arr)):.2f}, {arr.mean() + 1.96*arr.std()/np.sqrt(len(arr)):.2f}]")

if __name__ == "__main__":
    main()
