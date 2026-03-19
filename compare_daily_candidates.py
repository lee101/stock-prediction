#!/usr/bin/env python3
"""
Head-to-head comparison of all daily RL checkpoints.
Evaluates each on its matching val data with realistic fees/slippage.
"""

import subprocess
import sys
import json
import re
from pathlib import Path

BASE = Path("/nvme0n1-disk/code/stock-prediction/pufferlib_market")

# Candidates: (name, checkpoint, val_data, hidden_size, arch, fee_rate, extra_args)
CANDIDATES = [
    # === CURRENT PROD CANDIDATE (crypto5 daily) ===
    ("crypto5/trade_pen_05 [PROD]",
     "checkpoints/autoresearch_daily/trade_pen_05/best.pt",
     "data/crypto5_daily_val.bin", 1024, "mlp", 0.001, []),

    ("crypto5/cosine_lr",
     "checkpoints/autoresearch_daily/cosine_lr/best.pt",
     "data/crypto5_daily_val.bin", 1024, "mlp", 0.001, []),

    ("crypto5/ent_001",
     "checkpoints/autoresearch_daily/ent_001/best.pt",
     "data/crypto5_daily_val.bin", 1024, "mlp", 0.001, []),

    # === CRYPTO8 DAILY (0-fee training) ===
    ("crypto8/clip_anneal",
     "checkpoints/autoresearch_crypto8_daily/clip_anneal/best.pt",
     "data/crypto8_daily_val.bin", 1024, "mlp", 0.001, []),

    ("crypto8/slip_10bps",
     "checkpoints/autoresearch_crypto8_daily/slip_10bps/best.pt",
     "data/crypto8_daily_val.bin", 1024, "mlp", 0.001, []),

    ("crypto8/envs_256",
     "checkpoints/autoresearch_crypto8_daily/envs_256/best.pt",
     "data/crypto8_daily_val.bin", 1024, "mlp", 0.001, []),

    ("crypto8/wd_05",
     "checkpoints/autoresearch_crypto8_daily/wd_05/best.pt",
     "data/crypto8_daily_val.bin", 1024, "mlp", 0.001, []),

    ("crypto8/trade_pen_05",
     "checkpoints/autoresearch_crypto8_daily/trade_pen_05/best.pt",
     "data/crypto8_daily_val.bin", 1024, "mlp", 0.001, []),

    # === MIXED23 DAILY (23 symbols - stocks + crypto) ===
    ("mixed23/baseline_anneal_lr",
     "checkpoints/autoresearch_mixed23_daily/baseline_anneal_lr/best.pt",
     "data/mixed23_daily_val.bin", 1024, "mlp", 0.001, []),

    ("mixed23/ent_anneal",
     "checkpoints/autoresearch_mixed23_daily/ent_anneal/best.pt",
     "data/mixed23_daily_val.bin", 1024, "mlp", 0.001, []),

    ("mixed23/clip_anneal",
     "checkpoints/autoresearch_mixed23_daily/clip_anneal/best.pt",
     "data/mixed23_daily_val.bin", 1024, "mlp", 0.001, []),

    ("mixed23/envs_256",
     "checkpoints/autoresearch_mixed23_daily/envs_256/best.pt",
     "data/mixed23_daily_val.bin", 1024, "mlp", 0.001, []),

    ("mixed23/trade_pen_05",
     "checkpoints/autoresearch_mixed23_daily/trade_pen_05/best.pt",
     "data/mixed23_daily_val.bin", 1024, "mlp", 0.001, []),

    ("mixed23/wd_005",
     "checkpoints/autoresearch_mixed23_daily/wd_005/best.pt",
     "data/mixed23_daily_val.bin", 1024, "mlp", 0.001, []),

    # === DAILY COMBOS (crypto5, variants of trade_pen_05) ===
    ("combos/tp05_cosine",
     "checkpoints/autoresearch_daily_combos/tp05_cosine/best.pt",
     "data/crypto5_daily_val.bin", 1024, "mlp", 0.001, []),

    ("combos/tp05_ent001",
     "checkpoints/autoresearch_daily_combos/tp05_ent001/best.pt",
     "data/crypto5_daily_val.bin", 1024, "mlp", 0.001, []),

    ("combos/tp05_wd01",
     "checkpoints/autoresearch_daily_combos/tp05_wd01/best.pt",
     "data/crypto5_daily_val.bin", 1024, "mlp", 0.001, []),

    ("combos/tp06_cosine",
     "checkpoints/autoresearch_daily_combos/tp06_cosine/best.pt",
     "data/crypto5_daily_val.bin", 1024, "mlp", 0.001, []),

    ("combos/tp08_cosine",
     "checkpoints/autoresearch_daily_combos/tp08_cosine/best.pt",
     "data/crypto5_daily_val.bin", 1024, "mlp", 0.001, []),

    # === MASS DAILY (multi-seed trade_penalty sweep on crypto5) ===
    ("mass/tp0.05_s42",
     "checkpoints/mass_daily/tp0.05_s42/best.pt",
     "data/crypto5_daily_val.bin", 1024, "mlp", 0.001, []),

    ("mass/tp0.05_s123",
     "checkpoints/mass_daily/tp0.05_s123/best.pt",
     "data/crypto5_daily_val.bin", 1024, "mlp", 0.001, []),

    ("mass/tp0.10_s42",
     "checkpoints/mass_daily/tp0.10_s42/best.pt",
     "data/crypto5_daily_val.bin", 1024, "mlp", 0.001, []),

    ("mass/tp0.10_s123",
     "checkpoints/mass_daily/tp0.10_s123/best.pt",
     "data/crypto5_daily_val.bin", 1024, "mlp", 0.001, []),

    ("mass/tp0.15_s42",
     "checkpoints/mass_daily/tp0.15_s42/best.pt",
     "data/crypto5_daily_val.bin", 1024, "mlp", 0.001, []),

    ("mass/tp0.20_s42",
     "checkpoints/mass_daily/tp0.20_s42/best.pt",
     "data/crypto5_daily_val.bin", 1024, "mlp", 0.001, []),
]


def run_eval(name, checkpoint, data_path, hidden_size, arch, fee_rate, extra_args):
    ckpt_path = BASE / checkpoint
    data_full = BASE / data_path
    if not ckpt_path.exists():
        return name, {"error": f"checkpoint missing: {ckpt_path}"}
    if not data_full.exists():
        return name, {"error": f"data missing: {data_full}"}

    cmd = [
        sys.executable, "-u", "-m", "pufferlib_market.evaluate",
        "--checkpoint", str(ckpt_path),
        "--data-path", str(data_full),
        "--max-steps", "90",
        "--periods-per-year", "365",
        "--fill-slippage-bps", "8",
        "--fee-rate", str(fee_rate),
        "--hidden-size", str(hidden_size),
        "--arch", arch,
        "--deterministic",
        "--num-episodes", "500",
        "--num-envs", "64",
        "--no-drawdown-profit-early-exit",
        "--quiet-drawdown-profit-early-exit",
    ] + extra_args

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                cwd="/nvme0n1-disk/code/stock-prediction")
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return name, {"error": "timeout"}
    except Exception as e:
        return name, {"error": str(e)}

    # Parse results
    metrics = {}
    for line in output.split("\n"):
        if "Return:" in line and "mean=" in line:
            m = re.search(r"mean=([+-]?\d+\.\d+)", line)
            if m:
                metrics["mean_return"] = float(m.group(1))
            m = re.search(r"median=([+-]?\d+\.\d+)", line)
            if m:
                metrics["median_return"] = float(m.group(1))
            m = re.search(r">0:\s*(\d+)/(\d+)\s*\((\d+\.\d+)%\)", line)
            if m:
                metrics["profitable_pct"] = float(m.group(3))
        if "Sortino:" in line and "mean=" in line:
            m = re.search(r"mean=([+-]?\d+\.\d+)", line)
            if m:
                metrics["sortino"] = float(m.group(1))
        if "Win rate:" in line and "mean=" in line:
            m = re.search(r"mean=(\d+\.\d+)", line)
            if m:
                metrics["win_rate"] = float(m.group(1))
        if "Trades:" in line and "mean=" in line:
            m = re.search(r"mean=(\d+\.\d+)", line)
            if m:
                metrics["avg_trades"] = float(m.group(1))
        if "annualized return" in line:
            m = re.search(r"([+-]?\d+\.\d+)%", line)
            if m:
                metrics["annualized_pct"] = float(m.group(1))
        if "p05:" in line:
            m = re.search(r"p05:\s*([+-]?\d+\.\d+)", line)
            if m:
                metrics["p05_return"] = float(m.group(1))
        if "p95:" in line:
            m = re.search(r"p95:\s*([+-]?\d+\.\d+)", line)
            if m:
                metrics["p95_return"] = float(m.group(1))

    if not metrics:
        metrics["error"] = f"parse failed, output: {output[-500:]}"

    return name, metrics


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print(f"Evaluating {len(CANDIDATES)} candidates...")
    print(f"Settings: 90-day episodes, 8bps slippage, 0.1% fee, deterministic\n")

    results = {}
    # Run evaluations (parallel would be nice but GPU contention; run serial)
    for i, (name, ckpt, data, hs, arch, fee, extra) in enumerate(CANDIDATES):
        print(f"[{i+1}/{len(CANDIDATES)}] {name}...", flush=True)
        name, metrics = run_eval(name, ckpt, data, hs, arch, fee, extra)
        results[name] = metrics
        if "error" not in metrics and "mean_return" in metrics:
            print(f"  -> return={metrics.get('mean_return', 0):+.4f} "
                  f"sortino={metrics.get('sortino', 0):.2f} "
                  f"profitable={metrics.get('profitable_pct', 0):.0f}% "
                  f"annualized={metrics.get('annualized_pct', 0):+.1f}%")
        else:
            print(f"  -> ERROR: {metrics['error'][:100]}")

    # Print summary sorted by Sortino
    print(f"\n{'='*100}")
    print(f"DAILY RL CANDIDATE COMPARISON - SORTED BY SORTINO")
    print(f"{'='*100}")
    print(f"{'Name':<35} {'Return':>8} {'Sortino':>8} {'Profit%':>8} {'Annual%':>9} {'WR':>6} {'Trades':>7} {'p5':>8} {'p95':>8}")
    print(f"{'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*6} {'-'*7} {'-'*8} {'-'*8}")

    sorted_results = sorted(results.items(),
                            key=lambda x: x[1].get("sortino", -999),
                            reverse=True)

    for name, m in sorted_results:
        if "error" in m:
            print(f"{name:<35} {'ERROR':>8} {m['error'][:60]}")
            continue
        print(f"{name:<35} {m.get('mean_return', 0):>+8.4f} "
              f"{m.get('sortino', 0):>8.2f} "
              f"{m.get('profitable_pct', 0):>8.0f} "
              f"{m.get('annualized_pct', 0):>+9.1f} "
              f"{m.get('win_rate', 0):>6.2f} "
              f"{m.get('avg_trades', 0):>7.1f} "
              f"{m.get('p05_return', 0):>+8.4f} "
              f"{m.get('p95_return', 0):>+8.4f}")

    # Save results
    out_path = "/nvme0n1-disk/code/stock-prediction/daily_candidate_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
