#!/usr/bin/env python3
"""Sweep Chronos2 LoRA fine-tuning with augmentation strategies for SUIUSDT.

Tests: preaug strategies, context lengths, learning rates, LoRA ranks.
Rebuilds forecast cache with each new LoRA, then evaluates downstream policy.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = Path(__file__).resolve().parent


@dataclass
class LoRAConfig:
    symbol: str = "SUIUSDT"
    preaug: str = "baseline"
    context_length: int = 512
    prediction_length: int = 24
    learning_rate: float = 5e-5
    num_steps: int = 1000
    lora_r: int = 16
    lora_alpha: int = 32


def run_lora_train(cfg: LoRAConfig) -> dict:
    data_root = REPO_ROOT / "trainingdatahourlybinance"
    output_root = REPO_ROOT / "chronos2_finetuned"
    results_dir = REPO_ROOT / "hyperparams" / "sui_lora_sweep"
    results_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "train_crypto_lora_sweep.py"),
        "--symbol", cfg.symbol,
        "--data-root", str(data_root),
        "--output-root", str(output_root),
        "--results-dir", str(results_dir),
        "--context-length", str(cfg.context_length),
        "--prediction-length", str(cfg.prediction_length),
        "--learning-rate", str(cfg.learning_rate),
        "--num-steps", str(cfg.num_steps),
        "--lora-r", str(cfg.lora_r),
        "--preaug", cfg.preaug,
    ]

    print(f"\n{'='*60}")
    print(f"LoRA: {cfg.symbol} preaug={cfg.preaug} ctx={cfg.context_length} "
          f"lr={cfg.learning_rate:.0e} r={cfg.lora_r} steps={cfg.num_steps}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=str(REPO_ROOT))
        elapsed = time.time() - t0

        # Find the results file
        results_files = sorted(results_dir.glob(f"{cfg.symbol}*{cfg.preaug}*ctx{cfg.context_length}*.json"),
                               key=lambda p: p.stat().st_mtime, reverse=True)
        if results_files:
            with open(results_files[0]) as f:
                metrics = json.load(f)
            out = {
                "config": asdict(cfg),
                "run_name": metrics.get("run_name", ""),
                "output_dir": metrics.get("output_dir", ""),
                "val_mae_pct": metrics.get("val", {}).get("mae_percent_mean", float("inf")),
                "val_consistency": metrics.get("val_consistency_score", float("inf")),
                "test_mae_pct": metrics.get("test", {}).get("mae_percent_mean", float("inf")),
                "test_consistency": metrics.get("test_consistency_score", float("inf")),
                "elapsed_s": elapsed,
            }
            print(f"  -> val_mae={out['val_mae_pct']:.3f}% test_mae={out['test_mae_pct']:.3f}% "
                  f"consistency={out['val_consistency']:.3f} ({elapsed:.0f}s)")
            return out
        else:
            return {"config": asdict(cfg), "error": "no results file",
                    "stderr": result.stderr[-500:], "elapsed_s": elapsed}
    except Exception as e:
        return {"config": asdict(cfg), "error": str(e), "elapsed_s": time.time() - t0}


def rebuild_forecast_cache(lora_model_path: str, symbol: str = "SUIUSDT") -> bool:
    """Rebuild forecast cache with a specific LoRA model."""
    cache_dir = EXP_DIR / f"forecast_cache_{symbol.lower()}_sweep"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-c",
        f"""
import sys; sys.path.insert(0, '{REPO_ROOT}')
from binancechronossolexperiment.forecasts import build_forecast_bundle
from pathlib import Path
bundle = build_forecast_bundle(
    symbol='{symbol}',
    data_root=Path('trainingdatahourlybinance'),
    cache_root=Path('{cache_dir}'),
    horizons=(1, 4, 24),
    context_hours=512,
    quantile_levels=(0.1, 0.5, 0.9),
    batch_size=32,
    model_id='{lora_model_path}',
    device_map='cuda',
    cache_only=False,
)
print(f'Generated {{len(bundle)}} forecast rows')
""",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=str(REPO_ROOT))
        print(f"Cache rebuild: {result.stdout.strip()}")
        return result.returncode == 0
    except Exception as e:
        print(f"Cache rebuild failed: {e}")
        return False


def run_policy_with_cache(cache_dir: str, run_name: str) -> dict:
    """Train policy using a specific forecast cache."""
    cmd = [
        sys.executable, "-m", "binancechronossolexperiment.run_experiment",
        "--symbol", "SUIUSDT",
        "--return-weight", "0.012",
        "--epochs", "25",
        "--sequence-length", "72",
        "--learning-rate", "1e-4",
        "--horizons", "1,4,24",
        "--batch-size", "16",
        "--seed", "1337",
        "--forecast-cache-root", cache_dir,
        "--cache-only",
        "--no-compile",
        "--no-plot",
        "--run-name", run_name,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=str(REPO_ROOT))
        metrics_path = EXP_DIR / "results" / run_name / "simulation_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            test_m = m.get("metrics", {}).get("test", {})
            return {
                "name": run_name,
                "test_sortino": test_m.get("sortino", 0),
                "test_return": test_m.get("total_return", 0),
                "final_equity": test_m.get("final_equity", 10000),
            }
        return {"name": run_name, "error": "no metrics"}
    except Exception as e:
        return {"name": run_name, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SUIUSDT")
    parser.add_argument("--lora-only", action="store_true", help="Only train LoRAs, skip policy eval")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else EXP_DIR / f"lora_sweep_{args.symbol.lower()}_{timestamp}.json"

    configs = [
        # Augmentation strategies with ctx=512
        LoRAConfig(symbol=args.symbol, preaug="baseline", context_length=512),
        LoRAConfig(symbol=args.symbol, preaug="percent_change", context_length=512),
        LoRAConfig(symbol=args.symbol, preaug="log_returns", context_length=512),
        LoRAConfig(symbol=args.symbol, preaug="differencing", context_length=512),
        LoRAConfig(symbol=args.symbol, preaug="robust_scaling", context_length=512),
        # Context length variations with best aug (baseline from prior results)
        LoRAConfig(symbol=args.symbol, preaug="baseline", context_length=256),
        LoRAConfig(symbol=args.symbol, preaug="baseline", context_length=1024),
        # Learning rate variations
        LoRAConfig(symbol=args.symbol, preaug="baseline", context_length=512, learning_rate=1e-5),
        LoRAConfig(symbol=args.symbol, preaug="baseline", context_length=512, learning_rate=1e-4),
        LoRAConfig(symbol=args.symbol, preaug="baseline", context_length=512, learning_rate=2e-4),
        # LoRA rank variations
        LoRAConfig(symbol=args.symbol, preaug="baseline", context_length=512, lora_r=8),
        LoRAConfig(symbol=args.symbol, preaug="baseline", context_length=512, lora_r=32),
        # More steps
        LoRAConfig(symbol=args.symbol, preaug="baseline", context_length=512, num_steps=2000),
        LoRAConfig(symbol=args.symbol, preaug="baseline", context_length=512, num_steps=3000),
        # Best preaug combos
        LoRAConfig(symbol=args.symbol, preaug="percent_change", context_length=512, learning_rate=1e-4),
        LoRAConfig(symbol=args.symbol, preaug="log_returns", context_length=512, learning_rate=1e-4),
    ]

    results = []
    for cfg in configs:
        result = run_lora_train(cfg)
        results.append(result)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    # Sort by val MAE
    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda x: x.get("val_mae_pct", float("inf")))

    print(f"\n{'='*70}")
    print(f"LORA SWEEP COMPLETE: {args.symbol}")
    print(f"{'='*70}")
    print(f"{'Preaug':15s} {'Ctx':>5s} {'LR':>8s} {'R':>3s} {'ValMAE%':>8s} {'TestMAE%':>8s} {'Consist':>8s}")
    print("-" * 60)
    for r in valid:
        c = r["config"]
        print(f"{c['preaug']:15s} {c['context_length']:5d} {c['learning_rate']:.0e} "
              f"{c['lora_r']:3d} {r['val_mae_pct']:8.3f} {r['test_mae_pct']:8.3f} "
              f"{r['val_consistency']:8.3f}")

    if valid:
        best = valid[0]
        print(f"\nBest LoRA: {best['run_name']} (val MAE {best['val_mae_pct']:.3f}%)")

    if not args.lora_only and valid:
        print("\n\nPhase 2: Evaluating top LoRAs with policy training...")
        top_loras = valid[:3]
        policy_results = []
        for lr in top_loras:
            model_path = lr.get("output_dir", "")
            if not model_path:
                continue
            ckpt_path = Path(model_path) / "finetuned-ckpt"
            if not ckpt_path.exists():
                print(f"  Skipping {lr['run_name']}: no checkpoint")
                continue

            cache_dir = str(EXP_DIR / f"forecast_cache_{args.symbol.lower()}_sweep")
            print(f"\n  Rebuilding cache with {lr['run_name']}...")
            if rebuild_forecast_cache(str(ckpt_path), args.symbol):
                policy_name = f"policy_lora_{lr['config']['preaug']}_ctx{lr['config']['context_length']}"
                pr = run_policy_with_cache(cache_dir, policy_name)
                pr["lora_run"] = lr["run_name"]
                pr["lora_val_mae"] = lr["val_mae_pct"]
                policy_results.append(pr)
                print(f"  -> Policy sortino={pr.get('test_sortino', 'N/A')} return={pr.get('test_return', 'N/A')}")

        results.append({"policy_results": policy_results})
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
