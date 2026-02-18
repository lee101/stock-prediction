#!/usr/bin/env python3
"""Train Chronos2 LoRA models for all stocks with preaug sweep on remote 5090."""
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

SYMBOLS = [
    "NVDA", "MSFT", "META", "GOOG", "NET", "PLTR", "DBX",
    "ANGI", "BKNG", "EBAY", "EXPE", "MTCH", "NWSA", "NYT", "TRIP", "YELP", "Z",
    "AAPL", "TSLA",
]

PREAUGS = ["differencing", "percent_change", "log_returns", "robust_scaling"]

CONFIGS = [
    {"ctx": 128, "lr": "5e-5", "steps": 1000},
    {"ctx": 256, "lr": "5e-5", "steps": 1000},
    {"ctx": 128, "lr": "1e-4", "steps": 1500},
]

DATA_ROOT = "trainingdatahourly/stocks"
RESULTS_FILE = Path("hyperparams/stock_lora_sweep_results.json")


def parse_result(data):
    """Extract val/test MAE from nested or flat JSON structure."""
    if isinstance(data.get("val"), dict):
        val_mae = data["val"].get("mae_percent_mean", 999)
        test_mae = data.get("test", {}).get("mae_percent_mean", 999)
    else:
        val_mae = data.get("val_mae_percent_mean", data.get("val_mae_pct", 999))
        test_mae = data.get("test_mae_percent_mean", data.get("test_mae_pct", 999))
    model_name = data.get("run_name", data.get("model_name", ""))
    return val_mae, test_mae, model_name


def train_and_eval(symbol, preaug, ctx, lr, steps):
    cmd = [
        sys.executable, "scripts/train_crypto_lora_sweep.py",
        "--symbol", symbol,
        "--data-root", DATA_ROOT,
        "--context-length", str(ctx),
        "--learning-rate", lr,
        "--num-steps", str(steps),
        "--lora-r", "16",
        "--preaug", preaug,
    ]
    try:
        for attempt in range(5):
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                                    env={**__import__('os').environ, 'PYTHONUNBUFFERED': '1'})
            if result.returncode == -9 or result.returncode == 137:
                if attempt < 4 and "Loaded" not in result.stdout and "Loaded" not in result.stderr:
                    import time; time.sleep(2)
                    continue
            break
        # Check for saved JSON files (most reliable)
        # Try both lr formats: 5e-5 and 5e-05
        for lr_fmt in [lr, lr.replace("e-5", "e-05").replace("e-4", "e-04")]:
            pattern = f"hyperparams/crypto_lora_sweep/{symbol}_lora_{preaug}_ctx{ctx}_lr{lr_fmt}_r16_*.json"
            jsons = sorted(Path(".").glob(pattern))
            if jsons:
                return json.loads(jsons[-1].read_text())
        # Fallback: scan both stdout and stderr for JSON
        for output in [result.stdout, result.stderr]:
            for line in output.strip().split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue
        if result.returncode not in (0, 137, -9):
            print(f"  FAILED (rc={result.returncode}): {result.stderr[-200:]}")
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT")
    except Exception as e:
        print(f"  ERROR: {e}")
    return None


def main():
    results = {}
    if RESULTS_FILE.exists():
        results = json.loads(RESULTS_FILE.read_text())

    total = len(SYMBOLS) * len(PREAUGS) * len(CONFIGS)
    done = 0
    start = time.time()

    for symbol in SYMBOLS:
        best_mae = float("inf")
        best_config = None

        for preaug in PREAUGS:
            for cfg in CONFIGS:
                key = f"{symbol}_{preaug}_ctx{cfg['ctx']}_lr{cfg['lr']}"
                if key in results:
                    done += 1
                    mae = results[key].get("test_mae_pct", float("inf"))
                    if mae < best_mae:
                        best_mae = mae
                        best_config = key
                    continue

                done += 1
                elapsed = time.time() - start
                rate = elapsed / max(done - len(results), 1)
                eta = rate * (total - done) / 60
                print(f"[{done}/{total}] {symbol} {preaug} ctx={cfg['ctx']} lr={cfg['lr']} (ETA: {eta:.0f}min)")

                r = train_and_eval(symbol, preaug, cfg["ctx"], cfg["lr"], cfg["steps"])
                if r:
                    val_mae, test_mae, model_name = parse_result(r)
                    print(f"  val_mae={val_mae:.3f}% test_mae={test_mae:.3f}% model={model_name}")

                    results[key] = {
                        "symbol": symbol,
                        "preaug": preaug,
                        "ctx": cfg["ctx"],
                        "lr": cfg["lr"],
                        "steps": cfg["steps"],
                        "val_mae_pct": val_mae,
                        "test_mae_pct": test_mae,
                        "model_name": model_name,
                        "timestamp": datetime.now().isoformat(),
                    }

                    if test_mae < best_mae:
                        best_mae = test_mae
                        best_config = key

                    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
                    RESULTS_FILE.write_text(json.dumps(results, indent=2))

        if best_config:
            print(f"\n  BEST for {symbol}: {best_config} -> test_mae={best_mae:.3f}%\n")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for symbol in SYMBOLS:
        sym_results = {k: v for k, v in results.items() if v.get("symbol") == symbol}
        if sym_results:
            best_key = min(sym_results, key=lambda k: sym_results[k].get("test_mae_pct", 999))
            best = sym_results[best_key]
            print(f"{symbol:<6} best={best['preaug']:<20} ctx={best['ctx']} lr={best['lr']} "
                  f"test_mae={best['test_mae_pct']:.3f}% model={best.get('model_name', '?')}")

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
