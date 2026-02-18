#!/usr/bin/env python3
"""Train Chronos2 LoRA models for all stocks - in-process (no subprocess)."""
import json
import sys
import time
import gc
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
from loguru import logger
from preaug import get_augmentation
from chronos2_trainer import _load_pipeline, _fit_pipeline, _save_pipeline

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
RESULTS_DIR = Path("hyperparams/crypto_lora_sweep")
OUTPUT_ROOT = Path("chronos2_finetuned")


class FakeConfig:
    def __init__(self, ctx, lr, steps, lora_r=16):
        self.context_length = ctx
        self.prediction_length = 24
        self.batch_size = 32
        self.learning_rate = float(lr)
        self.num_steps = steps
        self.lora_rank = lora_r
        self.lora_target_modules = "q,v"
        self.device_map = "cuda"
        self.torch_dtype = None


def eval_model(pipeline, data, ctx, pred_len=24):
    from chronos import Chronos2Pipeline
    results = []
    for i in range(len(data) - ctx - pred_len):
        inp = torch.tensor(data[i:i+ctx], dtype=torch.float32)
        with torch.no_grad():
            preds = pipeline.predict([inp], prediction_length=pred_len, batch_size=1)
        pred_mean = preds[0].mean(dim=0).numpy()
        actual = data[i+ctx:i+ctx+pred_len]
        mae = np.abs(pred_mean - actual).mean()
        pct_mae = mae / (np.abs(actual).mean() + 1e-8) * 100
        results.append(pct_mae)
    return np.mean(results), np.std(results)


def train_single(symbol, preaug, ctx, lr, steps, lora_r=16):
    data_path = Path(DATA_ROOT) / f"{symbol}.csv"
    if not data_path.exists():
        print(f"  No data for {symbol}")
        return None

    df = pd.read_csv(data_path)
    if "close" not in df.columns:
        print(f"  No close column for {symbol}")
        return None

    close = df["close"].values.astype(np.float64)
    aug_fn = get_augmentation(preaug)
    aug_data = aug_fn(close)

    n = len(aug_data)
    val_size = min(168, n // 10)
    test_size = val_size
    train_data = aug_data[:n - val_size - test_size]
    val_data = aug_data[n - val_size - test_size:n - test_size]
    test_data = aug_data[n - test_size:]

    logger.info(f"Training {symbol} {preaug} ctx={ctx} lr={lr} steps={steps}: "
                f"train={len(train_data)} val={len(val_data)} test={len(test_data)}")

    train_inputs = [torch.tensor(train_data, dtype=torch.float32)]
    val_inputs = [torch.tensor(val_data, dtype=torch.float32)]

    run_name = f"{symbol}_lora_{preaug}_ctx{ctx}_lr{lr}_r{lora_r}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = OUTPUT_ROOT / run_name

    cfg = FakeConfig(ctx, lr, steps, lora_r)

    pipeline = _load_pipeline("amazon/chronos-2", "cuda", None)
    finetuned = _fit_pipeline(pipeline, train_inputs, val_inputs, cfg, str(output_dir))
    _save_pipeline(finetuned, str(output_dir), "finetuned-ckpt")

    val_mae, val_std = eval_model(finetuned, val_data, ctx)
    test_mae, test_std = eval_model(finetuned, test_data, ctx)

    result = {
        "run_name": run_name,
        "config": {"symbol": symbol, "preaug": preaug, "context_length": ctx,
                    "learning_rate": float(lr), "num_steps": steps, "lora_r": lora_r},
        "val": {"mae_percent_mean": float(val_mae), "mae_percent_std": float(val_std)},
        "test": {"mae_percent_mean": float(test_mae), "mae_percent_std": float(test_std)},
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"{run_name}.json"
    result_path.write_text(json.dumps(result, indent=2))

    del pipeline, finetuned
    gc.collect()
    torch.cuda.empty_cache()

    return result


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
                print(f"[{done}/{total}] {symbol} {preaug} ctx={cfg['ctx']} lr={cfg['lr']} (ETA: {eta:.0f}min)",
                      flush=True)

                try:
                    r = train_single(symbol, preaug, cfg["ctx"], cfg["lr"], cfg["steps"])
                except Exception as e:
                    print(f"  ERROR: {e}", flush=True)
                    r = None

                if r:
                    val_mae = r["val"]["mae_percent_mean"]
                    test_mae = r["test"]["mae_percent_mean"]
                    model_name = r["run_name"]
                    print(f"  val_mae={val_mae:.3f}% test_mae={test_mae:.3f}%", flush=True)

                    results[key] = {
                        "symbol": symbol, "preaug": preaug,
                        "ctx": cfg["ctx"], "lr": cfg["lr"], "steps": cfg["steps"],
                        "val_mae_pct": val_mae, "test_mae_pct": test_mae,
                        "model_name": model_name,
                        "timestamp": datetime.now().isoformat(),
                    }

                    if test_mae < best_mae:
                        best_mae = test_mae
                        best_config = key

                    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
                    RESULTS_FILE.write_text(json.dumps(results, indent=2))

        if best_config:
            print(f"\n  BEST for {symbol}: {best_config} -> test_mae={best_mae:.3f}%\n", flush=True)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for symbol in SYMBOLS:
        sym_results = {k: v for k, v in results.items() if v.get("symbol") == symbol}
        if sym_results:
            best_key = min(sym_results, key=lambda k: sym_results[k].get("test_mae_pct", 999))
            best = sym_results[best_key]
            print(f"{symbol:<6} best={best['preaug']:<20} ctx={best['ctx']} lr={best['lr']} "
                  f"test_mae={best['test_mae_pct']:.3f}%")

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
