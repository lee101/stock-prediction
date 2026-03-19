#!/usr/bin/env python3
"""Train a base Chronos2 LoRA on multiple stocks, then fine-tune per symbol.

Phase 1: Base LoRA trained on 10 diverse stocks simultaneously.
Phase 2: Fine-tune the base LoRA on individual symbols (AVAXUSD, NET, AMD).
Phase 3: Train from scratch on same symbols for comparison.

Usage:
    # Full pipeline (all 3 phases)
    python train_base_stocks_lora.py

    # Smoke test (100 steps)
    python train_base_stocks_lora.py --smoke-test

    # Phase 1 only
    python train_base_stocks_lora.py --phase 1

    # Custom steps
    python train_base_stocks_lora.py --base-steps 2000 --finetune-steps 200 --scratch-steps 1000
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chronos2_trainer import (
    TrainerConfig,
    _load_pipeline,
    _fit_pipeline,
    _save_pipeline,
    _load_hourly_frame,
    _split_windows,
    compute_mae_percent,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_SYMBOLS = ["AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "AMD", "NFLX", "ADBE"]

FINETUNE_SYMBOLS = [
    ("AVAXUSD", "crypto"),
    ("NET", "stock"),
    ("AMD", "stock"),
]

TARGET_COLS = ("open", "high", "low", "close")
_TARGET_COLS_LIST = list(TARGET_COLS)
VAL_HOURS = 168
TEST_HOURS = 168

MODEL_ID = "amazon/chronos-2"

# Resolve data roots: check local dir first, fall back to repo root
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR
# Walk up to find repo root (contains trainingdatahourly/)
for _p in [_SCRIPT_DIR] + list(_SCRIPT_DIR.parents):
    if (_p / "trainingdatahourly").exists():
        _REPO_ROOT = _p
        break


def _find_data_root(relative: str) -> Path:
    """Find data directory, checking script dir then repo root."""
    local = _SCRIPT_DIR / relative
    if local.exists():
        return local
    repo = _REPO_ROOT / relative
    if repo.exists():
        return repo
    return Path(relative)  # fallback to relative


STOCK_DATA_ROOT = _find_data_root("trainingdatahourly/stocks")
CRYPTO_DATA_ROOT = _find_data_root("trainingdatahourly/crypto")

OUTPUT_ROOT = Path("chronos2_finetuned")
BASE_OUTPUT = OUTPUT_ROOT / "base_stocks"
FINETUNE_OUTPUT = OUTPUT_ROOT / "base_finetune"
SCRATCH_OUTPUT = OUTPUT_ROOT / "scratch_comparison"
RESULTS_DIR = Path("hyperparams/base_stocks_lora")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_symbol_data(symbol: str, asset_type: str) -> pd.DataFrame:
    """Load hourly data for a symbol based on its asset type."""
    root = CRYPTO_DATA_ROOT if asset_type == "crypto" else STOCK_DATA_ROOT
    path = root / f"{symbol}.csv"
    return _load_hourly_frame(path, TARGET_COLS)


# Cache the augmentation instance -- stateless for differencing
_DIFFERENCING_AUG = None


def apply_differencing(df: pd.DataFrame) -> pd.DataFrame:
    """Apply first-order differencing to OHLC columns."""
    global _DIFFERENCING_AUG
    if _DIFFERENCING_AUG is None:
        from preaug_sweeps.augmentations import get_augmentation
        _DIFFERENCING_AUG = get_augmentation("differencing")
    return _DIFFERENCING_AUG.transform_dataframe(df.copy())


def prepare_inputs_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    """Convert a DataFrame to a single training input dict."""
    values = df[_TARGET_COLS_LIST].to_numpy(dtype=np.float32).T
    return {"target": values}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_mae_percent(
    pipeline: Any,
    df: pd.DataFrame,
    context_length: int,
    prediction_length: int = 1,
    n_samples: int = 50,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
) -> float:
    """Evaluate MAE% on a DataFrame using sliding window forecasts.

    When start_idx/end_idx are provided, evaluate only on that range
    (e.g., val or test window), using prior rows as context lookback.
    """
    data = df[_TARGET_COLS_LIST].to_numpy(dtype=np.float32)
    n = len(data)

    if start_idx is None:
        start_idx = context_length
    if end_idx is None:
        end_idx = n

    effective_start = max(start_idx, context_length)
    if effective_start >= end_idx:
        logger.warning("Not enough data for evaluation: start={} >= end={} (ctx={})",
                       effective_start, end_idx, context_length)
        return float("nan")

    available_windows = end_idx - effective_start
    step = max(1, available_windows // n_samples)

    quantiles = getattr(pipeline, "quantiles", None)
    if quantiles is None:
        logger.warning("Pipeline missing quantiles; using index 0")
        q_index = 0
    else:
        distances = [abs(float(q) - 0.5) for q in quantiles]
        q_index = int(np.argmin(distances))

    close_idx = _TARGET_COLS_LIST.index("close")
    all_actual = []
    all_pred = []

    for idx in range(effective_start, end_idx, step):
        ctx_start = idx - context_length
        inp = data[ctx_start:idx].T
        future_end = min(idx + prediction_length, n)
        actual = data[idx:future_end, close_idx]

        if len(actual) < prediction_length:
            continue

        with torch.no_grad():
            preds = pipeline.predict([inp], prediction_length=prediction_length, batch_size=1)

        if not preds:
            continue

        pred_tensor = preds[0].detach().cpu().numpy()
        pred_close = pred_tensor[close_idx, q_index, :]

        all_actual.extend(actual.tolist())
        all_pred.extend(pred_close.tolist())

    if not all_actual:
        return float("nan")

    return compute_mae_percent(np.array(all_pred), np.array(all_actual))


# ---------------------------------------------------------------------------
# Shared: train a LoRA on one symbol and evaluate
# ---------------------------------------------------------------------------

def _train_and_eval_single(
    symbol: str,
    asset_type: str,
    model_id: str,
    output_dir: Path,
    num_steps: int,
    learning_rate: float,
    context_length: int,
    lora_r: int,
    lora_alpha: int,
    batch_size: int,
    method_label: str,
) -> Dict[str, Any]:
    """Load data, train a LoRA, evaluate, and return results dict.

    Shared by both finetune_from_base and train_from_scratch.
    """
    df = load_symbol_data(symbol, asset_type)

    if len(df) < VAL_HOURS + TEST_HOURS + context_length:
        logger.error("Not enough data for {}: {} rows", symbol, len(df))
        return {"symbol": symbol, "error": "insufficient data"}

    train_df, val_df, test_df = _split_windows(df, VAL_HOURS, TEST_HOURS)

    # Augment full data once, then slice for train/val inputs
    full_aug = apply_differencing(pd.concat([train_df, val_df, test_df], ignore_index=True))
    train_len = len(train_df)
    val_len = len(val_df)
    train_aug = full_aug.iloc[:train_len]
    val_aug = full_aug.iloc[train_len:train_len + val_len]

    train_inputs = [prepare_inputs_from_df(train_aug)]
    val_inputs = [prepare_inputs_from_df(val_aug)]

    output_dir.mkdir(parents=True, exist_ok=True)

    config = TrainerConfig(
        symbol=symbol,
        data_root=CRYPTO_DATA_ROOT if asset_type == "crypto" else STOCK_DATA_ROOT,
        output_root=output_dir.parent,
        model_id=model_id,
        device_map="cuda",
        target_cols=TARGET_COLS,
        prediction_length=1,
        context_length=context_length,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_steps=num_steps,
        val_hours=VAL_HOURS,
        test_hours=TEST_HOURS,
        finetune_mode="lora",
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        lora_targets=("q", "k", "v", "o"),
        merge_lora=True,
        save_name=f"{method_label}_{symbol}",
    )

    pipeline = _load_pipeline(model_id, "cuda", None)
    finetuned = _fit_pipeline(pipeline, train_inputs, val_inputs, config, output_dir)
    _save_pipeline(finetuned, output_dir, "finetuned-ckpt")

    # Evaluate on val/test windows with full context lookback
    val_start = train_len
    val_end = val_start + val_len
    test_start = val_end
    test_end = len(full_aug)

    val_mae = evaluate_mae_percent(finetuned, full_aug, context_length,
                                   start_idx=val_start, end_idx=val_end)
    test_mae = evaluate_mae_percent(finetuned, full_aug, context_length,
                                    start_idx=test_start, end_idx=test_end)

    result = {
        "symbol": symbol,
        "asset_type": asset_type,
        "method": method_label,
        "val_mae_percent": val_mae,
        "test_mae_percent": test_mae,
        "num_steps": num_steps,
        "learning_rate": learning_rate,
        "output_dir": str(output_dir),
    }

    logger.info("{} {}: val_MAE%={:.4f} test_MAE%={:.4f}", symbol, method_label, val_mae, test_mae)

    del pipeline, finetuned
    gc.collect()
    torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Phase 1: Base LoRA training on multiple stocks
# ---------------------------------------------------------------------------

def train_base_lora(
    symbols: List[str],
    num_steps: int = 2000,
    learning_rate: float = 1e-5,
    context_length: int = 512,
    lora_r: int = 32,
    lora_alpha: int = 64,
    batch_size: int = 32,
) -> Path:
    """Train a single LoRA on multiple stock symbols."""
    logger.info("=" * 70)
    logger.info("PHASE 1: Base LoRA on {} stocks", len(symbols))
    logger.info("=" * 70)

    train_inputs = []
    val_inputs = []
    symbol_stats = {}

    for sym in symbols:
        logger.info("Loading {}", sym)
        try:
            df = load_symbol_data(sym, "stock")
        except FileNotFoundError as e:
            logger.warning("Skipping {}: {}", sym, e)
            continue

        if len(df) < VAL_HOURS + TEST_HOURS + context_length:
            logger.warning("Skipping {}: only {} rows (need {})", sym, len(df),
                           VAL_HOURS + TEST_HOURS + context_length)
            continue

        train_df, val_df, _ = _split_windows(df, VAL_HOURS, TEST_HOURS)

        train_aug = apply_differencing(train_df)
        val_aug = apply_differencing(val_df)

        train_inputs.append(prepare_inputs_from_df(train_aug))
        val_inputs.append(prepare_inputs_from_df(val_aug))

        symbol_stats[sym] = {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
        }
        logger.info("  {} train={} val={}", sym, len(train_df), len(val_df))

    if not train_inputs:
        raise RuntimeError("No symbols had enough data for training")

    logger.info("Training base LoRA on {} symbols", len(train_inputs))

    output_dir = BASE_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TrainerConfig(
        symbol="MULTI",
        data_root=STOCK_DATA_ROOT,
        output_root=OUTPUT_ROOT,
        model_id=MODEL_ID,
        device_map="cuda",
        target_cols=TARGET_COLS,
        prediction_length=1,
        context_length=context_length,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_steps=num_steps,
        val_hours=VAL_HOURS,
        test_hours=TEST_HOURS,
        finetune_mode="lora",
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        lora_targets=("q", "k", "v", "o"),
        merge_lora=True,
        save_name="base_stocks",
    )

    pipeline = _load_pipeline(MODEL_ID, "cuda", None)
    finetuned = _fit_pipeline(pipeline, train_inputs, val_inputs, config, output_dir)
    _save_pipeline(finetuned, output_dir, "finetuned-ckpt")

    meta = {
        "phase": "base_lora",
        "symbols": symbols,
        "symbol_stats": symbol_stats,
        "config": {
            "num_steps": num_steps,
            "learning_rate": learning_rate,
            "context_length": context_length,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "batch_size": batch_size,
            "preaug": "differencing",
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))

    logger.info("Base LoRA saved to {}", output_dir)

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir


# ---------------------------------------------------------------------------
# Phase 2: Fine-tune base LoRA on individual symbols
# ---------------------------------------------------------------------------

def finetune_from_base(
    base_dir: Path,
    symbol: str,
    asset_type: str,
    num_steps: int = 200,
    learning_rate: float = 5e-6,
    context_length: int = 512,
    lora_r: int = 32,
    lora_alpha: int = 64,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Fine-tune the base LoRA on a single symbol."""
    logger.info("-" * 50)
    logger.info("Fine-tuning base LoRA on {} ({})", symbol, asset_type)
    logger.info("-" * 50)

    base_ckpt = base_dir / "finetuned-ckpt"
    if base_ckpt.exists():
        model_id = str(base_ckpt)
    else:
        logger.warning("Base checkpoint not found at {}, falling back to base model", base_ckpt)
        model_id = MODEL_ID

    return _train_and_eval_single(
        symbol=symbol,
        asset_type=asset_type,
        model_id=model_id,
        output_dir=FINETUNE_OUTPUT / symbol,
        num_steps=num_steps,
        learning_rate=learning_rate,
        context_length=context_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        batch_size=batch_size,
        method_label="base_finetune",
    )


# ---------------------------------------------------------------------------
# Phase 3: Train from scratch on individual symbols
# ---------------------------------------------------------------------------

def train_from_scratch(
    symbol: str,
    asset_type: str,
    num_steps: int = 1000,
    learning_rate: float = 1e-5,
    context_length: int = 512,
    lora_r: int = 32,
    lora_alpha: int = 64,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Train LoRA from scratch on a single symbol for comparison."""
    logger.info("-" * 50)
    logger.info("Training from scratch on {} ({})", symbol, asset_type)
    logger.info("-" * 50)

    return _train_and_eval_single(
        symbol=symbol,
        asset_type=asset_type,
        model_id=MODEL_ID,
        output_dir=SCRATCH_OUTPUT / symbol,
        num_steps=num_steps,
        learning_rate=learning_rate,
        context_length=context_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        batch_size=batch_size,
        method_label="from_scratch",
    )


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def run_comparison(results: List[Dict[str, Any]]) -> str:
    """Generate a comparison report from all results."""
    lines = []
    lines.append("=" * 70)
    lines.append("BASE LoRA vs FROM-SCRATCH COMPARISON")
    lines.append("=" * 70)
    lines.append(f"{'Symbol':<10} {'Method':<16} {'Val MAE%':>10} {'Test MAE%':>10}")
    lines.append("-" * 50)

    by_symbol: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in results:
        sym = r["symbol"]
        by_symbol.setdefault(sym, {})[r["method"]] = r

    for sym in sorted(by_symbol.keys()):
        for method in ["base_finetune", "from_scratch"]:
            if method in by_symbol[sym]:
                r = by_symbol[sym][method]
                val = r.get("val_mae_percent", float("nan"))
                test = r.get("test_mae_percent", float("nan"))
                lines.append(f"{sym:<10} {method:<16} {val:>10.4f} {test:>10.4f}")

        ft = by_symbol[sym].get("base_finetune", {})
        sc = by_symbol[sym].get("from_scratch", {})
        if ft and sc and not ft.get("error") and not sc.get("error"):
            ft_test = ft.get("test_mae_percent", float("nan"))
            sc_test = sc.get("test_mae_percent", float("nan"))
            if not (np.isnan(ft_test) or np.isnan(sc_test)) and sc_test > 0:
                improvement = (sc_test - ft_test) / sc_test * 100
                direction = "BETTER" if improvement > 0 else "WORSE"
                lines.append(f"  -> base_finetune is {abs(improvement):.1f}% {direction} than scratch")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Base stocks LoRA + per-symbol fine-tuning")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick smoke test with 100 steps")
    parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Run specific phase (0=all, 1=base, 2=finetune, 3=scratch)")
    parser.add_argument("--base-steps", type=int, default=2000,
                        help="Training steps for base LoRA")
    parser.add_argument("--finetune-steps", type=int, default=200,
                        help="Training steps for fine-tuning")
    parser.add_argument("--scratch-steps", type=int, default=1000,
                        help="Training steps for from-scratch comparison")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--base-lr", type=float, default=1e-5)
    parser.add_argument("--finetune-lr", type=float, default=5e-6)
    parser.add_argument("--scratch-lr", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.smoke_test:
        args.base_steps = 100
        args.finetune_steps = 50
        args.scratch_steps = 100
        logger.info("SMOKE TEST MODE: reduced steps")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: List[Dict[str, Any]] = []

    # Phase 1: Base LoRA
    if args.phase in (0, 1):
        base_dir = train_base_lora(
            symbols=BASE_SYMBOLS,
            num_steps=args.base_steps,
            learning_rate=args.base_lr,
            context_length=args.context_length,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            batch_size=args.batch_size,
        )
    else:
        base_dir = BASE_OUTPUT
        if not (base_dir / "finetuned-ckpt").exists():
            logger.error("Base LoRA not found at {}. Run phase 1 first.", base_dir)
            return

    # Phase 2: Fine-tune from base
    if args.phase in (0, 2):
        for symbol, asset_type in FINETUNE_SYMBOLS:
            result = finetune_from_base(
                base_dir=base_dir,
                symbol=symbol,
                asset_type=asset_type,
                num_steps=args.finetune_steps,
                learning_rate=args.finetune_lr,
                context_length=args.context_length,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                batch_size=args.batch_size,
            )
            all_results.append(result)

    # Phase 3: From scratch comparison
    if args.phase in (0, 3):
        for symbol, asset_type in FINETUNE_SYMBOLS:
            result = train_from_scratch(
                symbol=symbol,
                asset_type=asset_type,
                num_steps=args.scratch_steps,
                learning_rate=args.scratch_lr,
                context_length=args.context_length,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                batch_size=args.batch_size,
            )
            all_results.append(result)

    # Save results
    if all_results:
        results_path = RESULTS_DIR / "comparison_results.json"
        results_path.write_text(json.dumps(all_results, indent=2))
        logger.info("Results saved to {}", results_path)

        report = run_comparison(all_results)
        print(report)

        report_path = RESULTS_DIR / "comparison_report.txt"
        report_path.write_text(report)
        logger.info("Report saved to {}", report_path)


if __name__ == "__main__":
    main()
