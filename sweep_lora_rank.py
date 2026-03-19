#!/usr/bin/env python3
"""LoRA rank sweep: test r=8,16,32,64 across multiple symbols.

Finds the optimal expressiveness/generalization tradeoff for Chronos2 LoRA
fine-tuning by varying rank while keeping alpha=2*r.

Sweep: 5 symbols x 4 ranks = 20 configs, ~2min each, ~40min total.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# Ensure repo root is on path for imports
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from preaug import get_augmentation, AUGMENTATION_REGISTRY

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_ROOT = Path("chronos2_finetuned/lora_rank_sweep")
DEFAULT_RESULTS_DIR = Path("hyperparams/lora_rank_sweep")
DEFAULT_CSV = Path("lora_rank_sweep_results.csv")
BEST_PREAUG_DIR = Path("preaugstrategies/best/hourly")

SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD", "AAPL", "NVDA"]
RANKS = [8, 16, 32, 64]
TARGET_COLS = ["open", "high", "low", "close"]

DATA_ROOTS: Dict[str, Path] = {
    "BTCUSD": Path("trainingdatahourly/crypto"),
    "ETHUSD": Path("trainingdatahourly/crypto"),
    "SOLUSD": Path("trainingdatahourly/crypto"),
    "AAPL": Path("trainingdatahourly/stocks"),
    "NVDA": Path("trainingdatahourly/stocks"),
}


@dataclass
class SweepConfig:
    symbol: str
    lora_r: int
    lora_alpha: int
    preaug: str
    context_length: int = 512
    prediction_length: int = 1
    batch_size: int = 64
    learning_rate: float = 1e-5
    num_steps: int = 1000
    lora_dropout: float = 0.05
    val_hours: int = 168
    test_hours: int = 168


@dataclass
class SweepResult:
    symbol: str
    lora_r: int
    lora_alpha: int
    preaug: str
    val_mae_pct: float
    test_mae_pct: float
    val_mae: float
    test_mae: float
    val_count: int
    test_count: int
    train_time_s: float
    run_name: str
    output_dir: str


class _FitConfig:
    """Adapter that exposes SweepConfig fields in the layout _fit_pipeline expects."""

    def __init__(self, c: SweepConfig):
        self.context_length = c.context_length
        self.prediction_length = c.prediction_length
        self.batch_size = c.batch_size
        self.learning_rate = c.learning_rate
        self.num_steps = c.num_steps
        self.finetune_mode = "lora"
        self.lora_r = c.lora_r
        self.lora_alpha = c.lora_alpha
        self.lora_dropout = c.lora_dropout
        self.lora_targets = ("q", "k", "v", "o")
        self.merge_lora = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_best_preaug(symbol: str) -> str:
    """Load best preaug strategy name for a symbol, falling back to baseline."""
    path = BEST_PREAUG_DIR / f"{symbol}.json"
    if not path.exists():
        logger.info("No best preaug for {} ({}), using baseline", symbol, path)
        return "baseline"
    try:
        data = json.loads(path.read_text())
        strategy = data.get("best_strategy", "baseline")
        if strategy not in AUGMENTATION_REGISTRY:
            logger.warning(
                "Best preaug '{}' for {} not in registry ({}), falling back to baseline",
                strategy, symbol, sorted(AUGMENTATION_REGISTRY.keys()),
            )
            return "baseline"
        return strategy
    except Exception as exc:
        logger.warning("Failed to read preaug for {}: {}", symbol, exc)
        return "baseline"


def resolve_data_path(symbol: str) -> Path:
    """Find the CSV data file for a symbol."""
    root = DATA_ROOTS.get(symbol)
    if root is not None:
        candidate = root / f"{symbol}.csv"
        if candidate.exists():
            return candidate
    # Fallback: check both directories
    for d in [Path("trainingdatahourly/crypto"), Path("trainingdatahourly/stocks")]:
        candidate = d / f"{symbol}.csv"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No data file for {symbol}")


def load_hourly_frame(csv_path: Path) -> pd.DataFrame:
    """Load and clean hourly OHLCV data."""
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise KeyError(f"{csv_path} missing 'timestamp' column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    for col in TARGET_COLS + ["volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=TARGET_COLS)
    return df


def split_data(
    df: pd.DataFrame, val_hours: int, test_hours: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train / val / test."""
    if len(df) <= val_hours + test_hours:
        raise ValueError(
            f"Not enough rows ({len(df)}) for val={val_hours} test={test_hours}"
        )
    train_end = len(df) - (val_hours + test_hours)
    val_end = len(df) - test_hours
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


def compute_eval_metrics(
    pipeline: Any,
    df: pd.DataFrame,
    context_length: int,
    prediction_length: int,
    start_idx: int,
    end_idx: int,
    preaug_name: str = "baseline",
) -> Dict[str, float]:
    """Evaluate pipeline on a window, return mae and mae_percent."""
    if start_idx < context_length:
        start_idx = context_length
    if start_idx >= end_idx - prediction_length:
        return {"mae": float("inf"), "mae_percent": float("inf"), "count": 0}

    close_idx = TARGET_COLS.index("close")

    quantiles = getattr(pipeline, "quantiles", None)
    if quantiles is None:
        raise RuntimeError("Pipeline missing quantiles attribute")
    q_index = int(np.argmin([abs(float(q) - 0.5) for q in quantiles]))

    aug = get_augmentation(preaug_name) if preaug_name != "baseline" else None

    actual_vals: List[float] = []
    pred_vals: List[float] = []
    step = max(1, prediction_length)

    for idx in range(start_idx, end_idx - prediction_length + 1, step):
        context = df.iloc[idx - context_length : idx]
        future = df.iloc[idx : idx + prediction_length]
        if len(context) < context_length or len(future) < prediction_length:
            continue

        ctx_data = context.copy()
        if aug:
            ctx_data = aug.transform_dataframe(ctx_data)

        inputs = ctx_data[TARGET_COLS].to_numpy(dtype=np.float32).T
        try:
            preds = pipeline.predict(
                [inputs], prediction_length=prediction_length, batch_size=1
            )
        except Exception as e:
            logger.warning("Prediction failed at idx {}: {}", idx, e)
            continue
        if not preds:
            continue

        pred_tensor = preds[0].detach().cpu().numpy()
        pred_matrix = pred_tensor[:, q_index, :].T  # (pred_len, channels)

        if aug:
            pred_matrix = aug.inverse_transform_predictions(
                pred_matrix, context, columns=TARGET_COLS
            )

        actual_close = future[TARGET_COLS].to_numpy(dtype=np.float32)[:, close_idx]
        pred_close = pred_matrix[:, close_idx]

        actual_vals.extend(actual_close.tolist())
        pred_vals.extend(pred_close.tolist())

    if not actual_vals:
        return {"mae": float("inf"), "mae_percent": float("inf"), "count": 0}

    actual_arr = np.array(actual_vals, dtype=np.float64)
    pred_arr = np.array(pred_vals, dtype=np.float64)
    abs_errors = np.abs(actual_arr - pred_arr)
    mae = float(np.mean(abs_errors))
    mae_pct = float(
        np.mean(abs_errors / np.clip(np.abs(actual_arr), 1e-8, None)) * 100
    )
    return {"mae": mae, "mae_percent": mae_pct, "count": len(actual_vals)}


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_single(
    cfg: SweepConfig,
    output_root: Path,
    base_pipeline: Any,
    data_splits: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> SweepResult:
    """Train one (symbol, rank) config and evaluate.

    Accepts a pre-loaded base pipeline and pre-split data to avoid redundant
    model loading and CSV parsing when sweeping ranks for the same symbol.
    """
    from chronos2_trainer import _fit_pipeline, _save_pipeline

    train_df, val_df, test_df = data_splits

    # Apply preaug to training and validation data
    aug = get_augmentation(cfg.preaug) if cfg.preaug != "baseline" else None
    train_aug = aug.transform_dataframe(train_df.copy()) if aug else train_df
    val_aug = aug.transform_dataframe(val_df.copy()) if aug else val_df

    train_inputs = [{"target": train_aug[TARGET_COLS].to_numpy(dtype=np.float32).T}]
    val_inputs = [{"target": val_aug[TARGET_COLS].to_numpy(dtype=np.float32).T}]

    run_name = (
        f"{cfg.symbol}_r{cfg.lora_r}_a{cfg.lora_alpha}_{cfg.preaug}"
        f"_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = output_root / run_name

    t0 = time.time()
    finetuned = _fit_pipeline(
        base_pipeline, train_inputs, val_inputs, _FitConfig(cfg), output_dir
    )
    train_time = time.time() - t0
    logger.info("Training took {:.1f}s", train_time)

    _save_pipeline(finetuned, output_dir, "finetuned-ckpt")

    # Evaluate on val and test windows
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    val_start = len(train_df)
    val_end = val_start + len(val_df)
    test_start = val_end
    test_end = len(full_df)

    val_metrics = compute_eval_metrics(
        finetuned, full_df, cfg.context_length, cfg.prediction_length,
        val_start, val_end, cfg.preaug,
    )
    test_metrics = compute_eval_metrics(
        finetuned, full_df, cfg.context_length, cfg.prediction_length,
        test_start, test_end, cfg.preaug,
    )

    logger.info(
        "{} r={}: val MAE%={:.4f} test MAE%={:.4f} ({:.1f}s)",
        cfg.symbol, cfg.lora_r,
        val_metrics["mae_percent"], test_metrics["mae_percent"],
        train_time,
    )

    return SweepResult(
        symbol=cfg.symbol,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        preaug=cfg.preaug,
        val_mae_pct=val_metrics["mae_percent"],
        test_mae_pct=test_metrics["mae_percent"],
        val_mae=val_metrics["mae"],
        test_mae=test_metrics["mae"],
        val_count=int(val_metrics["count"]),
        test_count=int(test_metrics["count"]),
        train_time_s=train_time,
        run_name=run_name,
        output_dir=str(output_dir),
    )


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(
    symbols: List[str],
    ranks: List[int],
    output_root: Path,
    results_dir: Path,
    csv_path: Path,
) -> List[SweepResult]:
    """Run full LoRA rank sweep."""
    from chronos2_trainer import _load_pipeline

    output_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Resolve best preaug per symbol
    preaug_map: Dict[str, str] = {}
    for sym in symbols:
        preaug_map[sym] = load_best_preaug(sym)
        logger.info("  {} -> preaug={}", sym, preaug_map[sym])

    total = len(symbols) * len(ranks)
    results: List[SweepResult] = []
    idx = 0

    for sym in symbols:
        # Load data once per symbol
        data_path = resolve_data_path(sym)
        df = load_hourly_frame(data_path)
        logger.info("Loaded {} rows for {} from {}", len(df), sym, data_path)
        data_splits = split_data(df, 168, 168)
        logger.info(
            "Split: train={} val={} test={}",
            len(data_splits[0]), len(data_splits[1]), len(data_splits[2]),
        )

        # Load base pipeline once per symbol (LoRA training mutates model
        # weights, so we reload per symbol but not per rank -- _fit_pipeline
        # copies the base model internally so the pipeline is safe to reuse).
        base_pipeline = _load_pipeline("amazon/chronos-2", "cuda", None)

        for r in ranks:
            idx += 1
            alpha = 2 * r
            preaug = preaug_map[sym]
            logger.info(
                "\n[{}/{}] {} r={} alpha={} preaug={}",
                idx, total, sym, r, alpha, preaug,
            )

            cfg = SweepConfig(
                symbol=sym,
                lora_r=r,
                lora_alpha=alpha,
                preaug=preaug,
            )

            try:
                result = train_single(cfg, output_root, base_pipeline, data_splits)
                results.append(result)

                # Save individual result JSON
                result_path = results_dir / f"{result.run_name}.json"
                result_path.write_text(
                    json.dumps(asdict(result), indent=2, default=str)
                )
            except Exception as exc:
                logger.error("FAILED {}/{} r={}: {}", sym, preaug, r, exc)
                traceback.print_exc()
                results.append(SweepResult(
                    symbol=sym, lora_r=r, lora_alpha=alpha, preaug=preaug,
                    val_mae_pct=float("inf"), test_mae_pct=float("inf"),
                    val_mae=float("inf"), test_mae=float("inf"),
                    val_count=0, test_count=0, train_time_s=0.0,
                    run_name=f"FAILED_{sym}_r{r}", output_dir="",
                ))

    # Write CSV summary
    rows = [asdict(r) for r in results]
    out_df = pd.DataFrame(rows)
    out_df.to_csv(csv_path, index=False)
    logger.info("Results CSV: {}", csv_path)

    # Print summary table
    print_summary(results)

    return results


def print_summary(results: List[SweepResult]) -> None:
    """Print a summary table of MAE% by rank for each symbol."""
    print("\n" + "=" * 80)
    print("LoRA Rank Sweep Summary -- Val MAE% / Test MAE%")
    print("=" * 80)

    # Collect unique symbols and ranks preserving insertion order
    symbols_seen: List[str] = []
    for r in results:
        if r.symbol not in symbols_seen:
            symbols_seen.append(r.symbol)
    ranks_seen = sorted({r.lora_r for r in results})

    # Header
    header = f"{'Symbol':<10}"
    for rank in ranks_seen:
        header += f"  r={rank:<5}"
    header += "  Best"
    print(header)
    print("-" * 80)

    for sym in symbols_seen:
        sym_results = [r for r in results if r.symbol == sym]
        line_val = f"{'  ' + sym:<10}"
        line_test = f"{'  (test)':<10}"
        best_rank = None
        best_val = float("inf")

        for rank in ranks_seen:
            match = [r for r in sym_results if r.lora_r == rank]
            if match:
                r = match[0]
                line_val += f"  {r.val_mae_pct:>6.3f}"
                line_test += f"  {r.test_mae_pct:>6.3f}"
                if r.val_mae_pct < best_val:
                    best_val = r.val_mae_pct
                    best_rank = rank
            else:
                line_val += f"  {'N/A':>6}"
                line_test += f"  {'N/A':>6}"

        line_val += f"  r={best_rank}" if best_rank else ""
        print(line_val)
        print(line_test)

    print("=" * 80)

    # Overall best per symbol
    print("\nBest rank per symbol (by val MAE%):")
    for sym in symbols_seen:
        sym_results = [
            r for r in results
            if r.symbol == sym and r.val_mae_pct < float("inf")
        ]
        if sym_results:
            best = min(sym_results, key=lambda x: x.val_mae_pct)
            print(
                f"  {sym}: r={best.lora_r} (val={best.val_mae_pct:.4f}%, "
                f"test={best.test_mae_pct:.4f}%, {best.train_time_s:.0f}s)"
            )
        else:
            print(f"  {sym}: all failed")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA rank sweep for Chronos2")
    parser.add_argument(
        "--symbols", type=str, default=",".join(SYMBOLS),
        help="Comma-separated symbols to sweep",
    )
    parser.add_argument(
        "--ranks", type=str, default=",".join(str(r) for r in RANKS),
        help="Comma-separated LoRA ranks to test",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help="Directory for model checkpoints",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help="Directory for per-run JSON results",
    )
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV,
        help="Path for summary CSV output",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    ranks = [int(r.strip()) for r in args.ranks.split(",") if r.strip()]

    logger.info("LoRA rank sweep: {} symbols x {} ranks = {} configs",
                len(symbols), len(ranks), len(symbols) * len(ranks))
    logger.info("Symbols: {}", symbols)
    logger.info("Ranks: {}", ranks)

    run_sweep(symbols, ranks, args.output_root, args.results_dir, args.csv)


if __name__ == "__main__":
    main()
