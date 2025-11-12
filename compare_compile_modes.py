#!/usr/bin/env python3
"""
Benchmark compiled vs. uncompiled evaluation for previously-selected configs.

This utility loads the best Kronos/Toto configurations from hyperparams_extended
and re-evaluates them twice: once in eager mode and once with torch.compile
enabled. Results are written to hyperparams_compile_compare/<model>/<symbol>.json.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

import test_hyperparameters_extended as hyper

DATA_DIR = Path("trainingdata")
CONFIG_ROOT = Path("hyperparams_extended")
OUTPUT_ROOT = Path("hyperparams_compile_compare")


def _window_indices(df_len: int) -> Tuple[range, range]:
    """Return validation and test index ranges matching the main harness."""
    if df_len < hyper.VAL_WINDOW + hyper.TEST_WINDOW + hyper.MIN_CONTEXT:
        raise ValueError("Not enough rows to build val/test windows.")
    val_start = df_len - (hyper.TEST_WINDOW + hyper.VAL_WINDOW)
    val_indices = range(val_start, df_len - hyper.TEST_WINDOW)
    test_indices = range(df_len - hyper.TEST_WINDOW, df_len)
    return val_indices, test_indices


def _result_payload(result: hyper.EvaluationResult) -> Dict[str, float]:
    return {
        "price_mae": float(result.price_mae),
        "pct_return_mae": float(result.pct_return_mae),
        "latency_s": float(result.latency_s),
    }


def _load_config(symbol: str, model: str):
    cfg_path = CONFIG_ROOT / model / f"{symbol}.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config found at {cfg_path}")
    with cfg_path.open() as f:
        payload = json.load(f)
    cfg_dict = payload["config"]
    if model == "kronos":
        return cfg_path, hyper.KronosRunConfig(**cfg_dict)
    return cfg_path, hyper.TotoRunConfig(**cfg_dict)


def _reset_model_cache(model: str) -> None:
    """Reset cached wrappers/pipelines so compile toggles take effect."""
    if model == "kronos":
        hyper.KRONOS_WRAPPER_CACHE.clear()
    else:
        hyper._TOTO_PIPELINE = None
        hyper._TOTO_PIPELINE_SETTINGS = None


def _evaluate_once(
    *,
    model: str,
    df: pd.DataFrame,
    val_indices: Iterable[int],
    test_indices: Iterable[int],
    config,
) -> Dict[str, Dict[str, float]]:
    """Run sequential evaluation for both windows."""
    if model == "kronos":
        val_res = hyper._sequential_kronos(df, val_indices, config)
        test_res = hyper._sequential_kronos(df, test_indices, config)
    else:
        val_res = hyper._sequential_toto(df, val_indices, config)
        test_res = hyper._sequential_toto(df, test_indices, config)
    return {
        "validation": _result_payload(val_res),
        "test": _result_payload(test_res),
    }


def evaluate_symbol(symbol: str, *, models: Iterable[str], args) -> None:
    """Compare compile modes for the requested symbol."""
    csv_path = DATA_DIR / f"{symbol}.csv"
    if not csv_path.exists():
        print(f"[WARN] Missing data for {symbol} ({csv_path}); skipping.")
        return
    df = hyper._prepare_series(csv_path)
    val_indices, test_indices = _window_indices(len(df))

    for model in models:
        try:
            cfg_path, cfg = _load_config(symbol, model)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}")
            continue

        results: Dict[str, Dict[str, Dict[str, float]]] = {}
        for label, compiled in (("uncompiled", False), ("compiled", True)):
            if model == "kronos":
                hyper.ENABLE_KRONOS_COMPILE = compiled
                hyper.KRONOS_COMPILE_MODE = args.kronos_compile_mode
                kronos_backend = (args.kronos_compile_backend or "").lower()
                hyper.KRONOS_COMPILE_BACKEND = None if kronos_backend in ("", "none") else args.kronos_compile_backend
            else:
                hyper.ENABLE_TOTO_COMPILE = compiled
                hyper.ENABLE_TOTO_TORCH_COMPILE = compiled
                hyper.TOTO_COMPILE_MODE = args.toto_compile_mode
                toto_backend = (args.toto_compile_backend or "").lower()
                hyper.TOTO_COMPILE_BACKEND = None if toto_backend in ("", "none") else args.toto_compile_backend

            _reset_model_cache(model)
            print(f"[INFO] {symbol}:{model} -> {label} (compile={compiled})")
            try:
                results[label] = _evaluate_once(
                    model=model,
                    df=df,
                    val_indices=val_indices,
                    test_indices=test_indices,
                    config=cfg,
                )
            except Exception as exc:  # pragma: no cover - diagnostic path
                message = str(exc).strip() or repr(exc)
                print(f"[WARN] {symbol}:{model} {label} failed: {message}")
                results[label] = {"error": message}

        model_dir = OUTPUT_ROOT / model
        model_dir.mkdir(parents=True, exist_ok=True)
        output_path = model_dir / f"{symbol}.json"
        payload = {
            "symbol": symbol,
            "model": model,
            "config": asdict(cfg),
            "config_source": str(cfg_path),
            "compile_settings": {
                "kronos": {
                    "mode": hyper.KRONOS_COMPILE_MODE,
                    "backend": hyper.KRONOS_COMPILE_BACKEND or "none",
                },
                "toto": {
                    "mode": hyper.TOTO_COMPILE_MODE,
                    "backend": hyper.TOTO_COMPILE_BACKEND or "none",
                },
            },
            "results": results,
        }
        with output_path.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"[INFO] Saved comparison -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Symbols to evaluate (default: infer from hyperparams_extended directories).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=("kronos", "toto"),
        default=("kronos", "toto"),
        help="Subset of models to evaluate (default: both).",
    )
    parser.add_argument(
        "--kronos-compile-mode",
        default="max-autotune",
        help="torch.compile mode for Kronos (default: max-autotune).",
    )
    parser.add_argument(
        "--kronos-compile-backend",
        default="inductor",
        help="torch.compile backend for Kronos (use 'none' to disable).",
    )
    parser.add_argument(
        "--toto-compile-mode",
        default="max-autotune",
        help="torch.compile mode for Toto (default: max-autotune).",
    )
    parser.add_argument(
        "--toto-compile-backend",
        default="inductor",
        help="torch.compile backend for Toto (use 'none' to disable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.symbols:
        symbols = args.symbols
    else:
        kronos_symbols = {p.stem for p in (CONFIG_ROOT / "kronos").glob("*.json")}
        toto_symbols = {p.stem for p in (CONFIG_ROOT / "toto").glob("*.json")}
        symbols = sorted(kronos_symbols | toto_symbols)
    for symbol in symbols:
        evaluate_symbol(symbol, models=args.models, args=args)


if __name__ == "__main__":
    main()
