from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import build_dataset
from .model import train_and_optimize, MODELS_DIR
from .backtest import run_backtest


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m boostbaseline.run_baseline <SYMBOL> [crypto:true|false]")
        sys.exit(1)
    symbol = sys.argv[1].upper()
    is_crypto = True
    if len(sys.argv) >= 3:
        is_crypto = sys.argv[2].lower() in ("1", "true", "yes")

    print(f"[boostbaseline] Building dataset for {symbol} (is_crypto={is_crypto})…")
    df = build_dataset(symbol, is_crypto=is_crypto)
    if df.empty:
        print("No dataset rows found. Ensure results/predictions-*.csv exist for this symbol and trainingdata CSV is present.")
        sys.exit(2)

    print(f"[boostbaseline] Dataset size: {len(df)} rows")
    model = train_and_optimize(df, is_crypto=is_crypto, fee=0.0023 if is_crypto else 0.0002)

    # Evaluate on the tail split used during training for quick reporting
    split = max(10, int(len(df) * 0.8))
    X_cols = model.feature_cols
    X_te = df[X_cols].astype(float).iloc[split:]
    y_te = df['y'].astype(float).iloc[split:]

    y_pred = model.predict(X_te)
    bt = run_backtest(y_true=y_te.values, y_pred=y_pred, is_crypto=is_crypto, fee=0.0023 if is_crypto else 0.0002, scale=model.scale, cap=model.cap)

    model.save(symbol)

    # Report
    total_return_pct = bt.total_return * 100.0
    sharpe = bt.sharpe
    cap = model.cap
    scale = model.scale

    summary = [
        f"BoostBaseline summary for {symbol}",
        f"Rows: {len(df)} | Test: {len(X_te)}",
        f"Model: {model.model_name} | Features: {len(X_cols)}",
        f"Sizing: scale={scale:.2f}, cap={cap:.2f}, is_crypto={is_crypto}",
        f"Backtest: total_return={total_return_pct:.2f}% | sharpe={sharpe:.3f}",
        f"Saved model → {MODELS_DIR / (symbol + '_boost.model')}",
    ]
    print("\n".join("[boostbaseline] " + s for s in summary))

    # Append to baselineperf.md for convenience
    try:
        with open("baselineperf.md", "a") as f:
            f.write("\n\n" + "\n".join(summary))
    except Exception:
        pass


if __name__ == "__main__":
    main()

