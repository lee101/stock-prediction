from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib  # type: ignore
import numpy as np
import pandas as pd

from .dataset import build_dataset, iter_prediction_rows
from .model import MODELS_DIR


def load_trained(symbol: str):
    meta_path = MODELS_DIR / f"{symbol}_boost.model"
    if not meta_path.exists():
        raise FileNotFoundError(f"Model not found: {meta_path}. Train first with boostbaseline.run_baseline.")
    meta = json.load(open(meta_path))
    feature_cols = meta['feature_cols']
    is_xgb = meta['is_xgb']
    scale = float(meta['scale'])
    cap = float(meta['cap'])
    mean = np.array(meta['scaler_mean']) if meta['scaler_mean'] is not None else None
    std = np.array(meta['scaler_std']) if meta['scaler_std'] is not None else None

    if is_xgb:
        import xgboost as xgb  # type: ignore
        model = xgb.Booster()
        model.load_model(str(MODELS_DIR / meta['xgb_json']))
        loader = ('xgb', model)
    else:
        model = joblib.load(str(MODELS_DIR / meta['sk_joblib']))
        loader = ('sk', model)
    return {
        'feature_cols': feature_cols,
        'is_xgb': is_xgb,
        'scale': scale,
        'cap': cap,
        'mean': mean,
        'std': std,
        'model': loader,
    }


def latest_feature_row(symbol: str) -> pd.DataFrame:
    # Build single-row feature frame from the latest snapshot
    rows = list(iter_prediction_rows(symbol))
    if not rows:
        raise RuntimeError(f"No cached prediction rows found in results/ for {symbol}")
    snap_time, s = rows[-1]
    from .dataset import _coerce_float
    close_now = _coerce_float(s.get('close_last_price'))
    close_pred_val = _coerce_float(s.get('close_predicted_price_value'))
    close_pred_raw = _coerce_float(s.get('close_predicted_price'))
    high_pred_val = _coerce_float(s.get('high_predicted_price_value'))
    low_pred_val = _coerce_float(s.get('low_predicted_price_value'))
    close_val_loss = _coerce_float(s.get('close_val_loss'))
    high_val_loss = _coerce_float(s.get('high_val_loss'))
    low_val_loss = _coerce_float(s.get('low_val_loss'))
    takeprofit_profit = _coerce_float(s.get('takeprofit_profit'))
    entry_takeprofit_profit = _coerce_float(s.get('entry_takeprofit_profit'))
    maxdiffprofit_profit = _coerce_float(s.get('maxdiffprofit_profit'))

    if close_now is None:
        raise RuntimeError("close_last_price missing in latest snapshot")
    if close_pred_val is not None:
        pred_close_delta = (close_pred_val - close_now) / close_now
    elif close_pred_raw is not None and abs(close_pred_raw) < 0.2:
        pred_close_delta = close_pred_raw
    else:
        pred_close_delta = 0.0

    feats = {
        'feature_pred_close_delta': pred_close_delta,
        'feature_pred_high_delta': (high_pred_val - close_now) / close_now if high_pred_val is not None else 0.0,
        'feature_pred_low_delta': (close_now - low_pred_val) / close_now if low_pred_val is not None else 0.0,
        'feature_close_val_loss': 0.0 if close_val_loss is None else close_val_loss,
        'feature_high_val_loss': 0.0 if high_val_loss is None else high_val_loss,
        'feature_low_val_loss': 0.0 if low_val_loss is None else low_val_loss,
        'feature_takeprofit_profit': 0.0 if takeprofit_profit is None else takeprofit_profit,
        'feature_entry_takeprofit_profit': 0.0 if entry_takeprofit_profit is None else entry_takeprofit_profit,
        'feature_maxdiffprofit_profit': 0.0 if maxdiffprofit_profit is None else maxdiffprofit_profit,
    }
    return pd.DataFrame([feats])


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m boostbaseline.recommend <SYMBOL> [crypto:true|false]")
        sys.exit(1)
    symbol = sys.argv[1].upper()
    is_crypto = True
    if len(sys.argv) >= 3:
        is_crypto = sys.argv[2].lower() in ("1", "true", "yes")
    meta = load_trained(symbol)
    feat_df = latest_feature_row(symbol)
    # Align feature columns
    missing = [c for c in meta['feature_cols'] if c not in feat_df.columns]
    for c in missing:
        feat_df[c] = 0.0
    feat_df = feat_df[meta['feature_cols']]

    Xv = feat_df.values
    if meta['mean'] is not None and meta['std'] is not None:
        Xv = (Xv - meta['mean']) / np.maximum(meta['std'], 1e-8)

    kind, model = meta['model']
    if kind == 'xgb':
        import xgboost as xgb  # type: ignore
        y_pred = model.predict(xgb.DMatrix(Xv))
    else:
        y_pred = model.predict(Xv)

    # Suggested position size (apply scaling/cap and crypto short rules)
    pos = float(np.clip(meta['scale'] * y_pred[0], -meta['cap'], meta['cap']))
    if is_crypto:
        pos = float(np.clip(pos, 0.0, meta['cap']))

    print(f"[boostbaseline] Suggested position fraction for {symbol}: {pos:+.4f} (cap={meta['cap']}, scale={meta['scale']})")


if __name__ == "__main__":
    main()

