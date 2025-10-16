from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


RESULTS_DIR = Path('results')
TRAINING_DIR = Path('trainingdata/train')


_PRED_FILE_RE = re.compile(r"predictions-(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\.csv$")


def _parse_snapshot_time_from_filename(path: Path) -> Optional[pd.Timestamp]:
    m = _PRED_FILE_RE.search(path.name)
    if not m:
        return None
    date_part, time_part = m.groups()
    # naive UTC
    try:
        return pd.Timestamp(f"{date_part} {time_part.replace('-', ':')}", tz='UTC')
    except Exception:
        return None


def _coerce_float(val) -> Optional[float]:
    if pd.isna(val):
        return None
    # handle strings like "(119.93,)"
    if isinstance(val, str):
        s = val.strip()
        if s.startswith('(') and s.endswith(')'):
            s = s.strip('()').rstrip(',').strip()
        try:
            return float(s)
        except Exception:
            return None
    try:
        return float(val)
    except Exception:
        return None


def load_price_series(symbol: str) -> pd.DataFrame:
    """Load OHLCV for symbol from trainingdata. Tries various filename conventions.

    Returns DataFrame indexed by UTC timestamp, with columns including 'Close'.
    """
    candidates = [
        TRAINING_DIR / f"{symbol}.csv",
        TRAINING_DIR / f"{symbol.replace('-', '')}.csv",
        TRAINING_DIR / f"{symbol.replace('/', '')}.csv",
        TRAINING_DIR / f"{symbol.replace('-', '_')}.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No training CSV found for {symbol} under {TRAINING_DIR}")

    df = pd.read_csv(path)
    # Flexible timestamp column handling
    ts_col = 'timestamp' if 'timestamp' in df.columns else 'Date' if 'Date' in df.columns else None
    if ts_col is None:
        # some files have first col name like 'Unnamed: 0' or index; try the second column
        ts_col = df.columns[1]
    ts = pd.to_datetime(df[ts_col], utc=True, errors='coerce')
    df = df.assign(timestamp=ts).dropna(subset=['timestamp']).set_index('timestamp').sort_index()
    return df


def iter_prediction_rows(symbol: str) -> Iterable[Tuple[pd.Timestamp, pd.Series]]:
    """Yield (snapshot_time, row) for each results/predictions-*.csv containing symbol.

    The row contains the parsed numeric fields for the symbol.
    """
    if not RESULTS_DIR.exists():
        return []
    files = sorted(RESULTS_DIR.glob('predictions-*.csv'))
    for path in files:
        snap_time = _parse_snapshot_time_from_filename(path)
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if 'instrument' not in df.columns:
            continue
        row = df.loc[df['instrument'] == symbol]
        if row.empty:
            continue
        s = row.iloc[0].copy()
        s['__snapshot_time__'] = snap_time
        yield snap_time, s


def build_dataset(symbol: str, is_crypto: bool = True) -> pd.DataFrame:
    """Build dataset with features X and next-day return y.

    Columns:
    - feature_*: engineered features from prediction row
    - y: realized next-day close-to-close return
    - snapshot_time: prediction snapshot time
    - price_time: aligned price timestamp used for y calculation
    """
    price = load_price_series(symbol)
    out_rows: List[dict] = []

    for snap_time, row in iter_prediction_rows(symbol):
        if snap_time is None:
            continue
        # Align to last price timestamp <= snapshot
        price_up_to = price[price.index <= snap_time]
        if price_up_to.empty:
            continue
        current_idx = price_up_to.index[-1]
        try:
            next_idx_pos = price.index.get_loc(current_idx) + 1
        except KeyError:
            # if index not found directly (shouldn't happen), skip
            continue
        if next_idx_pos >= len(price.index):
            continue  # no future point
        next_idx = price.index[next_idx_pos]

        close_now = float(price.loc[current_idx, 'Close'])
        close_next = float(price.loc[next_idx, 'Close'])
        y = (close_next - close_now) / close_now

        # Extract features robustly
        close_pred_val = _coerce_float(row.get('close_predicted_price_value'))
        high_pred_val = _coerce_float(row.get('high_predicted_price_value'))
        low_pred_val = _coerce_float(row.get('low_predicted_price_value'))
        close_val_loss = _coerce_float(row.get('close_val_loss'))
        high_val_loss = _coerce_float(row.get('high_val_loss'))
        low_val_loss = _coerce_float(row.get('low_val_loss'))

        # Some files have 'close_predicted_price' as delta; detect if value looks small (~-0.01..0.01)
        close_pred_raw = _coerce_float(row.get('close_predicted_price'))

        # Compute deltas
        if close_pred_val is not None:
            pred_close_delta = (close_pred_val - close_now) / close_now
        elif close_pred_raw is not None and abs(close_pred_raw) < 0.2:
            pred_close_delta = close_pred_raw  # already a fraction
        else:
            pred_close_delta = None

        pred_high_delta = (high_pred_val - close_now) / close_now if high_pred_val is not None else None
        pred_low_delta = (close_now - low_pred_val) / close_now if low_pred_val is not None else None

        # Profit metrics (optional)
        takeprofit_profit = _coerce_float(row.get('takeprofit_profit'))
        entry_takeprofit_profit = _coerce_float(row.get('entry_takeprofit_profit'))
        maxdiffprofit_profit = _coerce_float(row.get('maxdiffprofit_profit'))

        feat = {
            'feature_pred_close_delta': pred_close_delta,
            'feature_pred_high_delta': pred_high_delta,
            'feature_pred_low_delta': pred_low_delta,
            'feature_close_val_loss': close_val_loss,
            'feature_high_val_loss': high_val_loss,
            'feature_low_val_loss': low_val_loss,
            'feature_takeprofit_profit': takeprofit_profit,
            'feature_entry_takeprofit_profit': entry_takeprofit_profit,
            'feature_maxdiffprofit_profit': maxdiffprofit_profit,
        }

        # Drop if no core features
        if feat['feature_pred_close_delta'] is None and (
            feat['feature_pred_high_delta'] is None or feat['feature_pred_low_delta'] is None
        ):
            continue

        # Replace None with NaN for ML
        for k, v in list(feat.items()):
            feat[k] = np.nan if v is None else float(v)

        out_rows.append({
            **feat,
            'y': float(y),
            'snapshot_time': snap_time,
            'price_time': current_idx,
            'close_now': close_now,
            'close_next': close_next,
        })

    df = pd.DataFrame(out_rows).sort_values('price_time')
    # Basic NA handling: fill validation losses/profits with zeros, keep deltas with median
    if not df.empty:
        for col in df.columns:
            if col.startswith('feature_'):
                if 'delta' in col:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(0.0)
    return df

