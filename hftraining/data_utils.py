#!/usr/bin/env python3
"""
Data utilities for HuggingFace-style training
"""

import ast

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Sequence, Union, Set
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from hfshared import compute_training_style_features, training_feature_columns_list
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    from .toto_features import (
        TotoFeatureGenerator,
        TotoOptions,
        append_toto_columns,
    )
except ImportError:  # Allow running as a top-level script
    from toto_features import (  # type: ignore
        TotoFeatureGenerator,
        TotoOptions,
        append_toto_columns,
    )

try:
    from .asset_metadata import get_asset_class_id, get_trading_fee
except ImportError:  # pragma: no cover - script execution
    from asset_metadata import get_asset_class_id, get_trading_fee  # type: ignore


def _parse_numeric_scalar(value: Any) -> Optional[float]:
    """Best-effort conversion of Toto CSV cell contents into a scalar float."""
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return None
        return float(value)

    if isinstance(value, str):
        candidate = value.strip()
        if not candidate or candidate.lower() in {"na", "nan", "none"}:
            return None

        if candidate.startswith("tensor(") and candidate.endswith(")"):
            candidate = candidate[len("tensor("):-1].strip()
        if candidate.startswith("array(") and candidate.endswith(")"):
            candidate = candidate[len("array("):-1].strip()

        try:
            parsed = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            try:
                return float(candidate)
            except ValueError:
                tokens = [tok.strip() for tok in candidate.strip("[]").split(",") if tok.strip()]
                if not tokens:
                    return None
                try:
                    return float(tokens[-1])
                except ValueError:
                    return None

        if isinstance(parsed, (int, float, np.integer, np.floating)):
            return float(parsed)
        if isinstance(parsed, (list, tuple)):
            for item in reversed(parsed):
                scalar = _parse_numeric_scalar(item)
                if scalar is not None:
                    return scalar
            return None
        if isinstance(parsed, dict):
            return None
        try:
            return float(parsed)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    return None


def load_toto_prediction_history(
    predictions_dir: Union[str, Path]
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Load historical Toto strategy prediction CSVs and convert them into per-symbol feature frames.

    The loader focuses on rows whose `instrument` column embeds a timestamp
    (e.g., ``AAPL-2024-10-03 07:10:00``). Columns containing list- or tensor-type
    payloads (``*_values``, ``*_trade_values``, ``*_predictions``) are skipped to keep
    a consistent tabular feature shape.
    """
    base = Path(predictions_dir).expanduser()
    if not base.exists():
        raise FileNotFoundError(f"Toto prediction directory '{base}' does not exist")

    feature_records: Dict[str, List[Dict[str, Any]]] = {}
    feature_columns: Set[str] = set()
    banned_tokens = ("values", "trade_values", "predictions")

    for csv_path in sorted(base.glob("*.csv")):
        if not csv_path.is_file():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if "instrument" not in df.columns:
            continue

        for idx, raw_inst in df["instrument"].dropna().items():
            instrument = str(raw_inst)
            if "-" not in instrument:
                continue
            symbol_part, ts_part = instrument.split("-", 1)
            try:
                timestamp = pd.to_datetime(ts_part)
            except Exception:
                continue

            row = df.loc[idx]
            symbol = symbol_part.strip().upper()
            features: Dict[str, Any] = {}

            for col, raw_value in row.items():
                if col == "instrument":
                    continue
                lower = col.lower()
                if any(token in lower for token in banned_tokens) or lower == "generated_at":
                    continue
                scalar = _parse_numeric_scalar(raw_value)
                if scalar is None:
                    continue
                features[col] = float(scalar)

            if not features:
                continue

            features["prediction_time"] = timestamp.tz_localize(None) if isinstance(timestamp, pd.Timestamp) else timestamp
            features["source_file"] = csv_path.name

            feature_records.setdefault(symbol, []).append(features)
            feature_columns.update(features.keys() - {"prediction_time", "source_file"})

    ordered_feature_columns = sorted(feature_columns)
    prefixed_columns = [
        col if col.startswith("toto_pred_") else f"toto_pred_{col}"
        for col in ordered_feature_columns
    ]

    symbol_frames: Dict[str, pd.DataFrame] = {}
    for symbol, rows in feature_records.items():
        if not rows:
            continue
        sym_df = pd.DataFrame(rows)
        if sym_df.empty:
            continue
        sym_df["prediction_time"] = pd.to_datetime(sym_df["prediction_time"])
        sym_df["prediction_date"] = sym_df["prediction_time"].dt.normalize()
        sym_df.sort_values("prediction_time", inplace=True)
        sym_df = sym_df.drop_duplicates("prediction_date", keep="last")

        for col in ordered_feature_columns:
            if col not in sym_df.columns:
                sym_df[col] = np.nan

        keep_cols = ["prediction_date"] + ordered_feature_columns
        sym_df = sym_df[keep_cols]
        rename_map = {
            col: col if col.startswith("toto_pred_") else f"toto_pred_{col}"
            for col in ordered_feature_columns
        }
        sym_df.rename(columns=rename_map, inplace=True)
        sym_df.set_index("prediction_date", inplace=True)
        sym_df = sym_df.reindex(columns=prefixed_columns)
        symbol_frames[symbol] = sym_df.sort_index()

    return symbol_frames, prefixed_columns


class StockDataProcessor:
    """Advanced stock data processor with multiple features."""

    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizon: int = 5,
        features: Optional[List[str]] = None,
        use_toto_forecasts: bool = False,
        toto_options: Optional[TotoOptions] = None,
        toto_prediction_features: Optional[Dict[str, pd.DataFrame]] = None,
        toto_prediction_columns: Optional[Sequence[str]] = None,
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.features = features or ['open', 'high', 'low', 'close', 'volume']
        self.scalers: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.use_toto_forecasts = use_toto_forecasts
        self._toto_generator: Optional[TotoFeatureGenerator] = None
        self._toto_prediction_features: Dict[str, pd.DataFrame] = {
            symbol.upper(): df.copy()
            for symbol, df in (toto_prediction_features or {}).items()
        }
        self._toto_prediction_columns: List[str] = list(toto_prediction_columns or [])
        self._toto_availability_column = "toto_pred_available"

        if self.use_toto_forecasts:
            options = toto_options or TotoOptions(
                horizon=prediction_horizon,
                context_length=sequence_length,
            )
            self._toto_generator = TotoFeatureGenerator(options)

    def prepare_features(self, df: pd.DataFrame, symbol: Optional[str] = None) -> np.ndarray:
        """Prepare and select features for training."""
        normalized_df = df.copy()
        normalized_df.columns = normalized_df.columns.str.lower()

        feats_df = compute_training_style_features(normalized_df)
        ordered = [c for c in training_feature_columns_list() if c in feats_df.columns]
        feats_df = feats_df[ordered]

        if self.use_toto_forecasts and self._toto_generator is not None:
            price_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in price_columns if col not in normalized_df.columns]
            if missing_cols:
                for col in missing_cols:
                    if col == 'volume':
                        normalized_df[col] = 1.0
                    else:
                        raise ValueError(
                            f"Missing required columns for Toto forecasts: {missing_cols}"
                        )
            price_matrix = normalized_df[price_columns].to_numpy(dtype=np.float32)
            prefix = symbol.lower() if symbol else "toto"
            toto_feats, toto_names = self._toto_generator.compute_features(
                price_matrix,
                price_columns,
                symbol_prefix=prefix,
            )
            append_toto_columns(
                feats_df,
                toto_feats,
                column_names=toto_names,
            )

            # Add residual features against first-step Toto mean for close price
            mean_col = f"{prefix}_close_toto_mean_t+1"
            if mean_col in feats_df.columns:
                close_series = normalized_df['close'].to_numpy(dtype=np.float32)
                residual = close_series - feats_df[mean_col].to_numpy(dtype=np.float32)
                feats_df[f"{prefix}_close_toto_residual"] = residual

        if self._toto_prediction_features and symbol:
            feats_df = self._append_toto_prediction_features(feats_df, normalized_df, symbol)

        self.feature_names = list(feats_df.columns)
        return feats_df.values

    def _append_toto_prediction_features(
        self,
        feats_df: pd.DataFrame,
        normalized_df: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """Append precomputed Toto prediction summaries as additional features."""
        symbol_key = symbol.upper()
        pred_frame = self._toto_prediction_features.get(symbol_key)

        target_columns: List[str] = list(self._toto_prediction_columns)
        if not target_columns:
            sample_frame = next(
                (df for df in self._toto_prediction_features.values() if df is not None and not df.empty),
                None,
            )
            if sample_frame is not None:
                target_columns = list(sample_frame.columns)
                self._toto_prediction_columns = list(target_columns)

        def build_zero_frame() -> pd.DataFrame:
            if target_columns:
                zero = pd.DataFrame(
                    np.zeros((len(feats_df), len(target_columns)), dtype=np.float32),
                    columns=target_columns,
                )
            else:
                zero = pd.DataFrame(index=pd.RangeIndex(len(feats_df)), dtype=np.float32)
            zero[self._toto_availability_column] = np.zeros(len(feats_df), dtype=np.float32)
            return zero.astype(np.float32, copy=False)

        date_series: Optional[pd.Series] = None
        for candidate in ("date", "timestamp"):
            if candidate in normalized_df.columns:
                date_series = pd.to_datetime(normalized_df[candidate]).dt.normalize()
                break
        if date_series is None:
            aligned = build_zero_frame()
        elif pred_frame is None or pred_frame.empty:
            aligned = build_zero_frame()
        else:
            date_index = pd.DatetimeIndex(date_series)
            aligned = pred_frame.reindex(date_index)
            if target_columns:
                aligned = aligned.reindex(target_columns, axis=1)

            availability = (~aligned.isna()).any(axis=1).astype(np.float32)
            aligned = aligned.fillna(0.0).astype(np.float32, copy=False)
            aligned[self._toto_availability_column] = availability.values

        if self._toto_prediction_columns and target_columns and aligned.shape[1] - int(self._toto_availability_column in aligned.columns) != len(self._toto_prediction_columns):
            # Ensure column order matches the declared prediction column list when possible
            aligned = aligned.reindex(columns=[*self._toto_prediction_columns, self._toto_availability_column], fill_value=0.0)
        aligned = aligned.astype(np.float32, copy=False)

        combined = pd.concat(
            [feats_df.reset_index(drop=True), aligned.reset_index(drop=True)],
            axis=1,
        )
        return combined
    
    def fit_scalers(self, data):
        """Fit scalers on training data"""
        
        # Standard scaler for most features
        self.scalers['standard'] = StandardScaler()
        
        # MinMax scaler for bounded features (like RSI)
        self.scalers['minmax'] = MinMaxScaler()
        
        # Fit standard scaler on all features
        self.scalers['standard'].fit(data)
        
        return self
    
    def transform(self, data):
        """Transform data using fitted scalers"""
        if 'standard' not in self.scalers:
            raise ValueError("Scalers not fitted. Call fit_scalers first.")
        
        return self.scalers['standard'].transform(data)
    
    def inverse_transform(self, data):
        """Inverse transform data"""
        return self.scalers['standard'].inverse_transform(data)
    
    def save_scalers(self, path):
        """Save scalers to disk"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon
        }, path)
    
    def load_scalers(self, path):
        """Load scalers from disk"""
        data = joblib.load(path)
        self.scalers = data['scalers']
        self.feature_names = data['feature_names']
        self.sequence_length = data['sequence_length']
        self.prediction_horizon = data['prediction_horizon']
        return self


def load_local_stock_data(symbols: List[str], data_dir: str = "trainingdata") -> Dict[str, pd.DataFrame]:
    """Load per-symbol CSVs from trainingdata directory.

    Looks for files like SYMBOL.csv (case-insensitive). Returns a dict of dataframes
    with standardized lowercase columns and a 'date' column if present.
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    data: Dict[str, pd.DataFrame] = {}
    base = Path(data_dir)
    fallbacks = [Path("data"), Path("hftraining")/"data"/"raw"]
    for sym in symbols:
        # Try primary dir
        candidates = list(base.glob(f"{sym}.csv"))
        if not candidates:
            # Try case-insensitive / contains match
            candidates = [p for p in base.glob("*.csv") if sym.lower() in p.stem.lower()]
        # Try fallbacks
        if not candidates:
            for fb in fallbacks:
                candidates = list(fb.glob(f"{sym}.csv"))
                if not candidates:
                    candidates = [p for p in fb.glob("*.csv") if sym.lower() in p.stem.lower()]
                if candidates:
                    break
        if not candidates:
            print(f"Warning: no CSV found for symbol {sym} under {base} or fallbacks {fallbacks}")
            continue
        try:
            df = pd.read_csv(candidates[0])
            df.columns = df.columns.str.lower()
            if 'date' in df.columns:
                try:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                except Exception:
                    pass
            elif 'timestamp' in df.columns:
                try:
                    df['date'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('date')
                except Exception:
                    pass
            data[sym] = df
        except Exception as e:
            print(f"Error loading {sym} from {candidates[0]}: {e}")
    return data


def create_sequences(data, sequence_length, prediction_horizon, target_column='close'):
    """
    Create sequences for time series prediction
    
    Args:
        data: Input data array
        sequence_length: Length of input sequences
        prediction_horizon: Number of steps to predict
        target_column: Index of target column (default: 3 for close price)
        
    Returns:
        Tuple of (sequences, targets, action_labels)
    """
    
    if len(data) < sequence_length + prediction_horizon:
        raise ValueError(f"Data too short: {len(data)} < {sequence_length + prediction_horizon}")
    
    sequences = []
    targets = []
    action_labels = []
    
    for i in range(len(data) - sequence_length - prediction_horizon + 1):
        # Input sequence
        seq = data[i:i + sequence_length]
        sequences.append(seq)
        
        # Target sequence (future prices)
        target_start = i + sequence_length
        target_end = target_start + prediction_horizon
        target = data[target_start:target_end]
        targets.append(target)
        
        # Action label (buy/hold/sell based on next price movement)
        current_price = data[i + sequence_length - 1, 3]  # Last close price in sequence
        next_price = data[i + sequence_length, 3]  # Next close price
        
        price_change = (next_price - current_price) / current_price
        
        if price_change > 0.01:  # 1% threshold
            action_label = 0  # Buy
        elif price_change < -0.01:
            action_label = 2  # Sell
        else:
            action_label = 1  # Hold
            
        action_labels.append(action_label)
    
    return np.array(sequences), np.array(targets), np.array(action_labels)


def align_on_timestamp(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align two price dataframes on their timestamp or date column.

    The function performs an inner join to retain only overlapping dates and
    returns aligned copies sorted chronologically.
    """
    key = None
    for col in ['date', 'timestamp', 'Datetime', 'datetime']:
        if col in df_a.columns and col in df_b.columns:
            key = col
            break
    if key is None:
        raise ValueError("Unable to align stock data – no common timestamp column found.")

    df_a_ = df_a.copy()
    df_b_ = df_b.copy()

    # Ensure datetime dtype for alignment column.
    from pandas.api.types import is_datetime64_any_dtype

    if not is_datetime64_any_dtype(df_a_[key]):
        df_a_[key] = pd.to_datetime(df_a_[key])
    if not is_datetime64_any_dtype(df_b_[key]):
        df_b_[key] = pd.to_datetime(df_b_[key])

    def _try_merge(left: pd.DataFrame, right: pd.DataFrame, on: str) -> pd.DataFrame:
        return pd.merge(
            left,
            right,
            on=on,
            suffixes=('_a', '_b'),
            how='inner',
        ).sort_values(on)

    merged = _try_merge(df_a_, df_b_, key)

    if merged.empty:
        for freq in ['T', 'H']:
            align_col = f"__align_{freq}"
            df_a_[align_col] = df_a_[key].dt.floor(freq)
            df_b_[align_col] = df_b_[key].dt.floor(freq)
            merged = _try_merge(df_a_.drop(columns=[key]), df_b_.drop(columns=[key]), align_col)
            if not merged.empty:
                key = align_col
                break

    if merged.empty:
        min_len = min(len(df_a_), len(df_b_))
        df_a_trim = df_a_.iloc[:min_len].reset_index(drop=True)
        df_b_trim = df_b_.iloc[:min_len].reset_index(drop=True)
        return df_a_trim, df_b_trim

    cols_a = [c for c in merged.columns if c.endswith('_a')]
    cols_b = [c for c in merged.columns if c.endswith('_b')]

    df_a_aligned = merged[[key] + cols_a].copy()
    df_b_aligned = merged[[key] + cols_b].copy()

    def _strip_suffix(name: str, suffix: str) -> str:
        return name[:-len(suffix)] if name.endswith(suffix) else name

    df_a_aligned.columns = [_strip_suffix(c, '_a') for c in df_a_aligned.columns]
    df_b_aligned.columns = [_strip_suffix(c, '_b') for c in df_b_aligned.columns]

    effective_key = key
    if key.startswith("__align_"):
        effective_key = "date"
        df_a_aligned = df_a_aligned.rename(columns={key: effective_key})
        df_b_aligned = df_b_aligned.rename(columns={key: effective_key})

    if is_datetime64_any_dtype(df_a_aligned[effective_key]):
        try:
            df_a_aligned[effective_key] = df_a_aligned[effective_key].dt.tz_localize(None)
            df_b_aligned[effective_key] = df_b_aligned[effective_key].dt.tz_localize(None)
        except AttributeError:
            # Column is already tz-naive
            pass

    return df_a_aligned.reset_index(drop=True), df_b_aligned.reset_index(drop=True)


class PairStockDataset(torch.utils.data.Dataset):
    """
    Dataset that yields joint sequences for a pair of stocks so the model can
    learn cross-asset signals as well as allocation targets.
    """

    def __init__(
        self,
        stock_a: np.ndarray,
        stock_b: np.ndarray,
        sequence_length: int,
        prediction_horizon: int,
        name_a: str,
        name_b: str,
        raw_close_a: Optional[np.ndarray] = None,
        raw_close_b: Optional[np.ndarray] = None,
        close_feature_index: int = 3,
        epsilon: float = 1e-8,
    ):
        if len(stock_a) != len(stock_b):
            raise ValueError("Aligned stock arrays must share the same length.")
        if len(stock_a) < sequence_length + prediction_horizon:
            raise ValueError(
                f"Pair dataset too small: {len(stock_a)} < "
                f"{sequence_length + prediction_horizon}"
            )

        if raw_close_a is None or raw_close_b is None:
            raise ValueError(
                "Raw close price arrays are required to compute meaningful returns."
            )
        if len(raw_close_a) != len(stock_a) or len(raw_close_b) != len(stock_b):
            raise ValueError("Raw close price arrays must align with feature arrays.")
        if close_feature_index >= stock_a.shape[1] or close_feature_index >= stock_b.shape[1]:
            raise ValueError("close_feature_index is out of bounds for the provided features.")

        self.stock_a = torch.as_tensor(stock_a, dtype=torch.float32)
        self.stock_b = torch.as_tensor(stock_b, dtype=torch.float32)
        self.close_a = torch.as_tensor(raw_close_a, dtype=torch.float32)
        self.close_b = torch.as_tensor(raw_close_b, dtype=torch.float32)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.name_a = name_a
        self.name_b = name_b
        self.close_feature_index = close_feature_index
        self.epsilon = epsilon
        self.asset_class_ids = torch.tensor(
            [get_asset_class_id(name_a), get_asset_class_id(name_b)],
            dtype=torch.long,
        )
        self.per_asset_fees = torch.tensor(
            [float(get_trading_fee(name_a)), float(get_trading_fee(name_b))],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return self.stock_a.shape[0] - self.sequence_length - self.prediction_horizon + 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sl = self.sequence_length
        ph = self.prediction_horizon

        seq_a = self.stock_a[idx : idx + sl]
        seq_b = self.stock_b[idx : idx + sl]

        target_a = self.stock_a[idx + sl : idx + sl + ph]
        target_b = self.stock_b[idx + sl : idx + sl + ph]

        inputs = torch.cat([seq_a, seq_b], dim=1).contiguous()
        price_targets = torch.stack(
            [
                target_a[:, self.close_feature_index],
                target_b[:, self.close_feature_index],
            ],
            dim=0,
        )

        current_close = torch.stack([
            self.close_a[idx + sl - 1],
            self.close_b[idx + sl - 1],
        ])
        next_close = torch.stack([
            self.close_a[idx + sl],
            self.close_b[idx + sl],
        ])
        returns = (next_close - current_close) / (current_close + self.epsilon)

        action_labels = torch.ones_like(returns, dtype=torch.long)
        action_labels = torch.where(
            returns > 0.01,
            torch.zeros_like(action_labels),
            action_labels,
        )
        action_labels = torch.where(
            returns < -0.01,
            torch.full_like(action_labels, 2, dtype=torch.long),
            action_labels,
        )

        return {
            'input_ids': inputs,
            'labels': price_targets,
            'future_returns': returns,
            'action_labels': action_labels,
            'attention_mask': torch.ones(self.sequence_length, dtype=torch.float32),
            'asset_class_ids': self.asset_class_ids.clone(),
            'per_asset_fees': self.per_asset_fees.clone(),
        }


class MultiAssetPortfolioDataset(torch.utils.data.Dataset):
    """Dataset that yields aligned sequences for multiple assets."""

    def __init__(
        self,
        asset_arrays: List[np.ndarray],
        asset_names: List[str],
        asset_close_prices: List[np.ndarray],
        sequence_length: int,
        prediction_horizon: int,
        close_feature_index: int = 3,
        epsilon: float = 1e-8,
    ):
        if len(asset_arrays) != len(asset_names):
            raise ValueError("Asset arrays and names must be same length")
        if len(asset_arrays) != len(asset_close_prices):
            raise ValueError("Asset feature arrays and close price arrays must align")
        base_len = len(asset_arrays[0])
        for idx, arr in enumerate(asset_arrays):
            if len(arr) != base_len:
                raise ValueError("All assets must share identical length")
            if len(asset_close_prices[idx]) != base_len:
                raise ValueError("Close price arrays must match feature array length")
        if base_len < sequence_length + prediction_horizon:
            raise ValueError("Not enough data for requested sequence/horizon")
        if any(close_feature_index >= arr.shape[1] for arr in asset_arrays):
            raise ValueError("close_feature_index out of bounds for provided features")

        self.asset_arrays = [
            torch.as_tensor(arr, dtype=torch.float32) for arr in asset_arrays
        ]
        self.asset_close_prices = [
            torch.as_tensor(prices, dtype=torch.float32) for prices in asset_close_prices
        ]
        self.asset_names = asset_names
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.close_feature_index = close_feature_index
        self.epsilon = epsilon
        self.asset_class_ids = torch.tensor(
            [get_asset_class_id(name) for name in self.asset_names],
            dtype=torch.long,
        )
        self.per_asset_fees = torch.tensor(
            [float(get_trading_fee(name)) for name in self.asset_names],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return self.asset_arrays[0].shape[0] - self.sequence_length - self.prediction_horizon + 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sl = self.sequence_length
        ph = self.prediction_horizon
        seqs = []
        targets = []
        future_returns = []

        for features, close_prices in zip(self.asset_arrays, self.asset_close_prices):
            seq = features[idx : idx + sl]
            target = features[idx + sl : idx + sl + ph]
            seqs.append(seq)
            targets.append(target[:, self.close_feature_index])

            current_price = close_prices[idx + sl - 1]
            next_price = close_prices[idx + sl]
            ret = (next_price - current_price) / (current_price + self.epsilon)
            future_returns.append(ret)

        combined_inputs = torch.cat(seqs, dim=1).contiguous()
        price_targets = torch.stack(targets, dim=0)
        future_returns = torch.stack(future_returns, dim=0)

        return {
            'input_ids': combined_inputs,
            'labels': price_targets,
            'future_returns': future_returns,
            'attention_mask': torch.ones(self.sequence_length, dtype=torch.float32),
            'asset_class_ids': self.asset_class_ids.clone(),
            'per_asset_fees': self.per_asset_fees.clone(),
        }


def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split data into train/validation/test sets
    
    Args:
        data: Input data
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data


def augment_data(data, noise_factor=0.01, scaling_factor=0.05):
    """
    Augment time series data with noise and scaling
    
    Args:
        data: Input data array
        noise_factor: Standard deviation of Gaussian noise
        scaling_factor: Standard deviation of scaling factor
        
    Returns:
        Augmented data
    """
    
    augmented = data.copy()
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_factor, data.shape)
    augmented += noise
    
    # Random scaling
    scaling = np.random.normal(1.0, scaling_factor, (data.shape[0], 1))
    augmented *= scaling
    
    return augmented


def load_training_data(
    data_dir="trainingdata",
    symbols=None,
    start_date='2015-01-01',
    recursive: bool = True,
    min_rows: int = 50,
    use_toto_forecasts: bool = False,
    toto_options: Optional[TotoOptions] = None,
    sequence_length: int = 60,
    prediction_horizon: int = 5,
) -> np.ndarray:
    """
    Load training data from various sources
    
    Args:
        data_dir: Directory containing CSV files
        symbols: List of symbols to download if no local data
        start_date: Start date for downloading data
        
    Returns:
        Processed data array
    """
    
    data_path = Path(data_dir)
    
    # Prepare shared processor configuration so Toto features stay consistent.
    if use_toto_forecasts:
        resolved_toto_options = toto_options or TotoOptions(
            horizon=prediction_horizon,
            context_length=sequence_length,
        )
    else:
        resolved_toto_options = toto_options

    processor_kwargs = dict(
        use_toto_forecasts=use_toto_forecasts,
        toto_options=resolved_toto_options,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
    )

    # Try to load from local CSV files first (supports nested folders)
    if data_path.exists():
        csv_files = list(data_path.rglob("*.csv")) if recursive else list(data_path.glob("*.csv"))
        if csv_files:
            print(f"Found {len(csv_files)} CSV files under {data_path} (recursive={recursive})")
            
            all_data = []
            loaded_files = 0
            processor = StockDataProcessor(**processor_kwargs)
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    # Standardize columns
                    df.columns = df.columns.str.lower()
                    if 'date' in df.columns:
                        try:
                            df['date'] = pd.to_datetime(df['date'])
                            df = df.sort_values('date')
                        except Exception:
                            pass
                    
                    print(f"Loaded {csv_file.name}: {len(df)} rows")
                    
                    # Validate required columns
                    required = {'open', 'high', 'low', 'close', 'volume'}
                    if not required.issubset(set(df.columns)):
                        continue
                    if len(df) < min_rows:
                        continue
                    features = processor.prepare_features(df, symbol=Path(csv_file).stem)
                    all_data.append(features)
                    loaded_files += 1
                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")
            
            if all_data:
                combined_data = np.vstack(all_data)
                print(f"Combined local data shape: {combined_data.shape} from {loaded_files} files")
                return combined_data
    
    # If no local data, try symbols from local stock CSVs (no external download)
    if symbols:
        print(f"No aggregated CSVs found. Loading local per-symbol CSVs for: {symbols}")
        data_dict = load_local_stock_data(symbols, data_dir=str(data_path))
        if data_dict:
            all_data = []
            processor = StockDataProcessor(**processor_kwargs)
            for symbol, df in data_dict.items():
                features = processor.prepare_features(df, symbol=symbol)
                all_data.append(features)
            if all_data:
                combined_data = np.vstack(all_data)
                print(f"Combined local symbol data shape: {combined_data.shape}")
                return combined_data
    
    # Generate synthetic data as fallback
    print("No data sources available. Generating synthetic data...")
    return generate_synthetic_data()


def generate_synthetic_data(length=10000, n_features=25):
    """
    Generate synthetic stock-like data for testing
    
    Args:
        length: Number of time steps
        n_features: Number of features
        
    Returns:
        Synthetic data array
    """
    
    np.random.seed(42)
    
    # Generate realistic stock price movements
    initial_price = 100.0
    returns = np.random.normal(0.0005, 0.02, length)  # 0.05% daily return, 2% volatility
    prices = [initial_price]
    
    for i in range(1, length):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(max(new_price, 0.01))  # Prevent negative prices
    
    prices = np.array(prices)
    
    # Generate OHLCV data
    data = []
    for i in range(len(prices)):
        price = prices[i]
        
        # Generate realistic OHLC from close price
        volatility = abs(np.random.normal(0, 0.01))
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        open_price = np.random.uniform(low, high)
        
        # Volume (random but realistic)
        volume = np.random.exponential(1000000)
        
        # Additional synthetic features
        features = [open_price, high, low, price, volume]
        
        # Add more synthetic technical indicators
        for j in range(n_features - 5):
            features.append(np.random.normal(0, 1))
        
        data.append(features)
    
    data = np.array(data)
    print(f"Generated synthetic data: {data.shape}")
    
    return data


class DataCollator:
    """Data collator for batching sequences"""
    
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, examples):
        """Collate examples into a batch"""
        
        batch = {}
        
        # Get max sequence length in batch
        max_len = max(example['input_ids'].shape[0] for example in examples)
        
        # Pad sequences
        input_ids = []
        attention_masks = []
        labels = []
        action_labels = []
        
        for example in examples:
            seq_len = example['input_ids'].shape[0]
            
            # Pad input_ids
            padded_input = torch.zeros(max_len, example['input_ids'].shape[1])
            padded_input[:seq_len] = example['input_ids']
            input_ids.append(padded_input)
            
            # Create attention mask
            attention_mask = torch.zeros(max_len)
            attention_mask[:seq_len] = 1
            attention_masks.append(attention_mask)
            
            labels.append(example['labels'])
            action_labels.append(example['action_labels'])
        
        batch['input_ids'] = torch.stack(input_ids)
        batch['attention_mask'] = torch.stack(attention_masks)
        batch['labels'] = torch.stack(labels)
        batch['action_labels'] = torch.stack(action_labels)
        
        return batch
