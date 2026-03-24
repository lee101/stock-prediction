"""Market data loading and prompt construction for Qwen RL trading."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

SYMBOLS_30 = [
    "BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "AVAXUSD", "LINKUSD",
    "AAVEUSD", "LTCUSD", "XRPUSD", "DOTUSD", "UNIUSD", "NEARUSD",
    "APTUSD", "ICPUSD", "SHIBUSD", "ADAUSD", "FILUSD", "ARBUSD",
    "OPUSD", "INJUSD", "SUIUSD", "TIAUSD", "SEIUSD", "ATOMUSD",
    "ALGOUSD", "BCHUSD", "BNBUSD", "TRXUSD", "PEPEUSD", "MATICUSD",
]

SYSTEM_PROMPT = (
    "You are a crypto trading planner for Binance. "
    "Given market data for multiple symbols, output a JSON trading plan. "
    "The plan should specify which symbols to trade, direction (LONG/SHORT/FLAT), "
    "allocation percentages, entry/stop/target prices, and brief reasoning. "
    "Output ONLY valid JSON matching the TradingPlan schema."
)


@dataclass
class MarketSnapshot:
    """A point-in-time multi-symbol market state with forward bars for sim."""
    window_id: str
    timestamp: pd.Timestamp
    symbols: list[str]
    features: dict[str, dict]  # symbol -> {price, ret_1h, vol_24h, ...}
    chronos_forecasts: dict[str, dict]  # symbol -> {h1_close_delta, ...}
    forward_bars: pd.DataFrame  # OHLCV for next N hours (for sim reward)


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trading features on a single-symbol OHLCV frame."""
    df = df.copy()
    df["return_1h"] = df["close"].pct_change(1)
    df["return_4h"] = df["close"].pct_change(4)
    df["return_24h"] = df["close"].pct_change(24)
    df["volatility_24h"] = df["return_1h"].rolling(24).std()
    df["range_pct"] = (df["high"] - df["low"]).abs() / df["close"].replace(0, np.nan)
    vol = df["volume"].astype(float)
    log_vol = np.log1p(vol)
    roll_mean = log_vol.rolling(24, min_periods=1).mean()
    roll_std = log_vol.rolling(24, min_periods=1).std().replace(0, 1)
    df["volume_z"] = (log_vol - roll_mean) / roll_std
    return df


def load_symbol_data(data_dir: Path, symbol: str) -> Optional[pd.DataFrame]:
    """Load hourly OHLCV CSV for a symbol."""
    path = data_dir / f"{symbol}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["symbol"] = symbol.upper()
    return df


def load_forecast_cache(cache_dir: Path, symbol: str, horizon: int) -> Optional[pd.DataFrame]:
    """Load Chronos2 forecast cache parquet for a symbol+horizon."""
    path = cache_dir / f"h{horizon}" / f"{symbol}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


class PromptDataset:
    """Sliding window dataset producing (prompt_text, MarketSnapshot) pairs."""

    def __init__(
        self,
        data_dir: Path,
        symbols: Sequence[str],
        *,
        forecast_cache_dir: Optional[Path] = None,
        lookback: int = 24,
        eval_horizon: int = 24,
        stride: int = 4,
        prompt_variant: str = "detailed",
        val_fraction: float = 0.0,
        val_mode: bool = False,
    ):
        self.symbols = list(symbols)
        self.lookback = lookback
        self.eval_horizon = eval_horizon
        self.stride = stride
        self.prompt_variant = prompt_variant

        # load data per symbol
        self._data: dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            df = load_symbol_data(data_dir, sym)
            if df is not None and len(df) >= lookback + eval_horizon + 48:
                df = _compute_features(df)
                self._data[sym] = df

        self.active_symbols = sorted(self._data.keys())
        if not self.active_symbols:
            self._windows = []
            return

        # forecasts
        self._forecasts: dict[str, dict[int, pd.DataFrame]] = {}
        if forecast_cache_dir and forecast_cache_dir.exists():
            for sym in self.active_symbols:
                self._forecasts[sym] = {}
                for h in [1, 24]:
                    fc = load_forecast_cache(forecast_cache_dir, sym, h)
                    if fc is not None:
                        self._forecasts[sym][h] = fc

        # find common timestamp range
        all_ts = None
        for sym in self.active_symbols:
            ts_set = set(self._data[sym]["timestamp"])
            all_ts = ts_set if all_ts is None else all_ts & ts_set
        common_ts = sorted(all_ts) if all_ts else []

        # need lookback + eval_horizon contiguous bars
        min_idx = lookback + 24  # skip first 24h for feature warmup
        max_idx = len(common_ts) - eval_horizon
        if max_idx <= min_idx:
            self._windows = []
            return

        all_windows = list(range(min_idx, max_idx, stride))
        if val_fraction > 0:
            split = int(len(all_windows) * (1 - val_fraction))
            self._windows = all_windows[split:] if val_mode else all_windows[:split]
        else:
            self._windows = all_windows

        self._common_ts = common_ts

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> tuple[str, MarketSnapshot]:
        ts_idx = self._windows[idx]
        decision_ts = self._common_ts[ts_idx]

        # build per-symbol features at decision point
        features = {}
        chronos = {}
        forward_rows = []

        for sym in self.active_symbols:
            df = self._data[sym]
            mask = df["timestamp"] == decision_ts
            if not mask.any():
                continue
            row_idx = df.index[mask][0]
            row = df.iloc[row_idx]

            features[sym] = {
                "price": float(row["close"]),
                "return_1h": float(row.get("return_1h", 0) or 0),
                "return_4h": float(row.get("return_4h", 0) or 0),
                "return_24h": float(row.get("return_24h", 0) or 0),
                "volatility_24h": float(row.get("volatility_24h", 0) or 0),
                "range_pct": float(row.get("range_pct", 0) or 0),
                "volume_z": float(row.get("volume_z", 0) or 0),
            }

            # chronos forecasts
            sym_fc = {}
            for h, fc_df in self._forecasts.get(sym, {}).items():
                fc_mask = fc_df["timestamp"] == decision_ts if "timestamp" in fc_df.columns else None
                if fc_mask is not None and fc_mask.any():
                    fc_row = fc_df[fc_mask].iloc[0]
                    ref = float(row["close"])
                    for col in ["predicted_close_p50", "predicted_high_p50", "predicted_low_p50"]:
                        full_col = f"{col}_h{h}"
                        if full_col in fc_row.index:
                            val = float(fc_row[full_col])
                            key = col.replace("predicted_", "").replace("_p50", "") + f"_delta_h{h}"
                            sym_fc[key] = (val - ref) / ref if ref > 0 else 0.0
            chronos[sym] = sym_fc

            # forward bars for sim
            fwd_start = row_idx + 1
            fwd_end = min(row_idx + 1 + self.eval_horizon, len(df))
            fwd = df.iloc[fwd_start:fwd_end][["timestamp", "symbol", "open", "high", "low", "close", "volume"]].copy()
            forward_rows.append(fwd)

        forward_bars = pd.concat(forward_rows, ignore_index=True) if forward_rows else pd.DataFrame()

        window_id = hashlib.md5(f"{decision_ts}_{idx}".encode()).hexdigest()[:12]
        snapshot = MarketSnapshot(
            window_id=window_id,
            timestamp=decision_ts,
            symbols=self.active_symbols,
            features=features,
            chronos_forecasts=chronos,
            forward_bars=forward_bars,
        )

        prompt = format_prompt(snapshot, variant=self.prompt_variant)
        return prompt, snapshot


def format_prompt(snapshot: MarketSnapshot, variant: str = "detailed") -> str:
    """Format a MarketSnapshot into a user prompt string."""
    lines = [f"[window:{snapshot.window_id}] Market data at {snapshot.timestamp.isoformat()}:"]

    for sym in snapshot.symbols:
        feat = snapshot.features.get(sym, {})
        if not feat:
            continue

        if variant == "minimal":
            lines.append(
                f"{sym}: p={feat['price']:.2f} "
                f"r1h={feat['return_1h']:+.3%} "
                f"r24h={feat['return_24h']:+.3%}"
            )
        elif variant == "detailed":
            parts = [
                f"{sym}: p={feat['price']:.2f}",
                f"r1h={feat['return_1h']:+.3%}",
                f"r4h={feat['return_4h']:+.3%}",
                f"r24h={feat['return_24h']:+.3%}",
                f"vol24h={feat['volatility_24h']:.4f}",
                f"range={feat['range_pct']:.4f}",
                f"vz={feat['volume_z']:.2f}",
            ]
            lines.append(" ".join(parts))
        elif variant == "with_chronos2":
            parts = [
                f"{sym}: p={feat['price']:.2f}",
                f"r1h={feat['return_1h']:+.3%}",
                f"r4h={feat['return_4h']:+.3%}",
                f"r24h={feat['return_24h']:+.3%}",
                f"vol24h={feat['volatility_24h']:.4f}",
                f"vz={feat['volume_z']:.2f}",
            ]
            fc = snapshot.chronos_forecasts.get(sym, {})
            for key in sorted(fc.keys()):
                parts.append(f"{key}={fc[key]:+.3%}")
            lines.append(" ".join(parts))

    lines.append("Portfolio: cash=100% positions=none")
    lines.append("Output a JSON TradingPlan.")
    return "\n".join(lines)


def build_chat_messages(prompt: str) -> list[dict[str, str]]:
    """Build chat messages for tokenizer.apply_chat_template."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
