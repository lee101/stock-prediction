"""Live RL signal generator for Binance spot trading.

Loads the autoresearch-trained TradingPolicy and produces trading signals
from live kline data + Chronos2 forecast cache.

Symbols (fixed order matching training): BTCUSD, ETHUSD, DOGEUSD, AAVEUSD
Obs: 73-dim = 4*16 market features + 5 portfolio state + 4 position encoding
Actions: 0=flat, 1-4=long(BTC,ETH,DOGE,AAVE), 5-8=short (treated as flat for spot)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

SYMBOLS = ("BTCUSD", "ETHUSD", "DOGEUSD", "AAVEUSD")
FEATURES_PER_SYM = 16
NUM_SYMBOLS = 4
OBS_SIZE = NUM_SYMBOLS * FEATURES_PER_SYM + 5 + NUM_SYMBOLS  # 73
NUM_ACTIONS = 1 + 2 * NUM_SYMBOLS  # 9
INITIAL_CASH = 10000.0

ACTION_NAMES = [
    "FLAT",
    "LONG_BTC", "LONG_ETH", "LONG_DOGE", "LONG_AAVE",
    "SHORT_BTC", "SHORT_ETH", "SHORT_DOGE", "SHORT_AAVE",
]


# ---------------------------------------------------------------------------
# Policy architecture (must match training exactly)
# ---------------------------------------------------------------------------

class RunningObsNorm(nn.Module):
    def __init__(self, size, eps=1e-5, clip=10.0):
        super().__init__()
        self.eps = eps
        self.clip = clip
        self.register_buffer('running_mean', torch.zeros(size))
        self.register_buffer('running_var', torch.ones(size))
        self.register_buffer('count', torch.tensor(1e-4))

    def forward(self, x):
        return ((x - self.running_mean) / (self.running_var.sqrt() + self.eps)).clamp(-self.clip, self.clip)


class TradingPolicy(nn.Module):
    def __init__(self, obs_size: int, num_actions: int, hidden: int = 1024):
        super().__init__()
        self.obs_norm = RunningObsNorm(obs_size)
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        x = self.obs_norm(x)
        h = self.encoder(x)
        return self.actor(h), self.critic(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Feature computation (mirrors pufferlib_market/export_data_hourly_forecast.py)
# ---------------------------------------------------------------------------

def _load_forecast_parquet(cache_root: Path, horizon: int, symbol: str) -> pd.DataFrame:
    path = cache_root / f"h{horizon}" / f"{symbol.upper()}.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame.columns = [str(c).strip() for c in frame.columns]
    if "issued_at" in frame.columns:
        frame["issued_at"] = pd.to_datetime(frame["issued_at"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["issued_at"]).drop_duplicates(subset="issued_at", keep="last")
        frame = frame.set_index("issued_at")
    elif "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).drop_duplicates(subset="timestamp", keep="last")
        frame = frame.set_index("timestamp")
    frame.index = frame.index.floor("h")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def _forecast_delta(forecast: pd.DataFrame, index: pd.DatetimeIndex,
                    col: str, close: pd.Series) -> pd.Series:
    if col not in forecast.columns or forecast.empty:
        return pd.Series(0.0, index=index, dtype=np.float32)
    aligned = forecast[col].reindex(index)
    aligned = pd.to_numeric(aligned, errors="coerce")
    delta = (aligned - close) / close.clip(lower=1e-8)
    return delta.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def _forecast_confidence(forecast: pd.DataFrame, index: pd.DatetimeIndex,
                         close: pd.Series) -> pd.Series:
    p90, p10 = "predicted_close_p90", "predicted_close_p10"
    if p90 not in forecast.columns or p10 not in forecast.columns or forecast.empty:
        return pd.Series(0.5, index=index, dtype=np.float32)
    spread = (forecast[p90] - forecast[p10]).reindex(index).abs()
    spread = pd.to_numeric(spread, errors="coerce")
    conf = 1.0 / (1.0 + spread / close.clip(lower=1e-8))
    return conf.replace([np.inf, -np.inf], np.nan).fillna(0.5).astype(np.float32)


def compute_symbol_features(
    price_df: pd.DataFrame,
    fc_h1: pd.DataFrame,
    fc_h24: pd.DataFrame,
) -> np.ndarray:
    """Compute 16 features for one symbol. Returns [T, 16] float32 array."""
    close = price_df["close"].astype(float)
    high = price_df["high"].astype(float)
    low = price_df["low"].astype(float)
    idx = price_df.index

    feat = pd.DataFrame(index=idx)
    feat["chronos_close_delta_h1"] = _forecast_delta(fc_h1, idx, "predicted_close_p50", close)
    feat["chronos_high_delta_h1"] = _forecast_delta(fc_h1, idx, "predicted_high_p50", close)
    feat["chronos_low_delta_h1"] = _forecast_delta(fc_h1, idx, "predicted_low_p50", close)
    feat["chronos_close_delta_h24"] = _forecast_delta(fc_h24, idx, "predicted_close_p50", close)
    feat["chronos_high_delta_h24"] = _forecast_delta(fc_h24, idx, "predicted_high_p50", close)
    feat["chronos_low_delta_h24"] = _forecast_delta(fc_h24, idx, "predicted_low_p50", close)
    feat["forecast_confidence_h1"] = _forecast_confidence(fc_h1, idx, close)
    feat["forecast_confidence_h24"] = _forecast_confidence(fc_h24, idx, close)

    ret_1h = close.pct_change(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feat["return_1h"] = ret_1h.clip(-0.5, 0.5).astype(np.float32)
    feat["return_24h"] = close.pct_change(24).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0).astype(np.float32)
    feat["volatility_24h"] = ret_1h.rolling(24, min_periods=1).std(ddof=0).fillna(0.0).clip(0.0, 1.0).astype(np.float32)

    ma24 = close.rolling(24, min_periods=1).mean()
    ma72 = close.rolling(72, min_periods=1).mean()
    feat["ma_delta_24h"] = ((close - ma24) / ma24.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5).astype(np.float32)
    feat["ma_delta_72h"] = ((close - ma72) / ma72.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5).astype(np.float32)

    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr24 = tr.rolling(24, min_periods=1).mean()
    feat["atr_pct_24h"] = (atr24 / close.clip(lower=1e-8)).fillna(0.0).clip(0.0, 0.5).astype(np.float32)
    feat["trend_72h"] = close.pct_change(72).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0).astype(np.float32)

    roll_max = close.rolling(72, min_periods=1).max()
    feat["drawdown_72h"] = ((close - roll_max) / roll_max.clip(lower=1e-8)).fillna(0.0).clip(-1.0, 0.0).astype(np.float32)

    return feat.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)


# ---------------------------------------------------------------------------
# Portfolio state tracker
# ---------------------------------------------------------------------------

@dataclass
class PortfolioSnapshot:
    """Current portfolio state for obs construction."""
    cash_usd: float = 0.0
    total_value_usd: float = 0.0
    position_symbol: Optional[str] = None  # e.g. "BTCUSD" or None if flat
    position_value_usd: float = 0.0
    unrealized_pnl_usd: float = 0.0
    hold_hours: int = 0
    is_short: bool = False


# ---------------------------------------------------------------------------
# Main signal generator
# ---------------------------------------------------------------------------

@dataclass
class RLSignal:
    action: int
    action_name: str
    target_symbol: Optional[str]  # None if flat
    direction: str  # "long", "flat"
    logits: list[float] = field(default_factory=list)
    value: float = 0.0


class RLSignalGenerator:
    def __init__(
        self,
        checkpoint_path: str | Path,
        forecast_cache_root: str | Path = "binanceneural/forecast_cache",
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.forecast_cache_root = Path(forecast_cache_root)

        ckpt = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        hidden = ckpt.get("hidden_size", 1024)

        self.policy = TradingPolicy(OBS_SIZE, NUM_ACTIONS, hidden=hidden)
        self.policy.load_state_dict(ckpt["model"])
        self.policy.to(self.device).eval()
        logger.info(
            f"RL policy loaded: hidden={hidden}, update={ckpt.get('update')}, "
            f"steps={ckpt.get('global_step')}"
        )

    def _fetch_klines(self, binance_pair: str, limit: int = 96) -> pd.DataFrame:
        """Fetch hourly klines from Binance."""
        from src.binan import binance_wrapper as bw
        klines = bw.get_client().get_klines(symbol=binance_pair, interval="1h", limit=limit)
        rows = []
        for k in klines:
            rows.append({
                "timestamp": pd.Timestamp(k[0], unit="ms", tz="UTC").floor("h"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df

    def _build_market_features(self, klines_map: dict[str, pd.DataFrame]) -> np.ndarray:
        """Build [4, 16] feature matrix from klines + forecast cache."""
        features = np.zeros((NUM_SYMBOLS, FEATURES_PER_SYM), dtype=np.float32)

        for i, sym in enumerate(SYMBOLS):
            klines = klines_map.get(sym)
            if klines is None or klines.empty:
                continue

            fc_h1 = _load_forecast_parquet(self.forecast_cache_root, 1, sym)
            fc_h24 = _load_forecast_parquet(self.forecast_cache_root, 24, sym)

            feat_arr = compute_symbol_features(klines, fc_h1, fc_h24)
            if len(feat_arr) > 0:
                features[i] = feat_arr[-1]  # latest row

        return features

    def _build_obs(
        self,
        market_features: np.ndarray,
        portfolio: PortfolioSnapshot,
    ) -> np.ndarray:
        """Assemble 73-dim observation vector."""
        obs = np.zeros(OBS_SIZE, dtype=np.float32)

        # Market features [0:64]
        obs[:NUM_SYMBOLS * FEATURES_PER_SYM] = market_features.flatten()

        # Portfolio state [64:69]
        base = NUM_SYMBOLS * FEATURES_PER_SYM
        obs[base + 0] = portfolio.cash_usd / INITIAL_CASH
        pos_val = portfolio.position_value_usd
        if portfolio.is_short:
            pos_val = -pos_val
        obs[base + 1] = pos_val / INITIAL_CASH
        obs[base + 2] = portfolio.unrealized_pnl_usd / INITIAL_CASH
        obs[base + 3] = min(portfolio.hold_hours / 720.0, 1.0)
        obs[base + 4] = 0.25  # episode progress - fixed for live

        # Position encoding [69:73]
        if portfolio.position_symbol and portfolio.position_symbol in SYMBOLS:
            sym_idx = SYMBOLS.index(portfolio.position_symbol)
            obs[base + 5 + sym_idx] = -1.0 if portfolio.is_short else 1.0

        return obs

    def get_signal(
        self,
        portfolio: PortfolioSnapshot,
        klines_map: Optional[dict[str, pd.DataFrame]] = None,
        binance_pairs: Optional[dict[str, str]] = None,
    ) -> RLSignal:
        """Get RL trading signal.

        Either pass pre-fetched klines_map (symbol→DataFrame) or
        binance_pairs (symbol→pair) to fetch live.
        """
        if klines_map is None:
            if binance_pairs is None:
                binance_pairs = {
                    "BTCUSD": "BTCFDUSD",
                    "ETHUSD": "ETHFDUSD",
                    "DOGEUSD": "DOGEUSDT",
                    "AAVEUSD": "AAVEUSDT",
                }
            klines_map = {}
            for sym, pair in binance_pairs.items():
                try:
                    klines_map[sym] = self._fetch_klines(pair, limit=96)
                except Exception as e:
                    logger.warning(f"Failed to fetch klines for {sym} ({pair}): {e}")

        market_features = self._build_market_features(klines_map)
        obs = self._build_obs(market_features, portfolio)

        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.policy(obs_t)

        logits_np = logits[0].cpu().numpy()
        action = int(logits_np.argmax())
        value_f = float(value[0].cpu())

        # For spot trading: shorts map to flat
        if action >= 5:
            action = 0

        target_symbol = SYMBOLS[action - 1] if 1 <= action <= 4 else None
        direction = "long" if target_symbol else "flat"

        sig = RLSignal(
            action=action,
            action_name=ACTION_NAMES[action],
            target_symbol=target_symbol,
            direction=direction,
            logits=logits_np.tolist(),
            value=value_f,
        )
        logger.info(
            f"RL signal: {sig.action_name} (value={sig.value:.3f}, "
            f"logits={[f'{l:.2f}' for l in sig.logits]})"
        )
        return sig
