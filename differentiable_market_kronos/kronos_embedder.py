"""Frozen Kronos wrapper and rolling feature precomputation utilities."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch


def _maybe_append_kronos_to_path() -> Optional[str]:
    for candidate in ("external/kronos", "../external/kronos", "../../external/kronos"):
        model_dir = os.path.join(candidate, "model")
        if os.path.exists(model_dir):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return candidate
    return None


KRONOS_PATH = _maybe_append_kronos_to_path()

try:  # pragma: no cover
    from model import Kronos, KronosTokenizer, KronosPredictor  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Could not import Kronos classes. Clone 'shiyu-coder/Kronos' under external/kronos."
    ) from exc


@dataclass(slots=True)
class KronosFeatureSpec:
    horizons: Tuple[int, ...] = (1, 12, 48)
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)
    include_path_stats: bool = True


class KronosEmbedder:
    def __init__(
        self,
        model_id: str = "NeoQuasar/Kronos-base",
        tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base",
        device: str = "cuda",
        max_context: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 16,
        sample_chunk: int = 32,
        top_k: int = 0,
        clip: float = 5.0,
        feature_spec: Optional[KronosFeatureSpec] = None,
        bf16: bool = True,
    ) -> None:
        self.device = device
        self.max_context = max_context
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.sample_count = sample_count
        self.sample_chunk = max(1, sample_chunk)
        self.feature_spec = feature_spec or KronosFeatureSpec()
        self.bf16 = bf16 and device.startswith("cuda")
        self.clip = clip

        self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
        self.model = Kronos.from_pretrained(model_id)
        self.model.eval().to(self.device)
        try:
            self.model = torch.compile(self.model)
        except Exception:  # pragma: no cover
            pass
        self.predictor = KronosPredictor(
            self.model,
            self.tokenizer,
            device=self.device,
            max_context=self.max_context,
            clip=self.clip,
        )

    @torch.no_grad()
    def _predict_paths(self, x_df: pd.DataFrame, x_ts: pd.Series, horizon: int) -> Tuple[np.ndarray, float]:
        if len(x_ts) < 2:
            raise ValueError("Need at least two timestamps to infer frequency")
        delta = x_ts.iloc[-1] - x_ts.iloc[-2]
        y_ts = pd.Series(pd.date_range(start=x_ts.iloc[-1] + delta, periods=horizon, freq=delta))
        dtype_ctx = torch.bfloat16 if self.bf16 and torch.cuda.is_available() else torch.float32
        preds = []
        using_cuda = self.device.startswith("cuda")
        autocast_enabled = using_cuda and self.bf16
        with torch.autocast(device_type="cuda", dtype=dtype_ctx, enabled=autocast_enabled):
            for sample_idx in range(self.sample_count):
                self.predictor.clip = self.clip
                pred_df = self.predictor.predict(
                    df=x_df,
                    x_timestamp=x_ts,
                    y_timestamp=y_ts,
                    pred_len=horizon,
                    T=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    sample_count=1,
                )
                preds.append(pred_df["close"].to_numpy(dtype=np.float64))
                if using_cuda and ((sample_idx + 1) % self.sample_chunk == 0):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
        if using_cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        paths = np.stack(preds, axis=0)
        last_close = float(x_df["close"].iloc[-1])
        return paths, last_close

    def _summarize_paths(self, paths: np.ndarray, last_close: float) -> Dict[str, float]:
        end_prices = paths[:, -1]
        end_returns = (end_prices / (last_close + 1e-8)) - 1.0
        features: Dict[str, float] = {
            "mu_end": float(end_returns.mean()),
            "sigma_end": float(end_returns.std(ddof=1) if end_returns.size > 1 else 0.0),
            "up_prob": float((end_returns > 0).mean()),
        }
        for q in self.feature_spec.quantiles:
            features[f"q{int(q * 100)}_end"] = float(np.quantile(end_returns, q))
        if self.feature_spec.include_path_stats:
            log_prices = np.log(paths + 1e-8)
            path_vol = log_prices[:, 1:] - log_prices[:, :-1]
            features["path_vol_mean"] = float(path_vol.std(axis=1, ddof=1).mean())
            features["path_range_mean"] = float((paths.max(axis=1) - paths.min(axis=1)).mean() / (last_close + 1e-8))
        return features

    @torch.no_grad()
    def features_for_context(self, x_df: pd.DataFrame, x_ts: pd.Series) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for horizon in self.feature_spec.horizons:
            paths, last_close = self._predict_paths(x_df, x_ts, horizon)
            feats = self._summarize_paths(paths, last_close)
            out.update({f"H{horizon}_{k}": v for k, v in feats.items()})
        return out


def precompute_feature_table(
    df: pd.DataFrame,
    ts: pd.Series,
    lookback: int,
    horizon_main: int,
    embedder: KronosEmbedder,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
) -> pd.DataFrame:
    start = max(lookback, start_index or 0)
    end = min(len(df) - horizon_main, end_index or len(df) - horizon_main)
    rows: list[Dict[str, float]] = []
    idx: list[pd.Timestamp] = []
    for i in range(start, end):
        context_df = df.iloc[i - lookback : i].copy()
        context_ts = ts.iloc[i - lookback : i].copy()
        feats = embedder.features_for_context(context_df, context_ts)
        rows.append(feats)
        idx.append(pd.Timestamp(ts.iloc[i]))
        if (i - start) % 50 == 0:
            print(f"[precompute] {i - start}/{end - start} windows")
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))
