from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import Tensor

try:
    from totoembedding.embedding_model import TotoEmbeddingModel
except Exception:  # pragma: no cover - Toto dependencies are optional
    TotoEmbeddingModel = None  # type: ignore

from differentiable_market_totoembedding.config import TotoEmbeddingConfig


class TotoEmbeddingFeatureExtractor:
    """
    Materialises frozen Toto embeddings for every (timestamp, asset) pair in a
    pre-aligned OHLC tensor. The resulting tensor aligns with the differentiable
    market feature matrices and can be concatenated channel-wise.
    """

    def __init__(self, cfg: TotoEmbeddingConfig):
        self.cfg = cfg

    def compute(
        self,
        ohlc: Tensor,
        timestamps: Sequence,
        symbols: Sequence[str],
    ) -> Tensor:
        """
        Args:
            ohlc: Tensor shaped [T, A, F] containing OHLC features.
            timestamps: Sequence of pandas.Timestamp aligned to the time axis.
            symbols: Asset tickers aligned to the asset axis.

        Returns:
            Tensor shaped [T-1, A, D] with Toto embeddings per timestep/asset.
        """
        if ohlc.ndim != 3:
            raise ValueError(f"Expected [T, A, F] ohlc tensor, received {tuple(ohlc.shape)}")

        cache_path = self._cache_path(ohlc, timestamps, symbols)
        if cache_path is not None and cache_path.exists() and self.cfg.reuse_cache:
            payload = torch.load(cache_path)
            return payload["embeddings"]

        price = ohlc.detach().cpu()
        T, A, F = price.shape

        context = int(max(2, min(self.cfg.context_length, T)))
        feature_dim = int(self.cfg.input_feature_dim or F)
        if feature_dim < F:
            price = price[..., :feature_dim]
        elif feature_dim > F:
            pad_width = feature_dim - F
            pad = torch.zeros(T, A, pad_width, dtype=price.dtype)
            price = torch.cat([price, pad], dim=-1)

        model = self._build_model(feature_dim, len(symbols))
        embeddings = self._materialise_embeddings(price, model, context, timestamps, symbols)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"embeddings": embeddings}, cache_path)

        return embeddings

    # ------------------------------------------------------------------ helpers

    def _build_model(self, feature_dim: int, num_symbols: int) -> TotoEmbeddingModel | None:
        if TotoEmbeddingModel is None:
            return None
        try:
            model = TotoEmbeddingModel(
                pretrained_model_path=str(self.cfg.pretrained_model_path) if self.cfg.pretrained_model_path else None,
                embedding_dim=self.cfg.embedding_dim or 128,
                num_symbols=max(num_symbols, 1),
                freeze_backbone=self.cfg.freeze_backbone,
                input_feature_dim=feature_dim,
                use_toto=self.cfg.use_toto,
                toto_model_id=self.cfg.toto_model_id,
                toto_device=self.cfg.toto_device,
                toto_horizon=self.cfg.toto_horizon,
                toto_num_samples=self.cfg.toto_num_samples,
            )
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            return model
        except Exception:
            return None

    def _materialise_embeddings(
        self,
        price: Tensor,
        model: TotoEmbeddingModel | None,
        context: int,
        timestamps: Sequence,
        symbols: Sequence[str],
    ) -> Tensor:
        T, A, F = price.shape
        device = None
        if model is not None:
            device = torch.device(self.cfg.toto_device if torch.cuda.is_available() else "cpu")
            try:
                model.to(device)
            except Exception:
                device = torch.device("cpu")
                model.to(device)

        windows = []
        for asset in range(A):
            series = price[:, asset, :]
            pad_len = context - 1
            if pad_len > 0:
                if self.cfg.pad_mode == "repeat" and series.shape[0] > 1:
                    reps = pad_len // max(series.shape[0] - 1, 1) + 1
                    prefix = torch.cat([series[1:]] * reps, dim=0)[:pad_len]
                    prefix = torch.cat([series[:1], prefix], dim=0)[:pad_len]
                else:
                    prefix = series[:1].repeat(pad_len, 1)
                padded = torch.cat([prefix, series], dim=0)
            else:
                padded = series
            asset_windows = padded.unfold(0, context, 1).permute(0, 2, 1).contiguous()
            windows.append(asset_windows.unsqueeze(1))
        price_windows = torch.cat(windows, dim=1)  # [T, A, context, F]
        price_windows_flat = price_windows.reshape(T * A, context, F)

        symbol_ids = torch.arange(A, dtype=torch.long).unsqueeze(0).repeat(T, 1).reshape(-1)
        timestamp_tensor = self._build_timestamp_tensor(timestamps, T)
        timestamp_batch = timestamp_tensor.repeat_interleave(A, dim=0)
        regime_tensor = self._build_market_regime(price).reshape(-1)

        batch_size = max(1, int(self.cfg.batch_size))
        outputs: list[Tensor] = []
        with torch.no_grad():
            for start in range(0, price_windows_flat.shape[0], batch_size):
                end = min(start + batch_size, price_windows_flat.shape[0])
                price_batch = price_windows_flat[start:end]
                symbol_batch = symbol_ids[start:end]
                time_batch = timestamp_batch[start:end]
                regime_batch = regime_tensor[start:end]
                if model is None:
                    emb = price_batch.mean(dim=1)
                else:
                    price_batch = price_batch.to(device)
                    symbol_batch = symbol_batch.to(device)
                    time_batch = time_batch.to(device)
                    regime_batch = regime_batch.to(device)
                    out = model(
                        price_data=price_batch,
                        symbol_ids=symbol_batch,
                        timestamps=time_batch,
                        market_regime=regime_batch,
                    )
                    emb = out["embeddings"].detach().cpu()
                outputs.append(emb)
        stacked = torch.cat(outputs, dim=0)

        embed_dim = stacked.shape[-1]
        embeddings = stacked.reshape(T, A, embed_dim)

        # Drop the first timestep to align with forward returns (T-1)
        embeddings = embeddings[1:].contiguous()
        if self.cfg.detach:
            embeddings = embeddings.detach()
        return embeddings

    def _build_timestamp_tensor(self, timestamps: Sequence, T: int) -> Tensor:
        hours = torch.zeros(T, dtype=torch.long)
        day_of_week = torch.zeros(T, dtype=torch.long)
        month = torch.zeros(T, dtype=torch.long)
        for idx, ts in enumerate(timestamps[:T]):
            hour = getattr(ts, "hour", 0)
            dow = getattr(ts, "dayofweek", getattr(ts, "weekday", 0))
            month_val = getattr(ts, "month", 1)
            hours[idx] = max(0, min(23, int(hour)))
            day_of_week[idx] = max(0, min(6, int(dow)))
            month[idx] = max(0, min(11, int(month_val) - 1))
        return torch.stack([hours, day_of_week, month], dim=1)

    def _build_market_regime(self, price: Tensor) -> Tensor:
        close = price[..., 3] if price.shape[-1] >= 4 else price[..., -1]
        log_ret = torch.zeros_like(close)
        log_ret[1:] = torch.log(torch.clamp(close[1:] / close[:-1], min=1e-8, max=1e8))
        small, large = self.cfg.market_regime_thresholds
        regimes = torch.full_like(log_ret, 2, dtype=torch.long)
        regimes = torch.where(log_ret > small, torch.zeros_like(regimes), regimes)
        regimes = torch.where(log_ret < -small, torch.ones_like(regimes), regimes)
        regimes = torch.where(log_ret.abs() > large, torch.full_like(regimes, 3), regimes)
        regimes[0] = 2
        return regimes.to(torch.long)

    def _cache_path(self, ohlc: Tensor, timestamps: Sequence, symbols: Sequence[str]) -> Path | None:
        if self.cfg.cache_dir is None:
            return None
        try:
            cache_dir = Path(self.cfg.cache_dir)
            fingerprint = self._fingerprint(ohlc, timestamps, symbols)
            return cache_dir / f"embeddings_{fingerprint}.pt"
        except Exception:
            return None

    def _fingerprint(self, ohlc: Tensor, timestamps: Sequence, symbols: Sequence[str]) -> str:
        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(str(tuple(ohlc.shape)).encode())
        if len(timestamps):
            try:
                import numpy as np

                ts_values = np.array([getattr(ts, "value", int(idx)) for idx, ts in enumerate(timestamps)], dtype=np.int64)
                hasher.update(ts_values.tobytes())
            except Exception:
                pass
        sym_key = "|".join(symbols)
        hasher.update(sym_key.encode())
        tensor = torch.as_tensor(ohlc, dtype=torch.float32).contiguous()
        hasher.update(tensor.cpu().numpy().tobytes())
        return hasher.hexdigest()
