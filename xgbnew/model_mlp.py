"""Small PyTorch MLP for daily open-to-close direction.

A tabular feed-forward network matched to the 14-feature (or 14+rank/disp)
XGB frame. Same contract as ``XGBStockModel``: returns ``pd.Series`` in [0, 1].

Intentionally simple — 3 hidden layers, dropout, Adam, early stopping on val.
Used to probe whether extra model capacity (vs XGB) can extract more signal
from the identical feature set.
"""
from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np
import pandas as pd

from xgbnew.model_base import BaseBinaryDailyModel

logger = logging.getLogger(__name__)


def _check_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torch not installed. This venv should already have it."
        ) from exc


def _build_mlp(in_dim: int, hidden_dims: list[int], dropout: float):
    import torch
    import torch.nn as nn
    layers: list[nn.Module] = []
    d_prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(d_prev, h))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d_prev = h
    layers.append(nn.Linear(d_prev, 1))
    return nn.Sequential(*layers)


class MLPStockModel(BaseBinaryDailyModel):
    """Small tabular MLP on the XGB feature frame."""

    family = "mlp"

    DEFAULT_PARAMS = dict(
        hidden_dims=(256, 128, 64),
        dropout=0.2,
        learning_rate=1e-3,
        batch_size=8192,
        epochs=30,
        early_stop_patience=5,
        weight_decay=1e-5,
        random_state=42,
    )

    def __init__(self, device: str | None = None, **kwargs) -> None:
        super().__init__(device=device)
        _check_torch()
        params = {**self.DEFAULT_PARAMS, **kwargs}
        self.hidden_dims = [int(h) for h in params["hidden_dims"]]
        self.dropout = float(params["dropout"])
        self.learning_rate = float(params["learning_rate"])
        self.batch_size = int(params["batch_size"])
        self.epochs = int(params["epochs"])
        self.early_stop_patience = int(params["early_stop_patience"])
        self.weight_decay = float(params["weight_decay"])
        self.random_state = int(params["random_state"])
        # Feature-wise normalization stats (learnt from train fold).
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None
        self._in_dim: int | None = None
        self._resolved_device: str | None = None
        # self.clf is the torch module; populated in fit() and serialized
        # through _extra_state so that a CPU-only reload path exists.

    # ── Device helper ───────────────────────────────────────────────────────

    def _resolve_device(self) -> str:
        if self._resolved_device is not None:
            return self._resolved_device
        import torch
        if self.device and self.device.startswith("cuda") and torch.cuda.is_available():
            self._resolved_device = self.device
        elif self.device and self.device.startswith("cuda"):
            logger.warning("cuda requested but not available; using cpu")
            self._resolved_device = "cpu"
        else:
            self._resolved_device = self.device or "cpu"
        return self._resolved_device

    # ── fit ─────────────────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame,
            feature_cols: Sequence[str],
            val_df: pd.DataFrame | None = None,
            verbose: bool = True) -> "MLPStockModel":
        """GPU-resident training loop.

        Entire (X, y) sits on the GPU once; per-epoch shuffle is a single
        cuda RNG permute; batches are slices of the resident tensor. For a
        1M × 15 float32 frame (≈60 MB) on a 5090, this is ~100× faster than
        the previous DataLoader + TensorDataset path which paid host→device
        copy + python loop overhead per batch.
        """
        import torch
        import torch.nn as nn

        self._fit_medians(train_df, feature_cols)
        self._in_dim = len(self.feature_cols)

        X_train = train_df[self.feature_cols].values.astype(np.float32)
        y_train = train_df["target_oc_up"].values.astype(np.float32)
        nan_mask = np.isnan(X_train)
        if nan_mask.any():
            X_train[nan_mask] = np.take(self._col_medians, np.where(nan_mask)[1])

        self._feat_mean = X_train.mean(axis=0).astype(np.float32)
        self._feat_std = X_train.std(axis=0).astype(np.float32)
        self._feat_std[self._feat_std < 1e-6] = 1.0
        X_train = (X_train - self._feat_mean) / self._feat_std

        if val_df is not None and len(val_df) > 0:
            X_val = val_df[self.feature_cols].values.astype(np.float32)
            vnm = np.isnan(X_val)
            if vnm.any():
                X_val[vnm] = np.take(self._col_medians, np.where(vnm)[1])
            X_val = (X_val - self._feat_mean) / self._feat_std
            y_val = val_df["target_oc_up"].values.astype(np.float32)
        else:
            X_val = y_val = None

        torch.manual_seed(self.random_state)
        device = self._resolve_device()
        net = _build_mlp(self._in_dim, self.hidden_dims, self.dropout).to(device)
        opt = torch.optim.AdamW(net.parameters(),
                                lr=self.learning_rate,
                                weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        # Park the full training set on device once.
        X_dev = torch.from_numpy(X_train).to(device, non_blocking=True)
        y_dev = torch.from_numpy(y_train).unsqueeze(1).to(device, non_blocking=True)
        n_rows = X_dev.shape[0]
        bs = int(self.batch_size)

        if X_val is not None:
            X_val_dev = torch.from_numpy(X_val).to(device)
            y_val_dev = torch.from_numpy(y_val).unsqueeze(1).to(device)
        else:
            X_val_dev = y_val_dev = None

        g = torch.Generator(device=device)
        g.manual_seed(self.random_state)

        best_val_loss = float("inf")
        best_state = None
        patience_left = self.early_stop_patience

        t0 = time.perf_counter()
        for epoch in range(1, self.epochs + 1):
            net.train()
            perm = torch.randperm(n_rows, device=device, generator=g)
            running = 0.0
            n_seen = 0
            for i in range(0, n_rows, bs):
                idx = perm[i:i + bs]
                xb = X_dev.index_select(0, idx)
                yb = y_dev.index_select(0, idx)
                opt.zero_grad(set_to_none=True)
                logits = net(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                running += float(loss.detach()) * xb.size(0)
                n_seen += xb.size(0)
            train_loss = running / max(n_seen, 1)

            val_msg = ""
            if X_val_dev is not None:
                net.eval()
                with torch.no_grad():
                    vl = float(loss_fn(net(X_val_dev), y_val_dev).detach())
                val_msg = f" val={vl:.5f}"
                if vl < best_val_loss - 1e-6:
                    best_val_loss = vl
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in net.state_dict().items()}
                    patience_left = self.early_stop_patience
                else:
                    patience_left -= 1
                    if patience_left < 0:
                        logger.info("MLP early-stopped at epoch %d", epoch)
                        break
            if verbose:
                logger.info("MLP epoch %d/%d train=%.5f%s",
                            epoch, self.epochs, train_loss, val_msg)

        if best_state is not None:
            net.load_state_dict(best_state)
        logger.info("MLP fit in %.1fs", time.perf_counter() - t0)
        self.clf = net
        self._fitted = True
        return self

    # ── predict ─────────────────────────────────────────────────────────────

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        if self._feat_mean is None or self._feat_std is None:
            raise RuntimeError("MLP feat_mean/std missing — load/fit broken.")
        Xn = (X - self._feat_mean) / self._feat_std
        device = self._resolve_device()
        self.clf.eval()
        with torch.no_grad():
            xt = torch.from_numpy(Xn.astype(np.float32)).to(device)
            # Process in chunks to avoid blowing memory on large OOS sets
            CHUNK = 262144
            outs: list[np.ndarray] = []
            for i in range(0, xt.shape[0], CHUNK):
                logits = self.clf(xt[i:i + CHUNK])
                outs.append(torch.sigmoid(logits).squeeze(-1).cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0,), dtype=np.float32)

    # ── save / load extra state ─────────────────────────────────────────────

    def _extra_state(self) -> dict:
        # Serialize torch state_dict as CPU tensors + MLP config + norm stats.
        import torch
        sd = {k: v.detach().cpu() for k, v in self.clf.state_dict().items()}
        return {
            "state_dict": sd,
            "hidden_dims": list(self.hidden_dims),
            "dropout": float(self.dropout),
            "in_dim": int(self._in_dim) if self._in_dim is not None else None,
            "feat_mean": self._feat_mean,
            "feat_std": self._feat_std,
            "hparams": {
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "early_stop_patience": self.early_stop_patience,
                "weight_decay": self.weight_decay,
                "random_state": self.random_state,
            },
        }

    def _load_extra_state(self, state: dict | None) -> None:
        if state is None:
            raise ValueError("MLP pickle missing 'state' field — cannot reload.")
        # load() bypasses __init__, so set up instance attrs defensively.
        self._resolved_device = None
        self.hidden_dims = [int(h) for h in state["hidden_dims"]]
        self.dropout = float(state["dropout"])
        self._in_dim = int(state["in_dim"]) if state.get("in_dim") is not None else len(self.feature_cols)
        self._feat_mean = state["feat_mean"]
        self._feat_std = state["feat_std"]
        hp = state.get("hparams") or {}
        for k, v in hp.items():
            setattr(self, k, v)
        device = self._resolve_device()
        net = _build_mlp(self._in_dim, self.hidden_dims, self.dropout)
        net.load_state_dict(state["state_dict"])
        net = net.to(device)
        self.clf = net


__all__ = ["MLPStockModel"]
