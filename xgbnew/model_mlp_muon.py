"""Muon-optimized residual MLP for daily direction.

Architecture:
  - Input projection (Linear, AdamW)
  - N residual blocks: RMSNorm → Linear → SiLU → Linear (Muon on hidden weights)
  - Final RMSNorm + head (Linear, AdamW)

Optimizer split:
  - 2D parameters (all hidden Linear weights) → Muon (bf16 Newton-Schulz)
  - 1D and input/output projection weights → AdamW
    (Muon paper advises not using it on embeddings / final head.)

Same pickle contract as ``MLPStockModel`` so it slots into ``sweep_ensemble_grid``.
Family tag is ``"mlp_muon"`` so the registry can dispatch separately.
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
        raise ImportError("torch not installed") from exc


import torch
import torch.nn as nn


class _RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class _ResBlock(nn.Module):
    def __init__(self, d: int, drop: float) -> None:
        super().__init__()
        self.n1 = _RMSNorm(d)
        self.fc1 = nn.Linear(d, 2 * d, bias=False)   # 2D weight → Muon
        self.fc2 = nn.Linear(2 * d, d, bias=False)   # 2D weight → Muon
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = self.n1(x)
        h = torch.nn.functional.silu(self.fc1(h))
        h = self.fc2(h)
        return x + self.drop(h)


class _ResMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_blocks: int, dropout: float) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [_ResBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.n_final = _RMSNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        # Tag input projection + head weights so _split_params_muon routes
        # them to AdamW instead of Muon (Muon paper: avoid embedding/head).
        self.in_proj.weight._adam = True   # type: ignore[attr-defined]
        self.head.weight._adam = True      # type: ignore[attr-defined]

    def forward(self, x):
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.n_final(h)
        return self.head(h)


def _build_net(in_dim: int, hidden_dim: int, n_blocks: int, dropout: float):
    return _ResMLP(in_dim, hidden_dim, n_blocks, dropout)


def _split_params_muon(net):
    """Return (muon_params, adam_params) partitioning respecting the ``_adam`` tag."""
    muon, adam = [], []
    for p in net.parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and not getattr(p, "_adam", False):
            muon.append(p)
        else:
            adam.append(p)
    return muon, adam


class MuonMLPStockModel(BaseBinaryDailyModel):
    """Residual MLP with Muon optimizer for hidden weight matrices."""

    family = "mlp_muon"

    DEFAULT_PARAMS = dict(
        hidden_dim=128,
        n_blocks=3,
        dropout=0.10,
        muon_lr=0.02,
        muon_momentum=0.95,
        adam_lr=3e-4,
        adam_weight_decay=1e-4,
        batch_size=8192,
        epochs=40,
        early_stop_patience=6,
        random_state=42,
    )

    def __init__(self, device: str | None = None, **kwargs) -> None:
        super().__init__(device=device)
        _check_torch()
        params = {**self.DEFAULT_PARAMS, **kwargs}
        self.hidden_dim = int(params["hidden_dim"])
        self.n_blocks = int(params["n_blocks"])
        self.dropout = float(params["dropout"])
        self.muon_lr = float(params["muon_lr"])
        self.muon_momentum = float(params["muon_momentum"])
        self.adam_lr = float(params["adam_lr"])
        self.adam_weight_decay = float(params["adam_weight_decay"])
        self.batch_size = int(params["batch_size"])
        self.epochs = int(params["epochs"])
        self.early_stop_patience = int(params["early_stop_patience"])
        self.random_state = int(params["random_state"])
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None
        self._in_dim: int | None = None
        self._resolved_device: str | None = None

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

    def fit(self, train_df: pd.DataFrame,
            feature_cols: Sequence[str],
            val_df: pd.DataFrame | None = None,
            verbose: bool = True) -> "MuonMLPStockModel":
        import torch
        import torch.nn as nn
        from xgbnew.muon import Muon

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
        net = _build_net(self._in_dim, self.hidden_dim, self.n_blocks, self.dropout).to(device)

        muon_params, adam_params = _split_params_muon(net)
        opt_muon = Muon(muon_params, lr=self.muon_lr, momentum=self.muon_momentum)
        opt_adam = torch.optim.AdamW(adam_params,
                                     lr=self.adam_lr,
                                     weight_decay=self.adam_weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        logger.info(
            "MuonMLP: %d muon params (%d els) + %d adam params (%d els)",
            len(muon_params), sum(p.numel() for p in muon_params),
            len(adam_params), sum(p.numel() for p in adam_params),
        )

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
                opt_muon.zero_grad(set_to_none=True)
                opt_adam.zero_grad(set_to_none=True)
                logits = net(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt_muon.step()
                opt_adam.step()
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
                        logger.info("MuonMLP early-stopped at epoch %d", epoch)
                        break
            if verbose:
                logger.info("MuonMLP epoch %d/%d train=%.5f%s",
                            epoch, self.epochs, train_loss, val_msg)

        if best_state is not None:
            net.load_state_dict(best_state)
        logger.info("MuonMLP fit in %.1fs", time.perf_counter() - t0)
        self.clf = net
        self._fitted = True
        return self

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        if self._feat_mean is None or self._feat_std is None:
            raise RuntimeError("MuonMLP feat_mean/std missing — load/fit broken.")
        Xn = (X - self._feat_mean) / self._feat_std
        device = self._resolve_device()
        self.clf.eval()
        with torch.no_grad():
            xt = torch.from_numpy(Xn.astype(np.float32)).to(device)
            CHUNK = 262144
            outs: list[np.ndarray] = []
            for i in range(0, xt.shape[0], CHUNK):
                logits = self.clf(xt[i:i + CHUNK])
                outs.append(torch.sigmoid(logits).squeeze(-1).cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0,), dtype=np.float32)

    def _extra_state(self) -> dict:
        import torch
        sd = {k: v.detach().cpu() for k, v in self.clf.state_dict().items()}
        return {
            "state_dict": sd,
            "hidden_dim": int(self.hidden_dim),
            "n_blocks": int(self.n_blocks),
            "dropout": float(self.dropout),
            "in_dim": int(self._in_dim) if self._in_dim is not None else None,
            "feat_mean": self._feat_mean,
            "feat_std": self._feat_std,
            "hparams": {
                "muon_lr": self.muon_lr,
                "muon_momentum": self.muon_momentum,
                "adam_lr": self.adam_lr,
                "adam_weight_decay": self.adam_weight_decay,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "early_stop_patience": self.early_stop_patience,
                "random_state": self.random_state,
            },
        }

    def _load_extra_state(self, state: dict | None) -> None:
        if state is None:
            raise ValueError("MuonMLP pickle missing 'state' field.")
        self._resolved_device = None
        self.hidden_dim = int(state["hidden_dim"])
        self.n_blocks = int(state["n_blocks"])
        self.dropout = float(state["dropout"])
        self._in_dim = int(state["in_dim"]) if state.get("in_dim") is not None else len(self.feature_cols)
        self._feat_mean = state["feat_mean"]
        self._feat_std = state["feat_std"]
        hp = state.get("hparams") or {}
        for k, v in hp.items():
            setattr(self, k, v)
        device = self._resolve_device()
        net = _build_net(self._in_dim, self.hidden_dim, self.n_blocks, self.dropout)
        net.load_state_dict(state["state_dict"])
        net = net.to(device)
        self.clf = net


__all__ = ["MuonMLPStockModel"]
