"""GPU-native backtest + sizing for boostbaseline.

This module keeps the hot loop on-device (CuPy), so that once predictions
come off a GPU-trained XGBoost booster (``device='cuda'``) we never bounce
back to pandas/NumPy just to compute positions, fees, and Sharpe.

API is intentionally a small surface:
  - ``run_backtest_gpu(y_true, y_pred, …) -> (total_return, sharpe)``
  - ``grid_search_sizing_gpu(y_true, y_pred, …) -> (tr, sh, scale, cap)``
  - ``train_xgb_gpu(df_cudf_or_pandas, …) -> dict``  (convenience wrapper)

Fee model matches ``boostbaseline.backtest._compute_fee_changes`` with
``turnover_proportional=True`` — i.e. ``fee * |Δposition|``, not flat
per-change — so CuPy and NumPy paths agree numerically.

CPU fallback: if CuPy / cuDF are unavailable the functions raise a clear
``RuntimeError``. Call sites can ``try: gpu_core … except RuntimeError:
boostbaseline.backtest …`` to stay portable.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def _require_cupy():
    try:
        import cupy as cp  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "gpu_core requires cupy; install with `uv pip install cupy-cuda13x`"
        ) from exc
    return cp


def _as_cupy_1d(x: Any):
    """Accept CuPy / NumPy / pandas.Series / cuDF.Series → 1-D CuPy float32."""
    cp = _require_cupy()
    if isinstance(x, cp.ndarray):
        return x.astype(cp.float32, copy=False).reshape(-1)
    # cuDF / pandas / numpy all expose .to_cupy() (cuDF) or __array__ (pandas/numpy).
    to_cupy = getattr(x, "to_cupy", None)
    if callable(to_cupy):
        return to_cupy().astype(cp.float32, copy=False).reshape(-1)
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    return cp.asarray(arr)


def run_backtest_gpu(
    y_true,
    y_pred,
    *,
    is_crypto: bool = True,
    fee: float = 0.0023,
    scale: float = 1.0,
    cap: float = 0.3,
    ann_factor: float = 252.0,
) -> tuple[float, float]:
    """Single-path GPU backtest. Turnover-proportional fees, log-free compounding.

    Parameters match ``boostbaseline.backtest.run_backtest``. Returns
    ``(total_return, sharpe)`` as plain floats so downstream selection
    logic stays simple.
    """
    cp = _require_cupy()
    yt = _as_cupy_1d(y_true)
    yp = _as_cupy_1d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch y_true={yt.shape} y_pred={yp.shape}")

    pos = cp.clip(scale * yp, -cap, cap)
    if is_crypto:
        pos = cp.clip(pos, 0.0, cap)

    prev = cp.concatenate([cp.zeros(1, dtype=pos.dtype), pos[:-1]])
    turnover = cp.abs(pos - prev)
    fees = fee * turnover  # matches _compute_fee_changes(turnover_proportional=True)

    rets = pos * yt - fees
    total_return = float((cp.prod(1.0 + rets) - 1.0).get())
    std = float(rets.std().get())
    mean = float(rets.mean().get())
    sharpe = mean / std * math.sqrt(ann_factor) if std > 1e-12 else 0.0
    return total_return, sharpe


def grid_search_sizing_gpu(
    y_true,
    y_pred,
    *,
    is_crypto: bool = True,
    fee: float = 0.0023,
    scales: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0),
    caps: Sequence[float] = (0.1, 0.2, 0.3, 0.5, 1.0),
    ann_factor: float = 252.0,
) -> tuple[float, float, float, float]:
    """Sweep (scale, cap) on GPU. Small grid; overhead is the copy not the grid."""
    cp = _require_cupy()
    yt = _as_cupy_1d(y_true)
    yp = _as_cupy_1d(y_pred)

    best: tuple[float, float, float, float] | None = None
    for s in scales:
        for c in caps:
            tr, sh = run_backtest_gpu(
                yt, yp, is_crypto=is_crypto, fee=fee, scale=s, cap=c,
                ann_factor=ann_factor,
            )
            if best is None or tr > best[0]:
                best = (tr, sh, float(s), float(c))
    assert best is not None
    return best


def train_xgb_gpu(
    df,
    *,
    feature_cols: Sequence[str] | None = None,
    target_col: str = "y",
    is_crypto: bool = True,
    fee: float = 0.0023,
    n_estimators: int = 4000,
    max_depth: int = 6,
    learning_rate: float = 0.02,
    early_stopping_rounds: int = 200,
    random_state: int = 42,
) -> dict:
    """Train XGBRegressor with ``device='cuda'`` on a (cu)DataFrame.

    Accepts either a cuDF DataFrame (zero-copy via ``.to_cupy`` on series) or
    a pandas DataFrame (XGBoost internally transfers). 60/20/20 split is
    done on the *row order of ``df``* — caller is responsible for sorting
    by time before calling (this module does not own the time-order).

    Sizing is tuned on the validation slice only; the test slice is held
    out for the returned ``test_total_return`` / ``test_sharpe`` numbers.
    """
    import xgboost as xgb  # type: ignore

    if feature_cols is None:
        feature_cols = [c for c in df.columns if str(c).startswith("feature_")]
    feature_cols = list(feature_cols)
    if target_col not in df.columns:
        raise KeyError(f"target column {target_col!r} missing from df")
    if not feature_cols:
        raise ValueError("no feature columns found (prefix 'feature_' or pass feature_cols=)")

    n = len(df)
    i1 = max(256, int(n * 0.60))
    i2 = max(i1 + 64, int(n * 0.80))
    i2 = min(i2, n - 1)
    if i1 >= i2 or i2 >= n:
        raise ValueError(f"dataset too small for 60/20/20 split (n={n})")

    X = df[feature_cols]
    y = df[target_col]
    X_tr, y_tr = X.iloc[:i1], y.iloc[:i1]
    X_va, y_va = X.iloc[i1:i2], y.iloc[i1:i2]
    X_te, y_te = X.iloc[i2:], y.iloc[i2:]

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=8,
        subsample=0.70,
        colsample_bytree=0.70,
        reg_alpha=0.5,
        reg_lambda=2.0,
        max_bin=256,
        objective="reg:pseudohubererror",
        eval_metric="mae",
        tree_method="hist",
        device="cuda",
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    pred_va = model.predict(X_va)
    pred_te = model.predict(X_te)

    tr_va, sh_va, scale, cap = grid_search_sizing_gpu(
        y_va, pred_va, is_crypto=is_crypto, fee=fee,
    )
    test_total, test_sharpe = run_backtest_gpu(
        y_te, pred_te, is_crypto=is_crypto, fee=fee, scale=scale, cap=cap,
    )

    return {
        "model": model,
        "feature_cols": feature_cols,
        "scale": scale,
        "cap": cap,
        "val_total_return": tr_va,
        "val_sharpe": sh_va,
        "test_total_return": test_total,
        "test_sharpe": test_sharpe,
        "best_iteration": getattr(model, "best_iteration", None),
    }


__all__ = ["run_backtest_gpu", "grid_search_sizing_gpu", "train_xgb_gpu"]
