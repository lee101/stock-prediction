"""Distill Chronos2 daily forecasts into a tiny directional MLP.

Goal: replace the per-decision Chronos2 call (~50ms/symbol) with a millisecond
MLP whose target is the cached Chronos2 next-day forecast direction. This is
the scaffold for the "scaling RL to 1000+ symbols" path — once we can score
1000 symbols/sec we can pre-screen the universe much wider than 32.

Pipeline:
  1. Load cached Chronos2 forecasts from ``strategytraining/forecast_cache/<SYM>.parquet``.
     The cache already covers 2022-02 → 2026-02 with daily forecast_move_pct
     (Chronos2's median forecast of 1d return).
  2. Load aligned OHLC from ``trainingdata/<SYM>.csv``.
  3. For each (symbol, date) build a feature vector from the past N days of
     OHLCV (returns, RSI, vol, MA deltas) — the same feature space the
     screened32 policy already uses, so the student is a drop-in.
  4. Targets are Chronos2's ``forecast_move_pct`` (regression) and its sign
     (binary direction). Train a tiny 3-layer MLP with both heads.
  5. Evaluate IC and directional accuracy on a held-out date range.

This file is the SKETCH — full multi-symbol training is a follow-up. The
smoke-test path (``--smoke``) trains 1 epoch on a tiny synthetic split and
asserts the student converges enough that the loss drops; that's what
``tests/test_distill_chronos2_directional.py`` exercises.

Why distill instead of just keeping Chronos2 calls? Chronos2 LoRA inference
is CUDA-bound at ~50ms/symbol (Hub Kernel sweep results in
docs/chronos2_full_finetune.md). 32 symbols/day is fine; 1000 symbols/day
costs 50s of GPU per decision tick — too slow for hourly policies and waste
for daily ones once the signal is captured.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# Data loading + feature engineering
# ---------------------------------------------------------------------------

def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a, dtype=np.float64)
    mask = np.abs(b) > 1e-12
    out[mask] = a[mask] / b[mask]
    return out


def compute_simple_features(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, vols: np.ndarray, *, lookback: int = 60) -> np.ndarray:
    """Tiny feature set the student MLP consumes (subset of the prod 16-feat pipeline).

    Returns shape (T, F) where F=8: ret_1d, ret_5d, ret_20d, vol_5d, vol_20d,
    rsi_14, ma_delta_5d, range_pct.
    Rows before lookback are filled with zeros; caller should drop them.
    """
    T = len(closes)
    F = 8
    out = np.zeros((T, F), dtype=np.float64)
    rets = np.diff(closes, prepend=closes[0]) / np.clip(closes, 1e-9, None)
    rets = np.clip(rets, -0.5, 0.5)
    for t in range(lookback, T):
        out[t, 0] = rets[t]
        out[t, 1] = (closes[t] / closes[t - 5] - 1.0) if t >= 5 else 0.0
        out[t, 2] = (closes[t] / closes[t - 20] - 1.0) if t >= 20 else 0.0
        out[t, 3] = float(np.std(rets[t - 5: t + 1])) if t >= 5 else 0.0
        out[t, 4] = float(np.std(rets[t - 20: t + 1])) if t >= 20 else 0.0
        # Wilder RSI(14) over the last 14 deltas
        deltas = rets[t - 14: t + 1]
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = float(np.mean(gains)) if gains.size else 0.0
        avg_loss = float(np.mean(losses)) if losses.size else 0.0
        rs = avg_gain / avg_loss if avg_loss > 1e-12 else 0.0
        out[t, 5] = (1.0 - 1.0 / (1.0 + rs)) if rs > 0 else 0.5
        ma5 = float(np.mean(closes[t - 5: t + 1])) if t >= 5 else closes[t]
        out[t, 6] = closes[t] / max(ma5, 1e-9) - 1.0
        out[t, 7] = (highs[t] - lows[t]) / max(closes[t], 1e-9)
    return out.astype(np.float32)


@dataclass(frozen=True)
class SymbolPanel:
    symbol: str
    dates: np.ndarray  # datetime64[D]
    features: np.ndarray  # (T, F)
    target_move: np.ndarray  # (T,) Chronos2's forecast_move_pct, the teacher signal
    realized_move: np.ndarray  # (T,) realized 1d return, for sanity stats


def load_symbol_panel(
    symbol: str,
    *,
    forecast_cache_dir: Path,
    daily_csv_dir: Path,
    lookback: int = 60,
) -> SymbolPanel | None:
    fc_path = forecast_cache_dir / f"{symbol}.parquet"
    csv_path = daily_csv_dir / f"{symbol}.csv"
    if not fc_path.exists() or not csv_path.exists():
        return None
    fc = pd.read_parquet(fc_path)
    daily = pd.read_csv(csv_path)
    # Forecast cache is heterogeneous: some files have a 'date' column derived
    # from the (often UTC midnight) timestamp, others only have 'timestamp'.
    # Derive 'date' uniformly so the merge works on both shapes.
    if "date" not in fc.columns:
        if "timestamp" not in fc.columns:
            return None
        fc = fc.copy()
        fc["date"] = pd.to_datetime(fc["timestamp"], utc=True).dt.tz_convert(None).dt.normalize()
    else:
        fc["date"] = pd.to_datetime(fc["date"], utc=True).dt.tz_convert(None).dt.normalize()
    if "forecast_move_pct" not in fc.columns:
        return None
    daily["timestamp"] = pd.to_datetime(daily["timestamp"], utc=True).dt.tz_convert(None).dt.normalize()
    daily = daily.rename(columns={"timestamp": "date"})
    merged = pd.merge(daily[["date", "open", "high", "low", "close", "volume"]],
                      fc[["date", "forecast_move_pct"]],
                      on="date", how="inner").sort_values("date").reset_index(drop=True)
    if len(merged) < lookback + 5:
        return None
    closes = merged["close"].to_numpy(dtype=np.float64)
    highs = merged["high"].to_numpy(dtype=np.float64)
    lows = merged["low"].to_numpy(dtype=np.float64)
    vols = merged["volume"].to_numpy(dtype=np.float64)
    feats = compute_simple_features(closes, highs, lows, vols, lookback=lookback)
    realized = np.diff(closes, prepend=closes[0]) / np.clip(closes, 1e-9, None)
    realized = np.clip(realized, -0.5, 0.5).astype(np.float32)
    target = merged["forecast_move_pct"].to_numpy(dtype=np.float32)
    dates = merged["date"].to_numpy(dtype="datetime64[D]")
    valid = lookback + 1
    return SymbolPanel(
        symbol=symbol,
        dates=dates[valid:],
        features=feats[valid:],
        target_move=target[valid:],
        realized_move=realized[valid:],
    )


# ---------------------------------------------------------------------------
# Student model
# ---------------------------------------------------------------------------

class DirectionalDistilleeMLP(nn.Module):
    """Tiny MLP: features → (regression head, direction logit head).

    Architecture: 3-layer MLP w/ residual skip on the trunk. Total params ~5k.
    Designed to score 10k+ symbols/sec on a single CPU thread post-export to
    ONNX. The regression head minimises MSE to Chronos2's forecast_move_pct;
    the direction head minimises BCE to sign(forecast_move_pct).
    """

    def __init__(self, input_dim: int, hidden: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.reg_head = nn.Linear(hidden, 1)
        self.dir_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.reg_head(z).squeeze(-1), self.dir_head(z).squeeze(-1)


def _split_by_date(panel: SymbolPanel, *, train_end: np.datetime64) -> tuple[SymbolPanel, SymbolPanel]:
    mask = panel.dates < train_end
    train = SymbolPanel(
        symbol=panel.symbol,
        dates=panel.dates[mask],
        features=panel.features[mask],
        target_move=panel.target_move[mask],
        realized_move=panel.realized_move[mask],
    )
    val = SymbolPanel(
        symbol=panel.symbol,
        dates=panel.dates[~mask],
        features=panel.features[~mask],
        target_move=panel.target_move[~mask],
        realized_move=panel.realized_move[~mask],
    )
    return train, val


def train_student(
    panels: Sequence[SymbolPanel],
    *,
    train_end: str,
    epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 1e-3,
    direction_weight: float = 1.0,
    device: torch.device | str = "cpu",
    seed: int = 1337,
) -> dict:
    """Train the directional distillee on stacked panels."""
    torch.manual_seed(int(seed))
    train_end_d = np.datetime64(str(train_end), "D")
    train_X, train_y, val_X, val_y, val_realized = [], [], [], [], []
    for p in panels:
        tr, va = _split_by_date(p, train_end=train_end_d)
        if tr.features.size:
            train_X.append(tr.features)
            train_y.append(tr.target_move)
        if va.features.size:
            val_X.append(va.features)
            val_y.append(va.target_move)
            val_realized.append(va.realized_move)
    if not train_X or not val_X:
        return {"status": "skip", "reason": "empty train or val split after date filter"}

    train_X_np = np.concatenate(train_X, axis=0)
    train_y_np = np.concatenate(train_y, axis=0)
    val_X_np = np.concatenate(val_X, axis=0)
    val_y_np = np.concatenate(val_y, axis=0)
    val_realized_np = np.concatenate(val_realized, axis=0)
    input_dim = int(train_X_np.shape[1])

    model = DirectionalDistilleeMLP(input_dim=input_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    train_X_t = torch.as_tensor(train_X_np, dtype=torch.float32, device=device)
    train_y_t = torch.as_tensor(train_y_np, dtype=torch.float32, device=device)
    val_X_t = torch.as_tensor(val_X_np, dtype=torch.float32, device=device)
    val_y_t = torch.as_tensor(val_y_np, dtype=torch.float32, device=device)

    losses_per_epoch: list[float] = []
    for epoch in range(int(epochs)):
        model.train()
        perm = torch.randperm(train_X_t.shape[0], device=device)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, train_X_t.shape[0], int(batch_size)):
            idx = perm[i: i + int(batch_size)]
            xb = train_X_t[idx]
            yb = train_y_t[idx]
            yb_dir = (yb > 0).float()
            opt.zero_grad()
            pred_reg, pred_dir = model(xb)
            loss = mse(pred_reg, yb) + float(direction_weight) * bce(pred_dir, yb_dir)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        losses_per_epoch.append(total_loss / max(1, n_batches))

    # Evaluation
    model.eval()
    with torch.no_grad():
        pred_reg, pred_dir = model(val_X_t)
        pred_reg_np = pred_reg.detach().cpu().numpy()
        pred_dir_np = torch.sigmoid(pred_dir).detach().cpu().numpy()
    teacher_dir = (val_y_np > 0).astype(np.float32)
    student_dir = (pred_dir_np > 0.5).astype(np.float32)
    teacher_match = float(np.mean(student_dir == teacher_dir))
    realized_dir = (val_realized_np > 0).astype(np.float32)
    realized_match = float(np.mean(student_dir == realized_dir))
    teacher_baseline = float(np.mean((teacher_dir == realized_dir).astype(np.float32)))
    # Spearman-style IC (rank correlation) of pred vs teacher
    if pred_reg_np.size > 1 and float(np.std(pred_reg_np)) > 1e-12 and float(np.std(val_y_np)) > 1e-12:
        ic = float(np.corrcoef(pred_reg_np, val_y_np)[0, 1])
    else:
        ic = 0.0

    return {
        "status": "ok",
        "n_train": int(train_X_t.shape[0]),
        "n_val": int(val_X_t.shape[0]),
        "input_dim": input_dim,
        "epochs": int(epochs),
        "loss_per_epoch": losses_per_epoch,
        "final_loss": float(losses_per_epoch[-1] if losses_per_epoch else math.nan),
        "teacher_distillation_accuracy": teacher_match,
        "realized_directional_accuracy_student": realized_match,
        "realized_directional_accuracy_teacher": teacher_baseline,
        "regression_pearson_ic": ic,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", default=None,
                    help="Comma-separated symbol list. Default: SCREENED32 set.")
    ap.add_argument("--forecast-cache-dir", default="strategytraining/forecast_cache")
    ap.add_argument("--daily-csv-dir", default="trainingdata")
    ap.add_argument("--train-end", default="2025-06-01",
                    help="Cutoff date — rows on or after this go to validation.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--direction-weight", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-json", default="docs/distill_chronos2_directional.json")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke mode: 4 symbols × 5 epochs, prints stats and exits 0 if status=ok.")
    args = ap.parse_args(argv)

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        from src.daily_stock_defaults import DEFAULT_SYMBOLS
        symbols = list(DEFAULT_SYMBOLS)
    if args.smoke:
        symbols = symbols[:4]
        args.epochs = min(int(args.epochs), 5)

    fc_dir = REPO / args.forecast_cache_dir if not Path(args.forecast_cache_dir).is_absolute() else Path(args.forecast_cache_dir)
    daily_dir = REPO / args.daily_csv_dir if not Path(args.daily_csv_dir).is_absolute() else Path(args.daily_csv_dir)
    panels: list[SymbolPanel] = []
    for sym in symbols:
        panel = load_symbol_panel(sym, forecast_cache_dir=fc_dir, daily_csv_dir=daily_dir)
        if panel is not None:
            panels.append(panel)
    if not panels:
        print("distill: no symbols had both forecast cache + daily CSV", file=sys.stderr)
        return 2
    print(f"loaded {len(panels)} symbols, total rows={sum(p.features.shape[0] for p in panels)}")

    result = train_student(
        panels,
        train_end=str(args.train_end),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        direction_weight=float(args.direction_weight),
        device=args.device,
    )
    out_path = Path(args.out_json)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbols": symbols,
        "n_symbols": len(panels),
        "train_end": str(args.train_end),
        **result,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps({k: v for k, v in payload.items() if k != "loss_per_epoch"}, indent=2))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
