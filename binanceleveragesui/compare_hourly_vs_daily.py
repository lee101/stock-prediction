#!/usr/bin/env python3
"""Compare hourly vs daily SUI trading strategies.

Trains and evaluates both timeframes with honest lag=1 evaluation.
Uses Chronos2 forecasts at each timeframe's native resolution.
"""
from __future__ import annotations

import json, sys, time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from binanceneural.config import TrainingConfig, ForecastConfig
from binanceneural.trainer import BinanceHourlyTrainer
from binanceneural.inference import generate_actions_from_frame
from binanceneural.data import (
    BinanceHourlyDataset,
    FeatureNormalizer,
    build_feature_frame,
    build_default_feature_columns,
)
from binanceneural.forecasts import ChronosForecastManager
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceleveragesui.run_leverage_sweep import (
    LeverageConfig, SUI_HOURLY_MARGIN_RATE, MAKER_FEE_10BP,
    simulate_with_margin_cost,
)

SYMBOL = "SUIUSDT"
HOURLY_DATA = REPO / "trainingdatahourlybinance"
DAILY_DATA = REPO / "binance_daily"
FORECAST_CACHE_HOURLY = REPO / "binancechronossolexperiment" / "forecast_cache_sui_10bp"
FORECAST_CACHE_DAILY = REPO / "binanceleveragesui" / "forecast_cache_sui_daily"
CHECKPOINT_ROOT = REPO / "binanceleveragesui" / "checkpoints"

CHRONOS_MODEL = "amazon/chronos-t5-small"

# -- daily features adapted from hourly --
DAILY_BASE_FEATURES = (
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "range_pct",
    "volume_z",
    "dow_sin",
    "dow_cos",
)


def build_daily_feature_frame(frame: pd.DataFrame, horizons=(1,)) -> pd.DataFrame:
    """Build features for daily bars (no hour_sin/cos, adjusted lookbacks)."""
    frame = frame.copy()
    frame["reference_close"] = frame["close"].astype(float)
    frame["return_1d"] = frame["close"].pct_change(1)
    frame["return_5d"] = frame["close"].pct_change(5)
    frame["return_20d"] = frame["close"].pct_change(20)
    frame["volatility_20d"] = frame["return_1d"].rolling(20).std()
    frame["range_pct"] = (frame["high"] - frame["low"]).abs() / frame["close"].replace(0.0, np.nan)
    frame["volume_z"] = _zscore(frame["volume"].astype(float), window=60)

    dow = frame["timestamp"].dt.dayofweek
    frame["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    for h in horizons:
        suffix = f"_h{int(h)}"
        close_col = f"predicted_close_p50{suffix}"
        high_col = f"predicted_high_p50{suffix}"
        low_col = f"predicted_low_p50{suffix}"
        if close_col not in frame.columns:
            continue
        ref = frame["reference_close"].replace(0.0, np.nan)
        frame[f"chronos_close_delta{suffix}"] = (frame[close_col] - ref) / ref
        if high_col in frame.columns:
            frame[f"chronos_high_delta{suffix}"] = (frame[high_col] - ref) / ref
        if low_col in frame.columns:
            frame[f"chronos_low_delta{suffix}"] = (frame[low_col] - ref) / ref
    return frame


def build_daily_feature_columns(horizons=(1,)):
    cols = list(DAILY_BASE_FEATURES)
    for h in horizons:
        suffix = f"_h{int(h)}"
        cols.extend([
            f"chronos_close_delta{suffix}",
            f"chronos_high_delta{suffix}",
            f"chronos_low_delta{suffix}",
        ])
    return cols


def _zscore(series, window):
    rm = series.rolling(window).mean()
    rs = series.rolling(window).std().replace(0.0, np.nan)
    return (series - rm) / rs


def build_daily_chronos_cache():
    """Generate Chronos2 forecast cache on daily bars."""
    logger.info("Building daily Chronos2 forecast cache...")
    for horizon in [1]:
        horizon_dir = FORECAST_CACHE_DAILY / f"h{horizon}"
        cfg = ForecastConfig(
            symbol=SYMBOL,
            data_root=str(DAILY_DATA),
            context_hours=90,  # 90 daily bars = 3 months context
            prediction_horizon_hours=horizon,  # 1 day ahead
            quantile_levels=(0.1, 0.5, 0.9),
            batch_size=32,
            cache_dir=horizon_dir,
        )
        manager = ChronosForecastManager(cfg)
        result = manager.ensure_latest(cache_only=False)
        logger.info(f"Daily h{horizon}: {len(result)} forecasts")


def load_daily_data_module(seq_len=30, val_days=30, test_days=30):
    """Load and prepare daily data with forecasts."""
    csv_path = DAILY_DATA / f"{SYMBOL}.csv"
    price = pd.read_csv(csv_path)
    price.columns = [c.lower() for c in price.columns]
    price["timestamp"] = pd.to_datetime(price["timestamp"], utc=True)
    price["symbol"] = SYMBOL
    price = price[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
    price = price.sort_values("timestamp").reset_index(drop=True)

    # Load daily forecasts
    horizons = (1,)
    merged = None
    for h in horizons:
        from binanceneural.forecasts import ForecastCache
        cache = ForecastCache(FORECAST_CACHE_DAILY / f"h{h}")
        fc = cache.load(SYMBOL)
        if fc.empty:
            raise RuntimeError(f"No daily forecast cache for h{h}")
        fc["timestamp"] = pd.to_datetime(fc["timestamp"], utc=True)
        suffix = f"_h{h}"
        fc = fc.rename(columns={
            "predicted_close_p50": f"predicted_close_p50{suffix}",
            "predicted_high_p50": f"predicted_high_p50{suffix}",
            "predicted_low_p50": f"predicted_low_p50{suffix}",
        })
        keep = ["timestamp", "symbol",
                f"predicted_close_p50{suffix}",
                f"predicted_high_p50{suffix}",
                f"predicted_low_p50{suffix}"]
        fc = fc[[c for c in keep if c in fc.columns]]
        merged = fc if merged is None else merged.merge(fc, on=["timestamp", "symbol"], how="inner")

    frame = price.merge(merged, on=["timestamp", "symbol"], how="inner")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame = build_daily_feature_frame(frame, horizons=horizons)

    feature_cols = build_daily_feature_columns(horizons=horizons)
    frame = frame.dropna(subset=feature_cols).reset_index(drop=True)

    # Split: last test_days for test, val_days before that for val
    n = len(frame)
    test_start = n - test_days
    val_start = test_start - val_days
    if val_start <= seq_len:
        raise ValueError(f"Not enough daily data: {n} rows, need {seq_len + val_days + test_days}")

    train = frame.iloc[:val_start].copy()
    val = frame.iloc[val_start - seq_len:test_start].copy()
    test = frame.iloc[test_start - seq_len:].copy()

    val_start_ts = frame.iloc[val_start]["timestamp"]
    test_start_ts = frame.iloc[test_start]["timestamp"]

    return frame, train, val, test, val_start_ts, test_start_ts, feature_cols


def simulate_daily(bars, actions, config, max_hold_days=3):
    """Simulate daily trading with lag and fill buffer."""
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    actions = actions.sort_values("timestamp").reset_index(drop=True)
    merged = bars.merge(actions, on=["timestamp", "symbol"], how="inner", suffixes=("", "_act"))

    if config.decision_lag_bars > 0:
        for col in ['buy_price', 'sell_price', 'buy_amount', 'sell_amount']:
            if col in merged.columns:
                merged[col] = merged[col].shift(config.decision_lag_bars)
        merged = merged.dropna(subset=['buy_price']).reset_index(drop=True)

    cash = config.initial_cash
    inventory = 0.0
    entry_day = 0
    equity_curve = [cash]
    trades = []

    for i, row in merged.iterrows():
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        buy_price = float(row.get("buy_price", 0) or 0)
        sell_price = float(row.get("sell_price", 0) or 0)
        buy_amount = float(row.get("buy_amount", 0) or 0) / 100.0
        sell_amount = float(row.get("sell_amount", 0) or 0) / 100.0

        equity = cash + inventory * close

        # Force close after max hold days
        if max_hold_days and inventory != 0 and (i - entry_day) >= max_hold_days:
            if inventory > 0:
                cash += inventory * close * (1 - config.maker_fee)
            else:
                cash -= abs(inventory) * close * (1 + config.maker_fee)
            trades.append(("force_close", close, inventory))
            inventory = 0.0

        fill_buf = config.fill_buffer_pct

        if buy_amount > 0 and buy_price > 0 and low <= buy_price * (1 - fill_buf):
            max_buy = max(equity, 0) - inventory * buy_price
            if max_buy > 0 and inventory <= 0:
                buy_qty = buy_amount * max_buy / (buy_price * (1 + config.maker_fee))
                if buy_qty > 0:
                    cost = buy_qty * buy_price * (1 + config.maker_fee)
                    cash -= cost
                    inventory += buy_qty
                    entry_day = i
                    trades.append(("buy", buy_price, buy_qty))

        if sell_amount > 0 and sell_price > 0 and high >= sell_price * (1 + fill_buf):
            if inventory > 0:
                sell_qty = min(sell_amount * inventory, inventory)
                if sell_qty > 0:
                    cash += sell_qty * sell_price * (1 - config.maker_fee)
                    inventory -= sell_qty
                    trades.append(("sell", sell_price, sell_qty))

        equity_curve.append(cash + inventory * close)

    # Close remaining
    if len(merged) > 0 and inventory != 0:
        last_close = float(merged.iloc[-1]["close"])
        if inventory > 0:
            cash += inventory * last_close * (1 - config.maker_fee)
        else:
            cash -= abs(inventory) * last_close * (1 + config.maker_fee)
        inventory = 0
        equity_curve[-1] = cash

    eq = np.array(equity_curve)
    ret = np.diff(eq) / (np.abs(eq[:-1]) + 1e-10)
    neg = ret[ret < 0]
    sortino = (np.mean(ret) / (np.std(neg) + 1e-10)) * np.sqrt(365) if len(neg) > 0 else 0
    return {
        "total_return": (eq[-1] / eq[0]) - 1 if eq[0] > 0 else 0,
        "sortino": float(sortino),
        "num_trades": len(trades),
        "final_equity": float(eq[-1]),
    }


def train_and_eval_daily(rw=0.016, seq_len=30, epochs=25):
    """Train daily SUI model and evaluate."""
    logger.info(f"\n=== DAILY MODEL: rw={rw} seq={seq_len} epochs={epochs} ===")

    frame, train, val, test, val_ts, test_ts, feat_cols = load_daily_data_module(
        seq_len=seq_len, val_days=30, test_days=30)

    train_features = train[feat_cols].to_numpy(dtype=np.float32)
    normalizer = FeatureNormalizer.fit(train_features)
    norm_train = normalizer.transform(train_features)
    norm_val = normalizer.transform(val[feat_cols].to_numpy(dtype=np.float32))
    norm_test = normalizer.transform(test[feat_cols].to_numpy(dtype=np.float32))

    train_ds = BinanceHourlyDataset(train, norm_train, seq_len, primary_horizon=1)
    val_ds = BinanceHourlyDataset(val, norm_val, seq_len, primary_horizon=1)

    ckpt_root = CHECKPOINT_ROOT / f"daily_rw{str(rw).replace('.','')}_seq{seq_len}"
    tc = TrainingConfig(
        epochs=epochs,
        batch_size=16,
        sequence_length=seq_len,
        learning_rate=1e-4,
        weight_decay=1e-4,
        return_weight=rw,
        seed=1337,
        transformer_dim=256,
        transformer_layers=4,
        transformer_heads=8,
        maker_fee=MAKER_FEE_10BP,
        fill_temperature=0.1,
        periods_per_year=365.0,  # daily bars, not hourly
        checkpoint_root=ckpt_root,
        log_dir=Path("tensorboard_logs/binanceleveragesui"),
        use_compile=False,
    )

    # Create a simple data module wrapper
    class DailyDM:
        def __init__(self, train_ds, val_ds, norm, fcols, tframe, vframe, testframe, test_start):
            self.train_dataset = train_ds
            self.val_dataset = val_ds
            self.normalizer = norm
            self.feature_columns = tuple(fcols)
            self.train_frame = tframe
            self.val_frame = vframe
            self.test_frame = testframe
            self.test_window_start = test_start
            self.frame = tframe  # for trainer compatibility

        def train_dataloader(self, batch_size, num_workers=0):
            from torch.utils.data import DataLoader
            return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, drop_last=True, pin_memory=True)

        def val_dataloader(self, batch_size, num_workers=0):
            from torch.utils.data import DataLoader
            return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False, pin_memory=True)

    dm = DailyDM(train_ds, val_ds, normalizer, feat_cols, train, val, test, test_ts)

    trainer = BinanceHourlyTrainer(tc, dm)
    artifacts = trainer.train()

    ckpt_root.mkdir(parents=True, exist_ok=True)
    sd = artifacts.state_dict
    if artifacts.best_checkpoint and artifacts.best_checkpoint.exists():
        ckpt = torch.load(artifacts.best_checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
    ckpt_path = ckpt_root / "policy_checkpoint.pt"
    torch.save({
        "state_dict": sd,
        "config": asdict(tc),
        "feature_columns": feat_cols,
        "normalizer": normalizer.to_dict(),
    }, ckpt_path)

    model, norm_loaded, fcols_loaded, _ = load_policy_checkpoint(str(ckpt_path))
    actions = generate_actions_from_frame(
        model=model, frame=test, feature_columns=fcols_loaded,
        normalizer=norm_loaded, sequence_length=seq_len, horizon=1,
    )

    bars_test = test[test["timestamp"] >= test_ts].copy()
    actions_test = actions[actions["timestamp"] >= test_ts].copy()

    results = {}
    for lag in [0, 1]:
        for buf in [0.0, 0.001]:
            lcfg = LeverageConfig(
                max_leverage=1.0, initial_cash=5000.0,
                decision_lag_bars=lag, fill_buffer_pct=buf,
                margin_hourly_rate=0.0, maker_fee=MAKER_FEE_10BP,
            )
            r = simulate_daily(bars_test, actions_test, lcfg, max_hold_days=3)
            key = f"lag{lag}_buf{int(buf*10000)}"
            results[key] = r
            logger.info(f"  DAILY {key}: ret={r['total_return']:.4f} sort={r['sortino']:.2f} trades={r['num_trades']}")

    return {"model": "daily", "rw": rw, "seq_len": seq_len, "results": results, "checkpoint": str(ckpt_path)}


def train_and_eval_hourly(rw=0.016, epochs=25):
    """Train hourly SUI model and evaluate."""
    logger.info(f"\n=== HOURLY MODEL: rw={rw} epochs={epochs} ===")

    dm = ChronosSolDataModule(
        symbol=SYMBOL,
        data_root=HOURLY_DATA,
        forecast_cache_root=FORECAST_CACHE_HOURLY,
        forecast_horizons=(1, 4, 24),
        context_hours=512,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32,
        model_id=CHRONOS_MODEL,
        sequence_length=72,
        split_config=SplitConfig(val_days=30, test_days=30),
        cache_only=True,
        max_history_days=180,  # 6 months to speed up training
    )

    ckpt_root = CHECKPOINT_ROOT / f"hourly_rw{str(rw).replace('.','')}_cmp"
    tc = TrainingConfig(
        epochs=epochs,
        batch_size=16,
        sequence_length=72,
        learning_rate=1e-4,
        weight_decay=1e-4,
        return_weight=rw,
        seed=1337,
        transformer_dim=256,
        transformer_layers=4,
        transformer_heads=8,
        maker_fee=MAKER_FEE_10BP,
        fill_temperature=0.1,
        checkpoint_root=ckpt_root,
        log_dir=Path("tensorboard_logs/binanceleveragesui"),
        use_compile=False,
    )

    trainer = BinanceHourlyTrainer(tc, dm)
    artifacts = trainer.train()

    ckpt_root.mkdir(parents=True, exist_ok=True)
    sd = artifacts.state_dict
    if artifacts.best_checkpoint and artifacts.best_checkpoint.exists():
        ckpt = torch.load(artifacts.best_checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
    ckpt_path = ckpt_root / "policy_checkpoint.pt"
    torch.save({
        "state_dict": sd,
        "config": asdict(tc),
        "feature_columns": list(artifacts.feature_columns),
        "normalizer": artifacts.normalizer.to_dict(),
    }, ckpt_path)

    model, normalizer, feature_columns, _ = load_policy_checkpoint(str(ckpt_path))
    test_frame = dm.test_frame.copy()
    actions = generate_actions_from_frame(
        model=model, frame=test_frame, feature_columns=feature_columns,
        normalizer=normalizer, sequence_length=72, horizon=1,
    )
    test_start = dm.test_window_start
    bars = test_frame[test_frame["timestamp"] >= test_start].copy()
    actions_test = actions[actions["timestamp"] >= test_start].copy()

    results = {}
    for lag in [0, 1]:
        for buf in [0.0, 0.001]:
            lcfg = LeverageConfig(
                max_leverage=1.0, initial_cash=5000.0,
                decision_lag_bars=lag, fill_buffer_pct=buf,
                margin_hourly_rate=SUI_HOURLY_MARGIN_RATE,
                maker_fee=MAKER_FEE_10BP,
            )
            r = simulate_with_margin_cost(bars, actions_test, lcfg)
            key = f"lag{lag}_buf{int(buf*10000)}"
            results[key] = r
            logger.info(f"  HOURLY {key}: ret={r['total_return']:.4f} sort={r['sortino']:.2f} trades={r['num_trades']}")

    return {"model": "hourly", "rw": rw, "results": results, "checkpoint": str(ckpt_path)}


def main():
    logger.info("=== SUI Hourly vs Daily Comparison ===")

    # Step 1: Build daily Chronos2 forecast cache
    build_daily_chronos_cache()

    # Step 2: Sweep return weights for both timeframes
    rw_values = [0.016]
    all_results = []

    for rw in rw_values:
        hourly = train_and_eval_hourly(rw=rw, epochs=10)
        daily = train_and_eval_daily(rw=rw, seq_len=30, epochs=10)
        all_results.append({"rw": rw, "hourly": hourly, "daily": daily})

    # Summary
    logger.info("\n\n=== COMPARISON SUMMARY (lag=1, buf=10bp) ===")
    logger.info(f"{'RW':>6} | {'HOURLY Sort':>12} {'HOURLY Ret':>11} {'H Trades':>8} | {'DAILY Sort':>11} {'DAILY Ret':>10} {'D Trades':>8}")
    logger.info("-" * 85)
    for r in all_results:
        rw = r["rw"]
        h = r["hourly"]["results"].get("lag1_buf10", {})
        d = r["daily"]["results"].get("lag1_buf10", {})
        logger.info(f"{rw:>6.3f} | {h.get('sortino',0):>12.2f} {h.get('total_return',0):>10.4f} {h.get('num_trades',0):>8} | "
                    f"{d.get('sortino',0):>11.2f} {d.get('total_return',0):>9.4f} {d.get('num_trades',0):>8}")

    out_path = REPO / "binanceleveragesui" / "hourly_vs_daily_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
