#!/usr/bin/env python3
"""End-to-end 20-symbol Binance margin sweep: LoRA -> forecast cache -> policy -> eval."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np
import pandas as pd
from loguru import logger

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from preaug import get_augmentation
from chronos2_trainer import _load_pipeline, _fit_pipeline, _save_pipeline
from binanceneural.config import TrainingConfig
from binanceneural.trainer import BinanceHourlyTrainer
from binanceneural.inference import generate_actions_from_frame
from binanceneural.forecasts import ChronosForecastManager, ForecastConfig
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import _build_policy
from binanceleveragesui.run_leverage_sweep import (
    LeverageConfig, MAKER_FEE_10BP, simulate_with_margin_cost,
)
from src.models.chronos2_wrapper import Chronos2OHLCWrapper
from binance_data_wrapper import download_and_save_pair

# ── Config ──────────────────────────────────────────────────────────────────

DATA_ROOT = REPO / "trainingdatahourlybinance"
FORECAST_CACHE_ROOT = REPO / "binanceneural" / "forecast_cache"
LORA_OUTPUT_ROOT = REPO / "chronos2_finetuned"
CHECKPOINT_ROOT = REPO / "binanceleveragesui" / "checkpoints"
RESULTS_DIR = REPO / "binanceleveragesui" / "sweep_results"

DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT",
    "LTCUSDT", "UNIUSDT", "AAVEUSDT", "ATOMUSDT", "APTUSDT",
    "TRXUSDT", "SHIBUSDT", "SUIUSDT", "NEARUSDT",
    "OPUSDT", "ARBUSDT",
]

# data symbol = strip trailing T from pair (BTCUSDT->BTCUSD), except SUI
def pair_to_data_symbol(pair: str) -> str:
    """Resolve pair to data symbol, checking which CSV actually exists."""
    if pair == "SUIUSDT":
        return "SUIUSDT"
    # try USD suffix first (legacy), then USDT (newer downloads)
    usd = pair.rstrip("T") if pair.endswith("USDT") else pair
    if (DATA_ROOT / f"{usd}.csv").exists():
        return usd
    if (DATA_ROOT / f"{pair}.csv").exists():
        return pair
    return usd  # default

PREAUGS = ["differencing", "percent_change", "log_returns", "robust_scaling"]
LORA_CTX = 128
LORA_LR = "5e-5"
LORA_STEPS = 1000
LORA_R = 16

RW_VALUES = [0.10, 0.30, 0.50]
WD_VALUES = [0.03, 0.05]
POLICY_EPOCHS = 20
POLICY_SEQ = 72
POLICY_DIM = 256
POLICY_LAYERS = 4
POLICY_HEADS = 8
POLICY_LR = 1e-4
FILL_TEMP = 0.1
FILL_BUFFER = 0.0005

# eval params (prod-matching)
EVAL_LAG = 1
EVAL_MAX_HOLD = 6
EVAL_INTENSITY = 5.0
EVAL_INITIAL_CASH = 5000.0


# ── Phase A: Data Download ──────────────────────────────────────────────────

def ensure_data(symbols: List[str]) -> Dict[str, Path]:
    """Download/update hourly data for all symbols. Returns {data_symbol: csv_path}."""
    paths = {}
    for pair in symbols:
        dsym = pair_to_data_symbol(pair)
        csv_path = DATA_ROOT / f"{dsym}.csv"
        if csv_path.exists() and csv_path.stat().st_size > 0:
            logger.info(f"Data exists: {dsym} ({csv_path.stat().st_size // 1024}KB)")
            paths[dsym] = csv_path
            continue
        logger.info(f"Downloading {pair} -> {dsym}")
        try:
            result = download_and_save_pair(
                pair, DATA_ROOT, history_years=3,
                fallback_quotes=["USDT", "FDUSD", "BUSD"],
                skip_if_exists=False,
            )
            if result.status == "ok" and result.file and result.file.exists():
                paths[dsym] = result.file
                logger.info(f"  Downloaded {result.bars} bars -> {result.file}")
            else:
                logger.warning(f"  Download failed for {pair}: {result.status} {result.error}")
        except Exception as e:
            logger.error(f"  Download error for {pair}: {e}")
    return paths


# ── Phase B1: LoRA Sweep ────────────────────────────────────────────────────

class FakeConfig:
    def __init__(self, ctx, lr, steps, lora_r=16):
        self.context_length = ctx
        self.prediction_length = 24
        self.batch_size = 32
        self.learning_rate = float(lr)
        self.num_steps = steps
        self.lora_r = lora_r
        self.lora_rank = lora_r
        self.lora_alpha = 32
        self.lora_dropout = 0.05
        self.lora_targets = ("q", "k", "v", "o")
        self.lora_target_modules = "q,v"
        self.finetune_mode = "lora"
        self.merge_lora = True
        self.device_map = "cuda"
        self.torch_dtype = None
        self.seed = 1337


def eval_lora_model(pipeline, aug_df, target_cols, ctx, pred_len=24, n_samples=20):
    data = aug_df[target_cols].to_numpy(dtype=np.float32)
    n = len(data)
    if n < ctx + pred_len + n_samples:
        n_samples = max(1, n - ctx - pred_len)
    step = max(1, (n - ctx - pred_len) // n_samples)
    maes = []
    for i in range(0, n - ctx - pred_len, step):
        inp = data[i:i+ctx].T
        with torch.no_grad():
            preds = pipeline.predict([inp], prediction_length=pred_len, batch_size=1)
        pred_mean = preds[0].mean(dim=0).numpy()
        actual = data[i+ctx:i+ctx+pred_len, 3]  # close column
        mae = np.abs(pred_mean[3] - actual).mean() if pred_mean.ndim > 1 else np.abs(pred_mean - actual).mean()
        pct_mae = mae / (np.abs(actual).mean() + 1e-8) * 100
        maes.append(pct_mae)
    return float(np.mean(maes))


def run_lora_sweep(data_symbol: str, csv_path: Path) -> Optional[Dict[str, Any]]:
    """Sweep 4 preaugs, return best LoRA info."""
    logger.info(f"\n{'='*60}")
    logger.info(f"LoRA sweep: {data_symbol}")
    logger.info(f"{'='*60}")

    df = pd.read_csv(csv_path)
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            logger.error(f"Missing {col} in {csv_path}")
            return None

    val_hours, test_hours = 168, 168
    n = len(df)
    if n < val_hours + test_hours + LORA_CTX + 100:
        logger.warning(f"{data_symbol}: only {n} bars, skipping LoRA")
        return None

    train_df = df.iloc[:n - val_hours - test_hours]
    val_df = df.iloc[n - val_hours - test_hours:n - test_hours]
    test_df = df.iloc[n - test_hours:]
    target_cols = ["open", "high", "low", "close"]

    best_mae = float("inf")
    best_result = None

    for preaug_name in PREAUGS:
        logger.info(f"  {data_symbol} preaug={preaug_name}")
        try:
            augmentation = get_augmentation(preaug_name)
            train_aug = augmentation.transform_dataframe(train_df.copy())
            val_aug = augmentation.transform_dataframe(val_df.copy())
            test_aug = augmentation.transform_dataframe(test_df.copy())

            train_inputs = [{"target": train_aug[target_cols].to_numpy(dtype=np.float32).T}]
            val_inputs = [{"target": val_aug[target_cols].to_numpy(dtype=np.float32).T}]

            run_name = f"{data_symbol}_lora_{preaug_name}_ctx{LORA_CTX}_lr{LORA_LR}_r{LORA_R}"
            output_dir = LORA_OUTPUT_ROOT / run_name
            ckpt_dir = output_dir / "finetuned-ckpt"

            # skip if already done
            if ckpt_dir.exists() and any(ckpt_dir.iterdir()):
                logger.info(f"    already exists, loading for eval")
                pipeline = _load_pipeline(str(ckpt_dir), "cuda", None)
            else:
                cfg = FakeConfig(LORA_CTX, LORA_LR, LORA_STEPS, LORA_R)
                pipeline = _load_pipeline("amazon/chronos-2", "cuda", None)
                pipeline = _fit_pipeline(pipeline, train_inputs, val_inputs, cfg, output_dir)
                _save_pipeline(pipeline, output_dir, "finetuned-ckpt")

            test_mae = eval_lora_model(pipeline, test_aug, target_cols, LORA_CTX)
            logger.info(f"    test_mae={test_mae:.3f}%")

            if test_mae < best_mae:
                best_mae = test_mae
                best_result = {
                    "data_symbol": data_symbol,
                    "preaug": preaug_name,
                    "test_mae_pct": test_mae,
                    "model_path": str(ckpt_dir),
                    "run_name": run_name,
                }

            del pipeline
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"    FAILED: {e}")
            import traceback; traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()

    if best_result:
        logger.info(f"  BEST LoRA for {data_symbol}: {best_result['preaug']} mae={best_result['test_mae_pct']:.3f}%")
    return best_result


# ── Phase B2: Build Forecast Cache ──────────────────────────────────────────

def build_forecast_cache(data_symbol: str, csv_path: Path, model_id: str) -> bool:
    """Build h1 forecast cache for symbol using given model."""
    logger.info(f"Building h1 forecast cache for {data_symbol} with {model_id}")

    h1_dir = FORECAST_CACHE_ROOT / "h1"
    h1_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = h1_dir / f"{data_symbol.upper()}.parquet"

    # skip if already exists, recent, and has good coverage
    price_df = pd.read_csv(csv_path)
    price_rows = len(price_df)
    if "timestamp" in price_df.columns:
        first_ts = pd.to_datetime(price_df["timestamp"]).min()
        if first_ts.tzinfo is None:
            first_ts = first_ts.tz_localize("UTC")
    else:
        first_ts = None
    del price_df
    if parquet_path.exists():
        existing = pd.read_parquet(parquet_path)
        coverage = len(existing) / max(price_rows, 1)
        if len(existing) > 100 and coverage > 0.5:
            logger.info(f"  Cache exists ({len(existing)} rows, {coverage:.0%} coverage), checking freshness")
            last_ts = pd.to_datetime(existing["timestamp"]).max()
            if (pd.Timestamp.now(tz="UTC") - last_ts).days < 7:
                logger.info(f"  Cache fresh enough (last: {last_ts}), skipping")
                return True
        elif coverage <= 0.5:
            logger.info(f"  Cache too small ({len(existing)} rows, {coverage:.0%} coverage), rebuilding")

    try:
        wrapper = Chronos2OHLCWrapper.from_pretrained(
            model_id=model_id,
            device_map="cuda",
            default_context_length=512,
            default_batch_size=128,
            quantile_levels=(0.1, 0.5, 0.9),
        )

        cfg = ForecastConfig(
            symbol=data_symbol,
            data_root=csv_path.parent,
            context_hours=512,
            prediction_horizon_hours=1,
            quantile_levels=(0.1, 0.5, 0.9),
            batch_size=128,
            cache_dir=h1_dir,
        )
        manager = ChronosForecastManager(cfg, wrapper_factory=lambda: wrapper)

        end_ts = pd.Timestamp.now(tz="UTC")
        start_ts = first_ts if first_ts is not None else end_ts - pd.Timedelta(hours=60000)
        manager.ensure_latest(start=start_ts, end=end_ts, cache_only=False, force_rebuild=True)

        logger.info(f"  Cache built: {parquet_path}")
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        logger.error(f"  Cache build FAILED: {e}")
        import traceback; traceback.print_exc()
        gc.collect()
        torch.cuda.empty_cache()
        return False


# ── Phase B3: Policy Training ───────────────────────────────────────────────

def train_policies(data_symbol: str, forecast_cache: Path, horizons=(1,)) -> List[Dict[str, Any]]:
    """Train policy with multiple rw/wd combos, return checkpoint roots."""
    results = []
    for rw in RW_VALUES:
        for wd in WD_VALUES:
            tag = f"{data_symbol}_rw{int(rw*100):02d}_wd{int(wd*100):02d}"
            ckpt_root = CHECKPOINT_ROOT / tag
            logger.info(f"\n--- Training policy: {tag} ---")

            try:
                dm = ChronosSolDataModule(
                    symbol=data_symbol,
                    data_root=DATA_ROOT,
                    forecast_cache_root=forecast_cache,
                    forecast_horizons=horizons,
                    context_hours=512,
                    quantile_levels=(0.1, 0.5, 0.9),
                    batch_size=32,
                    model_id="amazon/chronos-t5-small",
                    sequence_length=POLICY_SEQ,
                    split_config=SplitConfig(val_days=30, test_days=30),
                    cache_only=True,
                    max_history_days=365,
                )

                tc = TrainingConfig(
                    epochs=POLICY_EPOCHS,
                    batch_size=16,
                    sequence_length=POLICY_SEQ,
                    learning_rate=POLICY_LR,
                    weight_decay=wd,
                    return_weight=rw,
                    seed=1337,
                    transformer_dim=POLICY_DIM,
                    transformer_layers=POLICY_LAYERS,
                    transformer_heads=POLICY_HEADS,
                    maker_fee=MAKER_FEE_10BP,
                    fill_temperature=FILL_TEMP,
                    fill_buffer_pct=FILL_BUFFER,
                    checkpoint_root=ckpt_root,
                    log_dir=Path("tensorboard_logs/sweep_20"),
                    use_compile=False,
                )

                trainer = BinanceHourlyTrainer(tc, dm)
                trainer.train()
                results.append({
                    "tag": tag, "rw": rw, "wd": wd,
                    "ckpt_root": str(ckpt_root),
                    "dm_symbol": data_symbol,
                })
                logger.info(f"  {tag} done, checkpoints at {ckpt_root}")

            except Exception as e:
                logger.error(f"  {tag} FAILED: {e}")
                import traceback; traceback.print_exc()

            gc.collect()
            torch.cuda.empty_cache()

    return results


# ── Phase B4: Epoch Evaluation ──────────────────────────────────────────────

def evaluate_all_epochs(
    data_symbol: str,
    policy_results: List[Dict[str, Any]],
    forecast_cache: Path,
    horizons=(1,),
) -> List[Dict[str, Any]]:
    """Evaluate all checkpoints with prod-matching simulator."""
    all_evals = []

    for pr in policy_results:
        tag = pr["tag"]
        ckpt_root = Path(pr["ckpt_root"])
        rw, wd = pr["rw"], pr["wd"]

        logger.info(f"\nEvaluating {tag}")

        try:
            dm = ChronosSolDataModule(
                symbol=data_symbol,
                data_root=DATA_ROOT,
                forecast_cache_root=forecast_cache,
                forecast_horizons=horizons,
                context_hours=512,
                quantile_levels=(0.1, 0.5, 0.9),
                batch_size=32,
                model_id="amazon/chronos-t5-small",
                sequence_length=POLICY_SEQ,
                split_config=SplitConfig(val_days=30, test_days=30),
                cache_only=True,
                max_history_days=365,
            )
        except Exception as e:
            logger.error(f"  DataModule failed: {e}")
            continue

        ckpt_files = sorted(ckpt_root.rglob("epoch_*.pt"))
        if not ckpt_files:
            p = ckpt_root / "policy_checkpoint.pt"
            if p.exists():
                ckpt_files = [p]

        feature_columns = list(dm.feature_columns)
        normalizer = dm.normalizer
        test_frame = dm.test_frame.copy()
        test_start = dm.test_window_start

        for ckpt_path in ckpt_files:
            ep_name = ckpt_path.stem
            try:
                payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                sd = payload.get("state_dict", payload)
                cfg = payload.get("config", {})
                model = _build_policy(sd, cfg, len(feature_columns))

                actions = generate_actions_from_frame(
                    model=model, frame=test_frame, feature_columns=feature_columns,
                    normalizer=normalizer, sequence_length=POLICY_SEQ, horizon=horizons[0],
                )
                bars = test_frame[test_frame["timestamp"] >= test_start].copy()
                actions_test = actions[actions["timestamp"] >= test_start].copy()

                lcfg = LeverageConfig(
                    max_leverage=1.0, initial_cash=EVAL_INITIAL_CASH,
                    decision_lag_bars=EVAL_LAG, fill_buffer_pct=FILL_BUFFER,
                    margin_hourly_rate=0.0, maker_fee=MAKER_FEE_10BP,
                    min_edge=0.0, max_hold_bars=EVAL_MAX_HOLD,
                    intensity_scale=EVAL_INTENSITY,
                )
                r = simulate_with_margin_cost(bars, actions_test, lcfg)

                dd = abs(r["max_drawdown"]) if r["max_drawdown"] != 0 else 1e-6
                calmar = r["total_return"] / dd
                # composite: penalize high drawdown heavily
                composite = r["sortino"] * (1 - min(dd, 1.0))

                entry = {
                    "symbol": data_symbol, "rw": rw, "wd": wd,
                    "epoch": ep_name, "tag": tag,
                    "sortino": r["sortino"], "total_return": r["total_return"],
                    "max_drawdown": r["max_drawdown"], "num_trades": r["num_trades"],
                    "final_equity": r["final_equity"],
                    "calmar": calmar, "composite": composite,
                }
                all_evals.append(entry)

                logger.info(
                    f"  {tag} {ep_name}: Sort={r['sortino']:.2f} "
                    f"Ret={r['total_return']*100:+.1f}% "
                    f"DD={r['max_drawdown']*100:.1f}% T={r['num_trades']}"
                )

                del model
            except Exception as e:
                logger.warning(f"  {tag} {ep_name}: FAILED {e}")

        gc.collect()
        torch.cuda.empty_cache()

    return all_evals


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="20-symbol Binance sweep")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-sep symbols (default: all 20)")
    parser.add_argument("--skip-lora", action="store_true", help="Skip LoRA sweep")
    parser.add_argument("--skip-cache", action="store_true", help="Skip forecast cache build")
    parser.add_argument("--skip-train", action="store_true", help="Skip policy training")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download")
    parser.add_argument("--only-eval", action="store_true", help="Only run evaluation on existing checkpoints")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else DEFAULT_SYMBOLS

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "sweep_20_results.json"
    all_results: Dict[str, Any] = {}
    if results_path.exists():
        all_results = json.loads(results_path.read_text())

    t0 = time.time()

    # Phase A: data download
    if not args.skip_download and not args.only_eval:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE A: Data Download")
        logger.info("=" * 70)
        data_paths = ensure_data(symbols)
    else:
        data_paths = {}
        for pair in symbols:
            dsym = pair_to_data_symbol(pair)
            csv_path = DATA_ROOT / f"{dsym}.csv"
            if csv_path.exists():
                data_paths[dsym] = csv_path

    # Phase B: per-symbol pipeline
    for pair in symbols:
        dsym = pair_to_data_symbol(pair)
        csv_path = data_paths.get(dsym)
        if csv_path is None or not csv_path.exists():
            logger.warning(f"No data for {dsym}, skipping")
            continue

        sym_key = dsym
        if sym_key not in all_results:
            all_results[sym_key] = {}

        logger.info(f"\n{'#' * 70}")
        logger.info(f"# SYMBOL: {dsym} ({pair})")
        logger.info(f"{'#' * 70}")

        # B1: LoRA sweep
        if not args.skip_lora and not args.only_eval:
            lora_result = run_lora_sweep(dsym, csv_path)
            if lora_result:
                all_results[sym_key]["lora"] = lora_result
                _save_results(results_path, all_results)
        else:
            lora_result = all_results.get(sym_key, {}).get("lora")

        # determine model_id for forecast cache
        if lora_result and "model_path" in lora_result:
            model_id = lora_result["model_path"]
        else:
            model_id = "amazon/chronos-2"

        # B2: forecast cache
        forecast_cache = FORECAST_CACHE_ROOT
        if not args.skip_cache and not args.only_eval:
            ok = build_forecast_cache(dsym, csv_path, model_id)
            if not ok:
                logger.error(f"Forecast cache failed for {dsym}, skipping policy training")
                continue

        # B3: policy training
        if not args.skip_train and not args.only_eval:
            policy_results = train_policies(dsym, forecast_cache)
            all_results[sym_key]["policies"] = policy_results
            _save_results(results_path, all_results)
        else:
            policy_results = all_results.get(sym_key, {}).get("policies", [])
            if not policy_results:
                # reconstruct from existing checkpoints
                policy_results = []
                for rw in RW_VALUES:
                    for wd in WD_VALUES:
                        tag = f"{dsym}_rw{int(rw*100):02d}_wd{int(wd*100):02d}"
                        ckpt_root = CHECKPOINT_ROOT / tag
                        if ckpt_root.exists():
                            policy_results.append({
                                "tag": tag, "rw": rw, "wd": wd,
                                "ckpt_root": str(ckpt_root), "dm_symbol": dsym,
                            })

        # B4: evaluation
        if not args.skip_eval:
            if policy_results:
                evals = evaluate_all_epochs(dsym, policy_results, forecast_cache)
                all_results[sym_key]["evals"] = evals
                _save_results(results_path, all_results)

    # Phase C: summary
    elapsed = time.time() - t0
    logger.info(f"\n{'=' * 70}")
    logger.info(f"SWEEP COMPLETE ({elapsed/3600:.1f}h)")
    logger.info(f"{'=' * 70}")

    print_summary(all_results)
    _save_results(results_path, all_results)
    logger.info(f"Results: {results_path}")


def _save_results(path: Path, results: Dict):
    path.write_text(json.dumps(results, indent=2, default=str))


def print_summary(all_results: Dict[str, Any]):
    rows = []
    for sym, data in all_results.items():
        evals = data.get("evals", [])
        if not evals:
            continue
        # best by composite (sortino * (1-dd))
        best = max(evals, key=lambda e: e.get("composite", e.get("sortino", -999)))
        rows.append(best)

    rows.sort(key=lambda r: r.get("composite", r.get("sortino", -999)), reverse=True)

    print(f"\n{'Symbol':<10} {'RW':>4} {'WD':>5} {'Epoch':<12} {'Sortino':>8} {'Return':>8} {'MaxDD':>8} {'Calmar':>8} {'Comp':>8} {'Trades':>7}")
    print("-" * 85)
    for r in rows:
        calmar = r.get("calmar", 0)
        comp = r.get("composite", r.get("sortino", 0))
        print(
            f"{r['symbol']:<10} {r['rw']:>4.2f} {r['wd']:>5.3f} {r['epoch']:<12} "
            f"{r['sortino']:>8.2f} {r['total_return']*100:>7.1f}% "
            f"{r['max_drawdown']*100:>7.1f}% {calmar:>8.2f} {comp:>8.2f} {r['num_trades']:>7}"
        )

    if rows:
        print(f"\nTop 5 by composite (Sort*(1-DD)):")
        for i, r in enumerate(rows[:5]):
            comp = r.get("composite", r.get("sortino", 0))
            print(f"  {i+1}. {r['symbol']} rw={r['rw']} wd={r['wd']} ep={r['epoch']} "
                  f"Comp={comp:.2f} Sort={r['sortino']:.2f} DD={r['max_drawdown']*100:.1f}% Ret={r['total_return']*100:+.1f}%")


if __name__ == "__main__":
    main()
