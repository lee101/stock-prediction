#!/usr/bin/env python3
"""
Autonomous multivariate group forecasting sweep for Chronos2.

Tests which groups of symbols benefit from joint (cross-learning) forecasting,
retunes hyperparameters (context_length, batch_size) per group, and evaluates
pre-augmentation strategies within joint groups.

Three axes of experimentation:
  1. Symbol groupings: correlation-based, sector-based, asset-class, manual
  2. Hyperparameters: context_length, batch_size swept per group
  3. Pre-augmentation: all strategies tested per group

Results are written to multivariate_group_results/ and the best configs
are persisted to reports/joint_forecast_config.json for production use.

Usage:
  # Full autonomous sweep (runs continuously)
  python sweep_multivariate_groups.py --autonomous

  # Test specific groups
  python sweep_multivariate_groups.py --groups "BTCUSD,ETHUSD,SOLUSD" "AAPL,MSFT,GOOGL"

  # Correlation-discovery mode
  python sweep_multivariate_groups.py --discover-groups --top-k 20

  # Crypto only
  python sweep_multivariate_groups.py --asset-class crypto --autonomous

  # Stocks only with specific sectors
  python sweep_multivariate_groups.py --asset-class stocks --autonomous
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_chronos2 import (
    TARGET_COLUMNS,
    VAL_WINDOW,
    TEST_WINDOW,
    _prepare_series,
    _window_indices,
)
from kronostraining.metrics_utils import compute_mae_percent
from preaug_sweeps.augmentations import AUGMENTATION_REGISTRY, get_augmentation
from sklearn.metrics import mean_absolute_error
from src.models.chronos2_postprocessing import (
    Chronos2AggregationSpec,
    ColumnScaler,
    aggregate_quantile_forecasts,
    resolve_quantile_levels,
)
from src.models.chronos2_wrapper import Chronos2OHLCWrapper, DEFAULT_QUANTILE_LEVELS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
STOCK_DATA_DIR = REPO_ROOT / "trainingdatahourly" / "stocks"
CRYPTO_DATA_DIR = REPO_ROOT / "trainingdatahourly" / "crypto"
RESULTS_DIR = REPO_ROOT / "multivariate_group_results"
CONFIG_OUTPUT = REPO_ROOT / "reports" / "joint_forecast_config.json"
LEADERBOARD_PATH = RESULTS_DIR / "leaderboard.csv"

# ---------------------------------------------------------------------------
# Predefined sector groups (stocks)
# ---------------------------------------------------------------------------
SECTOR_GROUPS = {
    "mega_tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
    "semis": ["NVDA", "AMD", "INTC", "AVGO", "QCOM", "AMAT", "ASML"],
    "cloud_saas": ["CRM", "ADBE", "NOW", "SNOW", "PLTR", "NET", "CRWD"],
    "fintech": ["V", "MA", "PYPL", "SQ", "COIN", "AXP", "JPM"],
    "banks": ["JPM", "BAC", "GS", "MS", "WFC", "C", "USB"],
    "healthcare": ["UNH", "JNJ", "LLY", "ABBV", "PFE", "MRK", "ABT"],
    "consumer": ["WMT", "COST", "HD", "TGT", "LOW", "SBUX", "MCD"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX"],
    "industrials": ["CAT", "DE", "HON", "GE", "MMM", "UPS", "RTX"],
    "reits": ["AMT", "PLD", "EQIX", "SPG", "O", "WELL", "DLR"],
}

CRYPTO_GROUPS = {
    "majors": ["BTCUSD", "ETHUSD"],
    "majors_extended": ["BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD"],
    "alt_l1": ["SOLUSD", "AVAXUSD", "ADAUSD", "DOTUSD", "ATOMUSD"],
    "defi": ["AAVEUSD", "LINKUSD", "UNIUSD"],
    "btc_eth_sol": ["BTCUSD", "ETHUSD", "SOLUSD"],
    "full_crypto": ["BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD",
                    "DOGEUSD", "ADAUSD", "LINKUSD", "DOTUSD", "BNBUSD"],
}

# Cross-asset groups
CROSS_ASSET_GROUPS = {
    "crypto_plus_coin": ["BTCUSD", "ETHUSD", "COIN"],
    "tech_plus_btc": ["AAPL", "MSFT", "NVDA", "BTCUSD"],
}

# Hyperparameter grid
CONTEXT_LENGTHS = [256, 512, 1024, 2048, 4096]
BATCH_SIZES = [64, 128, 256, 512]
AGGREGATIONS = ["median"]
SCALERS = ["none", "meanstd"]

PREAUG_STRATEGIES = list(AUGMENTATION_REGISTRY.keys())


@dataclass
class GroupExperiment:
    """A single experiment configuration."""
    group_name: str
    symbols: List[str]
    context_length: int
    batch_size: int
    predict_jointly: bool
    aggregation: str = "median"
    scaler: str = "none"
    preaug: str = "none"


@dataclass
class GroupResult:
    """Result of evaluating a group experiment."""
    group_name: str
    symbols: List[str]
    context_length: int
    batch_size: int
    predict_jointly: bool
    aggregation: str
    scaler: str
    preaug: str
    # Per-symbol metrics (val)
    val_mae_percent: Dict[str, float] = field(default_factory=dict)
    val_pct_return_mae: Dict[str, float] = field(default_factory=dict)
    # Per-symbol metrics (test)
    test_mae_percent: Dict[str, float] = field(default_factory=dict)
    test_pct_return_mae: Dict[str, float] = field(default_factory=dict)
    # Aggregate
    mean_val_mae_pct: float = 0.0
    mean_test_mae_pct: float = 0.0
    mean_val_pct_return_mae: float = 0.0
    mean_test_pct_return_mae: float = 0.0
    # Comparison to independent baseline
    improvement_vs_independent_pct: float = 0.0
    latency_s: float = 0.0
    timestamp: str = ""


def load_symbol_data(symbol: str) -> Optional[pd.DataFrame]:
    """Load hourly OHLC data for a symbol."""
    for data_dir in [STOCK_DATA_DIR, CRYPTO_DATA_DIR]:
        path = data_dir / f"{symbol}.csv"
        if path.exists():
            try:
                return _prepare_series(path)
            except Exception as e:
                logger.warning("Failed to load %s: %s", symbol, e)
    return None


def compute_correlation_matrix(symbols: List[str], lookback: int = 500) -> pd.DataFrame:
    """Compute pairwise return correlation between symbols."""
    returns = {}
    for symbol in symbols:
        df = load_symbol_data(symbol)
        if df is not None and len(df) > lookback:
            close = df["close"].iloc[-lookback:]
            returns[symbol] = close.pct_change().dropna()

    if not returns:
        return pd.DataFrame()

    return_df = pd.DataFrame(returns)
    # Align on common timestamps
    return_df = return_df.dropna()
    if len(return_df) < 50:
        return pd.DataFrame()
    return return_df.corr()


def discover_correlated_groups(
    symbols: List[str],
    min_correlation: float = 0.5,
    max_group_size: int = 6,
    top_k: int = 20,
) -> List[Tuple[str, List[str]]]:
    """Discover groups of correlated symbols."""
    corr_matrix = compute_correlation_matrix(symbols)
    if corr_matrix.empty:
        return []

    # Find highly correlated pairs
    pairs = []
    for i, s1 in enumerate(corr_matrix.columns):
        for j, s2 in enumerate(corr_matrix.columns):
            if i < j:
                corr = corr_matrix.iloc[i, j]
                if abs(corr) >= min_correlation:
                    pairs.append((s1, s2, corr))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    # Build groups via greedy clustering
    groups = []
    used = set()
    for s1, s2, corr in pairs[:top_k * 3]:
        if s1 in used and s2 in used:
            continue

        # Try to extend existing group
        extended = False
        for gname, members in groups:
            if len(members) >= max_group_size:
                continue
            if s1 in members and s2 not in members:
                # Check correlation with all existing members
                all_corr = [abs(corr_matrix.loc[s2, m]) for m in members if m in corr_matrix.columns]
                if all_corr and min(all_corr) >= min_correlation * 0.8:
                    members.append(s2)
                    used.add(s2)
                    extended = True
                    break
            elif s2 in members and s1 not in members:
                all_corr = [abs(corr_matrix.loc[s1, m]) for m in members if m in corr_matrix.columns]
                if all_corr and min(all_corr) >= min_correlation * 0.8:
                    members.append(s1)
                    used.add(s1)
                    extended = True
                    break

        if not extended:
            name = f"corr_{len(groups):03d}_{s1}_{s2}"
            groups.append((name, [s1, s2]))
            used.add(s1)
            used.add(s2)

        if len(groups) >= top_k:
            break

    return groups[:top_k]


def evaluate_group(
    wrapper: Chronos2OHLCWrapper,
    experiment: GroupExperiment,
    symbol_data: Dict[str, pd.DataFrame],
    val_window: int = VAL_WINDOW,
    test_window: int = TEST_WINDOW,
    rng: Optional[np.random.Generator] = None,
) -> Optional[GroupResult]:
    """Evaluate a group experiment (joint vs independent)."""
    if rng is None:
        rng = np.random.default_rng(42)

    result = GroupResult(
        group_name=experiment.group_name,
        symbols=experiment.symbols,
        context_length=experiment.context_length,
        batch_size=experiment.batch_size,
        predict_jointly=experiment.predict_jointly,
        aggregation=experiment.aggregation,
        scaler=experiment.scaler,
        preaug=experiment.preaug,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    # Get augmentation
    aug = get_augmentation(experiment.preaug) if experiment.preaug != "none" else None

    total_latency = 0.0

    for split_name, window_size in [("val", val_window), ("test", test_window)]:
        for symbol in experiment.symbols:
            df = symbol_data.get(symbol)
            if df is None:
                continue

            try:
                val_indices, test_indices = _window_indices(len(df), val_window, test_window)
            except ValueError:
                logger.warning("Skipping %s: insufficient data", symbol)
                continue

            indices = val_indices if split_name == "val" else test_indices
            preds = []
            actuals = []
            actual_returns = []
            pred_returns = []

            for idx in indices:
                ctx_start = max(0, idx - experiment.context_length)
                context_df = df.iloc[ctx_start:idx].copy()
                if context_df.empty:
                    continue

                # Apply pre-augmentation
                if aug is not None:
                    try:
                        context_df = aug.transform(context_df)
                    except Exception:
                        pass  # Fall back to raw

                scaler_obj = ColumnScaler(experiment.scaler, context_df[list(TARGET_COLUMNS)], TARGET_COLUMNS)
                transformed = scaler_obj.transform(context_df)

                start_t = time.perf_counter()

                if experiment.predict_jointly:
                    # Build contexts for all symbols in the group
                    joint_contexts = []
                    joint_symbols = []
                    for jsym in experiment.symbols:
                        jdf = symbol_data.get(jsym)
                        if jdf is None:
                            continue
                        jctx_start = max(0, idx - experiment.context_length)
                        jctx_end = min(idx, len(jdf))
                        if jctx_end <= jctx_start:
                            continue
                        jctx = jdf.iloc[jctx_start:jctx_end].copy()
                        if aug is not None:
                            try:
                                jctx = aug.transform(jctx)
                            except Exception:
                                pass
                        joint_contexts.append(jctx)
                        joint_symbols.append(jsym)

                    if len(joint_contexts) < 2:
                        # Not enough data for joint, fall back
                        batch = wrapper.predict_ohlc(
                            transformed,
                            symbol=symbol,
                            prediction_length=1,
                            context_length=experiment.context_length,
                            batch_size=experiment.batch_size,
                        )
                    else:
                        try:
                            results_list = wrapper.predict_ohlc_joint(
                                joint_contexts,
                                symbols=joint_symbols,
                                prediction_length=1,
                                context_length=experiment.context_length,
                                predict_batches_jointly=True,
                                batch_size=experiment.batch_size,
                            )
                            # Find result for current symbol
                            batch = None
                            for r in results_list:
                                if r.panel.symbol == symbol:
                                    batch = r
                                    break
                            if batch is None:
                                batch = wrapper.predict_ohlc(
                                    transformed,
                                    symbol=symbol,
                                    prediction_length=1,
                                    context_length=experiment.context_length,
                                    batch_size=experiment.batch_size,
                                )
                        except Exception as e:
                            logger.warning("Joint prediction failed: %s, falling back", e)
                            batch = wrapper.predict_ohlc(
                                transformed,
                                symbol=symbol,
                                prediction_length=1,
                                context_length=experiment.context_length,
                                batch_size=experiment.batch_size,
                            )
                else:
                    batch = wrapper.predict_ohlc(
                        transformed,
                        symbol=symbol,
                        prediction_length=1,
                        context_length=experiment.context_length,
                        batch_size=experiment.batch_size,
                    )

                total_latency += time.perf_counter() - start_t

                # Extract close prediction
                q_frames = {
                    level: scaler_obj.inverse(batch.quantile(level))
                    for level in DEFAULT_QUANTILE_LEVELS
                }
                spec = Chronos2AggregationSpec(
                    aggregation=experiment.aggregation,
                    sample_count=0,
                    scaler="none",
                )
                aggregated = aggregate_quantile_forecasts(
                    q_frames, columns=("close",), spec=spec, rng=rng,
                )
                price_pred = float(aggregated.get("close", np.nan))
                preds.append(price_pred)

                prev_price = float(df["close"].iloc[idx - 1])
                actual_price = float(df["close"].iloc[idx])
                actuals.append(actual_price)

                if prev_price != 0:
                    pred_returns.append((price_pred - prev_price) / prev_price)
                    actual_returns.append((actual_price - prev_price) / prev_price)
                else:
                    pred_returns.append(0.0)
                    actual_returns.append(0.0)

            if not preds:
                continue

            mae_pct = compute_mae_percent(
                mean_absolute_error(actuals, preds),
                np.array(actuals, dtype=np.float64),
            )
            pct_ret_mae = mean_absolute_error(actual_returns, pred_returns)

            if split_name == "val":
                result.val_mae_percent[symbol] = mae_pct
                result.val_pct_return_mae[symbol] = pct_ret_mae
            else:
                result.test_mae_percent[symbol] = mae_pct
                result.test_pct_return_mae[symbol] = pct_ret_mae

    result.latency_s = total_latency

    # Compute aggregates
    if result.val_mae_percent:
        result.mean_val_mae_pct = np.mean(list(result.val_mae_percent.values()))
    if result.test_mae_percent:
        result.mean_test_mae_pct = np.mean(list(result.test_mae_percent.values()))
    if result.val_pct_return_mae:
        result.mean_val_pct_return_mae = np.mean(list(result.val_pct_return_mae.values()))
    if result.test_pct_return_mae:
        result.mean_test_pct_return_mae = np.mean(list(result.test_pct_return_mae.values()))

    return result


def run_group_comparison(
    wrapper: Chronos2OHLCWrapper,
    group_name: str,
    symbols: List[str],
    symbol_data: Dict[str, pd.DataFrame],
    context_lengths: List[int],
    batch_sizes: List[int],
    preaugs: List[str],
    scalers: List[str],
    rng: Optional[np.random.Generator] = None,
) -> List[GroupResult]:
    """Run joint vs independent comparison for a symbol group."""
    results = []

    for ctx_len in context_lengths:
        for batch_size in batch_sizes:
            for scaler in scalers:
                for preaug in preaugs:
                    for jointly in [False, True]:
                        exp = GroupExperiment(
                            group_name=group_name,
                            symbols=symbols,
                            context_length=ctx_len,
                            batch_size=batch_size,
                            predict_jointly=jointly,
                            scaler=scaler,
                            preaug=preaug,
                        )

                        tag = "joint" if jointly else "indep"
                        print(
                            f"  [{group_name}] {tag} ctx={ctx_len} bs={batch_size} "
                            f"scaler={scaler} preaug={preaug}...",
                            end="",
                            flush=True,
                        )

                        try:
                            result = evaluate_group(wrapper, exp, symbol_data, rng=rng)
                            if result is not None:
                                results.append(result)
                                # Save incrementally after each experiment
                                append_to_leaderboard([result])
                                print(
                                    f" val_mae={result.mean_val_mae_pct:.4f}% "
                                    f"test_mae={result.mean_test_mae_pct:.4f}% "
                                    f"({result.latency_s:.1f}s)"
                                )
                            else:
                                print(" SKIPPED")
                        except Exception as e:
                            print(f" ERROR: {e}")
                            logger.exception("Failed experiment %s", exp)

    # Compute improvement vs independent baseline
    indep_results = [r for r in results if not r.predict_jointly]
    joint_results = [r for r in results if r.predict_jointly]

    if indep_results and joint_results:
        best_indep = min(indep_results, key=lambda r: r.mean_val_mae_pct)
        for jr in joint_results:
            if best_indep.mean_val_mae_pct > 0:
                jr.improvement_vs_independent_pct = (
                    (best_indep.mean_val_mae_pct - jr.mean_val_mae_pct)
                    / best_indep.mean_val_mae_pct * 100
                )

    return results


def append_to_leaderboard(results: List[GroupResult]) -> None:
    """Append results to the CSV leaderboard."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        rows.append({
            "timestamp": r.timestamp,
            "group_name": r.group_name,
            "symbols": ",".join(r.symbols),
            "n_symbols": len(r.symbols),
            "predict_jointly": r.predict_jointly,
            "context_length": r.context_length,
            "batch_size": r.batch_size,
            "scaler": r.scaler,
            "preaug": r.preaug,
            "mean_val_mae_pct": r.mean_val_mae_pct,
            "mean_test_mae_pct": r.mean_test_mae_pct,
            "mean_val_pct_return_mae": r.mean_val_pct_return_mae,
            "mean_test_pct_return_mae": r.mean_test_pct_return_mae,
            "improvement_vs_independent_pct": r.improvement_vs_independent_pct,
            "latency_s": r.latency_s,
        })

    new_df = pd.DataFrame(rows)
    if LEADERBOARD_PATH.exists():
        existing = pd.read_csv(LEADERBOARD_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(LEADERBOARD_PATH, index=False)


def save_best_configs(all_results: List[GroupResult]) -> None:
    """Save the best joint forecast configurations to reports/."""
    if not all_results:
        return

    joint_results = [r for r in all_results if r.predict_jointly and r.mean_val_mae_pct > 0]
    if not joint_results:
        return

    # Group by group_name, find best per group
    by_group: Dict[str, List[GroupResult]] = {}
    for r in joint_results:
        by_group.setdefault(r.group_name, []).append(r)

    configs = {}
    for group_name, group_results in by_group.items():
        # Also get best independent for comparison
        indep = [r for r in all_results if r.group_name == group_name and not r.predict_jointly]
        best_indep_mae = min((r.mean_val_mae_pct for r in indep), default=float("inf"))

        best = min(group_results, key=lambda r: r.mean_val_mae_pct)
        improvement = (best_indep_mae - best.mean_val_mae_pct) / best_indep_mae * 100 if best_indep_mae > 0 else 0

        configs[group_name] = {
            "symbols": best.symbols,
            "context_length": best.context_length,
            "batch_size": best.batch_size,
            "scaler": best.scaler,
            "preaug": best.preaug,
            "mean_val_mae_pct": best.mean_val_mae_pct,
            "mean_test_mae_pct": best.mean_test_mae_pct,
            "improvement_vs_independent_pct": improvement,
            "use_joint_forecasting": improvement > 0,
        }

    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_groups": len(configs),
        "groups": configs,
    }

    CONFIG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_OUTPUT.write_text(json.dumps(output, indent=2))
    print(f"\n[INFO] Saved best joint configs -> {CONFIG_OUTPUT}")


def get_all_available_symbols(asset_class: str = "all") -> List[str]:
    """Get all symbols with available data."""
    symbols = []
    if asset_class in ("all", "stocks"):
        for p in sorted(STOCK_DATA_DIR.glob("*.csv")):
            sym = p.stem
            if "," not in sym and len(sym) <= 6:
                symbols.append(sym)
    if asset_class in ("all", "crypto"):
        for p in sorted(CRYPTO_DATA_DIR.glob("*.csv")):
            sym = p.stem
            if "," not in sym and sym.endswith("USD"):
                symbols.append(sym)
    return symbols


def build_experiment_groups(
    args: argparse.Namespace,
) -> List[Tuple[str, List[str]]]:
    """Build the list of symbol groups to test."""
    groups = []

    if args.groups:
        for i, g in enumerate(args.groups):
            syms = [s.strip() for s in g.split(",")]
            groups.append((f"manual_{i:03d}", syms))
        return groups

    if args.discover_groups:
        available = get_all_available_symbols(args.asset_class)
        discovered = discover_correlated_groups(
            available,
            min_correlation=args.min_correlation,
            max_group_size=args.max_group_size,
            top_k=args.top_k,
        )
        groups.extend(discovered)
        print(f"Discovered {len(discovered)} correlated groups")

    if args.asset_class in ("all", "crypto"):
        for name, syms in CRYPTO_GROUPS.items():
            groups.append((f"crypto_{name}", syms))

    if args.asset_class in ("all", "stocks"):
        for name, syms in SECTOR_GROUPS.items():
            groups.append((f"sector_{name}", syms))

    if args.asset_class == "all":
        for name, syms in CROSS_ASSET_GROUPS.items():
            groups.append((f"cross_{name}", syms))

    # Generate intra-sector pairs (pairs from within each sector)
    if args.discover_pairs:
        all_group_defs = {}
        if args.asset_class in ("all", "stocks"):
            all_group_defs.update(SECTOR_GROUPS)
        if args.asset_class in ("all", "crypto"):
            all_group_defs.update(CRYPTO_GROUPS)
        for gname, syms in all_group_defs.items():
            for s1, s2 in itertools.combinations(syms[:5], 2):  # limit combos
                groups.append((f"pair_{gname}_{s1}_{s2}", [s1, s2]))
            # Also test triplets
            for combo in itertools.combinations(syms[:5], 3):
                groups.append((f"trio_{gname}_{'_'.join(combo)}", list(combo)))

    return groups


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--groups", nargs="*", help="Comma-separated symbol groups to test")
    p.add_argument("--asset-class", choices=["all", "stocks", "crypto"], default="all")
    p.add_argument("--discover-groups", action="store_true", help="Auto-discover correlated groups")
    p.add_argument("--discover-pairs", action="store_true", help="Also test sequential pairs")
    p.add_argument("--min-correlation", type=float, default=0.5, help="Min correlation for group discovery")
    p.add_argument("--max-group-size", type=int, default=6, help="Max symbols per discovered group")
    p.add_argument("--top-k", type=int, default=20, help="Top K groups to discover")
    p.add_argument("--context-lengths", type=int, nargs="+", default=[512, 1024])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[128, 256])
    p.add_argument("--scalers", nargs="+", default=["none", "meanstd"])
    p.add_argument("--preaugs", nargs="+", default=["none"],
                   help="Pre-augmentation strategies to sweep")
    p.add_argument("--sweep-all-preaugs", action="store_true",
                   help="Test all available pre-augmentation strategies")
    p.add_argument("--model-id", default="amazon/chronos-2")
    p.add_argument("--device-map", default="cuda")
    p.add_argument("--autonomous", action="store_true", help="Run continuously, expanding search")
    p.add_argument("--max-experiments", type=int, default=0, help="Max experiments (0=unlimited)")
    p.add_argument("--skip-completed", action="store_true",
                   help="Skip groups already in leaderboard")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.sweep_all_preaugs:
        args.preaugs = list(AUGMENTATION_REGISTRY.keys())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 70)
    print("CHRONOS2 MULTIVARIATE GROUP FORECASTING SWEEP")
    print("=" * 70)

    # Build groups
    groups = build_experiment_groups(args)
    print(f"\nTotal groups to evaluate: {len(groups)}")
    for name, syms in groups:
        print(f"  {name}: {', '.join(syms)}")

    # Load data for all needed symbols
    all_symbols = set()
    for _, syms in groups:
        all_symbols.update(syms)

    print(f"\nLoading data for {len(all_symbols)} symbols...")
    symbol_data: Dict[str, pd.DataFrame] = {}
    for sym in sorted(all_symbols):
        df = load_symbol_data(sym)
        if df is not None:
            symbol_data[sym] = df
        else:
            print(f"  WARNING: No data for {sym}")

    # Filter groups to only include symbols with data
    valid_groups = []
    for name, syms in groups:
        valid_syms = [s for s in syms if s in symbol_data]
        if len(valid_syms) >= 2:
            valid_groups.append((name, valid_syms))
        else:
            print(f"  Skipping {name}: only {len(valid_syms)} symbols have data")

    print(f"\nValid groups: {len(valid_groups)}")

    # Initialize wrapper
    print(f"\nLoading Chronos2 model ({args.model_id})...")
    wrapper = Chronos2OHLCWrapper.from_pretrained(
        model_id=args.model_id,
        device_map=args.device_map,
        target_columns=TARGET_COLUMNS,
        default_context_length=max(args.context_lengths),
        quantile_levels=DEFAULT_QUANTILE_LEVELS,
    )

    all_results: List[GroupResult] = []
    experiment_count = 0

    # Load completed groups for skip logic
    completed_groups: set = set()
    if args.skip_completed and LEADERBOARD_PATH.exists():
        try:
            lb = pd.read_csv(LEADERBOARD_PATH)
            completed_groups = set(lb["group_name"].unique())
            print(f"Skipping {len(completed_groups)} already-completed groups")
        except Exception:
            pass

    iteration = 0
    while True:
        iteration += 1
        print(f"\n{'=' * 70}")
        print(f"ITERATION {iteration}")
        print(f"{'=' * 70}")

        for group_name, symbols in valid_groups:
            if group_name in completed_groups:
                continue
            print(f"\n--- Group: {group_name} ({', '.join(symbols)}) ---")

            group_results = run_group_comparison(
                wrapper=wrapper,
                group_name=group_name,
                symbols=symbols,
                symbol_data=symbol_data,
                context_lengths=args.context_lengths,
                batch_sizes=args.batch_sizes,
                preaugs=args.preaugs,
                scalers=args.scalers,
                rng=rng,
            )

            all_results.extend(group_results)
            experiment_count += len(group_results)

            # Print summary for this group
            joint = [r for r in group_results if r.predict_jointly]
            indep = [r for r in group_results if not r.predict_jointly]
            if joint and indep:
                best_joint = min(joint, key=lambda r: r.mean_val_mae_pct)
                best_indep = min(indep, key=lambda r: r.mean_val_mae_pct)
                improvement = (best_indep.mean_val_mae_pct - best_joint.mean_val_mae_pct) / best_indep.mean_val_mae_pct * 100 if best_indep.mean_val_mae_pct > 0 else 0
                verdict = "JOINT WINS" if improvement > 0 else "INDEPENDENT WINS"
                print(
                    f"\n  >> {verdict}: joint={best_joint.mean_val_mae_pct:.4f}% "
                    f"vs indep={best_indep.mean_val_mae_pct:.4f}% "
                    f"({improvement:+.2f}% improvement)"
                )

            if args.max_experiments and experiment_count >= args.max_experiments:
                break

        # Save best configs
        save_best_configs(all_results)

        # Print overall leaderboard
        print(f"\n{'=' * 70}")
        print(f"LEADERBOARD (after {experiment_count} experiments)")
        print(f"{'=' * 70}")
        joint_sorted = sorted(
            [r for r in all_results if r.predict_jointly],
            key=lambda r: r.mean_val_mae_pct,
        )
        for i, r in enumerate(joint_sorted[:20]):
            print(
                f"  {i+1}. {r.group_name} ctx={r.context_length} bs={r.batch_size} "
                f"preaug={r.preaug} scaler={r.scaler}: "
                f"val={r.mean_val_mae_pct:.4f}% test={r.mean_test_mae_pct:.4f}% "
                f"improvement={r.improvement_vs_independent_pct:+.2f}%"
            )

        if not args.autonomous:
            break

        if args.max_experiments and experiment_count >= args.max_experiments:
            print(f"\nReached max experiments ({args.max_experiments}). Stopping.")
            break

        # In autonomous mode, expand search space
        print("\n[AUTONOMOUS] Expanding search space for next iteration...")

        # Add more context lengths
        if 2048 not in args.context_lengths:
            args.context_lengths.append(2048)
        if 4096 not in args.context_lengths:
            args.context_lengths.append(4096)

        # Add more scalers
        if "mean_std" not in args.scalers:
            args.scalers.append("mean_std")

        # Try more pre-augmentations
        if iteration == 2 and not args.sweep_all_preaugs:
            args.preaugs = list(AUGMENTATION_REGISTRY.keys())
            print(f"  Added all {len(args.preaugs)} pre-augmentation strategies")

        # Discover more groups from correlation
        if iteration >= 2:
            available = get_all_available_symbols(args.asset_class)
            new_groups = discover_correlated_groups(
                available,
                min_correlation=max(0.3, args.min_correlation - 0.1 * iteration),
                top_k=args.top_k + iteration * 5,
            )
            existing_names = {n for n, _ in valid_groups}
            for name, syms in new_groups:
                if name not in existing_names:
                    valid_syms = [s for s in syms if s in symbol_data]
                    if len(valid_syms) >= 2:
                        valid_groups.append((name, valid_syms))
                        existing_names.add(name)
                        print(f"  Added new group: {name} ({', '.join(valid_syms)})")

    # Final cleanup
    try:
        wrapper.unload()
    except Exception:
        pass

    print(f"\nDone. Total experiments: {experiment_count}")
    print(f"Results: {LEADERBOARD_PATH}")
    print(f"Best configs: {CONFIG_OUTPUT}")


if __name__ == "__main__":
    main()
