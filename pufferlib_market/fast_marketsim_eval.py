#!/usr/bin/env python3
"""Fast market simulator evaluation for daily RL checkpoints.

Optimized version of comprehensive_marketsim_eval.py with:
1. Shared data loading — each MKTD val file loaded once, reused across checkpoints
2. Single-load checkpoints — each .pt file loaded once for both metadata and weights
3. torch.compile — compiled forward pass for faster inference (optional)
4. Parallel evaluation — concurrent.futures for CPU-parallel checkpoint eval
5. Result caching — skip re-evaluation of unchanged checkpoints (by mtime)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Monkey-patch early exit so we never stop mid-simulation
# ---------------------------------------------------------------------------
import src.market_sim_early_exit as _mse


def _no_early_exit(*args, **kwargs):
    return _mse.EarlyExitDecision(
        should_stop=False,
        progress_fraction=0.0,
        total_return=0.0,
        max_drawdown=0.0,
    )


_mse.evaluate_drawdown_vs_profit_early_exit = _no_early_exit

from pufferlib_market.hourly_replay import MktdData, read_mktd, simulate_daily_policy
from pufferlib_market.metrics import annualize_total_return
from pufferlib_market.evaluate_tail import (
    TradingPolicy,
    ResidualTradingPolicy,
    _infer_num_actions,
    _infer_arch,
    _infer_hidden_size,
    _infer_resmlp_blocks,
    _slice_tail,
)

# ---------------------------------------------------------------------------
# Default config (same as comprehensive_marketsim_eval.py)
# ---------------------------------------------------------------------------

CHECKPOINT_DIRS = [
    "pufferlib_market/checkpoints/mass_daily",
    "pufferlib_market/checkpoints/autoresearch_daily",
    "pufferlib_market/checkpoints/autoresearch_daily_combos",
    "pufferlib_market/checkpoints/autoresearch_crypto8_daily",
    "pufferlib_market/checkpoints/autoresearch_mixed23_daily",
    "pufferlib_market/checkpoints/autoresearch_mixed32_daily",
    "pufferlib_market/checkpoints/autoresearch_daily_v2",
    "pufferlib_market/checkpoints/autoresearch_fdusd_daily",
    "pufferlib_market/checkpoints/tp_fine",
    "pufferlib_market/checkpoints/mass_daily_v2",
    "pufferlib_market/checkpoints/long_daily",
    "pufferlib_market/checkpoints/daily_crypto5_baseline",
    "pufferlib_market/checkpoints/stocks12_daily_tp05",
    "pufferlib_market/checkpoints/stocks12_daily_tp05_longonly",
    "pufferlib_market/checkpoints/mixed32_daily_ent_anneal",
]

DATA_DIR = "pufferlib_market/data"

OBS_SIZE_TO_VAL_DATA = {
    56: "fdusd3_daily_val.bin",
    90: "crypto5_daily_val.bin",
    141: "crypto8_daily_val.bin",
    175: "crypto10_daily_val.bin",
    192: "crypto11_daily_val.bin",
    209: "stocks12_daily_val.bin",
    260: "crypto15_daily_val.bin",
    396: "mixed23_daily_val.bin",
    549: "mixed32_daily_val.bin",
}

# SIM-PRODUCTION GAP ANALYSIS (2026-03-20):
# Production orchestrator (Alpaca crypto) uses:
#   - TRAILING_STOP_PCT = 0.003  (0.3% below peak since entry → force exit)
#   - MAX_HOLD_HOURS = 6         (force exit after 6 hourly bars / 6 daily steps)
#   - Fee: 0 commission but ~2-5bps adverse slippage per fill
#   - Min notional: $12 (skip positions too small to execute)
#   - Cancel-replace buy loop (creates coverage gaps between cancel and refill)
#
# Measured effect on slip_5bps checkpoint (90d crypto5 val, 2026-03-20):
#   trailing_stop_pct=0.003: -4.65pp vs no trailing stop (7.72% → 3.07%)
#     — fires in whipsaw markets and interrupts winning trends early
#   max_hold_bars=6:          0pp effect (avg hold is ~1.6 bars naturally)
#     — policy already exits positions quickly; 6-bar limit rarely binds
#   slippage_bps=3 (vs old fee_rate=0.001=10bps): +1.47pp
#     — replacing 10bps fee with 3bps slippage is less punishing overall
#   min_notional_usd=12:      negligible effect at $10k+ account size
#   Combined production-realistic: +0.32pp vs old unconstrained sim
#     (the fee correction dominates; trailing stop cost partially offset)
#
# Key insight: the old FEE_RATE=0.001 (10bps) was overly conservative for
# Alpaca crypto (0 commission). The corrected model (fee=0 + slippage=3bps)
# gives a more accurate picture. The trailing stop is the dominant constraint
# that distinguishes sim from production in volatile markets.
#
# The legacy FEE_RATE=0.001 (10bps) was too conservative for Alpaca crypto
# (0 commission); replaced with fee_rate=0.0 + slippage_bps=3 for accuracy.
FEE_RATE = 0.0           # Alpaca crypto: 0 commission
SLIPPAGE_BPS = 3.0       # ~3bps adverse slippage per fill (2-5bps realistic range)
TRAILING_STOP_PCT = 0.003  # 0.3% below peak since entry
MAX_HOLD_BARS = 6        # force exit after 6 bars (matches MAX_HOLD_HOURS=6)
MIN_NOTIONAL_USD = 12.0  # skip positions < $12 notional
FILL_BUFFER_BPS = 8.0
MAX_LEVERAGE = 1.0
PERIODS_PER_YEAR = 365.0


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def discover_checkpoints(root: Path, checkpoint_dirs: list[str]) -> list[Path]:
    """Find all best.pt files under the given checkpoint directories."""
    found = []
    for rel_dir in checkpoint_dirs:
        d = root / rel_dir
        if not d.is_dir():
            continue
        for dirpath, _, filenames in os.walk(str(d)):
            if "best.pt" in filenames:
                found.append(Path(dirpath) / "best.pt")
    return sorted(set(found))


def short_checkpoint_name(path: Path) -> str:
    """Turn a long path into a readable short name like 'mass_daily/tp0.05_s42'."""
    parts = path.parts
    try:
        idx = list(parts).index("checkpoints")
        relevant = parts[idx + 1 : -1]
        return "/".join(relevant)
    except ValueError:
        return str(path.parent.name)


# ---------------------------------------------------------------------------
# Checkpoint info extraction
# ---------------------------------------------------------------------------

@dataclass
class CheckpointInfo:
    path: Path
    obs_size: int
    num_actions: int
    arch: str
    hidden_size: int
    resmlp_blocks: int
    mtime: float  # file modification time for caching


def load_checkpoint_info(
    path: Path, device: torch.device, *, keep_state_dict: bool = False,
) -> tuple[CheckpointInfo, Optional[dict]]:
    """Load a checkpoint and extract architecture metadata.

    When *keep_state_dict* is True the state_dict is returned alongside the
    info so the caller can build the policy without a redundant disk load.
    """
    payload = torch.load(str(path), map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        state_dict = payload["model"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError(f"Unsupported checkpoint format in {path}")

    if "encoder.0.weight" in state_dict:
        obs_size = int(state_dict["encoder.0.weight"].shape[1])
    elif "input_proj.weight" in state_dict:
        obs_size = int(state_dict["input_proj.weight"].shape[1])
    else:
        raise ValueError(f"Cannot detect obs_size from {path}")

    arch = _infer_arch(state_dict)
    hidden_size = _infer_hidden_size(state_dict, arch=arch)
    num_actions = _infer_num_actions(state_dict, fallback=0)
    resmlp_blocks = _infer_resmlp_blocks(state_dict) if arch == "resmlp" else 0
    mtime = os.path.getmtime(str(path))

    info = CheckpointInfo(
        path=path,
        obs_size=obs_size,
        num_actions=num_actions,
        arch=arch,
        hidden_size=hidden_size,
        resmlp_blocks=resmlp_blocks,
        mtime=mtime,
    )
    return info, state_dict if keep_state_dict else None


# ---------------------------------------------------------------------------
# Policy construction & compilation
# ---------------------------------------------------------------------------

def build_policy(info: CheckpointInfo, state_dict: dict, device: torch.device) -> nn.Module:
    """Build and load a policy network from checkpoint info and state_dict."""
    if info.arch == "resmlp":
        policy = ResidualTradingPolicy(
            info.obs_size,
            info.num_actions,
            hidden=info.hidden_size,
            num_blocks=info.resmlp_blocks,
        ).to(device)
    elif info.arch == "mlp":
        policy = TradingPolicy(
            info.obs_size,
            info.num_actions,
            hidden=info.hidden_size,
        ).to(device)
    else:
        raise ValueError(f"Unsupported arch: {info.arch}")
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def try_compile_policy(policy: nn.Module) -> nn.Module:
    """Try to torch.compile the policy for faster inference. Falls back gracefully."""
    try:
        compiled = torch.compile(policy, mode="reduce-overhead")
        # Warmup with a dummy forward pass to trigger compilation
        dummy = torch.zeros(1, policy.encoder[0].in_features if hasattr(policy, 'encoder') else policy.input_proj.in_features)
        with torch.inference_mode():
            compiled(dummy)
        return compiled
    except Exception:
        # torch.compile may fail on some setups; fall back to eager
        return policy


def make_policy_fn(policy: nn.Module, device: torch.device) -> Callable[[np.ndarray], int]:
    """Create a deterministic policy function for simulate_daily_policy."""
    def _fn(obs: np.ndarray) -> int:
        obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).to(device=device).view(1, -1)
        with torch.inference_mode():
            logits, _ = policy(obs_t)
        return int(torch.argmax(logits, dim=-1).item())
    return _fn


# ---------------------------------------------------------------------------
# Shared data loading
# ---------------------------------------------------------------------------

def load_val_data(data_dir: Path) -> dict[int, MktdData]:
    """Load all val data files once, indexed by obs_size."""
    data_cache: dict[int, MktdData] = {}
    for obs_size, fname in OBS_SIZE_TO_VAL_DATA.items():
        fpath = data_dir / fname
        if fpath.exists():
            try:
                data_cache[obs_size] = read_mktd(fpath)
            except Exception as e:
                print(f"  WARNING: failed to load {fname}: {e}")
    return data_cache


# ---------------------------------------------------------------------------
# Result caching
# ---------------------------------------------------------------------------

def load_cache(cache_path: str) -> dict[str, dict]:
    """Load cached results from JSON. Returns {checkpoint_path: {mtime, results}}."""
    p = Path(cache_path)
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(cache_path: str, cache: dict[str, dict]) -> None:
    """Save cached results to JSON."""
    try:
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=1)
    except OSError as e:
        print(f"  WARNING: failed to save cache: {e}")


def is_cached(cache: dict[str, dict], ckpt_path: str, mtime: float) -> bool:
    """Check if a checkpoint result is cached and still valid (same mtime)."""
    entry = cache.get(ckpt_path)
    if entry is None:
        return False
    return abs(entry.get("mtime", 0.0) - mtime) < 0.01


# ---------------------------------------------------------------------------
# Single-checkpoint evaluation (used by workers)
# ---------------------------------------------------------------------------

def _universe_name(data: MktdData) -> str:
    syms = data.symbols
    n = len(syms)
    if n <= 5:
        return "+".join(syms)
    return f"{n}sym"


def _format_sim_result(sim_result, ckpt_name: str, universe: str, period_days: int, periods_per_year: float) -> dict:
    """Convert a DailySimResult into the standard result dict."""
    annualized = annualize_total_return(
        float(sim_result.total_return),
        periods=float(period_days),
        periods_per_year=periods_per_year,
    )
    return {
        "checkpoint": ckpt_name,
        "universe": universe,
        "period": period_days,
        "return_pct": round(sim_result.total_return * 100.0, 2),
        "annualized_pct": round(annualized * 100.0, 2),
        "sortino": round(sim_result.sortino, 3),
        "max_dd_pct": round(sim_result.max_drawdown * 100.0, 2),
        "num_trades": sim_result.num_trades,
        "win_rate": round(sim_result.win_rate * 100.0, 1),
        "avg_hold": round(sim_result.avg_hold_steps, 1),
    }


def _eval_all_periods(
    policy_fn: Callable[[np.ndarray], int],
    data: MktdData,
    periods: list[int],
    ckpt_name: str,
    universe: str,
    fee_rate: float,
    fill_buffer_bps: float,
    periods_per_year: float,
    slippage_bps: float = SLIPPAGE_BPS,
    trailing_stop_pct: float = TRAILING_STOP_PCT,
    max_hold_bars: int = MAX_HOLD_BARS,
    min_notional_usd: float = MIN_NOTIONAL_USD,
) -> list[dict]:
    """Run simulate_daily_policy for each period, return list of result dicts."""
    results = []
    for period_days in periods:
        max_steps = period_days
        if data.num_timesteps < max_steps + 1:
            continue
        try:
            tail = _slice_tail(data, steps=max_steps)
        except ValueError:
            continue
        sim_result = simulate_daily_policy(
            tail,
            policy_fn,
            max_steps=max_steps,
            fee_rate=fee_rate,
            slippage_bps=slippage_bps,
            fill_buffer_bps=fill_buffer_bps,
            max_leverage=MAX_LEVERAGE,
            periods_per_year=periods_per_year,
            trailing_stop_pct=trailing_stop_pct,
            max_hold_bars=max_hold_bars,
            min_notional_usd=min_notional_usd,
        )
        results.append(_format_sim_result(sim_result, ckpt_name, universe, period_days, periods_per_year))
    return results


def evaluate_single_checkpoint(
    ckpt_path: str,
    data_dir: str,
    periods: list[int],
    fee_rate: float,
    fill_buffer_bps: float,
    periods_per_year: float,
    use_compile: bool = True,
    slippage_bps: float = SLIPPAGE_BPS,
    trailing_stop_pct: float = TRAILING_STOP_PCT,
    max_hold_bars: int = MAX_HOLD_BARS,
    min_notional_usd: float = MIN_NOTIONAL_USD,
) -> list[dict]:
    """Evaluate a single checkpoint across all periods. Designed for use in worker processes.

    Each worker re-applies the early-exit monkey-patch and loads its own data,
    since subprocess state is not shared with the parent.
    """
    # Re-apply monkey-patch in subprocess
    import src.market_sim_early_exit as _mse_local
    _mse_local.evaluate_drawdown_vs_profit_early_exit = lambda *a, **k: _mse_local.EarlyExitDecision(
        should_stop=False, progress_fraction=0.0, total_return=0.0, max_drawdown=0.0)

    device = torch.device("cpu")
    path = Path(ckpt_path)

    try:
        info, state_dict = load_checkpoint_info(path, device, keep_state_dict=True)
    except Exception:
        return []

    # Find matching val data
    val_fname = OBS_SIZE_TO_VAL_DATA.get(info.obs_size)
    if val_fname is None:
        return []
    val_path = Path(data_dir) / val_fname
    if not val_path.exists():
        return []

    data = read_mktd(val_path)
    expected_actions = 1 + 2 * data.num_symbols
    if info.num_actions != expected_actions:
        return []

    try:
        policy = build_policy(info, state_dict, device)
    except Exception:
        return []
    del state_dict

    if use_compile:
        policy = try_compile_policy(policy)

    policy_fn = make_policy_fn(policy, device)
    return _eval_all_periods(
        policy_fn, data, periods,
        short_checkpoint_name(path), _universe_name(data),
        fee_rate, fill_buffer_bps, periods_per_year,
        slippage_bps=slippage_bps,
        trailing_stop_pct=trailing_stop_pct,
        max_hold_bars=max_hold_bars,
        min_notional_usd=min_notional_usd,
    )


# ---------------------------------------------------------------------------
# Fast evaluation (sequential, in-process — uses shared data + compiled policy)
# ---------------------------------------------------------------------------

def fast_eval_sequential(
    root: Path,
    checkpoint_dirs: list[str],
    periods: list[int],
    fee_rate: float,
    fill_buffer_bps: float,
    periods_per_year: float,
    cache_path: str,
    use_compile: bool = True,
    slippage_bps: float = SLIPPAGE_BPS,
    trailing_stop_pct: float = TRAILING_STOP_PCT,
    max_hold_bars: int = MAX_HOLD_BARS,
    min_notional_usd: float = MIN_NOTIONAL_USD,
) -> pd.DataFrame:
    """Evaluate all checkpoints sequentially with shared data loading and caching.

    This is the fastest single-process approach: data is loaded once per obs_size,
    policies are compiled once, and unchanged checkpoints are skipped via caching.
    """
    device = torch.device("cpu")
    data_dir = root / DATA_DIR

    # 1. Load all val data once
    print("Loading validation data...")
    data_cache = load_val_data(data_dir)
    print(f"  Loaded {len(data_cache)} datasets")

    # 2. Discover checkpoints
    all_checkpoints = discover_checkpoints(root, checkpoint_dirs)
    print(f"Discovered {len(all_checkpoints)} checkpoints")
    if not all_checkpoints:
        return pd.DataFrame()

    # 3. Load result cache
    cache = load_cache(cache_path)
    cache_hits = 0

    # 4. Evaluate each checkpoint (load once, check cache before building policy)
    all_results: list[dict] = []
    total_checkpoints = len(all_checkpoints)
    eval_count = 0
    for i, ckpt_path in enumerate(all_checkpoints):
        ckpt_key = str(ckpt_path)
        ckpt_name = short_checkpoint_name(ckpt_path)
        progress = f"[{i+1}/{total_checkpoints}]"

        # Quick mtime check before loading checkpoint
        try:
            mtime = os.path.getmtime(str(ckpt_path))
        except OSError:
            continue
        if is_cached(cache, ckpt_key, mtime):
            cached_results = cache[ckpt_key].get("results", [])
            if cached_results:
                all_results.extend(cached_results)
                cache_hits += 1
                continue

        # Load checkpoint once -- extract info AND keep state_dict
        try:
            info, state_dict = load_checkpoint_info(ckpt_path, device, keep_state_dict=True)
        except Exception as e:
            print(f"{progress} SKIP {ckpt_name}: {e}")
            continue

        # Check data availability
        data = data_cache.get(info.obs_size)
        if data is None:
            print(f"{progress} SKIP {ckpt_name}: no val data for obs_size={info.obs_size}")
            del state_dict
            continue

        expected_actions = 1 + 2 * data.num_symbols
        if info.num_actions != expected_actions:
            print(f"{progress} SKIP {ckpt_name}: actions mismatch {info.num_actions} != {expected_actions}")
            del state_dict
            continue

        # Build policy from the already-loaded state_dict (no second disk read)
        try:
            policy = build_policy(info, state_dict, device)
            del state_dict  # free memory
        except Exception as e:
            print(f"{progress} SKIP {ckpt_name}: policy build error: {e}")
            del state_dict
            continue
        eval_count += 1

        # Try torch.compile
        if use_compile:
            policy = try_compile_policy(policy)

        policy_fn = make_policy_fn(policy, device)
        universe = _universe_name(data)

        ckpt_results = _eval_all_periods(
            policy_fn, data, periods, ckpt_name, universe,
            fee_rate, fill_buffer_bps, periods_per_year,
            slippage_bps=slippage_bps,
            trailing_stop_pct=trailing_stop_pct,
            max_hold_bars=max_hold_bars,
            min_notional_usd=min_notional_usd,
        )
        all_results.extend(ckpt_results)

        # Update cache
        cache[ckpt_key] = {"mtime": info.mtime, "results": ckpt_results}

        if ckpt_results:
            show = next((r for r in ckpt_results if r["period"] == 120), ckpt_results[-1])
            print(
                f"{progress} {ckpt_name}: "
                f"{show['period']}d ret={show['return_pct']:+.1f}% "
                f"sortino={show['sortino']:.2f}"
            )

        del policy  # free memory

    # Save cache
    save_cache(cache_path, cache)
    print(f"Cache: {cache_hits} hits, {eval_count} evaluations")

    if not all_results:
        return pd.DataFrame()
    return pd.DataFrame(all_results)


# ---------------------------------------------------------------------------
# Fast evaluation (parallel, multi-process)
# ---------------------------------------------------------------------------

def fast_eval_parallel(
    root: Path,
    checkpoint_dirs: list[str],
    periods: list[int],
    fee_rate: float,
    fill_buffer_bps: float,
    periods_per_year: float,
    cache_path: str,
    max_workers: int = 4,
    use_compile: bool = True,
    slippage_bps: float = SLIPPAGE_BPS,
    trailing_stop_pct: float = TRAILING_STOP_PCT,
    max_hold_bars: int = MAX_HOLD_BARS,
    min_notional_usd: float = MIN_NOTIONAL_USD,
) -> pd.DataFrame:
    """Evaluate all checkpoints in parallel using ProcessPoolExecutor.

    Each worker loads its own checkpoint and data. This is effective when
    you have many checkpoints and want to utilize multiple CPU cores.
    """
    data_dir = root / DATA_DIR

    # 1. Discover checkpoints
    all_checkpoints = discover_checkpoints(root, checkpoint_dirs)
    print(f"Discovered {len(all_checkpoints)} checkpoints")
    if not all_checkpoints:
        return pd.DataFrame()

    # 2. Load result cache
    cache = load_cache(cache_path)
    cache_hits = 0

    # 3. Partition: cached vs. need-eval
    to_eval: list[Path] = []
    cached_results: list[dict] = []

    for ckpt_path in all_checkpoints:
        ckpt_key = str(ckpt_path)
        try:
            mtime = os.path.getmtime(str(ckpt_path))
        except OSError:
            continue
        if is_cached(cache, ckpt_key, mtime):
            entry_results = cache[ckpt_key].get("results", [])
            cached_results.extend(entry_results)
            cache_hits += 1
        else:
            to_eval.append(ckpt_path)

    print(f"Cache hits: {cache_hits}, to evaluate: {len(to_eval)}")

    all_results = list(cached_results)

    if to_eval:
        data_dir_str = str(data_dir)
        # Use min of max_workers and number of checkpoints to avoid waste
        actual_workers = min(max_workers, len(to_eval))
        print(f"Evaluating {len(to_eval)} checkpoints with {actual_workers} workers...")

        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            futures = {}
            for ckpt_path in to_eval:
                future = executor.submit(
                    evaluate_single_checkpoint,
                    str(ckpt_path),
                    data_dir_str,
                    periods,
                    fee_rate,
                    fill_buffer_bps,
                    periods_per_year,
                    use_compile,
                    slippage_bps,
                    trailing_stop_pct,
                    max_hold_bars,
                    min_notional_usd,
                )
                futures[future] = ckpt_path

            done_count = 0
            for future in as_completed(futures):
                ckpt_path = futures[future]
                ckpt_key = str(ckpt_path)
                done_count += 1
                try:
                    results = future.result()
                    all_results.extend(results)
                    mtime = os.path.getmtime(str(ckpt_path))
                    cache[ckpt_key] = {"mtime": mtime, "results": results}
                    if results:
                        show = next((r for r in results if r["period"] == 120), results[-1])
                        name = short_checkpoint_name(ckpt_path)
                        print(
                            f"  [{done_count}/{len(to_eval)}] {name}: "
                            f"{show['period']}d ret={show['return_pct']:+.1f}% "
                            f"sortino={show['sortino']:.2f}"
                        )
                except Exception as e:
                    print(f"  [{done_count}/{len(to_eval)}] ERROR {short_checkpoint_name(ckpt_path)}: {e}")

        # Save updated cache
        save_cache(cache_path, cache)

    if not all_results:
        return pd.DataFrame()
    return pd.DataFrame(all_results)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fast_eval_all(
    checkpoint_dirs: list[tuple[str, str]],  # (dir_name, val_data_path) — ignored, kept for interface compat
    periods: list[int] = [30, 60, 90, 120, 180],
    fee_rate: float = FEE_RATE,
    fill_buffer_bps: float = FILL_BUFFER_BPS,
    periods_per_year: float = PERIODS_PER_YEAR,
    max_workers: int = 4,
    cache_path: str = "marketsim_eval_cache.json",
    root: str = ".",
    use_compile: bool = True,
    parallel: bool = True,
    slippage_bps: float = SLIPPAGE_BPS,
    trailing_stop_pct: float = TRAILING_STOP_PCT,
    max_hold_bars: int = MAX_HOLD_BARS,
    min_notional_usd: float = MIN_NOTIONAL_USD,
) -> pd.DataFrame:
    """Evaluate all checkpoints, return DataFrame of results.

    This is the main entry point. It auto-discovers checkpoints under the
    standard CHECKPOINT_DIRS, loads shared val data, and evaluates either
    sequentially or in parallel.

    Args:
        checkpoint_dirs: List of (dir_name, val_data_path) tuples. If provided,
            overrides the default CHECKPOINT_DIRS with just the dir_name portion.
        periods: Evaluation periods in days.
        fee_rate: Trading fee rate (default 0.0 for Alpaca crypto: 0 commission).
        fill_buffer_bps: Fill buffer in basis points.
        periods_per_year: Periods per year for annualization.
        max_workers: Maximum parallel workers (only used if parallel=True).
        cache_path: Path to the JSON result cache.
        root: Project root directory.
        use_compile: Whether to try torch.compile on policies.
        parallel: Whether to use multi-process parallelism.
        slippage_bps: Adverse fill slippage in bps (default 3 for Alpaca crypto).
        trailing_stop_pct: Force exit when price drops this far below peak (default 0.003).
        max_hold_bars: Force exit after this many bars held (default 6).
        min_notional_usd: Skip positions below this dollar value (default 12.0).

    Returns:
        pd.DataFrame with columns: checkpoint, universe, period, return_pct,
        annualized_pct, sortino, max_dd_pct, num_trades, win_rate, avg_hold.
    """
    root_path = Path(root).resolve()

    # If checkpoint_dirs is provided as (dir, val_path) pairs, extract just dir names
    if checkpoint_dirs:
        dirs = [d if isinstance(d, str) else d[0] for d in checkpoint_dirs]
    else:
        dirs = list(CHECKPOINT_DIRS)

    if parallel and max_workers > 1:
        return fast_eval_parallel(
            root_path, dirs, periods, fee_rate, fill_buffer_bps,
            periods_per_year, cache_path, max_workers, use_compile,
            slippage_bps=slippage_bps,
            trailing_stop_pct=trailing_stop_pct,
            max_hold_bars=max_hold_bars,
            min_notional_usd=min_notional_usd,
        )
    else:
        return fast_eval_sequential(
            root_path, dirs, periods, fee_rate, fill_buffer_bps,
            periods_per_year, cache_path, use_compile,
            slippage_bps=slippage_bps,
            trailing_stop_pct=trailing_stop_pct,
            max_hold_bars=max_hold_bars,
            min_notional_usd=min_notional_usd,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fast market sim eval for all daily RL checkpoints"
    )
    parser.add_argument("--root", type=str, default=".", help="Project root")
    parser.add_argument("--output", type=str, default="fast_marketsim_results.csv")
    parser.add_argument("--periods", type=str, default="30,60,90,120,180")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--cache-path", type=str, default="marketsim_eval_cache.json")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--sequential", action="store_true", help="Force sequential evaluation")
    parser.add_argument(
        "--sort-period", type=int, default=120,
        help="Period to sort final leaderboard by Sortino",
    )
    parser.add_argument(
        "--checkpoint-dirs", type=str, default=None,
        help="Comma-separated checkpoint dirs (overrides defaults)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    periods = [int(p.strip()) for p in args.periods.split(",")]
    sort_period = args.sort_period
    use_compile = not args.no_compile
    parallel = not args.sequential

    if args.checkpoint_dirs:
        dirs = [d.strip() for d in args.checkpoint_dirs.split(",") if d.strip()]
    else:
        dirs = list(CHECKPOINT_DIRS)

    print(f"Root: {root}")
    print(f"Periods: {periods}")
    print(f"Workers: {args.max_workers}")
    print(f"Compile: {use_compile}")
    print(f"Parallel: {parallel}")
    print(f"Fee: {FEE_RATE*10000:.0f}bps + {SLIPPAGE_BPS:.1f}bps slippage")
    print(f"Trailing stop: {TRAILING_STOP_PCT*100:.2f}%  Max hold: {MAX_HOLD_BARS} bars  Min notional: ${MIN_NOTIONAL_USD:.0f}")
    print()

    t_start = time.time()

    if parallel and args.max_workers > 1:
        df = fast_eval_parallel(
            root, dirs, periods, FEE_RATE, FILL_BUFFER_BPS,
            PERIODS_PER_YEAR, args.cache_path, args.max_workers, use_compile,
            slippage_bps=SLIPPAGE_BPS,
            trailing_stop_pct=TRAILING_STOP_PCT,
            max_hold_bars=MAX_HOLD_BARS,
            min_notional_usd=MIN_NOTIONAL_USD,
        )
    else:
        df = fast_eval_sequential(
            root, dirs, periods, FEE_RATE, FILL_BUFFER_BPS,
            PERIODS_PER_YEAR, args.cache_path, use_compile,
            slippage_bps=SLIPPAGE_BPS,
            trailing_stop_pct=TRAILING_STOP_PCT,
            max_hold_bars=MAX_HOLD_BARS,
            min_notional_usd=MIN_NOTIONAL_USD,
        )

    elapsed = time.time() - t_start
    print(f"\nCompleted in {elapsed:.1f}s ({len(df)} result rows)")

    if df.empty:
        print("No results.")
        sys.exit(1)

    # Write CSV
    df.to_csv(args.output, index=False)
    print(f"CSV written to: {args.output}")

    # Print leaderboard
    period_df = df[df["period"] == sort_period]
    if period_df.empty:
        available = sorted(df["period"].unique(), reverse=True)
        if available:
            sort_period = available[0]
            period_df = df[df["period"] == sort_period]
            print(f"(No {args.sort_period}d results; showing {sort_period}d)")

    if not period_df.empty:
        period_df = period_df.sort_values("sortino", ascending=False)
        print(f"\n{'='*110}")
        print(f"  LEADERBOARD ({sort_period}d, sorted by Sortino)")
        print(f"{'='*110}")
        fmt = "{rank:>4}  {name:<45} {universe:<10} {ret:>8}  {ann:>8}  {sortino:>8}  {dd:>7}  {trades:>6}  {wr:>5}"
        print(fmt.format(
            rank="Rank", name="Checkpoint", universe="Universe",
            ret="Ret%", ann="Ann%", sortino="Sortino", dd="MaxDD%",
            trades="Trades", wr="WR%",
        ))
        print("-" * 110)
        for rank, (_, row) in enumerate(period_df.iterrows(), 1):
            print(fmt.format(
                rank=rank,
                name=str(row["checkpoint"])[:45],
                universe=str(row["universe"])[:10],
                ret=f"{row['return_pct']:+.1f}",
                ann=f"{row['annualized_pct']:+.1f}",
                sortino=f"{row['sortino']:.3f}",
                dd=f"{row['max_dd_pct']:.1f}",
                trades=int(row["num_trades"]),
                wr=f"{row['win_rate']:.0f}",
            ))


if __name__ == "__main__":
    main()
