"""Fast C work-stealing simulator with Python ctypes wrapper."""
from __future__ import annotations
import ctypes, subprocess
from ctypes import c_double, c_int, POINTER, Structure
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

_DIR = Path(__file__).parent
_SO = _DIR / "libworksteal.so"
_SRC = _DIR / "worksteal_sim.c"

FDUSD_SYMBOLS = {"BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD"}


def _ensure_compiled():
    if _SO.exists() and _SO.stat().st_mtime > _SRC.stat().st_mtime:
        return
    subprocess.check_call([
        "gcc", "-O3", "-march=native", "-fopenmp", "-ffast-math",
        "-shared", "-fPIC",
        "-o", str(_SO), str(_SRC), "-lm",
    ])


class CWorkStealConfig(Structure):
    _fields_ = [
        ("dip_pct", c_double),
        ("proximity_pct", c_double),
        ("profit_target_pct", c_double),
        ("stop_loss_pct", c_double),
        ("trailing_stop_pct", c_double),
        ("margin_annual_rate", c_double),
        ("max_position_pct", c_double),
        ("max_positions", c_int),
        ("max_hold_days", c_int),
        ("lookback_days", c_int),
        ("sma_filter_period", c_int),
        ("initial_cash", c_double),
        ("max_leverage", c_double),
        ("maker_fee", c_double),
        ("max_drawdown_exit", c_double),
        ("enable_shorts", c_int),
        ("short_pump_pct", c_double),
        ("reentry_cooldown_days", c_int),
        ("momentum_period", c_int),
        ("momentum_min", c_double),
    ]


class CSimResult(Structure):
    _fields_ = [
        ("total_return", c_double),
        ("sortino", c_double),
        ("sharpe", c_double),
        ("max_drawdown", c_double),
        ("win_rate", c_double),
        ("final_equity", c_double),
        ("mean_daily_return", c_double),
        ("total_trades", c_int),
        ("n_days", c_int),
    ]


_ensure_compiled()
_lib = ctypes.CDLL(str(_SO))

_lib.worksteal_simulate.argtypes = [
    POINTER(c_double),  # timestamps
    POINTER(c_int),     # valid_mask
    POINTER(c_double),  # opens
    POINTER(c_double),  # highs
    POINTER(c_double),  # lows
    POINTER(c_double),  # closes
    POINTER(c_double),  # fee_rates
    c_int,              # n_bars
    c_int,              # n_symbols
    POINTER(CWorkStealConfig),
    POINTER(CSimResult),
    POINTER(c_double),  # equity_curve
]
_lib.worksteal_simulate.restype = None

_lib.worksteal_simulate_batch.argtypes = [
    POINTER(c_double), POINTER(c_int),
    POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
    POINTER(c_double), c_int, c_int,
    POINTER(CWorkStealConfig), POINTER(CSimResult), c_int,
]
_lib.worksteal_simulate_batch.restype = None

_lib.worksteal_get_num_threads.argtypes = []
_lib.worksteal_get_num_threads.restype = c_int


def get_num_threads() -> int:
    return _lib.worksteal_get_num_threads()


def _to_ptr_d(arr):
    return arr.ctypes.data_as(POINTER(c_double))

def _to_ptr_i(arr):
    return arr.ctypes.data_as(POINTER(c_int))


def _prepare_data(
    all_bars: Dict[str, pd.DataFrame],
    config,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    symbols = sorted(all_bars.keys())
    processed = {}
    for sym in symbols:
        df = all_bars[sym].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        processed[sym] = df

    all_dates = sorted(set().union(*[set(df["timestamp"].tolist()) for df in processed.values()]))
    if start_date:
        start_ts = pd.Timestamp(start_date, tz="UTC")
        all_dates = [d for d in all_dates if d >= start_ts]
    if end_date:
        end_ts = pd.Timestamp(end_date, tz="UTC")
        all_dates = [d for d in all_dates if d <= end_ts]

    n_bars = len(all_dates)
    n_sym = len(symbols)
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    timestamps = np.zeros(n_bars, dtype=np.float64)
    for i, d in enumerate(all_dates):
        timestamps[i] = d.timestamp()

    valid = np.zeros(n_sym * n_bars, dtype=np.int32)
    opens = np.zeros(n_sym * n_bars, dtype=np.float64)
    highs_arr = np.zeros(n_sym * n_bars, dtype=np.float64)
    lows_arr = np.zeros(n_sym * n_bars, dtype=np.float64)
    closes_arr = np.zeros(n_sym * n_bars, dtype=np.float64)

    for si, sym in enumerate(symbols):
        df = processed[sym]
        for _, row in df.iterrows():
            ts = row["timestamp"]
            if ts in date_to_idx:
                idx = si * n_bars + date_to_idx[ts]
                valid[idx] = 1
                opens[idx] = float(row.get("open", row["close"]))
                highs_arr[idx] = float(row.get("high", row["close"]))
                lows_arr[idx] = float(row.get("low", row["close"]))
                closes_arr[idx] = float(row["close"])

    fee_rates = np.zeros(n_sym, dtype=np.float64)
    maker_fee = getattr(config, 'maker_fee', 0.001)
    fdusd_fee = getattr(config, 'fdusd_fee', 0.0)
    for si, sym in enumerate(symbols):
        fee_rates[si] = fdusd_fee if sym in FDUSD_SYMBOLS else maker_fee

    return timestamps, valid, opens, highs_arr, lows_arr, closes_arr, fee_rates, symbols


def _config_to_c(config) -> CWorkStealConfig:
    return CWorkStealConfig(
        dip_pct=config.dip_pct,
        proximity_pct=config.proximity_pct,
        profit_target_pct=config.profit_target_pct,
        stop_loss_pct=config.stop_loss_pct,
        trailing_stop_pct=config.trailing_stop_pct,
        margin_annual_rate=config.margin_annual_rate,
        max_position_pct=config.max_position_pct,
        max_positions=config.max_positions,
        max_hold_days=config.max_hold_days,
        lookback_days=config.lookback_days,
        sma_filter_period=config.sma_filter_period,
        initial_cash=config.initial_cash,
        max_leverage=config.max_leverage,
        maker_fee=config.maker_fee,
        max_drawdown_exit=config.max_drawdown_exit,
        enable_shorts=int(config.enable_shorts),
        short_pump_pct=config.short_pump_pct,
        reentry_cooldown_days=config.reentry_cooldown_days,
        momentum_period=config.momentum_period,
        momentum_min=config.momentum_min,
    )


def _result_to_dict(r: CSimResult) -> dict:
    return {
        "total_return": r.total_return,
        "total_return_pct": r.total_return * 100,
        "sortino": r.sortino,
        "sharpe": r.sharpe,
        "max_drawdown": r.max_drawdown,
        "max_drawdown_pct": r.max_drawdown * 100,
        "win_rate": r.win_rate,
        "final_equity": r.final_equity,
        "mean_daily_return": r.mean_daily_return,
        "n_days": r.n_days,
        "total_trades": r.total_trades,
    }


def run_worksteal_backtest_fast(
    all_bars: Dict[str, pd.DataFrame],
    config,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    timestamps, valid, opens, highs_arr, lows_arr, closes_arr, fee_rates, symbols = \
        _prepare_data(all_bars, config, start_date, end_date)

    n_bars = len(timestamps)
    n_sym = len(symbols)
    if n_bars == 0:
        return {"total_return": 0, "sortino": 0, "max_drawdown": 0, "final_equity": config.initial_cash}

    c_cfg = _config_to_c(config)
    result = CSimResult()
    eq = np.zeros(n_bars, dtype=np.float64)

    _lib.worksteal_simulate(
        _to_ptr_d(timestamps), _to_ptr_i(valid),
        _to_ptr_d(opens), _to_ptr_d(highs_arr),
        _to_ptr_d(lows_arr), _to_ptr_d(closes_arr),
        _to_ptr_d(fee_rates), n_bars, n_sym,
        ctypes.byref(c_cfg), ctypes.byref(result), _to_ptr_d(eq),
    )

    return _result_to_dict(result)


def run_worksteal_batch_fast(
    all_bars: Dict[str, pd.DataFrame],
    configs: list,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[dict]:
    if not configs:
        return []
    timestamps, valid, opens, highs_arr, lows_arr, closes_arr, fee_rates, symbols = \
        _prepare_data(all_bars, configs[0], start_date, end_date)

    n_bars = len(timestamps)
    n_sym = len(symbols)
    nc = len(configs)

    if n_bars == 0:
        return [{"total_return": 0, "sortino": 0, "max_drawdown": 0,
                 "final_equity": c.initial_cash} for c in configs]

    c_cfgs = (CWorkStealConfig * nc)()
    for i, cfg in enumerate(configs):
        c = _config_to_c(cfg)
        ctypes.memmove(ctypes.byref(c_cfgs[i]), ctypes.byref(c), ctypes.sizeof(CWorkStealConfig))

    results = (CSimResult * nc)()

    _lib.worksteal_simulate_batch(
        _to_ptr_d(timestamps), _to_ptr_i(valid),
        _to_ptr_d(opens), _to_ptr_d(highs_arr),
        _to_ptr_d(lows_arr), _to_ptr_d(closes_arr),
        _to_ptr_d(fee_rates), n_bars, n_sym,
        c_cfgs, results, nc,
    )

    return [_result_to_dict(results[i]) for i in range(nc)]
