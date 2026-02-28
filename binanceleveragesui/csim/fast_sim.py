"""Fast C market simulator with Python wrapper via ctypes."""
from __future__ import annotations
import ctypes, os, subprocess
from ctypes import c_double, c_int, POINTER, Structure
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

_DIR = Path(__file__).parent
_SO = _DIR / "libmarketsim.so"
_SRC = _DIR / "market_sim.c"


def _ensure_compiled():
    if _SO.exists() and _SO.stat().st_mtime > _SRC.stat().st_mtime:
        return
    subprocess.check_call([
        "gcc", "-O3", "-march=native", "-shared", "-fPIC",
        "-o", str(_SO), str(_SRC), "-lm",
    ])


class CSimConfig(Structure):
    _fields_ = [
        ("max_leverage", c_double),
        ("can_short", c_int),
        ("maker_fee", c_double),
        ("margin_hourly_rate", c_double),
        ("initial_cash", c_double),
        ("fill_buffer_pct", c_double),
        ("min_edge", c_double),
        ("max_hold_bars", c_int),
        ("intensity_scale", c_double),
    ]


class CSimResult(Structure):
    _fields_ = [
        ("total_return", c_double),
        ("sortino", c_double),
        ("max_drawdown", c_double),
        ("final_equity", c_double),
        ("num_trades", c_int),
        ("margin_cost_total", c_double),
    ]


_ensure_compiled()
_lib = ctypes.CDLL(str(_SO))

_lib.simulate.argtypes = [
    POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
    POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
    c_int, POINTER(CSimConfig), POINTER(CSimResult), POINTER(c_double),
]
_lib.simulate.restype = None

_lib.simulate_batch.argtypes = [
    POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
    POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
    c_int, POINTER(CSimConfig), POINTER(CSimResult), c_int,
]
_lib.simulate_batch.restype = None


def _to_ptr(arr):
    return arr.ctypes.data_as(POINTER(c_double))


def simulate_fast(
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    config,
    decision_lag_bars: int = 1,
) -> dict:
    """Drop-in replacement for simulate_with_margin_cost using C backend."""
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    actions = actions.sort_values("timestamp").reset_index(drop=True)
    merged = bars.merge(actions, on=["timestamp", "symbol"], how="inner", suffixes=("", "_act"))

    if decision_lag_bars > 0:
        for col in ['buy_price', 'sell_price', 'buy_amount', 'sell_amount']:
            if col in merged.columns:
                merged[col] = merged[col].shift(decision_lag_bars)
        merged = merged.dropna(subset=['buy_price']).reset_index(drop=True)

    n = len(merged)
    if n == 0:
        return {"total_return": 0, "sortino": 0, "max_drawdown": 0,
                "final_equity": config.initial_cash, "num_trades": 0,
                "margin_cost_total": 0, "margin_cost_pct": 0}

    o = merged["open"].values.astype(np.float64, copy=True)
    h = merged["high"].values.astype(np.float64, copy=True)
    l = merged["low"].values.astype(np.float64, copy=True)
    c = merged["close"].values.astype(np.float64, copy=True)
    bp = merged["buy_price"].fillna(0).values.astype(np.float64, copy=True)
    sp = merged["sell_price"].fillna(0).values.astype(np.float64, copy=True)
    ba = merged["buy_amount"].fillna(0).values.astype(np.float64, copy=True)
    sa = merged["sell_amount"].fillna(0).values.astype(np.float64, copy=True)

    cfg = CSimConfig(
        max_leverage=config.max_leverage,
        can_short=int(config.can_short),
        maker_fee=config.maker_fee,
        margin_hourly_rate=config.margin_hourly_rate,
        initial_cash=config.initial_cash,
        fill_buffer_pct=config.fill_buffer_pct,
        min_edge=config.min_edge,
        max_hold_bars=config.max_hold_bars,
        intensity_scale=config.intensity_scale,
    )
    result = CSimResult()
    eq = np.zeros(n + 1, dtype=np.float64)

    _lib.simulate(
        _to_ptr(o), _to_ptr(h), _to_ptr(l), _to_ptr(c),
        _to_ptr(bp), _to_ptr(sp), _to_ptr(ba), _to_ptr(sa),
        n, ctypes.byref(cfg), ctypes.byref(result), _to_ptr(eq),
    )

    return {
        "total_return": result.total_return,
        "sortino": result.sortino,
        "max_drawdown": result.max_drawdown,
        "final_equity": result.final_equity,
        "num_trades": result.num_trades,
        "margin_cost_total": result.margin_cost_total,
        "margin_cost_pct": result.margin_cost_total / config.initial_cash * 100 if config.initial_cash > 0 else 0,
    }


def simulate_batch_fast(
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    configs: list,
    decision_lag_bars: int = 1,
) -> list[dict]:
    """Run multiple configs on same data in C (no Python loop overhead)."""
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    actions = actions.sort_values("timestamp").reset_index(drop=True)
    merged = bars.merge(actions, on=["timestamp", "symbol"], how="inner", suffixes=("", "_act"))

    if decision_lag_bars > 0:
        for col in ['buy_price', 'sell_price', 'buy_amount', 'sell_amount']:
            if col in merged.columns:
                merged[col] = merged[col].shift(decision_lag_bars)
        merged = merged.dropna(subset=['buy_price']).reset_index(drop=True)

    n = len(merged)
    nc = len(configs)
    if n == 0:
        return [{"total_return": 0, "sortino": 0, "max_drawdown": 0,
                 "final_equity": c.initial_cash, "num_trades": 0,
                 "margin_cost_total": 0, "margin_cost_pct": 0} for c in configs]

    o = merged["open"].values.astype(np.float64, copy=True)
    h = merged["high"].values.astype(np.float64, copy=True)
    l = merged["low"].values.astype(np.float64, copy=True)
    c = merged["close"].values.astype(np.float64, copy=True)
    bp = merged["buy_price"].fillna(0).values.astype(np.float64, copy=True)
    sp = merged["sell_price"].fillna(0).values.astype(np.float64, copy=True)
    ba = merged["buy_amount"].fillna(0).values.astype(np.float64, copy=True)
    sa = merged["sell_amount"].fillna(0).values.astype(np.float64, copy=True)

    cfgs = (CSimConfig * nc)()
    for i, conf in enumerate(configs):
        cfgs[i].max_leverage = conf.max_leverage
        cfgs[i].can_short = int(conf.can_short)
        cfgs[i].maker_fee = conf.maker_fee
        cfgs[i].margin_hourly_rate = conf.margin_hourly_rate
        cfgs[i].initial_cash = conf.initial_cash
        cfgs[i].fill_buffer_pct = conf.fill_buffer_pct
        cfgs[i].min_edge = conf.min_edge
        cfgs[i].max_hold_bars = conf.max_hold_bars
        cfgs[i].intensity_scale = conf.intensity_scale

    results = (CSimResult * nc)()

    _lib.simulate_batch(
        _to_ptr(o), _to_ptr(h), _to_ptr(l), _to_ptr(c),
        _to_ptr(bp), _to_ptr(sp), _to_ptr(ba), _to_ptr(sa),
        n, cfgs, results, nc,
    )

    out = []
    for i in range(nc):
        r = results[i]
        out.append({
            "total_return": r.total_return,
            "sortino": r.sortino,
            "max_drawdown": r.max_drawdown,
            "final_equity": r.final_equity,
            "num_trades": r.num_trades,
            "margin_cost_total": r.margin_cost_total,
            "margin_cost_pct": r.margin_cost_total / configs[i].initial_cash * 100 if configs[i].initial_cash > 0 else 0,
        })
    return out
