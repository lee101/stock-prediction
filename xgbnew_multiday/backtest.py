"""Backtest for the multi-horizon meta-selector.

Key differences from xgbnew.backtest:
  * A trade may span multiple days.
  * On each day the meta-selector picks (symbol, horizon) with the highest
    expected return ``score = (p_up - 0.5) * abs_ret_ewm_N`` where
    ``abs_ret_ewm_N`` is the rolling mean of past forward-N-day abs-returns
    for that symbol (a per-symbol magnitude prior estimated from history).
  * Realism knobs mirror xgbnew: binary fill at open, fee=10bps per side,
    fill_buffer_bps=5 each side, margin rate 6.25%/yr on (L-1) for each
    calendar day the position is open, decision_lag support (scores for day
    D's trade are computed from features at end of day D-1).

Position model — start simple: K=1 slot (fully allocated). When a trade is
active, new picks are ignored until the trade closes. Total-return equity
curve is the product of (1 + net_return_per_trade).

Outputs: list of ``Trade`` records + ``BacktestResult`` summary (median
monthly PnL, p10, neg-window-rate, max DD, win rate, avg hold days).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ANNUAL_MARGIN_RATE = 0.0625
CAL_DAYS_PER_YEAR = 365


@dataclass
class MultiDayConfig:
    horizons: tuple[int, ...] = (1, 2, 3, 5, 10)
    fee_bps_per_side: float = 10.0     # 10 bps fee per side (Alpaca retail)
    fill_buffer_bps: float = 5.0       # adverse fill each side
    leverage: float = 1.0
    decision_lag: int = 2              # 1d feature lag is baked in; this is extra
    min_expected_ret: float = 0.0      # hurdle on E[r_N] (fraction). 0 disables.
    min_prob: float = 0.50             # require p_up > this for the chosen horizon
    min_dollar_vol: float = 5e6
    top_n_slots: int = 1               # 1 = single-slot sim; >1 = concurrent slots
    allocation_per_slot: float = 1.0   # scalar on leverage for each slot


@dataclass
class Trade:
    entry_date: date
    exit_date: date
    symbol: str
    horizon: int
    prob: float
    expected_ret: float
    gross_ret: float                 # pre-cost close/open
    net_ret: float                   # leveraged, post cost + margin
    hold_days: int


@dataclass
class MultiDayResult:
    trades: list[Trade] = field(default_factory=list)
    equity_by_date: pd.Series = field(default_factory=pd.Series)  # index=date, value=equity
    summary: dict = field(default_factory=dict)


def _symbol_abs_ret_prior(
    train_df: pd.DataFrame, horizons: Sequence[int],
) -> pd.DataFrame:
    """Per-symbol historical mean of |forward-N-day return|.

    Used by the meta-selector as a cheap magnitude prior (so longer horizons,
    which naturally have bigger |returns|, get credit automatically).
    Returns a DataFrame indexed by symbol with columns 'abs_prior_{N}d'.
    """
    rows = []
    for sym, sub in train_df.groupby("symbol", sort=False):
        row = {"symbol": sym}
        for n in horizons:
            col = f"abs_fwd_{n}d"
            vcol = f"valid_fwd_{n}d"
            if col in sub.columns and vcol in sub.columns:
                valid = sub[sub[vcol] == 1]
                row[f"abs_prior_{n}d"] = float(valid[col].mean()) if len(valid) else 0.0
            else:
                row[f"abs_prior_{n}d"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows).set_index("symbol")


def build_daily_candidate_table(
    test_df: pd.DataFrame,
    probs_by_horizon: dict[int, np.ndarray],
    abs_prior_by_symbol: pd.DataFrame,
    cfg: MultiDayConfig,
) -> pd.DataFrame:
    """Per row of test_df, attach p_up_{N}, expected_ret_{N}, and pick the
    argmax horizon. Returns a DataFrame with columns date, symbol, horizon_pick,
    prob_pick, expected_ret_pick, plus a score = expected_ret_pick for sorting.
    """
    df = test_df.copy().reset_index(drop=True)
    for n, probs in probs_by_horizon.items():
        df[f"prob_{n}d"] = probs
    # Attach priors
    prior = abs_prior_by_symbol.rename_axis("symbol").reset_index()
    df = df.merge(prior, on="symbol", how="left")
    for n in cfg.horizons:
        col_prior = f"abs_prior_{n}d"
        if col_prior not in df.columns:
            df[col_prior] = 0.0
        df[col_prior] = df[col_prior].fillna(0.0)
        df[f"expected_ret_{n}d"] = (df[f"prob_{n}d"] - 0.5) * 2.0 * df[col_prior]

    # For each row, pick horizon with max expected return AND prob > min_prob
    pick_exp = np.full(len(df), -np.inf, dtype=np.float64)
    pick_h = np.full(len(df), 0, dtype=np.int32)
    pick_p = np.full(len(df), np.nan, dtype=np.float64)
    pick_prior = np.full(len(df), np.nan, dtype=np.float64)
    for n in cfg.horizons:
        vcol = f"valid_fwd_{n}d"
        if vcol not in df.columns:
            continue
        ok = (df[vcol] == 1) & (df[f"prob_{n}d"] >= cfg.min_prob)
        exp = df[f"expected_ret_{n}d"].values
        mask = ok.values & (exp > pick_exp)
        pick_exp = np.where(mask, exp, pick_exp)
        pick_h = np.where(mask, n, pick_h)
        pick_p = np.where(mask, df[f"prob_{n}d"].values, pick_p)
        pick_prior = np.where(mask, df[f"abs_prior_{n}d"].values, pick_prior)

    df["horizon_pick"] = pick_h
    df["prob_pick"] = pick_p
    df["expected_ret_pick"] = pick_exp
    df["abs_prior_pick"] = pick_prior
    df["score"] = df["expected_ret_pick"]
    # Mask out rows where no horizon was valid
    df.loc[df["horizon_pick"] == 0, "score"] = -np.inf
    return df


def _compute_trade_return(
    entry_row: pd.Series,
    exit_row: pd.Series,
    horizon: int,
    cfg: MultiDayConfig,
) -> tuple[float, float]:
    """Return (gross_ret, net_ret_leveraged). ``entry_row`` is the chosen
    (symbol, date) pick row (it has actual_open). ``exit_row`` holds the
    exit bar's actual_close. horizon is for margin-cost computation only.
    """
    open_ = float(entry_row["actual_open"])
    close = float(exit_row["actual_close"])
    if open_ <= 0 or close <= 0:
        return 0.0, 0.0

    buf = cfg.fill_buffer_bps / 1e4
    fee = cfg.fee_bps_per_side / 1e4
    # Worst-case binary fill: enter above open by buf, exit below close by buf.
    fill_open = open_ * (1.0 + buf)
    fill_close = close * (1.0 - buf)

    gross = close / open_ - 1.0
    net_raw = fill_close / fill_open - 1.0
    round_trip_fee = 2.0 * fee
    # Margin cost on (leverage - 1) * calendar_days of holding
    lev = cfg.leverage
    margin_cost = max(lev - 1.0, 0.0) * ANNUAL_MARGIN_RATE * max(horizon - 1, 0) / CAL_DAYS_PER_YEAR
    net_ret_lev = lev * net_raw - round_trip_fee - margin_cost
    return gross, net_ret_lev


def simulate(
    test_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    cfg: MultiDayConfig,
    initial_cash: float = 10_000.0,
) -> MultiDayResult:
    """K-slot multi-day sim. Candidate table must have columns:
    date, symbol, horizon_pick, prob_pick, expected_ret_pick, score, actual_open.

    Realism:
      * Each day, up to ``cfg.top_n_slots`` concurrent positions are held.
        Each slot is allocated ``equity * allocation_per_slot / top_n_slots``
        fraction of current total equity at entry. Profits/losses from each
        slot compound independently (slot's notional at close / notional at
        open × equity_at_entry). When a slot closes, its released capital is
        re-added to the free pool and available for new entries.
      * Only one position per unique symbol may be open at once (avoid
        double-counting). If today's pick is already held, we skip.
      * decision_lag: dl=1 = enter today at open (on same day we see signal);
                      dl=2 = enter at tomorrow's open (recommended for realism
                      with end-of-day signal generation).
    """
    # Preprocessing: per-symbol sorted date→(open, close) DataFrame for lookups.
    td = test_df[["symbol", "date", "actual_open", "actual_close"]].copy()
    per_sym: dict[str, pd.DataFrame] = {
        sym: sub.sort_values("date").reset_index(drop=True)
        for sym, sub in td.groupby("symbol", sort=False)
    }

    all_days = sorted(td["date"].unique())
    day_to_idx = {d: i for i, d in enumerate(all_days)}

    # Organize candidates by date (top picks per day, score-sorted)
    cand = candidate_df.copy()
    cand = cand[cand["horizon_pick"] > 0]
    cand = cand.sort_values(["date", "score"], ascending=[True, False])
    cand_by_day: dict = {d: sub for d, sub in cand.groupby("date", sort=False)}

    trades: list[Trade] = []
    equity_history: list[tuple[date, float]] = []
    dl = max(int(cfg.decision_lag), 1)
    k = max(int(cfg.top_n_slots), 1)
    alloc_per_slot = float(cfg.allocation_per_slot) / k  # fraction of equity

    # Active positions: list of dicts
    # {symbol, horizon, entry_day_idx, exit_day_idx, entry_cash, exit_row_future, ...}
    active: list[dict] = []
    equity = initial_cash
    # Free cash pool — we keep the equity compounding by tracking total equity,
    # but gate new entries by the number of free slots.
    def free_slots() -> int:
        return k - len(active)

    # Pre-compute picks per signal-day (ordered by score desc)
    picks_per_sig_day: dict[int, list[tuple]] = {}
    for i, d in enumerate(all_days):
        todays = cand_by_day.get(d)
        if todays is None or len(todays) == 0:
            continue
        filt = todays[
            (todays["prob_pick"] >= cfg.min_prob)
            & (todays["expected_ret_pick"] >= cfg.min_expected_ret)
        ]
        if filt.empty:
            continue
        # De-dup by symbol (keep highest-scoring row per symbol)
        filt = filt.drop_duplicates(subset=["symbol"], keep="first")
        picks_per_sig_day[i] = [
            (str(r["symbol"]), int(r["horizon_pick"]),
             float(r["prob_pick"]), float(r["expected_ret_pick"]))
            for _, r in filt.head(k * 3).iterrows()  # keep 3K candidates to
                                                    # allow for skipping held syms
        ]

    def _close_one(pos):
        nonlocal equity
        sym_df = per_sym.get(pos["symbol"])
        if sym_df is None:
            return
        exit_row = sym_df.iloc[pos["exit_loc"]]
        entry_row = pos["entry_row"]
        n = pos["horizon"]
        gross, net_per_slot = _compute_trade_return(entry_row, exit_row, n, cfg)
        # Slot invested alloc_per_slot of equity_at_entry; slot returns
        # alloc_per_slot * (1 + net). Other (1 - alloc_per_slot) sat in cash.
        slot_notional_ratio = (1.0 - alloc_per_slot) + alloc_per_slot * (1.0 + net_per_slot)
        equity = equity * slot_notional_ratio
        trades.append(Trade(
            entry_date=pos["entry_date"],
            exit_date=exit_row["date"],
            symbol=pos["symbol"],
            horizon=n,
            prob=pos["prob"],
            expected_ret=pos["expected_ret"],
            gross_ret=float(gross),
            net_ret=float(net_per_slot),
            hold_days=n,
        ))

    for i, d in enumerate(all_days):
        # 1) Close any positions whose exit day index == i (horizon > 1)
        still_active = []
        for pos in active:
            if pos["exit_day_idx"] == i:
                _close_one(pos)
            else:
                still_active.append(pos)
        active = still_active

        equity_history.append((d, equity))

        # 2) Open new positions if free slots. Signals from sig_day = i - (dl-1).
        sig_day = i - (dl - 1)
        if sig_day < 0 or sig_day not in picks_per_sig_day:
            continue
        held_syms = {pos["symbol"] for pos in active}
        free = free_slots()
        if free <= 0:
            continue

        newly_opened = []
        opened_today = 0
        for sym, n, prob, exp_r in picks_per_sig_day[sig_day]:
            if opened_today >= free:
                break
            if sym in held_syms:
                continue  # already holding this symbol
            sym_df = per_sym.get(sym)
            if sym_df is None:
                continue
            entry_match = sym_df[sym_df["date"] == d]
            if entry_match.empty:
                continue
            entry_loc = int(entry_match.index[0])
            exit_loc = entry_loc + (n - 1)
            if exit_loc >= len(sym_df):
                continue
            exit_row = sym_df.iloc[exit_loc]
            exit_day = exit_row["date"]
            exit_day_idx = day_to_idx.get(exit_day, i + n - 1)
            pos = {
                "symbol": sym,
                "horizon": n,
                "prob": prob,
                "expected_ret": exp_r,
                "entry_date": d,
                "entry_day_idx": i,
                "exit_day_idx": exit_day_idx,
                "entry_row": entry_match.iloc[0],
                "exit_loc": exit_loc,
            }
            held_syms.add(sym)
            opened_today += 1
            if exit_day_idx == i:
                # Horizon-1 position: opens and closes same day
                _close_one(pos)
            else:
                active.append(pos)
                newly_opened.append(pos)

    equity_series = pd.Series(
        {d: v for d, v in equity_history}, name="equity",
    ).sort_index()

    summary = summarize(trades, equity_series, initial_cash)
    return MultiDayResult(trades=trades, equity_by_date=equity_series, summary=summary)


def summarize(
    trades: list[Trade],
    equity: pd.Series,
    initial_cash: float,
) -> dict:
    """Aggregate metrics for the 120d backtest target."""
    if not trades or len(equity) == 0:
        return {
            "n_trades": 0,
            "total_return_pct": 0.0,
            "median_monthly_pnl_pct": 0.0,
            "p10_monthly_pnl_pct": 0.0,
            "neg_window_frac": float("nan"),
            "max_dd_pct": 0.0,
            "win_rate": float("nan"),
            "avg_hold_days": 0.0,
            "total_days": len(equity),
        }

    returns = np.array([t.net_ret for t in trades], dtype=np.float64)
    hold_days = np.array([t.hold_days for t in trades], dtype=np.float64)
    total_return = float(equity.iloc[-1] / initial_cash - 1.0)

    # Rolling 30-cal-day windows for median monthly PnL
    eq_df = equity.to_frame("equity")
    eq_df.index = pd.to_datetime(eq_df.index)
    eq_df = eq_df.sort_index()
    # 21-trading-day rolling return
    eq_vals = eq_df["equity"].values
    roll = []
    for i in range(21, len(eq_vals)):
        roll.append(eq_vals[i] / eq_vals[i - 21] - 1.0)
    roll = np.array(roll, dtype=np.float64) if roll else np.array([0.0])
    med_monthly = float(np.median(roll)) if len(roll) else 0.0
    p10_monthly = float(np.percentile(roll, 10)) if len(roll) else 0.0
    neg_frac = float((roll < 0).mean()) if len(roll) else float("nan")

    # Max drawdown
    running_max = np.maximum.accumulate(eq_vals)
    dd = eq_vals / running_max - 1.0
    max_dd = float(-dd.min())

    return {
        "n_trades": len(trades),
        "total_return_pct": 100.0 * total_return,
        "median_monthly_pnl_pct": 100.0 * med_monthly,
        "p10_monthly_pnl_pct": 100.0 * p10_monthly,
        "neg_window_frac": neg_frac,
        "max_dd_pct": 100.0 * max_dd,
        "win_rate": float((returns > 0).mean()),
        "avg_hold_days": float(hold_days.mean()),
        "avg_trade_ret_pct": 100.0 * float(returns.mean()),
        "total_days": int(len(equity)),
        "horizons_used": {
            int(h): int((np.array([t.horizon for t in trades]) == h).sum())
            for h in sorted({t.horizon for t in trades})
        },
    }
