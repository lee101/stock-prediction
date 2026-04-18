"""Numpy reference for the multi-symbol daily PORTFOLIO bracket env step.

This is the executable spec a CUDA kernel must match bit-for-bit (within
fp32 rounding) on golden tests. It is intentionally written for clarity,
not speed — the CUDA version will be the production path.

NOTE — distinguish from `gpu_trading_env/csrc/env_step_multisym.cu`, which
is a Discrete(1+2S) "pick one symbol to long/short" kernel (top_n=1 style).
This module implements the FULL PORTFOLIO action where each symbol gets its
own (limit_buy, limit_sell, buy_qty, sell_qty) and Σ|notional| is bounded
by max_leverage × equity. New design 2026-04-18.

Action per (env, symbol):
  limit_buy_offset_bps   : float  -- buy fills if bar_low  <= prev_close * (1 + offset/1e4)
  limit_sell_offset_bps  : float  -- sell fills if bar_high >= prev_close * (1 + offset/1e4)
  buy_qty_pct            : float >= 0  -- target buy notional as pct of equity (>1 OK; leverage cap handles overflow)
  sell_qty_pct           : float >= 0  -- target sell notional as pct of equity

State per env:
  cash      : float
  positions : float[num_symbols]   -- shares held (signed, neg for short)

Per-step economics:
  - Limit fills are binary against bar [low, high] (no slippage beyond limit).
  - After fills, scale positions proportionally so sum(|notional|) <= max_lev * equity.
  - Trading fee = fee_bps * |notional_traded|.
  - Margin interest = (annual_rate / 252) * max(0, sum(|notional|) - equity).

Returns NEW state + reward (= delta equity / equity_prev).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class MultiSymBracketConfig:
    fee_bps: float = 0.278           # Alpaca real fee
    fill_buffer_bps: float = 5.0     # buyer must beat low by this many bps
    max_leverage: float = 2.0
    annual_margin_rate: float = 0.0625  # 6.25% APR
    trading_days_per_year: int = 252


def _per_day_margin_rate(cfg: MultiSymBracketConfig) -> float:
    return cfg.annual_margin_rate / float(cfg.trading_days_per_year)


def step(
    cash: np.ndarray,            # [B]
    positions: np.ndarray,       # [B, S]
    actions: np.ndarray,         # [B, S, 4]: (lim_buy_bps, lim_sell_bps, buy_pct, sell_pct)
    prev_close: np.ndarray,      # [B, S]
    bar_open: np.ndarray,        # [B, S]
    bar_high: np.ndarray,        # [B, S]
    bar_low: np.ndarray,         # [B, S]
    bar_close: np.ndarray,       # [B, S]
    tradable_mask: np.ndarray,   # [B, S] bool — false rows skipped (delisted/halted)
    cfg: MultiSymBracketConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Apply one daily bracket-trading step.

    Returns (new_cash[B], new_positions[B, S], reward[B], info_dict).
    """
    B, S = positions.shape
    assert cash.shape == (B,)
    assert actions.shape == (B, S, 4)
    assert prev_close.shape == bar_open.shape == bar_high.shape == bar_low.shape == bar_close.shape == (B, S)
    assert tradable_mask.shape == (B, S)

    fee_rate = cfg.fee_bps * 1e-4
    fb = cfg.fill_buffer_bps * 1e-4

    lim_buy_bps  = actions[..., 0]
    lim_sell_bps = actions[..., 1]
    buy_pct      = np.maximum(actions[..., 2], 0.0)
    sell_pct     = np.maximum(actions[..., 3], 0.0)

    # Equity *before* the step: cash + Σ positions * prev_close.
    pos_value_prev = (positions * prev_close).sum(axis=1)  # [B]
    equity_prev = cash + pos_value_prev                    # [B]
    equity_prev_safe = np.where(equity_prev > 1e-9, equity_prev, 1e-9)

    limit_buy_px  = prev_close * (1.0 + lim_buy_bps  * 1e-4)
    limit_sell_px = prev_close * (1.0 + lim_sell_bps * 1e-4)

    # Binary bracket fills against the bar with a fill_buffer crossing margin.
    buy_could_fill  = tradable_mask & (bar_low  <= limit_buy_px  * (1.0 - fb))
    sell_could_fill = tradable_mask & (bar_high >= limit_sell_px * (1.0 + fb))

    # Same-bar dual fills are unrealistic without temporal ordering — they let
    # a policy capture H-L spread risk-free by placing buy-low + sell-high on
    # the same instrument. Conservative resolution: only the side closer to
    # bar_open executes (the side bar_open passes first as price walks). Only
    # contest when both sides have nonzero size — a zero-size side is no order.
    buy_dist  = np.abs(bar_open - limit_buy_px)
    sell_dist = np.abs(bar_open - limit_sell_px)
    both_could = buy_could_fill & sell_could_fill & (buy_pct > 0) & (sell_pct > 0)
    buy_first  = buy_dist <= sell_dist
    buy_filled  = buy_could_fill  & (~both_could | buy_first)
    sell_filled = sell_could_fill & (~both_could | ~buy_first)

    # Notional targets per symbol (fraction of *prior* equity).
    buy_notional_target  = buy_pct  * equity_prev_safe[:, None]
    sell_notional_target = sell_pct * equity_prev_safe[:, None]
    # Convert notional → shares using the limit price (the marginal fill price).
    buy_shares  = np.where(buy_filled,  buy_notional_target  / np.maximum(limit_buy_px,  1e-9), 0.0)
    sell_shares = np.where(sell_filled, sell_notional_target / np.maximum(limit_sell_px, 1e-9), 0.0)

    # Leverage clip: scale ONLY new buy/sell shares so that
    # Σ|positions + new_delta| * bar_close <= max_lev * equity_prev.
    # Crucially we do NOT shrink the existing position — that would create
    # equity from nothing (the shrunk size has no corresponding cash flow,
    # which a short-only policy can compound into runaway equity growth on a
    # flat tape). Real broker semantics: new orders that breach margin get
    # partial-filled, existing positions are untouched.
    candidate_new_pos = positions + buy_shares - sell_shares
    candidate_notional = (np.abs(candidate_new_pos) * bar_close).sum(axis=1)
    cap = cfg.max_leverage * equity_prev_safe
    over = candidate_notional > cap
    if over.any():
        existing_notional = (np.abs(positions) * bar_close).sum(axis=1)
        delta_notional = (np.abs(buy_shares - sell_shares) * bar_close).sum(axis=1)
        headroom = np.maximum(0.0, cap - existing_notional)
        # alpha is the max fraction of the requested trade that fits the cap.
        alpha = np.where(delta_notional > 1e-9,
                         headroom / np.maximum(delta_notional, 1e-9),
                         1.0)
        alpha = np.clip(alpha, 0.0, 1.0)
        # Only apply to envs that actually breached.
        alpha = np.where(over, alpha, 1.0)
        buy_shares  = buy_shares  * alpha[:, None]
        sell_shares = sell_shares * alpha[:, None]

    new_positions = positions + buy_shares - sell_shares

    # Cash flow: pay for buys (at limit_buy_px), receive sells (at limit_sell_px).
    cash_out = (buy_shares  * limit_buy_px ).sum(axis=1)
    cash_in  = (sell_shares * limit_sell_px).sum(axis=1)
    fee_cost = ((buy_shares * limit_buy_px) + (sell_shares * limit_sell_px)).sum(axis=1) * fee_rate
    new_cash = cash - cash_out + cash_in - fee_cost

    # Margin interest on borrowed dollars (post-fill notional vs equity).
    notional_close = (np.abs(new_positions) * bar_close).sum(axis=1)
    borrowed = np.maximum(0.0, notional_close - equity_prev_safe)
    margin_cost = borrowed * _per_day_margin_rate(cfg)
    new_cash = new_cash - margin_cost

    # New equity uses bar_close to MTM.
    new_equity = new_cash + (new_positions * bar_close).sum(axis=1)
    reward = (new_equity - equity_prev) / equity_prev_safe

    info = {
        "equity_prev": equity_prev,
        "new_equity": new_equity,
        "fees": fee_cost,
        "margin_cost": margin_cost,
        "borrowed": borrowed,
        "buy_filled": buy_filled,
        "sell_filled": sell_filled,
        "leverage_clip_active": over,
    }
    return new_cash, new_positions, reward, info
