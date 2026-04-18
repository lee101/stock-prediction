"""GPU-parallel realism-gate sim for the prod screened32 ensemble.

Single-function entry point `run_cell_gpu` that evaluates one
(fill_buffer_bps, max_leverage, slippage_bps) cell across N windows in
LOCKSTEP on CUDA. The N windows advance one sim-day at a time; all
state (cash, position, equity, drawdown) lives in per-window parallel
tensors, and the 12-policy ensemble forward runs once per step via
`batched_ensemble.StackedEnsemble`.

Pinned-parity with the CPU reference (`hourly_replay.simulate_daily_policy`)
on the DEPLOY-GATE PATH:
  * decision_lag=2
  * disable_shorts=True   (long-only policy)
  * deterministic=True    (argmax over softmax-avg probs)
  * alloc_bins=1, level_bins=1, action_max_offset_bps=0
  * binary_fills=True (default)
  * no trailing_stop, no max_hold, no min_notional, no short_borrow
  * no early_exit branches

Paths outside this envelope fall back to the CPU sim automatically via
`realism_gate_accelerated` — we do NOT silently widen the GPU path.

Bit-identical parity is NOT the promise — it's "within fp32 epsilon on
final total_return, sortino, and max_drawdown". Tests in
`tests/test_gpu_realism_gate_parity.py` assert max abs delta < 1e-5 on
total_return vs the CPU sim across 50 random starts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from pufferlib_market.batched_ensemble import StackedEnsemble, can_batch
from pufferlib_market.evaluate_holdout import load_policy
from pufferlib_market.hourly_replay import INITIAL_CASH, MktdData, P_CLOSE, P_HIGH, P_LOW

# Indices of prices tensor columns (must match hourly_replay constants)
_P_OPEN = 0  # noqa: F841 (keep for clarity)
_P_HIGH = P_HIGH
_P_LOW = P_LOW
_P_CLOSE = P_CLOSE


@dataclass
class GpuCellResult:
    total_returns: np.ndarray   # [N]
    sortinos: np.ndarray        # [N]
    max_drawdowns: np.ndarray   # [N]


def _stage_windows(
    data: MktdData,
    starts: Sequence[int],
    window_days: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice all N windows into contiguous [N, T+1, S, *] CUDA tensors.

    We include window_days+1 rows because the sim reads prices at both
    `t` (for action execution at current close) and `t+1` (for equity
    after advance). The CPU sim internally clamps t at T-1 when needed —
    we mirror that by duplicating the last row into slot T if starts[i]+T
    runs past data.num_timesteps-1.

    Returns (prices, features, tradable) where tradable is a bool tensor of
    shape [N, T, S] matching `_is_tradable` (all-True if the input data has
    no tradable mask).
    """
    T = int(window_days) + 1
    S = int(data.num_symbols)
    F = int(data.features.shape[2])
    N = len(starts)
    prices_np = np.empty((N, T, S, data.prices.shape[2]), dtype=np.float32)
    feats_np = np.empty((N, T, S, F), dtype=np.float32)
    has_tradable = data.tradable is not None
    tradable_np = np.ones((N, T, S), dtype=np.bool_)
    for i, s in enumerate(starts):
        end = int(s) + T
        if end <= data.num_timesteps:
            prices_np[i] = data.prices[int(s):end]
            feats_np[i] = data.features[int(s):end]
            if has_tradable:
                tradable_np[i] = data.tradable[int(s):end].astype(np.bool_)
        else:
            # clamp-at-last-bar behaviour mirrors CPU sim
            rows = data.num_timesteps - int(s)
            prices_np[i, :rows] = data.prices[int(s):]
            prices_np[i, rows:] = data.prices[-1:]  # broadcast last bar
            feats_np[i, :rows] = data.features[int(s):]
            feats_np[i, rows:] = data.features[-1:]
            if has_tradable:
                tradable_np[i, :rows] = data.tradable[int(s):].astype(np.bool_)
                tradable_np[i, rows:] = data.tradable[-1:].astype(np.bool_)
    prices_t = torch.from_numpy(prices_np).to(device=device, dtype=torch.float32).contiguous()
    feats_t = torch.from_numpy(feats_np).to(device=device, dtype=torch.float32).contiguous()
    tradable_t = torch.from_numpy(tradable_np).to(device=device).contiguous()
    return prices_t, feats_t, tradable_t


def _build_obs_batch(
    feats: torch.Tensor,          # [N, T+1, S, F]
    prices: torch.Tensor,         # [N, T+1, S, 5]
    t: int,
    cash: torch.Tensor,           # [N]
    pos_sym: torch.Tensor,        # [N] int32, -1 = flat
    pos_qty: torch.Tensor,        # [N] float
    pos_entry: torch.Tensor,      # [N] float
    hold_steps: torch.Tensor,     # [N] int32
    step: int,
    max_steps: int,
    portfolio_scale: float,
) -> torch.Tensor:
    """Build obs[N, S*F + 5 + S] matching `hourly_replay._build_obs` exactly.

    CPU obs uses features at t_obs = max(0, t-1). Position contribution
    uses prices at t_obs too. One-hot position encoding with +1 long / -1 short.
    """
    N, Tp1, S, F = feats.shape
    t_obs = max(0, t - 1)
    base = S * F
    obs_dim = base + 5 + S
    obs = torch.zeros(N, obs_dim, device=feats.device, dtype=torch.float32)
    # Features at t_obs: [N, S, F] → [N, S*F]
    obs[:, :base] = feats[:, t_obs].reshape(N, base)

    # Per-window held price at t_obs for pos-holding entries.
    # gather along S dim with pos_sym; use -1 sentinel safely.
    held_mask = pos_sym >= 0
    price_held = torch.zeros(N, device=feats.device, dtype=torch.float32)
    if held_mask.any():
        sym_idx = pos_sym.clamp_min(0).long()
        price_at_tobs = prices[:, t_obs, :, _P_CLOSE]  # [N, S]
        price_held = price_at_tobs.gather(1, sym_idx.unsqueeze(1)).squeeze(1)
        price_held = torch.where(held_mask, price_held, torch.zeros_like(price_held))

    # Long-only deploy path: is_short always False. pos_val = qty * price.
    pos_val = torch.where(held_mask, pos_qty * price_held, torch.zeros_like(price_held))
    unreal = torch.where(held_mask, pos_qty * (price_held - pos_entry), torch.zeros_like(price_held))

    denom = max(abs(float(portfolio_scale)), 1e-12)
    obs[:, base + 0] = cash / denom
    obs[:, base + 1] = pos_val / denom
    obs[:, base + 2] = unreal / denom
    obs[:, base + 3] = hold_steps.to(torch.float32) / max(max_steps, 1)
    obs[:, base + 4] = float(step) / max(max_steps, 1)
    # One-hot position: +1 long, -1 short. Long-only here.
    if held_mask.any():
        sym_idx = pos_sym.clamp_min(0).long()
        row_idx = torch.arange(N, device=feats.device)
        onehot_vals = torch.where(held_mask, torch.ones_like(price_held), torch.zeros_like(price_held))
        obs[row_idx, base + 5 + sym_idx] = onehot_vals
    return obs


def _argmax_with_short_mask(
    logits: torch.Tensor,  # [N, A]
    num_symbols: int,
    per_symbol_actions: int,
    disable_shorts: bool,
) -> torch.Tensor:
    """Deterministic argmax with an optional shorts mask.

    Mirrors `_mask_all_shorts` in evaluate_holdout: zero probability
    (-inf logit) on all short actions. Long-only path used by prod.
    """
    if disable_shorts:
        side_block = num_symbols * per_symbol_actions
        logits = logits.clone()
        logits[:, 1 + side_block:] = float("-inf")
    return logits.argmax(dim=-1)


def run_cell_gpu(
    data: MktdData,
    *,
    checkpoints: Sequence[str],
    num_symbols: int,
    features_per_sym: int,
    starts: Sequence[int],
    window_days: int,
    fill_buffer_bps: float,
    max_leverage: float,
    fee_rate: float,
    slippage_bps: float,
    decision_lag: int,
    device: torch.device | None = None,
    ensemble_mode: str = "softmax_avg",
) -> GpuCellResult:
    """Evaluate one (fb, lev, slip) cell across all `starts` windows in parallel."""
    if decision_lag != 2:
        raise ValueError(f"GPU path requires decision_lag=2 (got {decision_lag})")
    if ensemble_mode not in ("softmax_avg", "logit_avg"):
        raise ValueError(f"unsupported ensemble_mode {ensemble_mode!r}")
    if device is None:
        device = torch.device("cuda")

    # Load policies + build stacked ensemble
    loaded = [load_policy(str(p), num_symbols, features_per_sym=features_per_sym, device=device)
              for p in checkpoints]
    head = loaded[0]
    alloc_bins = int(head.action_allocation_bins)
    level_bins = int(head.action_level_bins)
    action_max_offset_bps = float(head.action_max_offset_bps)
    per_symbol_actions = max(1, alloc_bins) * max(1, level_bins)
    if alloc_bins != 1 or level_bins != 1 or action_max_offset_bps != 0.0:
        raise ValueError(
            "GPU path restricted to alloc_bins=level_bins=1, action_max_offset_bps=0"
            f" (got {alloc_bins},{level_bins},{action_max_offset_bps})"
        )
    policies = [lp.policy.eval() for lp in loaded]
    if not can_batch(policies):
        raise ValueError("policies are not stack-compatible (per_sym_norm or shape mismatch)")
    stacked = StackedEnsemble.from_policies(policies, device)

    # Stage window data
    N = len(starts)
    prices, feats, tradable = _stage_windows(data, starts, window_days, device)  # [N, W+1, S, *]
    W = int(window_days)
    S = int(num_symbols)
    side_block = S * per_symbol_actions

    # Sim state
    slip_frac = max(0.0, float(slippage_bps)) / 10_000.0
    effective_fee = float(fee_rate) + slip_frac
    init_cash = float(INITIAL_CASH)
    cash = torch.full((N,), init_cash, device=device, dtype=torch.float32)
    pos_sym = torch.full((N,), -1, device=device, dtype=torch.int32)
    pos_qty = torch.zeros(N, device=device, dtype=torch.float32)
    pos_entry = torch.zeros(N, device=device, dtype=torch.float32)
    hold_steps = torch.zeros(N, device=device, dtype=torch.int32)
    peak_equity = torch.full((N,), init_cash, device=device, dtype=torch.float32)
    max_dd = torch.zeros(N, device=device, dtype=torch.float32)
    sum_neg_sq = torch.zeros(N, device=device, dtype=torch.float64)
    sum_ret = torch.zeros(N, device=device, dtype=torch.float64)
    ret_count = 0

    # Decision-lag buffer: deque-of-length decision_lag+1 emulated as ring buffer.
    # Matches CPU semantics:
    #   pending.append(action_now)
    #   if len(pending) <= decision_lag: return 0
    #   return pending.popleft()
    # Equivalently: emit the action from (decision_lag) steps ago; before the
    # buffer fills, emit 0 (flat).
    lag = int(decision_lag)
    action_buf = torch.zeros(N, lag + 1, device=device, dtype=torch.int32)
    buf_count = 0  # number of actions appended so far

    fill_buffer_frac = max(0.0, float(fill_buffer_bps)) / 10_000.0

    obs_scale = init_cash  # matches CPU _build_initial_portfolio_state default for INITIAL_CASH

    for step in range(W):
        t = step
        # --- Observation ---
        obs = _build_obs_batch(
            feats, prices, t, cash, pos_sym, pos_qty, pos_entry,
            hold_steps, step, W, obs_scale,
        )
        # --- Forward ---
        with torch.no_grad():
            logits_stacked = stacked.forward(obs)  # [M, N, A]
            if ensemble_mode == "softmax_avg":
                probs = torch.softmax(logits_stacked, dim=-1).mean(dim=0)  # [N, A]
                logits = torch.log(probs + 1e-8)
            else:
                logits = logits_stacked.mean(dim=0)
        action_now = _argmax_with_short_mask(
            logits, num_symbols, per_symbol_actions, disable_shorts=True
        ).to(torch.int32)

        # --- decision_lag buffer ---
        # append action_now at slot (buf_count % (lag+1))
        slot = buf_count % (lag + 1)
        action_buf[:, slot] = action_now
        buf_count += 1
        if buf_count <= lag:
            action = torch.zeros(N, device=device, dtype=torch.int32)
        else:
            # emit action from lag steps ago: slot (buf_count-1-lag) % (lag+1)
            emit_slot = (buf_count - 1 - lag) % (lag + 1)
            action = action_buf[:, emit_slot]

        # --- equity BEFORE at time t (uses price_cur at close(t)) ---
        held = pos_sym >= 0
        price_cur = torch.zeros(N, device=device, dtype=torch.float32)
        if held.any():
            sym_idx = pos_sym.clamp_min(0).long()
            pc = prices[:, t, :, _P_CLOSE].gather(1, sym_idx.unsqueeze(1)).squeeze(1)
            price_cur = torch.where(held, pc, price_cur)
        equity_before = torch.where(held, cash + pos_qty * price_cur, cash)

        # --- Apply action (mirrors simulate_daily_policy, long-only path) ---
        # Must respect per-bar tradability: when data.tradable is non-None
        # (e.g. screened32 val_full marks weekends with 0), CPU sim refuses
        # to close the current pos OR open a new one when the relevant symbol
        # is not tradable on the current bar. Ignoring this silently drifts
        # the GPU sim by ~20% over 50 days because ~1/7 of steps are "market
        # closed → hold" in CPU but "switch freely" in GPU.
        pos_sym_pre = pos_sym.clone()
        was_held = pos_sym_pre >= 0

        flat_mask = action == 0
        long_mask = (action >= 1) & (action <= S)
        target_sym = (action - 1).clamp_min(0).long()
        same_sym = long_mask & was_held & (target_sym == pos_sym_pre.long())

        row_idx = torch.arange(N, device=device)
        trad_t = tradable[:, t, :]  # [N, S]
        cur_sym_safe = pos_sym_pre.clamp_min(0).long()
        cur_tradable = trad_t.gather(1, cur_sym_safe.unsqueeze(1)).squeeze(1)
        cur_tradable = torch.where(was_held, cur_tradable, torch.ones_like(cur_tradable))
        target_tradable = trad_t.gather(1, target_sym.unsqueeze(1)).squeeze(1)

        # Close cases:
        #   flat_close: action=0 & was_held & cur_tradable
        #   switch_close: long & was_held & ~same_sym & cur_tradable & target_tradable
        # (switch requires BOTH tradable because CPU skips close-then-open
        #  if either side is non-tradable and just increments hold_steps.)
        flat_close = flat_mask & was_held & cur_tradable
        switch_ready = long_mask & was_held & ~same_sym & cur_tradable & target_tradable
        switch_close = switch_ready  # close happens only when both tradable

        # Carry-hold branches (pos preserved, hold_steps += 1):
        #   * flat_mask & was_held & ~cur_tradable
        #   * long_mask & was_held & same_sym (any tradability — CPU checks same_sym first)
        #   * long_mask & was_held & ~same_sym & (~cur_tradable | ~target_tradable)
        flat_hold = flat_mask & was_held & ~cur_tradable
        switch_blocked_hold = long_mask & was_held & ~same_sym & (~cur_tradable | ~target_tradable)

        closing = flat_close | switch_close
        proceeds = pos_qty * price_cur * (1.0 - effective_fee)
        cash = torch.where(closing, cash + proceeds, cash)
        pos_sym = torch.where(closing, torch.full_like(pos_sym, -1), pos_sym)
        pos_qty = torch.where(closing, torch.zeros_like(pos_qty), pos_qty)
        pos_entry = torch.where(closing, torch.zeros_like(pos_entry), pos_entry)
        # flat-close resets hold_steps immediately (CPU's `action==0 & tradable` branch).
        hold_steps = torch.where(flat_close, torch.zeros_like(hold_steps), hold_steps)

        # Open branches:
        #   from_flat:    long_mask & ~was_held & target_tradable
        #   from_switch:  switch_close (already closed above; target_tradable is implied)
        # CPU skips open when target is not tradable (outer `if not target_tradable` branch).
        want_open = (
            (long_mask & ~was_held & target_tradable)
            | switch_close
        )
        close_tgt = prices[row_idx, t, target_sym, _P_CLOSE]
        low_tgt = prices[row_idx, t, target_sym, _P_LOW]
        trigger = close_tgt * (1.0 - fill_buffer_frac)
        fillable = low_tgt <= trigger
        fill_px = close_tgt  # level_offset=0 → target_price = close
        buy_budget = cash * float(max_leverage) * 1.0
        denom = fill_px * (1.0 + effective_fee)
        qty_new = torch.where(denom > 0, buy_budget / denom, torch.zeros_like(denom))
        cost_new = qty_new * denom
        can_open = want_open & fillable & (close_tgt > 0) & (cash > 0) & (qty_new > 0) & (cost_new > 0)
        cash = torch.where(can_open, cash - cost_new, cash)
        pos_sym = torch.where(can_open, target_sym.to(torch.int32), pos_sym)
        pos_qty = torch.where(can_open, qty_new, pos_qty)
        pos_entry = torch.where(can_open, fill_px, pos_entry)
        hold_steps = torch.where(can_open, torch.zeros_like(hold_steps), hold_steps)

        # Increment hold_steps for carry-hold cases and same_sym
        carry_hold = same_sym | flat_hold | switch_blocked_hold
        hold_steps = torch.where(carry_hold, hold_steps + 1, hold_steps)

        # --- Advance to t+1 and compute equity_after ---
        t_new = step + 1
        if t_new >= prices.size(1):
            t_new = prices.size(1) - 1
        held2 = pos_sym >= 0
        price_new = torch.zeros(N, device=device, dtype=torch.float32)
        if held2.any():
            sym_idx = pos_sym.clamp_min(0).long()
            pn = prices[:, t_new, :, _P_CLOSE].gather(1, sym_idx.unsqueeze(1)).squeeze(1)
            price_new = torch.where(held2, pn, price_new)
        equity_after = torch.where(held2, cash + pos_qty * price_new, cash)

        # --- Record return, peak, dd ---
        ret = torch.where(
            equity_before > 1e-6,
            (equity_after - equity_before) / equity_before.clamp_min(1e-12),
            torch.zeros_like(equity_before),
        )
        sum_ret += ret.to(torch.float64)
        sum_neg_sq += torch.where(ret < 0, (ret * ret).to(torch.float64), torch.zeros_like(ret, dtype=torch.float64))
        ret_count += 1
        peak_equity = torch.maximum(peak_equity, equity_after)
        dd = torch.where(peak_equity > 0, (peak_equity - equity_after) / peak_equity.clamp_min(1e-12), torch.zeros_like(peak_equity))
        max_dd = torch.maximum(max_dd, dd)

    # --- Final close at t_new (CPU sim closes held position when done=True) ---
    # Mirrors `simulate_daily_policy`: `cash, win = _close_position(cash, pos, price_end, effective_fee)`
    # then `final_equity = float(cash)`. Without this, GPU total_return omits the final exit fee
    # and drifts by ~effective_fee × notional from CPU.
    held_final = pos_sym >= 0
    t_last = prices.size(1) - 1
    if held_final.any():
        sym_idx = pos_sym.clamp_min(0).long()
        price_end = prices[:, t_last, :, _P_CLOSE].gather(1, sym_idx.unsqueeze(1)).squeeze(1)
        proceeds_final = pos_qty * price_end * (1.0 - effective_fee)
        cash = torch.where(held_final, cash + proceeds_final, cash)
        pos_qty = torch.where(held_final, torch.zeros_like(pos_qty), pos_qty)
        pos_sym = torch.where(held_final, torch.full_like(pos_sym, -1), pos_sym)

    # --- Final metrics ---
    total_return = (cash / init_cash) - 1.0
    # Sortino: mean(ret) / stddev_neg(ret), scaled by sqrt(periods_per_year).
    # Use the CPU sim's exact definition: sqrt(sum_neg_sq / count) as downside dev.
    mean_ret = sum_ret / max(1, ret_count)
    downside_var = sum_neg_sq / max(1, ret_count)
    downside_std = torch.sqrt(downside_var.clamp_min(0.0))
    # The CPU sim uses periods_per_year=365 by default. Sortino is mean/downside scaled.
    # Production gate annualises; we follow suit for parity.
    periods_per_year = 365.0
    safe = downside_std > 1e-12
    sortino = torch.where(safe, (mean_ret / downside_std.clamp_min(1e-12)) * np.sqrt(periods_per_year), torch.zeros_like(mean_ret))

    return GpuCellResult(
        total_returns=total_return.detach().to(torch.float64).cpu().numpy(),
        sortinos=sortino.detach().to(torch.float64).cpu().numpy(),
        max_drawdowns=max_dd.detach().to(torch.float64).cpu().numpy(),
    )
