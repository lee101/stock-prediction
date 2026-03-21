"""Triton-fused per-timestep portfolio simulation kernel.

Fuses ~20 element-wise tensor ops per timestep into a single GPU kernel launch.
Sequential across timesteps (state dependency), parallel across batch elements.
Remains fully differentiable via custom autograd Function.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

from differentiable_loss_utils import (
    HourlySimulationResult,
    HOURLY_PERIODS_PER_YEAR,
    _EPS,
    _as_tensor,
    _check_shapes,
)
from trainingefficiency.fast_differentiable_sim import _precompute_fill_probs

if HAS_TRITON:
    @triton.jit
    def _sim_step_fwd_kernel(
        # State (read/write)
        cash_ptr,
        inv_ptr,
        prev_val_ptr,
        # Pre-computed per-step inputs (read)
        close_ptr,
        bp_ptr,
        sp_ptr,
        buy_prob_ptr,
        sell_prob_ptr,
        bf_ptr,
        sf_ptr,
        sl_ptr,
        # Batch-level inputs (read)
        can_short_ptr,
        can_long_ptr,
        # Outputs for this step (write)
        pnl_ptr,
        ret_ptr,
        val_ptr,
        exec_buy_ptr,
        exec_sell_ptr,
        inv_out_ptr,
        # Scalars
        fee_buy,
        fee_sell,
        margin_cost_per_step,
        has_margin: tl.constexpr,
        has_can_long: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        N,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        cash = tl.load(cash_ptr + offs, mask=mask)
        inv = tl.load(inv_ptr + offs, mask=mask)
        prev_v = tl.load(prev_val_ptr + offs, mask=mask)
        close = tl.load(close_ptr + offs, mask=mask)
        bp = tl.load(bp_ptr + offs, mask=mask)
        sp = tl.load(sp_ptr + offs, mask=mask)
        b_prob = tl.load(buy_prob_ptr + offs, mask=mask)
        s_prob = tl.load(sell_prob_ptr + offs, mask=mask)
        bf = tl.load(bf_ptr + offs, mask=mask)
        sf = tl.load(sf_ptr + offs, mask=mask)
        sl = tl.load(sl_ptr + offs, mask=mask)
        can_short = tl.load(can_short_ptr + offs, mask=mask)

        equity = cash + inv * close
        equity_pos = tl.maximum(equity, EPS)

        bp_fee = bp * fee_buy + EPS
        max_buy_cash = tl.where(bp > 0.0, cash / bp_fee, 0.0)
        target_notional = sl * equity_pos
        current_notional = inv * bp
        room = tl.maximum(target_notional - current_notional, 0.0)
        leveraged_cap = tl.where(bp > 0.0, room / bp_fee, 0.0)
        buy_cap = tl.where(sl <= 1.0 + 1e-6, tl.maximum(max_buy_cash, 0.0), leveraged_cap)
        buy_qty = bf * buy_cap

        if has_can_long:
            can_long = tl.load(can_long_ptr + offs, mask=mask)
            cover_cap = tl.maximum(-inv, 0.0)
            buy_qty = tl.where(can_long > 0.5, buy_qty, tl.minimum(buy_qty, cover_cap))

        long_to_close = tl.maximum(inv, 0.0)
        sp_fee = sp * fee_buy + EPS
        max_short = tl.where(sp > 0.0, target_notional / sp_fee, 0.0)
        cur_short = tl.maximum(-inv, 0.0)
        short_open = tl.maximum(max_short - cur_short, 0.0)
        sell_cap = long_to_close + tl.where(can_short > 0.5, short_open, 0.0)
        sell_qty = sf * sell_cap

        exec_buys = buy_qty * b_prob
        exec_sells = sell_qty * s_prob

        cash = cash - exec_buys * bp * fee_buy + exec_sells * sp * fee_sell
        inv = inv + exec_buys - exec_sells

        if has_margin:
            pos_val = tl.abs(inv * close)
            eq = cash + inv * close
            margin_used = tl.maximum(pos_val - tl.maximum(eq, 0.0), 0.0)
            cash = cash - margin_used * margin_cost_per_step

        portfolio_value = cash + inv * close
        pnl = portfolio_value - prev_v
        ret = pnl / tl.maximum(prev_v, EPS)

        tl.store(cash_ptr + offs, cash, mask=mask)
        tl.store(inv_ptr + offs, inv, mask=mask)
        tl.store(prev_val_ptr + offs, portfolio_value, mask=mask)

        tl.store(pnl_ptr + offs, pnl, mask=mask)
        tl.store(ret_ptr + offs, ret, mask=mask)
        tl.store(val_ptr + offs, portfolio_value, mask=mask)
        tl.store(exec_buy_ptr + offs, exec_buys, mask=mask)
        tl.store(exec_sell_ptr + offs, exec_sells, mask=mask)
        tl.store(inv_out_ptr + offs, inv, mask=mask)


def _triton_sim_forward(
    closes, b_prices, s_prices, buy_probs, sell_probs,
    b_frac, s_frac, max_lev,
    can_short_t, can_long_t,
    initial_cash, initial_inventory,
    fee_buy, fee_sell, margin_cost_per_step,
):
    batch_shape = closes.shape[:-1]
    steps = closes.shape[-1]
    device = closes.device
    dtype = closes.dtype
    N = batch_shape.numel()

    has_margin = margin_cost_per_step > 0
    has_can_long = can_long_t.numel() > 0

    # Flatten batch dims and transpose to (steps, N) so [idx] slices are contiguous
    closes_t = closes.reshape(N, steps).t().contiguous()
    bp_t = b_prices.reshape(N, steps).t().contiguous()
    sp_t = s_prices.reshape(N, steps).t().contiguous()
    buy_probs_t = buy_probs.reshape(N, steps).t().contiguous()
    sell_probs_t = sell_probs.reshape(N, steps).t().contiguous()
    bf_t = b_frac.reshape(N, steps).t().contiguous()
    sf_t = s_frac.reshape(N, steps).t().contiguous()
    sl_t = max_lev.reshape(N, steps).t().contiguous()
    cs_flat = can_short_t.reshape(N).contiguous()
    cl_flat = can_long_t.reshape(N).contiguous() if has_can_long else can_long_t.reshape(-1).contiguous()

    # State tensors
    cash = torch.full((N,), initial_cash, dtype=dtype, device=device)
    inv = torch.full((N,), initial_inventory, dtype=dtype, device=device)
    prev_val = cash + inv * closes_t[0]

    # Output tensors (steps, N) layout for contiguous writes
    pnl_out = torch.empty(steps, N, dtype=dtype, device=device)
    ret_out = torch.empty(steps, N, dtype=dtype, device=device)
    val_out = torch.empty(steps, N, dtype=dtype, device=device)
    exec_buy_out = torch.empty(steps, N, dtype=dtype, device=device)
    exec_sell_out = torch.empty(steps, N, dtype=dtype, device=device)
    inv_out = torch.empty(steps, N, dtype=dtype, device=device)

    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    for idx in range(steps):
        _sim_step_fwd_kernel[grid](
            cash, inv, prev_val,
            closes_t[idx], bp_t[idx], sp_t[idx],
            buy_probs_t[idx], sell_probs_t[idx],
            bf_t[idx], sf_t[idx], sl_t[idx],
            cs_flat, cl_flat,
            pnl_out[idx], ret_out[idx], val_out[idx],
            exec_buy_out[idx], exec_sell_out[idx], inv_out[idx],
            fee_buy, fee_sell, margin_cost_per_step,
            has_margin=has_margin,
            has_can_long=has_can_long,
            EPS=_EPS,
            BLOCK_SIZE=BLOCK_SIZE,
            N=N,
        )

    # Transpose back to (N, steps) then reshape to batch_shape + (steps,)
    pnl_out = pnl_out.t().reshape(*batch_shape, steps)
    ret_out = ret_out.t().reshape(*batch_shape, steps)
    val_out = val_out.t().reshape(*batch_shape, steps)
    exec_buy_out = exec_buy_out.t().reshape(*batch_shape, steps)
    exec_sell_out = exec_sell_out.t().reshape(*batch_shape, steps)
    inv_out = inv_out.t().reshape(*batch_shape, steps)

    return pnl_out, ret_out, val_out, exec_buy_out, exec_sell_out, inv_out, cash.reshape(batch_shape), inv.reshape(batch_shape)


class _TritonSimAutograd(torch.autograd.Function):
    """Triton forward, PyTorch backward for differentiable sim."""

    @staticmethod
    def forward(
        ctx,
        closes, b_prices, s_prices, buy_probs, sell_probs,
        b_frac, s_frac, max_lev,
        can_short_t, can_long_t,
        initial_cash, initial_inventory,
        fee_buy, fee_sell, margin_cost_per_step,
    ):
        pnl, ret, val, eb, es, inv_path, final_cash, final_inv = _triton_sim_forward(
            closes, b_prices, s_prices, buy_probs, sell_probs,
            b_frac, s_frac, max_lev,
            can_short_t, can_long_t,
            initial_cash, initial_inventory,
            fee_buy, fee_sell, margin_cost_per_step,
        )
        ctx.save_for_backward(
            closes, b_prices, s_prices, buy_probs, sell_probs,
            b_frac, s_frac, max_lev, can_short_t, can_long_t,
        )
        ctx.initial_cash = initial_cash
        ctx.initial_inventory = initial_inventory
        ctx.fee_buy = fee_buy
        ctx.fee_sell = fee_sell
        ctx.margin_cost_per_step = margin_cost_per_step
        return pnl, ret, val, eb, es, inv_path, final_cash, final_inv

    @staticmethod
    def backward(ctx, grad_pnl, grad_ret, grad_val, grad_eb, grad_es, grad_inv_path, grad_fc, grad_fi):
        (
            closes, b_prices, s_prices, buy_probs, sell_probs,
            b_frac, s_frac, max_lev, can_short_t, can_long_t,
        ) = ctx.saved_tensors

        # Re-run forward in PyTorch for autograd backward
        # This is the standard pattern: Triton fwd for speed, PyTorch bwd for correctness
        with torch.enable_grad():
            closes_d = closes.detach().requires_grad_(closes.requires_grad)
            bp_d = b_prices.detach().requires_grad_(b_prices.requires_grad)
            sp_d = s_prices.detach().requires_grad_(s_prices.requires_grad)
            bprob_d = buy_probs.detach().requires_grad_(buy_probs.requires_grad)
            sprob_d = sell_probs.detach().requires_grad_(sell_probs.requires_grad)
            bf_d = b_frac.detach().requires_grad_(b_frac.requires_grad)
            sf_d = s_frac.detach().requires_grad_(s_frac.requires_grad)
            ml_d = max_lev.detach().requires_grad_(max_lev.requires_grad)

            pnl, ret, val, eb, es, inv_p, fc, fi = _pytorch_sim_forward(
                closes_d, bp_d, sp_d, bprob_d, sprob_d,
                bf_d, sf_d, ml_d,
                can_short_t, can_long_t,
                ctx.initial_cash, ctx.initial_inventory,
                ctx.fee_buy, ctx.fee_sell, ctx.margin_cost_per_step,
            )

            outputs = (pnl, ret, val, eb, es, inv_p, fc, fi)
            grads = (grad_pnl, grad_ret, grad_val, grad_eb, grad_es, grad_inv_path, grad_fc, grad_fi)

            # Filter to only grad-requiring inputs
            inputs = []
            for t in [closes_d, bp_d, sp_d, bprob_d, sprob_d, bf_d, sf_d, ml_d]:
                if t.requires_grad:
                    inputs.append(t)

            if not inputs:
                return (None,) * 15

            # Filter outputs/grads to non-None grads
            valid_outputs = []
            valid_grads = []
            for o, g in zip(outputs, grads):
                if g is not None:
                    valid_outputs.append(o)
                    valid_grads.append(g)

            if not valid_outputs:
                return (None,) * 15

            computed_grads = torch.autograd.grad(
                valid_outputs, inputs, valid_grads,
                allow_unused=True,
            )

            result = [None] * 15
            grad_idx = 0
            for i, t in enumerate([closes_d, bp_d, sp_d, bprob_d, sprob_d, bf_d, sf_d, ml_d]):
                if t.requires_grad:
                    result[i] = computed_grads[grad_idx]
                    grad_idx += 1
            return tuple(result)


def _pytorch_sim_forward(
    closes, b_prices, s_prices, buy_probs, sell_probs,
    b_frac, s_frac, max_lev,
    can_short_t, can_long_t,
    initial_cash, initial_inventory,
    fee_buy, fee_sell, margin_cost_per_step,
):
    """Pure PyTorch forward for backward pass recomputation."""
    batch_shape = closes.shape[:-1]
    steps = closes.shape[-1]
    dtype = closes.dtype
    device = closes.device
    has_margin = margin_cost_per_step > 0
    has_can_long = can_long_t.numel() > 0

    cash = torch.full(batch_shape, initial_cash, dtype=dtype, device=device)
    inv = torch.full(batch_shape, initial_inventory, dtype=dtype, device=device)
    prev_val = cash + inv * closes[..., 0]

    pnl_list, ret_list, val_list = [], [], []
    eb_list, es_list, inv_list = [], [], []

    for idx in range(steps):
        close = closes[..., idx]
        bp = b_prices[..., idx]
        sp = s_prices[..., idx]
        sl = max_lev[..., idx]
        bf = b_frac[..., idx]
        sf = s_frac[..., idx]

        equity = cash + inv * close
        equity_pos = torch.clamp(equity, min=_EPS)

        bp_fee = bp * fee_buy + _EPS
        max_buy_cash = torch.where(bp > 0, cash / bp_fee, torch.zeros_like(cash))
        target_notional = sl * equity_pos
        current_notional = inv * bp
        leveraged_cap = torch.where(
            bp > 0,
            torch.clamp(target_notional - current_notional, min=0.0) / bp_fee,
            torch.zeros_like(cash),
        )
        buy_cap = torch.where(sl <= 1.0 + 1e-6, torch.clamp(max_buy_cash, min=0.0), leveraged_cap)
        buy_qty = bf * buy_cap

        if has_can_long:
            cover_cap = torch.clamp(-inv, min=0.0)
            buy_qty = torch.where(can_long_t > 0.5, buy_qty, torch.minimum(buy_qty, cover_cap))

        long_to_close = torch.clamp(inv, min=0.0)
        sp_fee_denom = sp * fee_buy + _EPS
        max_short = torch.where(sp > 0, target_notional / sp_fee_denom, torch.zeros_like(cash))
        short_open = torch.clamp(max_short - torch.clamp(-inv, min=0.0), min=0.0)
        sell_cap = long_to_close + torch.where(can_short_t > 0.5, short_open, torch.zeros_like(cash))
        sell_qty = sf * sell_cap

        exec_buys = buy_qty * buy_probs[..., idx]
        exec_sells = sell_qty * sell_probs[..., idx]

        cash = cash - exec_buys * bp * fee_buy + exec_sells * sp * fee_sell
        inv = inv + exec_buys - exec_sells

        if has_margin:
            pos_val = torch.abs(inv * close)
            eq = cash + inv * close
            margin_used = torch.clamp(pos_val - torch.clamp(eq, min=0.0), min=0.0)
            cash = cash - margin_used * margin_cost_per_step

        portfolio_value = cash + inv * close
        pnl = portfolio_value - prev_val
        ret = pnl / torch.clamp(prev_val, min=_EPS)
        prev_val = portfolio_value

        pnl_list.append(pnl)
        ret_list.append(ret)
        val_list.append(portfolio_value)
        eb_list.append(exec_buys)
        es_list.append(exec_sells)
        inv_list.append(inv)

    return (
        torch.stack(pnl_list, dim=-1),
        torch.stack(ret_list, dim=-1),
        torch.stack(val_list, dim=-1),
        torch.stack(eb_list, dim=-1),
        torch.stack(es_list, dim=-1),
        torch.stack(inv_list, dim=-1),
        cash,
        inv,
    )


def simulate_hourly_trades_triton(
    *,
    highs: torch.Tensor,
    lows: torch.Tensor,
    closes: torch.Tensor,
    opens: torch.Tensor | None = None,
    buy_prices: torch.Tensor,
    sell_prices: torch.Tensor,
    trade_intensity: torch.Tensor,
    buy_trade_intensity: torch.Tensor | None = None,
    sell_trade_intensity: torch.Tensor | None = None,
    maker_fee: float = 0.0008,
    initial_cash: float = 1.0,
    initial_inventory: float = 0.0,
    temperature: float = 5e-4,
    max_leverage: float | torch.Tensor = 1.0,
    can_short: bool | float | torch.Tensor = False,
    can_long: bool | float | torch.Tensor = True,
    decision_lag_bars: int = 0,
    market_order_entry: bool = False,
    fill_buffer_pct: float = 0.0,
    margin_annual_rate: float = 0.0,
) -> HourlySimulationResult:
    """Triton-fused simulate_hourly_trades - same interface as fast version."""

    _check_shapes(highs, lows, closes, buy_prices, sell_prices, trade_intensity)
    if closes.ndim == 0:
        raise ValueError("Input tensors must include a time dimension")

    buy_intensity = buy_trade_intensity if buy_trade_intensity is not None else trade_intensity
    sell_intensity = sell_trade_intensity if sell_trade_intensity is not None else trade_intensity
    _check_shapes(highs, buy_intensity, sell_intensity)
    if opens is not None:
        _check_shapes(highs, opens)

    if decision_lag_bars > 0:
        lag = decision_lag_bars
        highs = highs[..., lag:]
        lows = lows[..., lag:]
        closes = closes[..., lag:]
        if opens is not None:
            opens = opens[..., lag:]
        buy_prices = buy_prices[..., :-lag]
        sell_prices = sell_prices[..., :-lag]
        trade_intensity = trade_intensity[..., :-lag]
        buy_intensity = buy_intensity[..., :-lag]
        sell_intensity = sell_intensity[..., :-lag]
        if torch.is_tensor(max_leverage) and max_leverage.ndim > 0:
            max_leverage = max_leverage[..., lag:]

    margin_cost_per_step = margin_annual_rate / HOURLY_PERIODS_PER_YEAR

    batch_shape = closes.shape[:-1]
    steps = closes.shape[-1]
    device = closes.device
    dtype = closes.dtype
    fee = torch.as_tensor(maker_fee, dtype=dtype, device=device)

    max_lev = torch.broadcast_to(_as_tensor(max_leverage, closes), closes.shape).contiguous()
    can_short_t = torch.broadcast_to(_as_tensor(can_short, closes), batch_shape).contiguous()
    can_long_t = torch.broadcast_to(_as_tensor(can_long, closes), batch_shape).contiguous()

    b_prices, s_prices, buy_probs, sell_probs = _precompute_fill_probs(
        buy_prices, sell_prices, lows, highs, closes,
        temperature, fill_buffer_pct, market_order_entry, opens,
    )

    b_int = torch.minimum(torch.clamp(buy_intensity, min=0.0), max_lev)
    s_int = torch.minimum(torch.clamp(sell_intensity, min=0.0), max_lev)
    b_frac = b_int / torch.clamp(max_lev, min=_EPS)
    s_frac = s_int / torch.clamp(max_lev, min=_EPS)

    fee_buy = 1.0 + fee.item()
    fee_sell = 1.0 - fee.item()

    if not HAS_TRITON or not closes.is_cuda:
        pnl, ret, val, eb, es, inv_p, fc, fi = _pytorch_sim_forward(
            closes, b_prices, s_prices, buy_probs, sell_probs,
            b_frac, s_frac, max_lev, can_short_t, can_long_t,
            initial_cash, initial_inventory,
            fee_buy, fee_sell, margin_cost_per_step,
        )
        return HourlySimulationResult(
            pnl=pnl, returns=ret, portfolio_values=val,
            cash=fc, inventory=fi,
            buy_fill_probability=buy_probs, sell_fill_probability=sell_probs,
            executed_buys=eb, executed_sells=es, inventory_path=inv_p,
        )

    pnl, ret, val, eb, es, inv_p, fc, fi = _TritonSimAutograd.apply(
        closes, b_prices, s_prices, buy_probs, sell_probs,
        b_frac, s_frac, max_lev,
        can_short_t, can_long_t,
        initial_cash, initial_inventory,
        fee_buy, fee_sell, margin_cost_per_step,
    )

    return HourlySimulationResult(
        pnl=pnl, returns=ret, portfolio_values=val,
        cash=fc, inventory=fi,
        buy_fill_probability=buy_probs, sell_fill_probability=sell_probs,
        executed_buys=eb, executed_sells=es, inventory_path=inv_p,
    )
