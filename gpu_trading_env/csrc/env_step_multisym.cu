// Multi-symbol fused env_step CUDA kernel — one thread per parallel env.
//
// Semantics match the pufferlib_market C env at a high level:
//   * Per-env state: single position at a time (sym_id, qty, entry_px, cash,
//     equity, dd_peak, drawdown, t_idx, done). Matches the "top_n=1"
//     behaviour that XGB + current RL prod use.
//   * Action: Discrete(1 + 2S)
//       0           -> flat (close any open position, cash-only)
//       1..S        -> long  sym_i
//       S+1..2S     -> short sym_i
//   * Fill model: market-style open of the NEXT bar, buffered by fill_buffer
//     (slightly worse than open). This mirrors the "buy @ open, sell @ close"
//     backtest semantic the XGB champion uses, minus the spread_bps extra.
//   * Reward: log(equity_t / equity_{t-1}).
//   * Data tapes are read-only, broadcast across episodes:
//       prices [T, S, 5]  (O, H, L, C, V)
//
// Blackwell SM120 target. Compiled via torch.utils.cpp_extension.load.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include "env_params.h"

namespace gpu_trading_env {

struct MultiEnvParams {
    float fee_bps;
    float buffer_bps;      // fill-side slippage vs open (bps)
    float max_leverage;
    float maint_margin;
    float liq_penalty;
    float init_cash;
    int   T_total;
    int   S;               // num symbols
    int   episode_len;
};

__device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return fmaxf(lo, fminf(hi, x));
}

__global__ void env_step_multisym_kernel(
    const float* __restrict__ prices,     // [T, S, 5]
    // SoA per-env state (read/write)
    int*   __restrict__ pos_sym,          // [B] current symbol id, -1 if flat
    int*   __restrict__ pos_side,         // [B] +1 long, -1 short, 0 flat
    float* __restrict__ pos_qty,          // [B]
    float* __restrict__ pos_entry_px,     // [B]
    float* __restrict__ cash,             // [B]
    float* __restrict__ equity,           // [B]
    float* __restrict__ dd_peak,          // [B]
    float* __restrict__ drawdown,         // [B]
    int*   __restrict__ t_idx,            // [B]
    int*   __restrict__ done_flag,        // [B]
    // Action per-env
    const int*   __restrict__ action,     // [B] discrete id in [0, 1+2S)
    // Outputs
    float* __restrict__ reward,           // [B]
    float* __restrict__ cost,             // [B, 4] optional (nullable)
    int    B,
    MultiEnvParams p
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    // Auto-reset completed episodes before stepping.
    if (done_flag[b]) {
        pos_sym[b]      = -1;
        pos_side[b]     = 0;
        pos_qty[b]      = 0.f;
        pos_entry_px[b] = 0.f;
        cash[b]         = p.init_cash;
        equity[b]       = p.init_cash;
        dd_peak[b]      = p.init_cash;
        drawdown[b]     = 0.f;
        t_idx[b]        = 0;
        done_flag[b]    = 0;
    }

    int ti = t_idx[b];
    if (ti >= p.T_total - 1) ti = p.T_total - 2;

    // Decode discrete action.
    int aid = action[b];
    int want_sym = -1, want_side = 0;
    if (aid > 0 && aid <= p.S) {
        want_sym = aid - 1;
        want_side = +1;
    } else if (aid > p.S && aid <= 2 * p.S) {
        want_sym = aid - p.S - 1;
        want_side = -1;
    }
    // else aid == 0 (or invalid): flat.

    const int  cur_sym  = pos_sym[b];
    const int  cur_side = pos_side[b];
    float      qty      = pos_qty[b];
    float      entry    = pos_entry_px[b];
    float      csh      = cash[b];
    const float eq0     = equity[b];

    const float fee     = p.fee_bps * 1e-4f;
    const float buf     = p.buffer_bps * 1e-4f;

    // Bar ti's price row for whatever symbol(s) we touch.
    auto price_lookup = [&] (int s, int which) -> float {
        return prices[((ti) * p.S + s) * 5 + which];
    };

    // ---- 1) Close current position (if any and changing) ----
    bool close_now = (cur_side != 0) && ( (want_sym != cur_sym) || (want_side != cur_side) );
    float close_fees = 0.f;
    float realized_pnl = 0.f;
    if (close_now) {
        // Close at bar ti open, buffered against us.
        float o = price_lookup(cur_sym, 0);
        // Selling a long: fill @ open * (1 - buf). Buying to cover short: @ open * (1 + buf).
        float fill_px = (cur_side > 0) ? o * (1.f - buf) : o * (1.f + buf);
        float notional = qty * fill_px;
        close_fees = notional * fee;
        if (cur_side > 0) {
            realized_pnl = qty * (fill_px - entry);
            csh += qty * fill_px;        // receive
        } else {
            realized_pnl = qty * (entry - fill_px);
            csh -= qty * fill_px;        // pay to cover
        }
        csh -= close_fees;
        qty = 0.f; entry = 0.f;
        // Zero out the symbol state to avoid stale references.
        pos_sym[b]  = -1;
        pos_side[b] = 0;
    }

    // ---- 2) Open new position (if flat and action says trade) ----
    float open_fees = 0.f;
    float turnover_notional = 0.f;
    int   new_sym  = (want_side != 0) ? want_sym : ((cur_side != 0 && !close_now) ? cur_sym : -1);
    int   new_side = (want_side != 0) ? want_side : ((!close_now) ? cur_side : 0);
    float new_qty  = (!close_now && cur_side != 0) ? qty : 0.f;
    float new_entry= (!close_now && cur_side != 0) ? entry : 0.f;

    if (want_side != 0 && (close_now || cur_side == 0)) {
        // Entering fresh at bar ti open, buffered against us.
        float o = price_lookup(want_sym, 0);
        float fill_px = (want_side > 0) ? o * (1.f + buf) : o * (1.f - buf);
        float budget  = fmaxf(csh, 1e-3f);  // size off cash post-close
        float lev_notional = p.max_leverage * fmaxf(eq0, 1e-6f);
        float notional = fminf(budget, lev_notional);
        float enter_qty = notional / fmaxf(fill_px, 1e-6f);
        open_fees = notional * fee;
        if (want_side > 0) {
            csh -= notional;
        } else {
            csh += notional;
        }
        csh -= open_fees;
        new_qty   = enter_qty;
        new_entry = fill_px;
        turnover_notional = notional;
    }

    // ---- 3) Mark-to-market at close of bar ti ----
    float position_value = 0.f;
    if (new_side != 0 && new_sym >= 0) {
        float c = price_lookup(new_sym, 3);
        position_value = (new_side > 0)
            ? new_qty * c
            : new_qty * (2.f * new_entry - c);  // short MTM = qty*(entry - (c-entry))
        // Equivalent: short_equity = cash already captured at open; MTM PnL = qty*(entry - c).
        // We'll compute equity as cash + qty*c for long, cash + qty*(2*entry - c) for short
        // so equity stays consistent with realized when we later close.
    }
    float eq1 = csh + position_value;

    // ---- 4) Maintenance / liquidation check (same logic as single-sym) ----
    int liquidated = 0;
    const float gross_exposure = (new_side != 0 && new_sym >= 0)
        ? fabsf(new_qty) * price_lookup(new_sym, 3)
        : 0.f;
    const float maint = p.maint_margin * gross_exposure;
    if (eq1 < maint && gross_exposure > 0.f) {
        // Punitive liquidation at the worst bar extreme for our side.
        float lo = price_lookup(new_sym, 2);
        float hi = price_lookup(new_sym, 1);
        float worst_px = (new_side > 0) ? lo : hi;
        float liq_notional = fabsf(new_qty) * worst_px;
        float liq_fee = liq_notional * fee;
        if (new_side > 0) csh += new_qty * worst_px;
        else              csh -= new_qty * worst_px;
        csh -= liq_fee;
        csh -= p.liq_penalty * fmaxf(eq0, 1e-6f);
        new_qty = 0.f; new_entry = 0.f; new_sym = -1; new_side = 0;
        position_value = 0.f;
        eq1 = csh;
        liquidated = 1;
    }

    // ---- 5) Drawdown ----
    float peak = fmaxf(dd_peak[b], eq1);
    float dd = (peak - eq1) / fmaxf(peak, 1e-6f);
    float dd_incr = fmaxf(0.f, dd - drawdown[b]);

    // ---- 6) Log-equity reward ----
    float r = 0.f;
    if (eq0 > 1e-6f && eq1 > 1e-6f) {
        r = logf(eq1 / eq0);
    } else if (eq1 <= 1e-6f) {
        r = -10.f;
    }

    // ---- 7) Write back state ----
    pos_sym[b]      = new_sym;
    pos_side[b]     = new_side;
    pos_qty[b]      = new_qty;
    pos_entry_px[b] = new_entry;
    cash[b]         = csh;
    equity[b]       = eq1;
    dd_peak[b]      = peak;
    drawdown[b]     = dd;

    int new_t = ti + 1;
    int new_done = 0;
    if (liquidated) new_done = 1;
    if (new_t >= p.episode_len || new_t >= p.T_total - 1) new_done = 1;
    t_idx[b]     = new_t;
    done_flag[b] = new_done;

    reward[b] = r;

    if (cost != nullptr) {
        float turnover = turnover_notional / fmaxf(eq0, 1e-6f);
        float gross_lev = gross_exposure / fmaxf(eq1, 1e-6f);
        cost[b * 4 + 0] = dd_incr;
        cost[b * 4 + 1] = (float)liquidated;
        cost[b * 4 + 2] = fmaxf(0.f, gross_lev - p.max_leverage);
        cost[b * 4 + 3] = turnover;
    }
}

void launch_env_step_multisym(
    const at::Tensor& prices,
    at::Tensor& pos_sym,
    at::Tensor& pos_side,
    at::Tensor& pos_qty,
    at::Tensor& pos_entry_px,
    at::Tensor& cash,
    at::Tensor& equity,
    at::Tensor& dd_peak,
    at::Tensor& drawdown,
    at::Tensor& t_idx,
    at::Tensor& done_flag,
    const at::Tensor& action,
    at::Tensor& reward,
    at::Tensor& cost,        // may be empty
    MultiEnvParams p
) {
    const int B = (int)pos_sym.size(0);
    const int TPB = 128;
    const int nblocks = (B + TPB - 1) / TPB;
    float* cost_ptr = (cost.defined() && cost.numel() > 0)
        ? cost.data_ptr<float>() : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream();
    env_step_multisym_kernel<<<nblocks, TPB, 0, stream.stream()>>>(
        prices.data_ptr<float>(),
        pos_sym.data_ptr<int>(),
        pos_side.data_ptr<int>(),
        pos_qty.data_ptr<float>(),
        pos_entry_px.data_ptr<float>(),
        cash.data_ptr<float>(),
        equity.data_ptr<float>(),
        dd_peak.data_ptr<float>(),
        drawdown.data_ptr<float>(),
        t_idx.data_ptr<int>(),
        done_flag.data_ptr<int>(),
        action.data_ptr<int>(),
        reward.data_ptr<float>(),
        cost_ptr,
        B,
        p
    );
}

} // namespace gpu_trading_env
