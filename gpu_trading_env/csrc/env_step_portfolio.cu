// Multi-symbol PORTFOLIO bracket env step CUDA kernel.
//
// One thread per parallel env. Each thread loops over S symbols to compute
// fills, leverage clip, fees, margin interest, and reward.
//
// Mirror of the numpy reference at
//   pufferlib_cpp_market_sim/python/market_sim_py/multisym_bracket_ref.py
// Kept structurally identical so golden tests can hold to fp32 rounding.
//
// Action per (env, symbol) is [B, S, 4]:
//   action[b, s, 0] = lim_buy_offset_bps
//   action[b, s, 1] = lim_sell_offset_bps
//   action[b, s, 2] = buy_qty_pct  (>= 0)
//   action[b, s, 3] = sell_qty_pct (>= 0)
//
// State per env:
//   cash[B]       : float
//   positions[B*S]: float (signed shares; neg = short)
//
// Distinct from env_step_multisym.cu (which is Discrete(1+2S) pick-one-symbol).
// Blackwell SM120 target.
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>

namespace gpu_trading_env {

struct PortfolioBracketParams {
    float fee_bps;
    float fill_buffer_bps;
    float max_leverage;
    float annual_margin_rate;
    int   trading_days_per_year;
    int   S;                       // num symbols
};

__device__ __forceinline__ float maxf0(float x) { return fmaxf(0.f, x); }

__global__ void portfolio_bracket_step_kernel(
    const float* __restrict__ cash_in,        // [B]
    const float* __restrict__ pos_in,         // [B, S]
    const float* __restrict__ actions,        // [B, S, 4]
    const float* __restrict__ prev_close,     // [B, S]
    const float* __restrict__ bar_open,       // [B, S]   (unused in spec but kept for API parity)
    const float* __restrict__ bar_high,       // [B, S]
    const float* __restrict__ bar_low,        // [B, S]
    const float* __restrict__ bar_close,      // [B, S]
    const uint8_t* __restrict__ tradable,     // [B, S] bool
    float* __restrict__ cash_out,             // [B]
    float* __restrict__ pos_out,              // [B, S]
    float* __restrict__ reward,               // [B]
    float* __restrict__ info_eq_prev,         // [B] nullable
    float* __restrict__ info_new_eq,          // [B] nullable
    float* __restrict__ info_fees,            // [B] nullable
    float* __restrict__ info_margin,          // [B] nullable
    float* __restrict__ info_borrowed,        // [B] nullable
    int B,
    PortfolioBracketParams p
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const int S = p.S;
    const float fee_rate     = p.fee_bps        * 1e-4f;
    const float fb           = p.fill_buffer_bps * 1e-4f;
    const float per_day_rate = p.annual_margin_rate / (float)p.trading_days_per_year;

    // ----- Pass 1: compute equity_prev and provisional fills -----
    float pos_value_prev = 0.f;
    for (int s = 0; s < S; ++s) {
        pos_value_prev += pos_in[b * S + s] * prev_close[b * S + s];
    }
    const float csh0       = cash_in[b];
    const float equity_prev = csh0 + pos_value_prev;
    const float eq_prev_safe = (equity_prev > 1e-9f) ? equity_prev : 1e-9f;

    // We need two passes: (1) compute provisional buy/sell shares + new positions,
    // (2) compute leverage scale, then a final pass to apply scale and accumulate
    // cash flow + fees + margin. To avoid heap, we recompute fills in the second
    // pass — cheap (a few mul/cmp per symbol).

    // First pass: tentative new_positions and notional summary.
    // We need three quantities for the leverage-clip alpha:
    //   candidate_notional = Σ |old + buy - sell| * bar_close   (post-trade)
    //   existing_notional  = Σ |old|             * bar_close   (pre-trade)
    //   delta_notional     = Σ |buy - sell|      * bar_close   (trade size)
    float candidate_notional = 0.f;
    float existing_notional  = 0.f;
    float delta_notional     = 0.f;
    for (int s = 0; s < S; ++s) {
        const int idx = b * S + s;
        const float pc = prev_close[idx];
        const bool tr  = tradable[idx] != 0;
        const float lim_buy_bps  = actions[idx * 4 + 0];
        const float lim_sell_bps = actions[idx * 4 + 1];
        const float buy_pct      = maxf0(actions[idx * 4 + 2]);
        const float sell_pct     = maxf0(actions[idx * 4 + 3]);

        const float lim_buy_px  = pc * (1.f + lim_buy_bps  * 1e-4f);
        const float lim_sell_px = pc * (1.f + lim_sell_bps * 1e-4f);
        bool  buy_filled  = tr && (bar_low[idx]  <= lim_buy_px  * (1.f - fb));
        bool  sell_filled = tr && (bar_high[idx] >= lim_sell_px * (1.f + fb));
        if (buy_filled && sell_filled && buy_pct > 0.f && sell_pct > 0.f) {
            const float bo = bar_open[idx];
            const float buy_dist  = fabsf(bo - lim_buy_px);
            const float sell_dist = fabsf(bo - lim_sell_px);
            if (buy_dist <= sell_dist) sell_filled = false;
            else                        buy_filled  = false;
        }

        const float buy_notional  = buy_pct  * eq_prev_safe;
        const float sell_notional = sell_pct * eq_prev_safe;
        const float buy_shares  = buy_filled  ? (buy_notional  / fmaxf(lim_buy_px,  1e-9f)) : 0.f;
        const float sell_shares = sell_filled ? (sell_notional / fmaxf(lim_sell_px, 1e-9f)) : 0.f;

        const float new_p_tentative = pos_in[idx] + buy_shares - sell_shares;
        candidate_notional += fabsf(new_p_tentative) * bar_close[idx];
        existing_notional  += fabsf(pos_in[idx])     * bar_close[idx];
        delta_notional     += fabsf(buy_shares - sell_shares) * bar_close[idx];
    }

    const float cap = p.max_leverage * eq_prev_safe;
    // Clip ONLY the new buy/sell shares (the trade delta). Do NOT shrink the
    // existing position — that creates equity from nothing because the shrunk
    // size has no corresponding cash flow. A short-only policy on a flat tape
    // can compound that into runaway equity growth otherwise.
    float alpha = 1.f;
    if (candidate_notional > cap) {
        const float headroom = fmaxf(0.f, cap - existing_notional);
        if (delta_notional > 1e-9f) {
            alpha = headroom / delta_notional;
            if (alpha > 1.f) alpha = 1.f;
            if (alpha < 0.f) alpha = 0.f;
        }
    }

    // Second pass: apply alpha to trade legs only, accumulate cash flow + fees.
    float cash_flow = 0.f;     // net (cash_in - cash_out)
    float fee_total = 0.f;
    float notional_close = 0.f;
    for (int s = 0; s < S; ++s) {
        const int idx = b * S + s;
        const float pc = prev_close[idx];
        const bool tr  = tradable[idx] != 0;
        const float lim_buy_bps  = actions[idx * 4 + 0];
        const float lim_sell_bps = actions[idx * 4 + 1];
        const float buy_pct      = maxf0(actions[idx * 4 + 2]);
        const float sell_pct     = maxf0(actions[idx * 4 + 3]);

        const float lim_buy_px  = pc * (1.f + lim_buy_bps  * 1e-4f);
        const float lim_sell_px = pc * (1.f + lim_sell_bps * 1e-4f);
        bool  buy_filled  = tr && (bar_low[idx]  <= lim_buy_px  * (1.f - fb));
        bool  sell_filled = tr && (bar_high[idx] >= lim_sell_px * (1.f + fb));
        if (buy_filled && sell_filled && buy_pct > 0.f && sell_pct > 0.f) {
            const float bo = bar_open[idx];
            const float buy_dist  = fabsf(bo - lim_buy_px);
            const float sell_dist = fabsf(bo - lim_sell_px);
            if (buy_dist <= sell_dist) sell_filled = false;
            else                        buy_filled  = false;
        }

        float buy_shares  = buy_filled  ? ((buy_pct  * eq_prev_safe) / fmaxf(lim_buy_px,  1e-9f)) : 0.f;
        float sell_shares = sell_filled ? ((sell_pct * eq_prev_safe) / fmaxf(lim_sell_px, 1e-9f)) : 0.f;
        // Scale only the trade delta, NOT the existing position.
        buy_shares  *= alpha;
        sell_shares *= alpha;

        const float buy_notional_actual  = buy_shares  * lim_buy_px;
        const float sell_notional_actual = sell_shares * lim_sell_px;
        cash_flow += sell_notional_actual - buy_notional_actual;
        fee_total += (buy_notional_actual + sell_notional_actual) * fee_rate;

        const float new_p = pos_in[idx] + buy_shares - sell_shares;
        pos_out[idx] = new_p;
        notional_close += fabsf(new_p) * bar_close[idx];
    }

    float new_cash = csh0 + cash_flow - fee_total;

    const float borrowed = fmaxf(0.f, notional_close - eq_prev_safe);
    const float margin_cost = borrowed * per_day_rate;
    new_cash -= margin_cost;

    // New equity: cash + Σ pos * close
    float new_pos_value = 0.f;
    for (int s = 0; s < S; ++s) {
        new_pos_value += pos_out[b * S + s] * bar_close[b * S + s];
    }
    const float new_equity = new_cash + new_pos_value;
    const float r = (new_equity - equity_prev) / eq_prev_safe;

    cash_out[b] = new_cash;
    reward[b]   = r;
    if (info_eq_prev)  info_eq_prev[b]  = equity_prev;
    if (info_new_eq)   info_new_eq[b]   = new_equity;
    if (info_fees)     info_fees[b]     = fee_total;
    if (info_margin)   info_margin[b]   = margin_cost;
    if (info_borrowed) info_borrowed[b] = borrowed;
}

void launch_portfolio_bracket_step(
    const at::Tensor& cash_in,
    const at::Tensor& pos_in,
    const at::Tensor& actions,
    const at::Tensor& prev_close,
    const at::Tensor& bar_open,
    const at::Tensor& bar_high,
    const at::Tensor& bar_low,
    const at::Tensor& bar_close,
    const at::Tensor& tradable,
    at::Tensor& cash_out,
    at::Tensor& pos_out,
    at::Tensor& reward,
    at::Tensor& info_eq_prev,
    at::Tensor& info_new_eq,
    at::Tensor& info_fees,
    at::Tensor& info_margin,
    at::Tensor& info_borrowed,
    PortfolioBracketParams p
) {
    const int B = (int)cash_in.size(0);
    const int TPB = 128;
    const int nblocks = (B + TPB - 1) / TPB;

    auto opt_ptr = [](at::Tensor& t) -> float* {
        return (t.defined() && t.numel() > 0) ? t.data_ptr<float>() : nullptr;
    };

    auto stream = at::cuda::getCurrentCUDAStream();
    portfolio_bracket_step_kernel<<<nblocks, TPB, 0, stream.stream()>>>(
        cash_in.data_ptr<float>(),
        pos_in.data_ptr<float>(),
        actions.data_ptr<float>(),
        prev_close.data_ptr<float>(),
        bar_open.data_ptr<float>(),
        bar_high.data_ptr<float>(),
        bar_low.data_ptr<float>(),
        bar_close.data_ptr<float>(),
        tradable.data_ptr<uint8_t>(),
        cash_out.data_ptr<float>(),
        pos_out.data_ptr<float>(),
        reward.data_ptr<float>(),
        opt_ptr(info_eq_prev),
        opt_ptr(info_new_eq),
        opt_ptr(info_fees),
        opt_ptr(info_margin),
        opt_ptr(info_borrowed),
        B,
        p
    );
}

} // namespace gpu_trading_env
