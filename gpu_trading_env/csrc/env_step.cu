// SoA fused env_step CUDA kernel — one thread per episode.
// Blackwell SM120 target. Buffered limit-fill (5 bps), leverage cap 5x,
// maintenance margin -> liquidation with punitive fill, log-equity reward.
//
// Action layout per-episode: [p_bid_frac, p_ask_frac, q_bid_frac, q_ask_frac]
//   p_bid_frac, p_ask_frac in [-1, 1] — quote price offset vs mid,
//       scaled by MAX_QUOTE_OFFSET_BPS (default 50 bps).
//   q_bid_frac, q_ask_frac in [0, 1] — fraction of max allowable size.
//
// OHLC layout: [T_total, 4] float32, columns = (open, high, low, close).
// Same OHLC tape is broadcast across all B episodes in v1 (single instrument,
// multi-episode parallel rollouts with independent t_idx).

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include "env_params.h"

namespace gpu_trading_env {

__device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return fmaxf(lo, fminf(hi, x));
}

__global__ void env_step_kernel(
    // OHLC tape, shared across episodes (single instrument v1)
    const float* __restrict__ ohlc,   // [T_total, 4]
    // SoA per-episode state (read/write)
    float* __restrict__ pos_qty,      // [B]
    float* __restrict__ pos_entry_px, // [B]
    float* __restrict__ cash,         // [B]
    float* __restrict__ equity,       // [B]
    float* __restrict__ dd_peak,      // [B]
    float* __restrict__ drawdown,     // [B]
    int*   __restrict__ t_idx,        // [B]
    int*   __restrict__ done_flag,    // [B]
    // Action per-episode
    const float* __restrict__ action, // [B, 4]
    // Outputs
    float* __restrict__ reward,       // [B]
    float* __restrict__ cost,         // [B, 4] optional (nullable)
    int    B,
    EnvParams p
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    // Auto-reset completed episodes before stepping.
    if (done_flag[b]) {
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
    // Clamp tape index safely (broadcast same tape across episodes).
    if (ti >= p.T_total - 1) ti = p.T_total - 2;
    const float O = ohlc[ti * 4 + 0];
    const float H = ohlc[ti * 4 + 1];
    const float L = ohlc[ti * 4 + 2];
    const float C = ohlc[ti * 4 + 3];
    const float mid = 0.5f * (H + L);

    // Decode action.
    const float a_pb = clampf(action[b * 4 + 0], -1.f, 1.f);
    const float a_pa = clampf(action[b * 4 + 1], -1.f, 1.f);
    const float a_qb = clampf(action[b * 4 + 2],  0.f, 1.f);
    const float a_qa = clampf(action[b * 4 + 3],  0.f, 1.f);

    const float off = p.max_quote_offset_bps * 1e-4f;
    const float p_bid = mid * (1.f + a_pb * off);  // limit BUY price
    const float p_ask = mid * (1.f + a_pa * off);  // limit SELL price

    // Current state (pre-step).
    float qty   = pos_qty[b];
    float entry = pos_entry_px[b];
    float csh   = cash[b];
    float eq0   = equity[b];

    // Size budget: leverage cap at max_leverage * equity, per side.
    // Max notional each side = leverage * equity; convert to quantity at mid.
    const float mid_pos = fmaxf(mid, 1e-6f);
    const float max_qty_side = (p.max_leverage * fmaxf(eq0, 1e-6f)) / mid_pos;
    float q_buy  = a_qb * max_qty_side;
    float q_sell = a_qa * max_qty_side;

    // Buffered limit-fill rule: fills only if the bar's extreme crosses the
    // limit by at least the fill buffer.
    const float buf = p.buffer_bps * 1e-4f;
    const bool fill_buy  = (p_bid >= L * (1.f + buf));
    const bool fill_sell = (p_ask <= H * (1.f - buf));

    // Conservative fill price: worst for us within the feasible range.
    const float fill_px_buy  = fmaxf(p_bid, L * (1.f + buf));
    const float fill_px_sell = fminf(p_ask, H * (1.f - buf));

    float buy_notional = 0.f, sell_notional = 0.f, fees = 0.f;
    if (fill_buy && q_buy > 0.f) {
        buy_notional = q_buy * fill_px_buy;
        fees += buy_notional * p.fee_bps * 1e-4f;
        // Update cost basis.
        float new_qty = qty + q_buy;
        if (qty >= 0.f) {
            float denom = fmaxf(fabsf(new_qty), 1e-9f);
            entry = (qty * entry + q_buy * fill_px_buy) / denom;
        } else {
            // Covering short: realize PnL on the covered portion.
            float cover = fminf(q_buy, -qty);
            csh += cover * (entry - fill_px_buy);
            if (new_qty > 0.f) entry = fill_px_buy;
        }
        csh -= q_buy * fill_px_buy;
        qty = new_qty;
    }
    if (fill_sell && q_sell > 0.f) {
        sell_notional = q_sell * fill_px_sell;
        fees += sell_notional * p.fee_bps * 1e-4f;
        float new_qty = qty - q_sell;
        if (qty <= 0.f) {
            float denom = fmaxf(fabsf(new_qty), 1e-9f);
            entry = (fabsf(qty) * entry + q_sell * fill_px_sell) / denom;
        } else {
            float reduce = fminf(q_sell, qty);
            csh += reduce * (fill_px_sell - entry);
            if (new_qty < 0.f) entry = fill_px_sell;
        }
        csh += q_sell * fill_px_sell;
        qty = new_qty;
    }
    csh -= fees;

    // Mark-to-market at close of the bar.
    float mark_px = C;
    float position_value = qty * mark_px;
    float eq1 = csh + position_value;

    // Leverage violation cost signal (gross exposure / equity).
    const float gross_lev = fabsf(position_value) / fmaxf(eq1, 1e-6f);
    const float lev_violation = fmaxf(0.f, gross_lev - p.max_leverage);

    // Force-reduce if we somehow ended the bar over-leveraged.
    if (gross_lev > p.max_leverage && qty != 0.f) {
        float target_notional = p.max_leverage * fmaxf(eq1, 1e-6f);
        float current_notional = fabsf(position_value);
        float scale = target_notional / fmaxf(current_notional, 1e-9f);
        float reduce_qty = qty * (1.f - scale);
        // Market reduce at close with extra fee.
        float reduce_notional = fabsf(reduce_qty) * mark_px;
        float extra_fee = reduce_notional * p.fee_bps * 1e-4f;
        if (qty > 0.f) {
            csh += reduce_qty * mark_px;   // sell positive chunk
        } else {
            csh += reduce_qty * mark_px;   // buy positive (reduce_qty<0) to cover
        }
        csh -= extra_fee;
        qty -= reduce_qty;
        position_value = qty * mark_px;
        eq1 = csh + position_value;
        fees += extra_fee;
    }

    // Maintenance margin / liquidation check.
    int liquidated = 0;
    const float maint = p.maint_margin * fabsf(position_value);
    if (eq1 < maint) {
        // Punitive fill at the worst bar extreme.
        float worst_px = (qty > 0.f) ? L : (qty < 0.f ? H : mark_px);
        float liq_notional = fabsf(qty) * worst_px;
        float liq_fee = liq_notional * (p.fee_bps * 1e-4f);
        csh += qty * worst_px;       // close to cash
        csh -= liq_fee;
        csh -= p.liq_penalty * fmaxf(eq0, 1e-6f);  // penalty as fraction of prior equity
        qty = 0.f;
        entry = 0.f;
        position_value = 0.f;
        eq1 = csh;
        liquidated = 1;
    }

    // Drawdown tracking.
    float peak = fmaxf(dd_peak[b], eq1);
    float dd = (peak - eq1) / fmaxf(peak, 1e-6f);
    float dd_incr = fmaxf(0.f, dd - drawdown[b]);

    // Log-equity reward. Guard negative/zero equity.
    float r = 0.f;
    if (eq0 > 1e-6f && eq1 > 1e-6f) {
        r = logf(eq1 / eq0);
    } else if (eq1 <= 1e-6f) {
        r = -10.f;  // catastrophic
    }

    // Write back SoA state.
    pos_qty[b]      = qty;
    pos_entry_px[b] = entry;
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
        float turnover = (buy_notional + sell_notional) / fmaxf(eq0, 1e-6f);
        cost[b * 4 + 0] = dd_incr;
        cost[b * 4 + 1] = (float)liquidated;
        cost[b * 4 + 2] = lev_violation;
        cost[b * 4 + 3] = turnover;
    }
}

void launch_env_step(
    const at::Tensor& ohlc,
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
    at::Tensor& cost,        // may be empty (numel()==0) to signal "null"
    EnvParams p
) {
    int B = (int)pos_qty.size(0);
    const int TPB = 128;
    int nblocks = (B + TPB - 1) / TPB;
    float* cost_ptr = (cost.defined() && cost.numel() > 0) ? cost.data_ptr<float>() : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream();
    env_step_kernel<<<nblocks, TPB, 0, stream.stream()>>>(
        ohlc.data_ptr<float>(),
        pos_qty.data_ptr<float>(),
        pos_entry_px.data_ptr<float>(),
        cash.data_ptr<float>(),
        equity.data_ptr<float>(),
        dd_peak.data_ptr<float>(),
        drawdown.data_ptr<float>(),
        t_idx.data_ptr<int>(),
        done_flag.data_ptr<int>(),
        action.data_ptr<float>(),
        reward.data_ptr<float>(),
        cost_ptr,
        B,
        p
    );
}

} // namespace gpu_trading_env
