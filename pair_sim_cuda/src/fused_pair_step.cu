// Fused daily pair-trading step, forward + backward.
//
// Templated on scalar_t (float32, bfloat16, float16). Internal math is
// always float32 for numerical stability (sigmoid, division, fabs).
// Memory bandwidth benefit comes from storing tensors as bf16/fp16.
//
// Model per (b, p):
//   trade           = target_pos - prev_pos
//   threshold_bps   = fee_bp + half_spread_bps + offset_bps
//   signal_bps      = reach_side_bps - threshold_bps
//   fill            = session_mask * sigmoid(signal_bps / fill_temp_bps)
//   exec            = fill * trade
//   next_pos        = prev_pos + exec
//   turnover        = |exec|
//   cost_frac       = turnover * (commission_bps + half_spread_bps + fee_bp
//                                 + 0.5 * offset_bps) * 1e-4
//   pair_pnl        = next_pos * pair_ret - cost_frac

#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace pair_sim {

__device__ inline float sigmoidf(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ inline float signf(float x) {
    return (x > 0.f) - (x < 0.f);
}

template <typename scalar_t>
__global__ void pair_step_fwd_kernel(
    const scalar_t* __restrict__ target_pos,      // [B, P]
    const scalar_t* __restrict__ offset_bps,      // [B, P]
    const scalar_t* __restrict__ prev_pos,        // [B, P]
    const scalar_t* __restrict__ pair_ret,        // [B, P]
    const scalar_t* __restrict__ reach_side_bps,  // [B, P]
    const scalar_t* __restrict__ half_spread_bps, // [B, P]
    const scalar_t* __restrict__ session_mask,    // [B]
    scalar_t* __restrict__ next_pos,              // [B, P]
    scalar_t* __restrict__ pair_pnl,              // [B, P]
    scalar_t* __restrict__ turnover,              // [B, P]
    scalar_t* __restrict__ fill_out,              // [B, P]  (saved)
    scalar_t* __restrict__ sign_trade_out,        // [B, P]  (saved)
    int B, int P,
    float commission_bps, float fill_temp_bps, float fee_bp
) {
    int b = blockIdx.y;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || p >= P) return;

    int idx = b * P + p;

    float tp = static_cast<float>(target_pos[idx]);
    float pp = static_cast<float>(prev_pos[idx]);
    float ob = static_cast<float>(offset_bps[idx]);
    float rt = static_cast<float>(pair_ret[idx]);
    float rs = static_cast<float>(reach_side_bps[idx]);
    float hs = static_cast<float>(half_spread_bps[idx]);
    float sm = static_cast<float>(session_mask[b]);

    float trade = tp - pp;
    float threshold = fee_bp + hs + ob;
    float signal_bps = rs - threshold;
    float sig = sigmoidf(signal_bps / fill_temp_bps);
    float fill = sm * sig;

    float exec = fill * trade;
    float npos = pp + exec;
    float tov = fabsf(exec);
    float cost_per_unit = (commission_bps + hs + fee_bp + 0.5f * ob) * 1e-4f;
    float cost = tov * cost_per_unit;
    float pnl = npos * rt - cost;

    next_pos[idx] = static_cast<scalar_t>(npos);
    pair_pnl[idx] = static_cast<scalar_t>(pnl);
    turnover[idx] = static_cast<scalar_t>(tov);
    fill_out[idx] = static_cast<scalar_t>(fill);
    sign_trade_out[idx] = static_cast<scalar_t>(signf(exec));
}

template <typename scalar_t>
__global__ void pair_step_bwd_kernel(
    const scalar_t* __restrict__ g_next_pos,      // [B, P]
    const scalar_t* __restrict__ g_pair_pnl,      // [B, P]
    const scalar_t* __restrict__ g_turnover,      // [B, P]
    const scalar_t* __restrict__ target_pos,      // [B, P]
    const scalar_t* __restrict__ offset_bps,      // [B, P]
    const scalar_t* __restrict__ prev_pos,        // [B, P]
    const scalar_t* __restrict__ pair_ret,        // [B, P]
    const scalar_t* __restrict__ half_spread_bps, // [B, P]
    const scalar_t* __restrict__ session_mask,    // [B]
    const scalar_t* __restrict__ fill_save,       // [B, P]
    const scalar_t* __restrict__ sign_trade_save, // [B, P]
    scalar_t* __restrict__ d_target_pos,          // [B, P]
    scalar_t* __restrict__ d_offset_bps,          // [B, P]
    scalar_t* __restrict__ d_prev_pos,            // [B, P]
    int B, int P,
    float commission_bps, float fill_temp_bps, float fee_bp
) {
    int b = blockIdx.y;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || p >= P) return;

    int idx = b * P + p;

    float tp = static_cast<float>(target_pos[idx]);
    float pp = static_cast<float>(prev_pos[idx]);
    float ob = static_cast<float>(offset_bps[idx]);
    float rt = static_cast<float>(pair_ret[idx]);
    float hs = static_cast<float>(half_spread_bps[idx]);
    float sm = static_cast<float>(session_mask[b]);
    float fill = static_cast<float>(fill_save[idx]);
    float sign_exec = static_cast<float>(sign_trade_save[idx]);

    float trade = tp - pp;
    float exec = fill * trade;

    float g_np = static_cast<float>(g_next_pos[idx]);
    float g_pn = static_cast<float>(g_pair_pnl[idx]);
    float g_tv = static_cast<float>(g_turnover[idx]);

    // pair_pnl = npos * rt - cost
    float u_npos = g_np + g_pn * rt;
    float u_cost = -g_pn;

    float c_unit = (commission_bps + hs + fee_bp + 0.5f * ob) * 1e-4f;
    float u_tov = g_tv + u_cost * c_unit;
    float u_cunit = u_cost * fabsf(exec);

    float u_exec = u_tov * sign_exec + u_npos;

    float u_fill = u_exec * trade;
    float u_trade = u_exec * fill;

    float dfill_dsignal = 0.f;
    if (sm > 0.f) {
        float sigma = fill / sm;
        dfill_dsignal = sm * sigma * (1.0f - sigma) / fill_temp_bps;
    }
    float u_signal = u_fill * dfill_dsignal;
    float u_ob_from_fill = -u_signal;
    float u_ob_from_cost = u_cunit * (0.5f * 1e-4f);
    float u_ob = u_ob_from_fill + u_ob_from_cost;

    float u_tp = u_trade;
    float u_pp = u_npos - u_trade;

    d_target_pos[idx] = static_cast<scalar_t>(u_tp);
    d_offset_bps[idx] = static_cast<scalar_t>(u_ob);
    d_prev_pos[idx]   = static_cast<scalar_t>(u_pp);
}


// ---- host wrappers ----

std::vector<torch::Tensor> pair_step_fwd_cuda(
    torch::Tensor target_pos, torch::Tensor offset_bps, torch::Tensor prev_pos,
    torch::Tensor pair_ret, torch::Tensor reach_side_bps,
    torch::Tensor half_spread_bps, torch::Tensor session_mask,
    double commission_bps, double fill_temp_bps, double fee_bp
) {
    TORCH_CHECK(target_pos.is_cuda(), "target_pos must be CUDA");
    auto B = target_pos.size(0);
    auto P = target_pos.size(1);

    auto opts = target_pos.options();
    auto next_pos  = torch::empty({B, P}, opts);
    auto pair_pnl  = torch::empty({B, P}, opts);
    auto turnover  = torch::empty({B, P}, opts);
    auto fill      = torch::empty({B, P}, opts);
    auto sign_exec = torch::empty({B, P}, opts);

    const int threads = 256;
    dim3 grid((P + threads - 1) / threads, B);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        target_pos.scalar_type(), "pair_step_fwd_cuda", ([&] {
            pair_step_fwd_kernel<scalar_t><<<grid, threads>>>(
                target_pos.data_ptr<scalar_t>(),
                offset_bps.data_ptr<scalar_t>(),
                prev_pos.data_ptr<scalar_t>(),
                pair_ret.data_ptr<scalar_t>(),
                reach_side_bps.data_ptr<scalar_t>(),
                half_spread_bps.data_ptr<scalar_t>(),
                session_mask.data_ptr<scalar_t>(),
                next_pos.data_ptr<scalar_t>(),
                pair_pnl.data_ptr<scalar_t>(),
                turnover.data_ptr<scalar_t>(),
                fill.data_ptr<scalar_t>(),
                sign_exec.data_ptr<scalar_t>(),
                static_cast<int>(B), static_cast<int>(P),
                static_cast<float>(commission_bps),
                static_cast<float>(fill_temp_bps),
                static_cast<float>(fee_bp)
            );
        })
    );
    return {next_pos, pair_pnl, turnover, fill, sign_exec};
}

std::vector<torch::Tensor> pair_step_bwd_cuda(
    torch::Tensor g_next_pos, torch::Tensor g_pair_pnl, torch::Tensor g_turnover,
    torch::Tensor target_pos, torch::Tensor offset_bps, torch::Tensor prev_pos,
    torch::Tensor pair_ret, torch::Tensor half_spread_bps, torch::Tensor session_mask,
    torch::Tensor fill_save, torch::Tensor sign_trade_save,
    double commission_bps, double fill_temp_bps, double fee_bp
) {
    TORCH_CHECK(g_next_pos.is_cuda(), "grads must be CUDA");
    auto B = g_next_pos.size(0);
    auto P = g_next_pos.size(1);

    auto opts = g_next_pos.options();
    auto d_target_pos = torch::empty({B, P}, opts);
    auto d_offset_bps = torch::empty({B, P}, opts);
    auto d_prev_pos   = torch::empty({B, P}, opts);

    const int threads = 256;
    dim3 grid((P + threads - 1) / threads, B);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        g_next_pos.scalar_type(), "pair_step_bwd_cuda", ([&] {
            pair_step_bwd_kernel<scalar_t><<<grid, threads>>>(
                g_next_pos.data_ptr<scalar_t>(),
                g_pair_pnl.data_ptr<scalar_t>(),
                g_turnover.data_ptr<scalar_t>(),
                target_pos.data_ptr<scalar_t>(),
                offset_bps.data_ptr<scalar_t>(),
                prev_pos.data_ptr<scalar_t>(),
                pair_ret.data_ptr<scalar_t>(),
                half_spread_bps.data_ptr<scalar_t>(),
                session_mask.data_ptr<scalar_t>(),
                fill_save.data_ptr<scalar_t>(),
                sign_trade_save.data_ptr<scalar_t>(),
                d_target_pos.data_ptr<scalar_t>(),
                d_offset_bps.data_ptr<scalar_t>(),
                d_prev_pos.data_ptr<scalar_t>(),
                static_cast<int>(B), static_cast<int>(P),
                static_cast<float>(commission_bps),
                static_cast<float>(fill_temp_bps),
                static_cast<float>(fee_bp)
            );
        })
    );
    return {d_target_pos, d_offset_bps, d_prev_pos};
}

} // namespace pair_sim


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pair_step_fwd", &pair_sim::pair_step_fwd_cuda, "Fused pair step forward (fp32/bf16/fp16)");
    m.def("pair_step_bwd", &pair_sim::pair_step_bwd_cuda, "Fused pair step backward (fp32/bf16/fp16)");
}
