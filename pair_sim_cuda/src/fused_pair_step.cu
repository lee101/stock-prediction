// Fused daily pair-trading step, forward + backward.
//
// One thread per (b, p). Produces per-pair outputs; per-batch reductions
// (total_pnl, gross, interest, equity_next) are done in Python via torch.sum
// so they stay differentiable through autograd without needing a second
// kernel for the reduction grads.
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
//
// Backward gradients computed: d/dtarget_pos, d/doffset_bps, d/dprev_pos.
// reach_side_bps, half_spread_bps, pair_ret, session_mask treated as
// constants (no grad flows through market-data inputs).

#include <torch/extension.h>
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

__global__ void pair_step_fwd_kernel(
    const float* __restrict__ target_pos,      // [B, P]
    const float* __restrict__ offset_bps,      // [B, P]
    const float* __restrict__ prev_pos,        // [B, P]
    const float* __restrict__ pair_ret,        // [B, P]
    const float* __restrict__ reach_side_bps,  // [B, P]
    const float* __restrict__ half_spread_bps, // [B, P]
    const float* __restrict__ session_mask,    // [B]
    float* __restrict__ next_pos,              // [B, P]
    float* __restrict__ pair_pnl,              // [B, P]
    float* __restrict__ turnover,              // [B, P]
    float* __restrict__ fill_out,              // [B, P]  (saved)
    float* __restrict__ sign_trade_out,        // [B, P]  (saved)
    int B, int P,
    float commission_bps, float fill_temp_bps, float fee_bp
) {
    int b = blockIdx.y;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || p >= P) return;

    int idx = b * P + p;

    float tp = target_pos[idx];
    float pp = prev_pos[idx];
    float ob = offset_bps[idx];
    float rt = pair_ret[idx];
    float rs = reach_side_bps[idx];
    float hs = half_spread_bps[idx];
    float sm = session_mask[b];

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

    next_pos[idx] = npos;
    pair_pnl[idx] = pnl;
    turnover[idx] = tov;
    fill_out[idx] = fill;
    sign_trade_out[idx] = signf(exec);   // sign of exec, for d|exec|/dexec
}

__global__ void pair_step_bwd_kernel(
    // upstream grads (one per forward output)
    const float* __restrict__ g_next_pos,      // [B, P]
    const float* __restrict__ g_pair_pnl,      // [B, P]
    const float* __restrict__ g_turnover,      // [B, P]
    // saved forward state
    const float* __restrict__ target_pos,      // [B, P]
    const float* __restrict__ offset_bps,      // [B, P]
    const float* __restrict__ prev_pos,        // [B, P]
    const float* __restrict__ pair_ret,        // [B, P]
    const float* __restrict__ half_spread_bps, // [B, P]
    const float* __restrict__ session_mask,    // [B]
    const float* __restrict__ fill_save,       // [B, P]
    const float* __restrict__ sign_trade_save, // [B, P]
    // output grads
    float* __restrict__ d_target_pos,          // [B, P]
    float* __restrict__ d_offset_bps,          // [B, P]
    float* __restrict__ d_prev_pos,            // [B, P]
    int B, int P,
    float commission_bps, float fill_temp_bps, float fee_bp
) {
    int b = blockIdx.y;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || p >= P) return;

    int idx = b * P + p;

    float tp = target_pos[idx];
    float pp = prev_pos[idx];
    float ob = offset_bps[idx];
    float rt = pair_ret[idx];
    float hs = half_spread_bps[idx];
    float sm = session_mask[b];
    float fill = fill_save[idx];            // = sm * sigmoid(signal/T)
    float sign_exec = sign_trade_save[idx];

    float trade = tp - pp;
    float exec = fill * trade;

    // Upstream grads
    float g_np = g_next_pos[idx];
    float g_pn = g_pair_pnl[idx];
    float g_tv = g_turnover[idx];

    // pair_pnl = npos * rt - cost → upstream flowing into npos and cost
    float u_npos = g_np + g_pn * rt;
    float u_cost = -g_pn;

    float c_unit = (commission_bps + hs + fee_bp + 0.5f * ob) * 1e-4f;
    float u_tov = g_tv + u_cost * c_unit;
    float u_cunit = u_cost * fabsf(exec);

    // d tov = sign(exec) * d exec
    float u_exec = u_tov * sign_exec + u_npos;

    // exec = fill * trade
    float u_fill = u_exec * trade;
    float u_trade = u_exec * fill;

    // fill = sm * sigmoid(signal/T), signal = rs - fee_bp - hs - ob
    //   sigma = fill / sm (when sm > 0)
    //   dfill/dsignal = sm * sigma * (1-sigma) / T
    //   dsignal/dob = -1
    // When sm == 0, fill == 0 and the derivative through fill is 0 anyway.
    float dfill_dsignal = 0.f;
    if (sm > 0.f) {
        float sigma = fill / sm;
        dfill_dsignal = sm * sigma * (1.0f - sigma) / fill_temp_bps;
    }
    float u_signal = u_fill * dfill_dsignal;
    float u_ob_from_fill = -u_signal;

    // c_unit depends on ob: dc_unit/dob = 0.5 * 1e-4
    float u_ob_from_cost = u_cunit * (0.5f * 1e-4f);

    float u_ob = u_ob_from_fill + u_ob_from_cost;

    // trade = tp - pp → dtp = +u_trade, dpp += -u_trade
    float u_tp = u_trade;
    float u_pp = u_npos - u_trade;

    d_target_pos[idx] = u_tp;
    d_offset_bps[idx] = u_ob;
    d_prev_pos[idx]   = u_pp;
}


// ---- host wrappers ----

std::vector<torch::Tensor> pair_step_fwd_cuda(
    torch::Tensor target_pos, torch::Tensor offset_bps, torch::Tensor prev_pos,
    torch::Tensor pair_ret, torch::Tensor reach_side_bps,
    torch::Tensor half_spread_bps, torch::Tensor session_mask,
    double commission_bps, double fill_temp_bps, double fee_bp
) {
    TORCH_CHECK(target_pos.is_cuda(), "target_pos must be CUDA");
    TORCH_CHECK(target_pos.scalar_type() == torch::kFloat32, "float32 only");
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
    pair_step_fwd_kernel<<<grid, threads>>>(
        target_pos.data_ptr<float>(),
        offset_bps.data_ptr<float>(),
        prev_pos.data_ptr<float>(),
        pair_ret.data_ptr<float>(),
        reach_side_bps.data_ptr<float>(),
        half_spread_bps.data_ptr<float>(),
        session_mask.data_ptr<float>(),
        next_pos.data_ptr<float>(),
        pair_pnl.data_ptr<float>(),
        turnover.data_ptr<float>(),
        fill.data_ptr<float>(),
        sign_exec.data_ptr<float>(),
        static_cast<int>(B), static_cast<int>(P),
        static_cast<float>(commission_bps),
        static_cast<float>(fill_temp_bps),
        static_cast<float>(fee_bp)
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
    pair_step_bwd_kernel<<<grid, threads>>>(
        g_next_pos.data_ptr<float>(),
        g_pair_pnl.data_ptr<float>(),
        g_turnover.data_ptr<float>(),
        target_pos.data_ptr<float>(),
        offset_bps.data_ptr<float>(),
        prev_pos.data_ptr<float>(),
        pair_ret.data_ptr<float>(),
        half_spread_bps.data_ptr<float>(),
        session_mask.data_ptr<float>(),
        fill_save.data_ptr<float>(),
        sign_trade_save.data_ptr<float>(),
        d_target_pos.data_ptr<float>(),
        d_offset_bps.data_ptr<float>(),
        d_prev_pos.data_ptr<float>(),
        static_cast<int>(B), static_cast<int>(P),
        static_cast<float>(commission_bps),
        static_cast<float>(fill_temp_bps),
        static_cast<float>(fee_bp)
    );
    return {d_target_pos, d_offset_bps, d_prev_pos};
}

} // namespace pair_sim


// Pybind registration
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pair_step_fwd", &pair_sim::pair_step_fwd_cuda, "Fused pair step forward");
    m.def("pair_step_bwd", &pair_sim::pair_step_bwd_cuda, "Fused pair step backward");
}
